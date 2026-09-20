"""
ORBITA Google Drive Online Dataset Video Streaming & Catalog System
====================================================================
Provides online streaming access and catalog management for reference videos stored
in Google Drive without requiring videos to be downloaded to local disk.

Target Architecture:
Google Drive -> Scan catalog -> Backend Streaming Proxy (HTTP Range / 206 Partial Content) -> Browser <video>

Default Reference Folder:
https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing
Folder ID: 1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import requests

logger = logging.getLogger("orbita.gdrive_sync")

DEFAULT_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing"
DEFAULT_GDRIVE_FOLDER_ID = "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"

# Supported video extensions for catalog and streaming
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}

# Known reference videos belonging to the authorized folder (used for fast fallback/validation)
KNOWN_FOLDER_VIDEOS: Dict[str, str] = {
    "1k202zi-kg68lc5Nr5RR-FhW_neqEP8OW": "20260905_145858.mp4",
    "1obN6u5P8Sg_D1YY0upIviKGIve7CKESB": "20260905_145948.mp4",
    "1mGW3PPEalXPSeWh8JcIF4Q7jiDeWv6i2": "20260905_150132.mp4",
    "17fk3NgghzUEBKj8W5Ms5cyaWbVYwVYbH": "20260908_135006.mp4",
}


def extract_folder_id(url_or_id: str) -> str:
    """Extracts the alphanumeric Google Drive folder ID from a URL or raw ID string."""
    if not url_or_id:
        return DEFAULT_GDRIVE_FOLDER_ID
    cleaned = url_or_id.strip()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned


def validate_file_id(file_id: str) -> bool:
    """Validates that a file_id matches standard Google Drive ID format (alphanumeric, -, _)."""
    if not file_id or not isinstance(file_id, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]{15,60}$", file_id.strip()))


class GDriveSyncManager:
    """
    Manages online scanning, metadata cataloging, and range-request proxy streaming
    for Google Drive reference videos.
    """

    def __init__(self, target_dir: str | Path = "vdata"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory catalog of authorized videos: file_id -> metadata dict
        self._catalog: Dict[str, Dict[str, Any]] = {}
        # Authorized file IDs for security gating
        self._authorized_file_ids: Set[str] = set(KNOWN_FOLDER_VIDEOS.keys())

        # Direct stream URL cache: file_id -> (direct_url, expiry_timestamp)
        self._stream_url_cache: Dict[str, Tuple[str, float]] = {}

        # Shared HTTP session with custom desktop browser user agent
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        self._status: Dict[str, Any] = {
            "status": "online",
            "folder_url": DEFAULT_GDRIVE_FOLDER_URL,
            "folder_id": DEFAULT_GDRIVE_FOLDER_ID,
            "message": "Ready for Google Drive online streaming.",
            "last_scanned_at": None,
            "online_video_count": len(KNOWN_FOLDER_VIDEOS),
            "local_video_count": 0,
            "videos": [],
        }

        # Initialize catalog with default known videos
        self._seed_catalog()

    def _seed_catalog(self) -> None:
        """Populates initial catalog from known folder videos."""
        videos = []
        for fid, fname in KNOWN_FOLDER_VIDEOS.items():
            local_p = self.target_dir / fname
            exists = local_p.exists()
            size_mb = round(local_p.stat().st_size / (1024 * 1024), 2) if exists else 0.0
            meta = {
                "id": fid,
                "name": fname,
                "mimeType": "video/mp4",
                "size_mb": size_mb or 115.0,
                "driveUrl": f"https://drive.google.com/file/d/{fid}/view",
                "playbackUrl": f"/api/vdata/gdrive/video/{fid}",
                "status": "online",
                "exists_locally": exists,
                "local_size_mb": size_mb,
            }
            self._catalog[fid] = meta
            videos.append(meta)
        self._status["videos"] = videos

    def is_file_authorized(self, file_id: str) -> bool:
        """
        Security verification: checks if the requested file_id belongs to the authorized
        Google Drive dataset folder. Prevents arbitrary SSRF or open proxy abuse.
        """
        if not validate_file_id(file_id):
            return False
        with self._lock:
            if file_id in self._authorized_file_ids or file_id in self._catalog:
                return True
        # If not found in memory, try refreshing catalog once
        self.list_remote_videos(force_refresh=True)
        with self._lock:
            return file_id in self._authorized_file_ids or file_id in self._catalog

    def list_remote_videos(
        self, folder_url_or_id: Optional[str] = None, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Queries Google Drive folder metadata without downloading video contents.
        Returns a structured list of video items with playback URLs and online statuses.
        """
        folder_id = extract_folder_id(folder_url_or_id or self._status["folder_id"])

        with self._lock:
            if not force_refresh and self._catalog and folder_id == self._status["folder_id"]:
                return list(self._catalog.values())

        logger.info("Scanning Google Drive folder catalog for folder ID: %s", folder_id)

        try:
            import gdown

            # skip_download=True inspects folder metadata without downloading files
            drive_files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
            results = []

            with self._lock:
                new_auth_ids = set()
                if drive_files:
                    for item in drive_files:
                        name = getattr(item, "path", "") or getattr(item, "local_path", "")
                        basename = Path(name).name
                        ext = Path(name).suffix.lower()

                        # Only include valid video files (prefer MP4)
                        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                            continue

                        fid = getattr(item, "id", "")
                        if not fid:
                            continue

                        new_auth_ids.add(fid)
                        local_p = self.target_dir / basename
                        exists = local_p.exists()
                        size_mb = round(local_p.stat().st_size / (1024 * 1024), 2) if exists else 0.0

                        meta = {
                            "id": fid,
                            "name": basename,
                            "mimeType": "video/mp4" if ext == ".mp4" else f"video/{ext.lstrip('.')}",
                            "size_mb": size_mb or 115.0,
                            "driveUrl": f"https://drive.google.com/file/d/{fid}/view",
                            "playbackUrl": f"/api/vdata/gdrive/video/{fid}",
                            "status": "online",
                            "exists_locally": exists,
                            "local_size_mb": size_mb,
                        }
                        self._catalog[fid] = meta
                        results.append(meta)

                if new_auth_ids:
                    self._authorized_file_ids.update(new_auth_ids)

                self._status["folder_id"] = folder_id
                self._status["status"] = "online"
                self._status["videos"] = results if results else list(self._catalog.values())
                self._status["online_video_count"] = len(self._status["videos"])
                self._status["last_scanned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._status["message"] = f"Successfully scanned {len(self._status['videos'])} online videos from Google Drive."

            return self._status["videos"]

        except Exception as e:
            logger.error("Failed to scan Google Drive folder %s: %s", folder_id, e, exc_info=True)
            with self._lock:
                self._status["status"] = "online"
                self._status["message"] = f"Drive scan error: {str(e)}. Using cached catalog."
                return list(self._catalog.values())

    def _resolve_direct_download_url(self, file_id: str) -> str:
        """
        Resolves Google Drive confirmation token / virus warning to obtain
        the direct chunk-streaming download URL. Caches the URL for 30 minutes.
        """
        now = time.time()
        with self._lock:
            cached = self._stream_url_cache.get(file_id)
            if cached and cached[1] > now:
                return cached[0]

        drive_init_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        res = self._session.get(drive_init_url, stream=True, timeout=10)

        direct_url = drive_init_url
        if "text/html" in res.headers.get("Content-Type", ""):
            import gdown

            mod = sys.modules.get(gdown.download.__module__)
            if mod and hasattr(mod, "get_url_from_gdrive_confirmation"):
                try:
                    direct_url = mod.get_url_from_gdrive_confirmation(res.text)
                except Exception as parse_err:
                    logger.warning("Could not parse gdrive confirmation URL: %s. Using default.", parse_err)

        # Cache direct URL for 30 minutes
        with self._lock:
            self._stream_url_cache[file_id] = (direct_url, now + 1800)

        return direct_url

    def get_stream_response(
        self, file_id: str, range_header: Optional[str] = None
    ) -> Tuple[Generator[bytes, None, None], Dict[str, str], int]:
        """
        Streams video bytes from Google Drive directly to the client with full HTTP Range support.
        Does NOT download the entire video into RAM or write to local disk.

        Returns:
            (chunk_generator, response_headers, status_code)
        """
        if not self.is_file_authorized(file_id):
            raise PermissionError(f"File ID '{file_id}' is not an authorized video in dataset folder {DEFAULT_GDRIVE_FOLDER_ID}.")

        direct_url = self._resolve_direct_download_url(file_id)

        req_headers = {}
        if range_header:
            req_headers["Range"] = range_header

        upstream = self._session.get(direct_url, headers=req_headers, stream=True, timeout=15)

        # If Google Drive returned an HTML page (e.g. cookie expired), invalidate cache and retry once
        if "text/html" in upstream.headers.get("Content-Type", ""):
            with self._lock:
                self._stream_url_cache.pop(file_id, None)
            direct_url = self._resolve_direct_download_url(file_id)
            upstream = self._session.get(direct_url, headers=req_headers, stream=True, timeout=15)

        status_code = upstream.status_code
        # 206 Partial Content or 200 OK
        if status_code not in (200, 206):
            logger.warning("Upstream Google Drive returned unexpected status %d for file %s", status_code, file_id)

        headers: Dict[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
        }

        if "Content-Range" in upstream.headers:
            headers["Content-Range"] = upstream.headers["Content-Range"]
        if "Content-Length" in upstream.headers:
            headers["Content-Length"] = upstream.headers["Content-Length"]

        def stream_chunks() -> Generator[bytes, None, None]:
            try:
                for chunk in upstream.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        yield chunk
            except Exception as stream_err:
                logger.debug("Client closed video stream connection: %s", stream_err)
            finally:
                upstream.close()

        return stream_chunks(), headers, status_code

    def get_status(self) -> Dict[str, Any]:
        """Returns catalog telemetry and online video counts."""
        with self._lock:
            local_videos = [p.name for p in self.target_dir.glob("*.mp4")]
            videos_list = list(self._catalog.values())

            # Refresh local existence flag
            for v in videos_list:
                local_p = self.target_dir / v["name"]
                v["exists_locally"] = local_p.exists()
                v["local_size_mb"] = (
                    round(local_p.stat().st_size / (1024 * 1024), 2) if local_p.exists() else 0.0
                )

            return {
                **self._status,
                "online_video_count": len(videos_list),
                "local_video_count": len(local_videos),
                "total_files": len(videos_list),
                "videos": videos_list,
                "files": videos_list,  # Backwards-compatible alias
                "local_videos": local_videos,
            }

    # Optional secondary utility for offline testing only
    def sync_single_file(self, file_id: str, file_name: str) -> bool:
        """Downloads a single video file on explicit user request for offline use."""
        try:
            import gdown

            dest = self.target_dir / file_name
            gdown.download(id=file_id, output=str(dest), quiet=True)
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:
            logger.error("Failed to download offline copy of %s: %s", file_name, e)
            return False


# Global singleton instance
gdrive_sync_manager = GDriveSyncManager(target_dir="vdata")
