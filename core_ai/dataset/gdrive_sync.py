"""
ORBITA Google Drive Dataset Video Sync
======================================
Provides seamless synchronization and cloud ingestion of reference videos from Google Drive
into the local `vdata/` catalog. This enables deployed instances of ORBITA (on Jetson,
remote servers, cloud VMs, or containerized environments) to access the complete video dataset
without requiring manual local copies.

Default Reference Folder:
https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing
Folder ID: 1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("orbita.gdrive_sync")

DEFAULT_GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing"
DEFAULT_GDRIVE_FOLDER_ID = "1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C"


def extract_folder_id(url_or_id: str) -> str:
    """Extracts the alphanumeric Google Drive folder ID from a URL or raw ID string."""
    if not url_or_id:
        return DEFAULT_GDRIVE_FOLDER_ID
    cleaned = url_or_id.strip()
    # Match /folders/<ID>
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    # Match id=<ID>
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)
    # If no URL pattern, assume raw ID
    return cleaned


class GDriveSyncManager:
    """
    Manages querying, syncing, and downloading reference videos from Google Drive into vdata/.
    """

    def __init__(self, target_dir: str | Path = "vdata"):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None

        self._status: Dict[str, Any] = {
            "status": "idle",  # idle, in_progress, completed, error
            "folder_url": DEFAULT_GDRIVE_FOLDER_URL,
            "folder_id": DEFAULT_GDRIVE_FOLDER_ID,
            "progress_pct": 0,
            "current_file": None,
            "downloaded_count": 0,
            "total_files": 0,
            "message": "Ready to sync videos from Google Drive.",
            "last_synced_at": None,
            "files": [],
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current sync state, progress, and file catalog."""
        with self._lock:
            # Refresh local existence for each known file
            files_status = []
            for f in self._status.get("files", []):
                local_path = self.target_dir / f.get("name", "")
                exists = local_path.exists()
                size_mb = round(local_path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
                files_status.append({
                    **f,
                    "exists_locally": exists,
                    "local_size_mb": size_mb,
                })

            # Count local files in vdata
            local_videos = [
                p.name for p in self.target_dir.glob("*.mp4")
            ]

            return {
                **self._status,
                "files": files_status,
                "local_video_count": len(local_videos),
                "local_videos": local_videos,
            }

    def list_remote_files(self, folder_url_or_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Queries Google Drive for the files located in the target folder without downloading.
        """
        folder_id = extract_folder_id(folder_url_or_id or self._status["folder_id"])
        logger.info(f"Querying Google Drive folder ID: {folder_id}")

        try:
            import gdown

            # Use skip_download=True to inspect metadata without downloading contents
            drive_files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True)
            results = []
            if drive_files:
                for item in drive_files:
                    name = getattr(item, "path", "") or getattr(item, "local_path", "")
                    basename = Path(name).name
                    file_id = getattr(item, "id", "")
                    local_p = self.target_dir / basename
                    exists = local_p.exists()
                    size_mb = round(local_p.stat().st_size / (1024 * 1024), 2) if exists else 0.0

                    results.append({
                        "id": file_id,
                        "name": basename,
                        "exists_locally": exists,
                        "local_size_mb": size_mb,
                        "download_url": f"https://drive.google.com/uc?id={file_id}",
                    })

            with self._lock:
                self._status["folder_id"] = folder_id
                self._status["files"] = results
                self._status["total_files"] = len(results)

            return results
        except Exception as e:
            logger.error(f"Error querying Google Drive folder {folder_id}: {e}", exc_info=True)
            with self._lock:
                self._status["message"] = f"Failed to list Drive files: {str(e)}"
            return []

    def start_sync(self, folder_url_or_id: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Starts an asynchronous background sync to download missing reference videos from Google Drive.
        """
        with self._lock:
            if self._status["status"] == "in_progress":
                return {"status": "in_progress", "message": "Sync is already in progress."}

            folder_id = extract_folder_id(folder_url_or_id or self._status["folder_id"])
            folder_url = folder_url_or_id or DEFAULT_GDRIVE_FOLDER_URL
            self._status["status"] = "in_progress"
            self._status["folder_id"] = folder_id
            self._status["folder_url"] = folder_url
            self._status["progress_pct"] = 5
            self._status["message"] = "Initializing Google Drive sync..."
            self._status["downloaded_count"] = 0

        self._sync_thread = threading.Thread(
            target=self._run_sync_worker,
            args=(folder_id, force),
            daemon=True,
            name="GDriveSyncWorker"
        )
        self._sync_thread.start()

        return {"status": "started", "folder_id": folder_id, "message": "Google Drive sync started in background."}

    def _run_sync_worker(self, folder_id: str, force: bool) -> None:
        """Background worker thread function."""
        try:
            import gdown

            with self._lock:
                self._status["message"] = f"Resolving file list for Google Drive folder {folder_id}..."
                self._status["progress_pct"] = 10

            logger.info(f"Starting GDrive sync worker for folder: {folder_id}")
            remote_files = self.list_remote_files(folder_id)

            if not remote_files:
                logger.info("Attempting direct folder download via gdown...")
                with self._lock:
                    self._status["message"] = "Downloading folder contents..."
                    self._status["progress_pct"] = 25

                gdown.download_folder(
                    id=folder_id,
                    output=str(self.target_dir),
                    quiet=False,
                    use_cookies=False,
                )
                with self._lock:
                    self._status["status"] = "completed"
                    self._status["progress_pct"] = 100
                    self._status["message"] = "Google Drive folder sync completed."
                    self._status["last_synced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return

            total = len(remote_files)
            downloaded = 0

            for idx, file_info in enumerate(remote_files):
                file_id = file_info["id"]
                file_name = file_info["name"]
                dest_path = self.target_dir / file_name

                with self._lock:
                    self._status["current_file"] = file_name
                    self._status["progress_pct"] = int(15 + (idx / total) * 80)
                    self._status["message"] = f"Checking file {idx + 1}/{total}: {file_name}"

                if dest_path.exists() and not force:
                    if dest_path.stat().st_size > 1024 * 1024:
                        logger.info(f"File {file_name} already exists locally ({dest_path.stat().st_size} bytes). Skipping.")
                        downloaded += 1
                        continue

                logger.info(f"Downloading {file_name} (ID: {file_id}) from Google Drive...")
                with self._lock:
                    self._status["message"] = f"Downloading ({idx + 1}/{total}): {file_name}..."

                gdown.download(
                    id=file_id,
                    output=str(dest_path),
                    quiet=False,
                )

                downloaded += 1
                with self._lock:
                    self._status["downloaded_count"] = downloaded
                    self._status["progress_pct"] = int(15 + (downloaded / total) * 80)

            with self._lock:
                self._status["status"] = "completed"
                self._status["progress_pct"] = 100
                self._status["current_file"] = None
                self._status["downloaded_count"] = downloaded
                self._status["total_files"] = total
                self._status["message"] = f"Successfully synchronized {downloaded}/{total} dataset videos."
                self._status["last_synced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"GDrive sync completed: {downloaded}/{total} videos synced into {self.target_dir}")

        except Exception as e:
            logger.error(f"GDrive sync worker error: {e}", exc_info=True)
            with self._lock:
                self._status["status"] = "error"
                self._status["message"] = f"Drive sync failed: {str(e)}"
                self._status["current_file"] = None

    def sync_single_file(self, file_id: str, file_name: str) -> bool:
        """Synchronizes an individual file by ID."""
        try:
            import gdown
            dest_path = self.target_dir / file_name
            logger.info(f"Synchronizing single file {file_name} ({file_id})...")
            gdown.download(id=file_id, output=str(dest_path), quiet=False)
            return dest_path.exists() and dest_path.stat().st_size > 0
        except Exception as e:
            logger.error(f"Failed to sync file {file_name}: {e}")
            return False


# Global singleton instance for application-wide use
gdrive_sync_manager = GDriveSyncManager(target_dir="vdata")
