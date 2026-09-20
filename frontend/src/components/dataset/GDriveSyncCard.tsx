import { useState, useEffect, useRef } from 'react';
import {
  Cloud,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  FileVideo,
  Play,
  X,
  Radio,
  Download,
} from 'lucide-react';
import { BACKEND_BASE } from '../../api/camera';

export interface DriveVideo {
  id: string;
  name: string;
  mimeType: string;
  size_mb: number;
  driveUrl: string;
  playbackUrl: string;
  status: string;
  exists_locally?: boolean;
  local_size_mb?: number;
}

interface GDriveStatus {
  status: 'online' | 'error' | 'idle';
  folder_url: string;
  folder_id: string;
  online_video_count: number;
  local_video_count: number;
  message: string;
  last_scanned_at: string | null;
  videos: DriveVideo[];
}

const DEFAULT_GDRIVE_URL =
  'https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing';

export default function GDriveSyncCard({
  onSyncComplete,
  onSelectVideoForExperiment,
}: {
  onSyncComplete?: () => void;
  onSelectVideoForExperiment?: (video: DriveVideo) => void;
}) {
  const [status, setStatus] = useState<GDriveStatus | null>(null);
  const [folderUrl, setFolderUrl] = useState(DEFAULT_GDRIVE_URL);
  const [scanning, setScanning] = useState(false);
  const [activeVideo, setActiveVideo] = useState<DriveVideo | null>(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${BACKEND_BASE}/api/vdata/gdrive/status`);
      if (res.ok) {
        const data: GDriveStatus = await res.json();
        setStatus(data);
        if (data.folder_url && data.folder_url !== folderUrl && folderUrl === DEFAULT_GDRIVE_URL) {
          setFolderUrl(data.folder_url);
        }
      }
    } catch (err: any) {
      console.warn('Backend unavailable or reconnecting:', err.message);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleScan = async () => {
    setScanning(true);
    setNotification({ type: 'info', text: 'Scanning Google Drive folder catalog...' });
    try {
      const res = await fetch(`${BACKEND_BASE}/api/vdata/gdrive/list`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_url: folderUrl }),
      });
      if (res.ok) {
        const data = await res.json();
        const count = data.count || data.videos?.length || 0;
        setNotification({
          type: 'success',
          text: `Found ${count} reference videos available online from Google Drive.`,
        });
        fetchStatus();
        onSyncComplete?.();
      } else {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
      }
    } catch (e: any) {
      const msg = e.message.includes('Failed to fetch')
        ? 'Backend unavailable. Please verify the ORBITA server is running.'
        : `Drive scan failed: ${e.message}`;
      setNotification({ type: 'error', text: msg });
    } finally {
      setScanning(false);
    }
  };

  const handlePlayVideo = (video: DriveVideo) => {
    setActiveVideo(video);
    setVideoLoading(true);
    setVideoError(null);
    onSelectVideoForExperiment?.(video);
  };

  const handleDownloadOffline = async (fileId: string, fileName: string) => {
    setNotification({ type: 'info', text: `Initiating offline download for ${fileName}...` });
    try {
      const res = await fetch(`${BACKEND_BASE}/api/vdata/gdrive/pull_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: fileId, file_name: fileName }),
      });
      if (res.ok) {
        setNotification({ type: 'success', text: `Saved ${fileName} locally for offline development.` });
        fetchStatus();
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (e: any) {
      setNotification({ type: 'error', text: `Offline download error: ${e.message}` });
    }
  };

  const videos = status?.videos || [];
  const onlineCount = status?.online_video_count ?? videos.length;
  const localCount = status?.local_video_count ?? 0;

  return (
    <div className="bg-[#0e141f] border border-cyan-900/50 rounded-xl p-5 text-white space-y-4 shadow-lg shadow-cyan-950/20">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1b2536] pb-3">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-cyan-950/80 border border-cyan-800/60 text-cyan-400 shadow-sm">
            <Cloud className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold tracking-wider uppercase text-cyan-300">
                Google Drive Online Dataset
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-700 flex items-center gap-1">
                <Radio className="w-2.5 h-2.5 text-cyan-400 animate-pulse" />
                ONLINE STREAMING
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Stream reference experiment videos online from Google Drive without local disk storage.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/80 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            {onlineCount} ONLINE VIDEOS
          </span>
          {localCount > 0 && (
            <span className="px-2 py-1 text-xs font-mono rounded bg-space-800 text-slate-300 border border-space-600">
              {localCount} LOCAL
            </span>
          )}
        </div>
      </div>

      {/* Notification Toast */}
      {notification && (
        <div
          className={`p-2.5 rounded-lg text-xs font-mono flex items-center justify-between transition-all ${
            notification.type === 'success'
              ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-700'
              : notification.type === 'error'
              ? 'bg-rose-950/80 text-rose-300 border border-rose-700'
              : 'bg-cyan-950/80 text-cyan-300 border border-cyan-700'
          }`}
        >
          <span>{notification.text}</span>
          <button onClick={() => setNotification(null)} className="ml-3 hover:opacity-75 font-bold text-sm">
            ×
          </button>
        </div>
      )}

      {/* Folder Configuration Bar */}
      <div className="space-y-2.5 bg-[#080d14] p-3.5 rounded-lg border border-[#182333]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
          <label className="text-xs font-mono text-slate-300 flex items-center gap-1.5">
            <span>GOOGLE DRIVE FOLDER:</span>
            <span className="text-[10px] text-slate-500 font-normal">(Shared reference folder)</span>
          </label>
          <div className="flex items-center gap-2 text-[11px] font-mono text-cyan-400">
            <span>FOLDER ID:</span>
            <span className="px-1.5 py-0.5 rounded bg-cyan-950/70 border border-cyan-800 text-cyan-300">
              {status?.folder_id || '1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C'}
            </span>
            <a
              href={folderUrl}
              target="_blank"
              rel="noreferrer"
              className="hover:underline flex items-center gap-1 text-cyan-400 ml-1 font-semibold"
              title="Open Google Drive folder in new tab"
            >
              <span>Drive</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
          <input
            type="text"
            value={folderUrl}
            onChange={(e) => setFolderUrl(e.target.value)}
            placeholder="https://drive.google.com/drive/folders/..."
            className="flex-1 bg-[#05080e] border border-cyan-900/60 rounded px-3 py-2 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-400"
          />

          <button
            onClick={handleScan}
            disabled={scanning}
            className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-sky-600 hover:from-cyan-500 hover:to-sky-500 text-slate-950 font-bold rounded text-xs font-mono transition flex items-center justify-center gap-2 disabled:opacity-50 shrink-0 shadow-md shadow-cyan-950/50"
            title="Scan Google Drive folder metadata"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${scanning ? 'animate-spin' : ''}`} />
            <span>{scanning ? 'Scanning Drive...' : 'SCAN GOOGLE DRIVE'}</span>
          </button>
        </div>
      </div>

      {/* Online Video Player Modal / Active Preview Panel */}
      {activeVideo && (
        <div className="p-4 bg-[#070b12] border border-cyan-500/40 rounded-xl space-y-3 shadow-2xl animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-cyan-950 pb-2.5">
            <div className="flex items-center gap-2 min-w-0">
              <span className="p-1.5 rounded bg-cyan-950 text-cyan-400">
                <Play className="w-3.5 h-3.5 fill-cyan-400" />
              </span>
              <div className="min-w-0">
                <div className="text-xs font-mono font-bold text-white truncate flex items-center gap-2">
                  <span>{activeVideo.name}</span>
                  <span className="px-2 py-0.5 rounded text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800">
                    {activeVideo.size_mb} MB
                  </span>
                </div>
                <div className="text-[10px] font-mono text-cyan-400 flex items-center gap-1.5 mt-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  Streaming from Google Drive (HTTP Range / 206 Partial Content)
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <a
                href={activeVideo.driveUrl}
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1 rounded bg-[#111927] hover:bg-[#182337] border border-space-600 text-[11px] font-mono text-cyan-400 flex items-center gap-1"
              >
                <span>Drive</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              <button
                onClick={() => setActiveVideo(null)}
                className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition"
                title="Close player"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* HTML5 Video Element with Full HTTP Range Seeking */}
          <div className="relative bg-black rounded-lg overflow-hidden aspect-video max-h-[460px] flex items-center justify-center">
            {videoLoading && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60 backdrop-blur-xs text-xs font-mono text-cyan-400 gap-2">
                <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
                <span>Buffering Google Drive stream...</span>
              </div>
            )}

            {videoError ? (
              <div className="p-6 text-center text-xs font-mono text-rose-300 space-y-2">
                <AlertCircle className="w-8 h-8 text-rose-400 mx-auto" />
                <div className="font-bold">Streaming Error</div>
                <p className="text-slate-400 max-w-md">{videoError}</p>
                <button
                  onClick={() => handlePlayVideo(activeVideo)}
                  className="px-3 py-1 bg-rose-950 border border-rose-700 rounded hover:bg-rose-900 text-xs text-rose-200"
                >
                  Retry Playback
                </button>
              </div>
            ) : (
              <video
                ref={videoRef}
                controls
                autoPlay
                preload="metadata"
                playsInline
                src={`${BACKEND_BASE}${activeVideo.playbackUrl}`}
                onLoadStart={() => setVideoLoading(true)}
                onLoadedData={() => setVideoLoading(false)}
                onWaiting={() => setVideoLoading(true)}
                onPlaying={() => setVideoLoading(false)}
                onError={() => {
                  setVideoLoading(false);
                  setVideoError(
                    'Unable to stream video from Google Drive. Please check network connectivity or backend streaming endpoint.'
                  );
                }}
                className="w-full h-full object-contain"
              />
            )}
          </div>
        </div>
      )}

      {/* Online Video Catalog Cards */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-300 flex items-center gap-1.5">
            <FileVideo className="w-3.5 h-3.5 text-cyan-400" />
            <span>ONLINE VIDEOS IN GOOGLE DRIVE</span>
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            {onlineCount} videos ready for online playback
          </span>
        </div>

        <div className="border border-[#1b2536] rounded-lg overflow-hidden bg-[#0a0f17] divide-y divide-[#182333]">
          {videos.length > 0 ? (
            videos.map((video) => (
              <div
                key={video.id}
                className={`p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 transition ${
                  activeVideo?.id === video.id ? 'bg-cyan-950/30' : 'hover:bg-[#0f1724]'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="p-2 rounded bg-space-900 border border-space-700 text-cyan-400 shrink-0">
                    <FileVideo className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-mono font-bold text-slate-100 truncate">
                      {video.name}
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 flex items-center gap-2 mt-0.5">
                      <span>{video.size_mb} MB</span>
                      <span>•</span>
                      <span>Google Drive</span>
                      <span>•</span>
                      <span className="text-emerald-400 font-semibold flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                        ONLINE
                      </span>
                      {video.exists_locally && (
                        <>
                          <span>•</span>
                          <span className="text-slate-400">Local Copy</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
                  <button
                    onClick={() => handlePlayVideo(video)}
                    className="px-3 py-1.5 bg-gradient-to-r from-cyan-600 to-sky-600 hover:from-cyan-500 hover:to-sky-500 text-slate-950 font-bold rounded text-xs font-mono flex items-center gap-1.5 shadow-sm transition"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    <span>PLAY</span>
                  </button>

                  <a
                    href={video.driveUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="px-2.5 py-1.5 bg-space-800 hover:bg-space-700 text-slate-300 border border-space-600 rounded text-xs font-mono flex items-center gap-1 transition"
                    title="Open directly in Google Drive"
                  >
                    <span>DRIVE</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>

                  {/* Optional secondary offline download */}
                  {!video.exists_locally && (
                    <button
                      onClick={() => handleDownloadOffline(video.id, video.name)}
                      className="px-2 py-1.5 text-slate-400 hover:text-slate-200 hover:bg-space-800 rounded text-xs font-mono transition"
                      title="Save optional offline copy for local testing"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="p-6 text-center text-xs font-mono text-slate-400 space-y-2">
              <div>No online videos listed yet.</div>
              <button
                onClick={handleScan}
                disabled={scanning}
                className="px-3 py-1.5 bg-cyan-900/40 hover:bg-cyan-900/60 border border-cyan-700 text-cyan-300 rounded font-bold"
              >
                Scan Google Drive Now
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
