import { useState, useEffect } from 'react';
import {
  Cloud,
  Download,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  FileVideo,
  ArrowDownToLine,
} from 'lucide-react';
import { BACKEND_BASE } from '../../api/camera';

interface DriveFile {
  id: string;
  name: string;
  exists_locally: boolean;
  local_size_mb: number;
  download_url: string;
}

interface GDriveStatus {
  status: 'idle' | 'in_progress' | 'completed' | 'error';
  folder_url: string;
  folder_id: string;
  progress_pct: number;
  current_file: string | null;
  downloaded_count: number;
  total_files: number;
  message: string;
  last_synced_at: string | null;
  files: DriveFile[];
  local_video_count: number;
  local_videos: string[];
}

const DEFAULT_GDRIVE_URL =
  'https://drive.google.com/drive/folders/1kGpbXynDD5PdaklODHo09ZIS-8sdVk2C?usp=sharing';

export default function GDriveSyncCard({ onSyncComplete }: { onSyncComplete?: () => void }) {
  const [status, setStatus] = useState<GDriveStatus | null>(null);
  const [folderUrl, setFolderUrl] = useState(DEFAULT_GDRIVE_URL);
  const [force, setForce] = useState(false);
  const [loading, setLoading] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [notification, setNotification] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);

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
    } catch (err) {
      console.error('Failed to query GDrive sync status:', err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
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
        setNotification({
          type: 'success',
          text: `Found ${data.count} dataset videos in Google Drive folder.`,
        });
        fetchStatus();
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (e: any) {
      setNotification({ type: 'error', text: `Scan failed: ${e.message}` });
    } finally {
      setScanning(false);
    }
  };

  const handleSyncAll = async () => {
    setLoading(true);
    setNotification({ type: 'info', text: 'Initiating Google Drive video synchronization...' });
    try {
      const res = await fetch(`${BACKEND_BASE}/api/vdata/gdrive/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_url: folderUrl, force }),
      });
      const data = await res.json();
      if (res.ok) {
        setNotification({
          type: 'success',
          text: data.message || 'Drive synchronization running in background.',
        });
        fetchStatus();
        onSyncComplete?.();
      } else {
        throw new Error(data.message || `HTTP ${res.status}`);
      }
    } catch (e: any) {
      setNotification({ type: 'error', text: `Sync failed: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  const handlePullFile = async (fileId: string, fileName: string) => {
    setNotification({ type: 'info', text: `Downloading ${fileName} from Google Drive...` });
    try {
      const res = await fetch(`${BACKEND_BASE}/api/vdata/gdrive/pull_file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: fileId, file_name: fileName }),
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setNotification({ type: 'success', text: `Successfully downloaded ${fileName} into vdata/.` });
        fetchStatus();
        onSyncComplete?.();
      } else {
        throw new Error(data.message || 'Download error');
      }
    } catch (e: any) {
      setNotification({ type: 'error', text: `Download failed: ${e.message}` });
    }
  };

  const isSyncing = status?.status === 'in_progress';
  const totalFiles = status?.total_files || (status?.files?.length ?? 4);
  const localCount = status?.local_video_count ?? 0;

  return (
    <div className="bg-[#0e141f] border border-cyan-900/50 rounded-xl p-5 text-white space-y-4 shadow-lg shadow-cyan-950/20">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#1b2536] pb-3">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-cyan-950/80 border border-cyan-800/60 text-cyan-400">
            <Cloud className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold tracking-wider uppercase text-cyan-300">
                Google Drive Dataset Ingestion
              </h3>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-700">
                CLOUD SYNC
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Access and sync reference videos directly from Google Drive for deployed environments.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isSyncing ? (
            <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded bg-amber-950/80 text-amber-300 border border-amber-700/80 flex items-center gap-1.5 animate-pulse">
              <RefreshCw className="w-3 h-3 animate-spin" />
              SYNCING ({status?.progress_pct ?? 0}%)
            </span>
          ) : localCount > 0 ? (
            <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/80 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              {localCount} VIDEOS ACTIVE
            </span>
          ) : (
            <span className="px-2.5 py-1 text-xs font-mono font-semibold rounded bg-rose-950/80 text-rose-300 border border-rose-700/80 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
              NO LOCAL VIDEOS
            </span>
          )}
        </div>
      </div>

      {/* Notification Toast */}
      {notification && (
        <div
          className={`p-2.5 rounded-lg text-xs font-mono flex items-center justify-between ${
            notification.type === 'success'
              ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-700'
              : notification.type === 'error'
              ? 'bg-rose-950/80 text-rose-300 border border-rose-700'
              : 'bg-cyan-950/80 text-cyan-300 border border-cyan-700'
          }`}
        >
          <span>{notification.text}</span>
          <button onClick={() => setNotification(null)} className="ml-3 hover:opacity-75 font-bold">
            ×
          </button>
        </div>
      )}

      {/* Folder Configuration & Trigger */}
      <div className="space-y-3 bg-[#080d14] p-3.5 rounded-lg border border-[#182333]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
          <label className="text-xs font-mono text-slate-300 flex items-center gap-1.5">
            <span>GOOGLE DRIVE FOLDER URL:</span>
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
              className="hover:underline flex items-center gap-1 text-cyan-400 ml-1"
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

          <div className="flex items-center gap-2 shrink-0">
            <button
              onClick={handleScan}
              disabled={scanning || isSyncing}
              className="px-3 py-2 bg-[#141e2e] hover:bg-[#1b293d] border border-cyan-800/60 rounded text-xs font-mono text-cyan-300 transition flex items-center gap-1.5 disabled:opacity-50"
              title="Scan Drive files without downloading"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${scanning ? 'animate-spin' : ''}`} />
              <span>{scanning ? 'Scanning...' : 'Scan Catalog'}</span>
            </button>

            <button
              onClick={handleSyncAll}
              disabled={loading || isSyncing}
              className="px-3.5 py-2 bg-gradient-to-r from-cyan-600 to-sky-600 hover:from-cyan-500 hover:to-sky-500 text-slate-950 font-bold rounded text-xs font-mono transition flex items-center gap-1.5 disabled:opacity-50 shadow-md shadow-cyan-950/50"
            >
              <ArrowDownToLine className="w-3.5 h-3.5" />
              <span>{isSyncing ? 'Syncing...' : 'Sync All Videos'}</span>
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-1">
          <label className="flex items-center gap-1.5 cursor-pointer hover:text-slate-300">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-0 focus:ring-offset-0"
            />
            <span>Force re-download (overwrite existing local files)</span>
          </label>

          {status?.last_synced_at && (
            <span>Last synced: <strong className="text-slate-300">{status.last_synced_at}</strong></span>
          )}
        </div>
      </div>

      {/* Sync Progress Bar (Active when syncing) */}
      {isSyncing && (
        <div className="p-3 bg-cyan-950/30 border border-cyan-800/60 rounded-lg space-y-2">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-cyan-300 font-semibold flex items-center gap-1.5">
              <RefreshCw className="w-3 h-3 animate-spin text-cyan-400" />
              {status?.message || 'Syncing from Google Drive...'}
            </span>
            <span className="text-cyan-400 font-bold">{status?.progress_pct ?? 0}%</span>
          </div>
          <div className="w-full bg-[#05080e] rounded-full h-2 overflow-hidden border border-cyan-900/50">
            <div
              className="bg-gradient-to-r from-cyan-500 to-sky-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(5, status?.progress_pct ?? 0)}%` }}
            />
          </div>
          {status?.current_file && (
            <div className="text-[11px] font-mono text-slate-400 truncate">
              Currently downloading: <span className="text-white font-bold">{status.current_file}</span>
            </div>
          )}
        </div>
      )}

      {/* Files Catalog Table */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono font-bold text-slate-300 flex items-center gap-1.5">
            <FileVideo className="w-3.5 h-3.5 text-cyan-400" />
            <span>DRIVE REFERENCE VIDEOS IN VDATA/</span>
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            {localCount} of {totalFiles} available locally
          </span>
        </div>

        <div className="border border-[#1b2536] rounded-lg overflow-hidden bg-[#0a0f17]">
          {status?.files && status.files.length > 0 ? (
            <div className="divide-y divide-[#182333]">
              {status.files.map((file) => (
                <div
                  key={file.id || file.name}
                  className="p-2.5 flex items-center justify-between text-xs font-mono hover:bg-[#0f1724] transition"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <FileVideo className={`w-4 h-4 shrink-0 ${file.exists_locally ? 'text-emerald-400' : 'text-slate-500'}`} />
                    <div className="min-w-0">
                      <div className="text-slate-200 font-medium truncate">{file.name}</div>
                      <div className="text-[10px] text-slate-500">
                        {file.exists_locally ? (
                          <span className="text-emerald-400">Ready locally ({file.local_size_mb} MB)</span>
                        ) : (
                          <span className="text-amber-400">Stored on Google Drive</span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {file.exists_locally ? (
                      <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" />
                        <span>Synced</span>
                      </span>
                    ) : (
                      <button
                        onClick={() => handlePullFile(file.id, file.name)}
                        disabled={isSyncing}
                        className="px-2.5 py-1 bg-cyan-900/60 hover:bg-cyan-800 text-cyan-200 rounded text-[11px] font-medium flex items-center gap-1 transition disabled:opacity-50"
                      >
                        <Download className="w-3 h-3" />
                        <span>Pull</span>
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 text-center text-xs font-mono text-slate-400 space-y-1">
              <div>No remote file list cached yet.</div>
              <button
                onClick={handleScan}
                disabled={scanning}
                className="text-cyan-400 hover:underline font-bold"
              >
                Scan Google Drive folder now
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
