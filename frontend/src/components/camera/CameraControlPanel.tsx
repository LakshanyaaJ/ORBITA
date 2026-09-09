import { useState, useEffect, useRef } from 'react';
import { Camera, Settings, RefreshCw, QrCode, ExternalLink, Smartphone } from 'lucide-react';
import CameraSourceSelector, { type CameraSourceType } from './CameraSourceSelector';
import IPWebcamConfig from './IPWebcamConfig';
import CameraConnectionStatus from './CameraConnectionStatus';
import CameraMetrics from './CameraMetrics';
import PhoneCameraQRModal from './PhoneCameraQRModal';
import {
  getCameraStatus,
  connectCamera,
  disconnectCamera,
  BACKEND_BASE,
  type CameraStatus,
  type ConnectCameraParams,
} from '../../api/camera';

interface CameraControlPanelProps {
  onCameraChange?: (status: CameraStatus) => void;
  className?: string;
}

export default function CameraControlPanel({
  onCameraChange,
  className = '',
}: CameraControlPanelProps) {
  const [status, setStatus] = useState<CameraStatus>({
    connected: true,
    source: 'sim',
    fps: 30,
    latency_ms: 0,
    status: 'connected',
    error: null,
  });
  const [selectedSource, setSelectedSource] = useState<CameraSourceType>('sim');
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [showQrModal, setShowQrModal] = useState(false);
  const [vdataVideos, setVdataVideos] = useState<any[]>([]);

  const userEditingRef = useRef(false);
  const initialLoadRef = useRef(false);

  // Fetch available vdata videos
  useEffect(() => {
    fetch(`${BACKEND_BASE}/api/vdata/videos`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setVdataVideos(data);
        }
      })
      .catch((err) => console.error('Failed to load vdata videos in control panel:', err));
  }, []);

  // Poll camera status periodically
  const fetchStatus = async () => {
    const current = await getCameraStatus();
    setStatus(current);

    if (!initialLoadRef.current) {
      if (['phone_webcam', 'ip_camera', 'jetson_camera', 'sim', 'video_file'].includes(current.source)) {
        setSelectedSource(current.source as CameraSourceType);
      }
      initialLoadRef.current = true;
    } else if (!userEditingRef.current) {
      if (['phone_webcam', 'ip_camera', 'jetson_camera', 'sim', 'video_file'].includes(current.source)) {
        setSelectedSource(current.source as CameraSourceType);
      }
    }

    onCameraChange?.(current);
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 2500);
    return () => clearInterval(timer);
  }, []);

  const handleSourceChange = async (newSource: CameraSourceType, customPath?: string) => {
    setSelectedSource(newSource);

    if (newSource === 'sim') {
      userEditingRef.current = false;
      setLoading(true);
      await connectCamera({ source: 'sim' });
      await fetchStatus();
      setLoading(false);
    } else if (newSource === 'video_file') {
      userEditingRef.current = false;
      setLoading(true);
      const vpath = customPath || 'vdata/20260905_145858.mp4';
      await connectCamera({ source: 'video_file', path: vpath, loop: true, reset_fsm: true });
      await fetchStatus();
      setLoading(false);
    } else if (newSource === 'jetson_camera') {
      userEditingRef.current = false;
      setLoading(true);
      await connectCamera({ source: 'jetson_camera', device_index: 0 });
      await fetchStatus();
      setLoading(false);
    } else if (newSource === 'phone_webcam') {
      userEditingRef.current = false;
      setLoading(true);
      await connectCamera({ source: 'phone_webcam' });
      await fetchStatus();
      setLoading(false);
      setShowQrModal(true);
    } else if (newSource === 'ip_camera') {
      userEditingRef.current = true;
      setIsExpanded(true);
    }
  };

  const handleConnect = async (params: ConnectCameraParams) => {
    setLoading(true);
    const res = await connectCamera(params);
    if (!res.success) {
      setStatus((prev) => ({
        ...prev,
        connected: false,
        status: 'error',
        error: res.error || 'Connection failed. Please check IP and port.',
      }));
    } else {
      userEditingRef.current = false;
      await fetchStatus();
    }
    setLoading(false);
  };

  const handleDisconnect = async () => {
    setLoading(true);
    userEditingRef.current = false;
    await disconnectCamera();
    await fetchStatus();
    setLoading(false);
  };

  return (
    <div className={`bg-space-800 border border-space-600 rounded flex flex-col font-sans ${className}`}>
      {/* Header Bar */}
      <div className="p-3 border-b border-space-600/70 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Camera size={16} className="text-accent-cyan" />
          <span className="font-mono text-xs font-bold tracking-widest text-space-100 uppercase">
            CAMERA FEED CONTROL
          </span>
        </div>

        <div className="flex items-center gap-3">
          <CameraConnectionStatus status={status.status} error={status.error} />
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            title="Configure Camera Source"
            className={`p-1.5 rounded transition-colors ${
              isExpanded
                ? 'bg-space-700 text-accent-cyan'
                : 'text-space-400 hover:text-space-100 hover:bg-space-700/60'
            }`}
          >
            <Settings size={14} />
          </button>
        </div>
      </div>

      {/* Metrics Row (Always Visible) */}
      <div className="p-3">
        <CameraMetrics status={status} />
      </div>

      {/* Expandable Configuration Section */}
      {isExpanded && (
        <div className="p-3 pt-0 space-y-3 border-t border-space-600/50 mt-1 animate-in fade-in duration-150">
          <div className="pt-2 flex items-center justify-between">
            <span className="font-mono text-[10px] text-space-400 font-bold uppercase tracking-wider">
              INPUT SELECTION
            </span>
            <button
              type="button"
              onClick={fetchStatus}
              title="Refresh Camera Status"
              className="text-space-400 hover:text-accent-cyan p-1 text-xs transition-colors"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>

          <CameraSourceSelector
            selectedSource={selectedSource}
            onChange={handleSourceChange}
            disabled={loading}
          />

          {/* Phone Web App Stream Configuration Card */}
          {selectedSource === 'phone_webcam' && (
            <div className="p-3 bg-space-900/80 border border-space-600 rounded flex flex-col gap-2.5 font-mono text-xs">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Smartphone size={15} className="text-accent-cyan" />
                  <span className="text-space-200 font-bold">Wireless Phone Web Cam</span>
                </div>
                <span className={status.connected ? 'text-status-success font-bold' : 'text-status-warning font-bold'}>
                  {status.connected ? '● CONNECTED' : '○ READY TO PAIR'}
                </span>
              </div>
              <p className="text-[11px] text-space-400">
                Scan QR code with your phone camera or visit <span className="text-accent-cyan font-bold">/cam</span>. No app required.
              </p>

              <div className="flex items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setShowQrModal(true)}
                  className="flex-1 py-1.5 px-3 rounded bg-accent-cyan text-space-950 hover:bg-sky-400 font-bold flex items-center justify-center gap-1.5 transition-all text-xs"
                >
                  <QrCode size={14} />
                  <span>PAIR PHONE CAMERA (QR)</span>
                </button>

                <a
                  href="/cam"
                  target="_blank"
                  rel="noreferrer"
                  title="Open phone camera view in new browser tab"
                  className="py-1.5 px-3 rounded bg-space-700 hover:bg-space-600 text-space-200 flex items-center justify-center gap-1 transition-colors text-xs"
                >
                  <span>Open Here</span>
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>
          )}

          {selectedSource === 'ip_camera' && (
            <IPWebcamConfig
              status={status}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              loading={loading}
            />
          )}

          {selectedSource === 'jetson_camera' && (
            <div className="p-3 bg-space-800/80 border border-space-600 rounded flex items-center justify-between text-xs font-mono">
              <span className="text-space-300">Device: /dev/video0 (Local Camera)</span>
              <button
                type="button"
                disabled={loading}
                onClick={() => handleSourceChange('jetson_camera')}
                className="px-2.5 py-1 rounded bg-space-700 hover:bg-space-600 text-accent-cyan text-[11px] font-bold"
              >
                Reconnect
              </button>
            </div>
          )}

          {selectedSource === 'video_file' && (
            <div className="p-3 bg-space-900/80 border border-space-600 rounded flex flex-col gap-2.5 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="text-space-200 font-bold">Reference Video (vdata/)</span>
                <span className={status.connected ? 'text-status-success font-bold' : 'text-status-warning font-bold'}>
                  {status.connected ? '● PLAYING / CONNECTED' : '○ CONNECTING'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={status.url?.replace(/\\/g, '/').split('/').pop() || '20260905_145858.mp4'}
                  onChange={(e) => handleSourceChange('video_file', `vdata/${e.target.value}`)}
                  disabled={loading}
                  className="flex-1 bg-space-950 border border-space-600 text-space-100 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-accent-cyan"
                >
                  {vdataVideos.length > 0 ? (
                    vdataVideos.map((v, idx) => (
                      <option key={v.filename} value={v.filename}>
                        {v.filename} {v.duration_seconds ? `(Trial ${idx + 1} · ${v.duration_seconds}s)` : `(Video ${idx + 1})`}
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="20260905_145858.mp4">20260905_145858.mp4 (Yellow & Blue Box Trial 1)</option>
                      <option value="20260905_145948.mp4">20260905_145948.mp4 (Yellow & Blue Box Trial 2)</option>
                      <option value="20260905_150132.mp4">20260905_150132.mp4 (Yellow & Blue Box Trial 3)</option>
                      <option value="20260908_135006.mp4">20260908_135006.mp4 (Trial 4)</option>
                    </>
                  )}
                </select>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => handleSourceChange('video_file', status.url || 'vdata/20260905_145858.mp4')}
                  className="px-3 py-1.5 rounded bg-accent-cyan text-space-950 hover:bg-sky-400 font-bold text-xs"
                >
                  Replay
                </button>
              </div>
              {status.total_frames ? (
                <div className="text-[10px] text-space-400 flex justify-between">
                  <span>Frame: {status.frame_index || 0} / {status.total_frames}</span>
                  <span className="text-accent-cyan font-bold">Continuous Loop Active</span>
                </div>
              ) : null}
            </div>
          )}
        </div>
      )}

      {/* Phone Camera QR Pairing Modal */}
      <PhoneCameraQRModal
        isOpen={showQrModal}
        onClose={() => setShowQrModal(false)}
        onConnected={() => {
          fetchStatus();
          setShowQrModal(false);
        }}
      />
    </div>
  );
}
