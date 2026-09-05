import { useState, useEffect, useRef } from 'react';
import { Camera, Settings, RefreshCw } from 'lucide-react';
import CameraSourceSelector, { type CameraSourceType } from './CameraSourceSelector';
import IPWebcamConfig from './IPWebcamConfig';
import CameraConnectionStatus from './CameraConnectionStatus';
import CameraMetrics from './CameraMetrics';
import {
  getCameraStatus,
  connectCamera,
  disconnectCamera,
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
  const [isExpanded, setIsExpanded] = useState(true); // Open by default for easy access

  const userEditingRef = useRef(false);
  const initialLoadRef = useRef(false);

  // Poll camera status periodically
  const fetchStatus = async () => {
    const current = await getCameraStatus();
    setStatus(current);

    // Only update selectedSource if this is the first load or if user is not actively selecting another source
    if (!initialLoadRef.current) {
      if (current.source === 'ip_camera' || current.source === 'jetson_camera' || current.source === 'sim') {
        setSelectedSource(current.source);
      }
      initialLoadRef.current = true;
    } else if (!userEditingRef.current) {
      // If user is not actively editing/configuring, sync to backend source
      if (current.source === 'ip_camera' || current.source === 'jetson_camera' || current.source === 'sim') {
        setSelectedSource(current.source);
      }
    }

    onCameraChange?.(current);
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 2500);
    return () => clearInterval(timer);
  }, []);

  const handleSourceChange = async (newSource: CameraSourceType) => {
    setSelectedSource(newSource);

    if (newSource === 'sim') {
      userEditingRef.current = false;
      setLoading(true);
      await connectCamera({ source: 'sim' });
      await fetchStatus();
      setLoading(false);
    } else if (newSource === 'jetson_camera') {
      userEditingRef.current = false;
      setLoading(true);
      await connectCamera({ source: 'jetson_camera', device_index: 0 });
      await fetchStatus();
      setLoading(false);
    } else if (newSource === 'ip_camera') {
      // Keep selected on ip_camera and do not let background poll revert it!
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
        </div>
      )}
    </div>
  );
}
