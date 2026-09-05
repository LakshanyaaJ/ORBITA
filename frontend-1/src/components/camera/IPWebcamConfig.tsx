import { useState, useEffect } from 'react';
import { Wifi, Link as LinkIcon, AlertCircle, Play, Square, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import type { CameraStatus, ConnectCameraParams } from '../../api/camera';

interface IPWebcamConfigProps {
  status: CameraStatus;
  onConnect: (params: ConnectCameraParams) => Promise<void>;
  onDisconnect: () => Promise<void>;
  loading: boolean;
}

export default function IPWebcamConfig({
  status,
  onConnect,
  onDisconnect,
  loading,
}: IPWebcamConfigProps) {
  // Input fields
  const [ip, setIp] = useState('192.168.1.105');
  const [port, setPort] = useState('8080');
  const [path, setPath] = useState('/video');
  const [completeUrl, setCompleteUrl] = useState('');
  const [inputMode, setInputMode] = useState<'fields' | 'url'>('fields');
  const [validationError, setValidationError] = useState<string | null>(null);

  // Sync completeUrl when in fields mode
  useEffect(() => {
    if (inputMode === 'fields') {
      const cleanIp = ip.trim().replace(/^https?:\/\//, '').replace(/\/.*$/, '');
      const cleanPort = port.trim() || '8080';
      const cleanPath = path.trim().startsWith('/') ? path.trim() : `/${path.trim()}`;
      if (cleanIp) {
        setCompleteUrl(`http://${cleanIp}:${cleanPort}${cleanPath}`);
      }
    }
  }, [ip, port, path, inputMode]);

  const handleConnect = async () => {
    setValidationError(null);

    if (inputMode === 'url') {
      if (!completeUrl.trim()) {
        setValidationError('Please enter a stream URL.');
        return;
      }
      await onConnect({
        source: 'ip_camera',
        url: completeUrl.trim(),
      });
    } else {
      const rawIp = ip.trim();
      if (!rawIp) {
        setValidationError('Please enter Phone IP Address.');
        return;
      }

      // If full URL was entered in IP field
      if (rawIp.startsWith('http://') || rawIp.startsWith('https://') || rawIp.startsWith('rtsp://')) {
        await onConnect({
          source: 'ip_camera',
          url: rawIp,
        });
        return;
      }

      // Parse possible port or path embedded in the IP field
      let cleanIp = rawIp;
      let cleanPort = port.trim() || '8080';
      let cleanPath = path.trim() || '/video';
      if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath;

      if (cleanIp.includes('/')) {
        const slashIdx = cleanIp.indexOf('/');
        cleanPath = cleanIp.slice(slashIdx);
        cleanIp = cleanIp.slice(0, slashIdx);
      }

      if (cleanIp.includes(':')) {
        const colonIdx = cleanIp.indexOf(':');
        cleanPort = cleanIp.slice(colonIdx + 1);
        cleanIp = cleanIp.slice(0, colonIdx);
      }

      const directUrl = `http://${cleanIp}:${cleanPort}${cleanPath}`;
      await onConnect({
        source: 'ip_camera',
        url: directUrl,
        ip: cleanIp,
        port: cleanPort,
        path: cleanPath,
      });
    }
  };

  const handleApplyPreset = (presetPort: string, presetPath: string) => {
    setPort(presetPort);
    setPath(presetPath);
  };

  const isConnected = status.connected && status.source === 'ip_camera';

  return (
    <div className="flex flex-col gap-3 p-3.5 bg-space-800/90 border border-space-600 rounded">
      {/* Mode Selector Tabs */}
      <div className="flex items-center justify-between border-b border-space-600/60 pb-2">
        <span className="text-[11px] font-mono font-bold tracking-widest text-space-400 uppercase flex items-center gap-1.5">
          <Wifi size={13} className="text-accent-cyan" />
          Phone Stream Setup
        </span>
        <div className="flex items-center gap-1 text-[10px] font-mono">
          <button
            type="button"
            onClick={() => setInputMode('fields')}
            className={clsx(
              "px-2 py-0.5 rounded transition-colors",
              inputMode === 'fields' ? "bg-space-700 text-accent-cyan font-bold" : "text-space-400 hover:text-space-100"
            )}
          >
            IP / Port
          </button>
          <button
            type="button"
            onClick={() => setInputMode('url')}
            className={clsx(
              "px-2 py-0.5 rounded transition-colors",
              inputMode === 'url' ? "bg-space-700 text-accent-cyan font-bold" : "text-space-400 hover:text-space-100"
            )}
          >
            Full URL
          </button>
        </div>
      </div>

      {/* Discrete Fields */}
      {inputMode === 'fields' ? (
        <div className="space-y-2.5">
          <div>
            <label className="block text-[10px] font-mono font-bold text-space-400 uppercase mb-1">
              Phone IP Address
            </label>
            <input
              type="text"
              value={ip}
              onChange={(e) => setIp(e.target.value)}
              placeholder="e.g. 192.168.1.105"
              disabled={loading || isConnected}
              className="w-full bg-space-900 border border-space-600 rounded px-2.5 py-1.5 font-mono text-xs text-space-100 placeholder:text-space-600 focus:outline-none focus:border-accent-cyan"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-[10px] font-mono font-bold text-space-400 uppercase mb-1">
                Stream Port
              </label>
              <input
                type="text"
                value={port}
                onChange={(e) => setPort(e.target.value)}
                placeholder="8080"
                disabled={loading || isConnected}
                className="w-full bg-space-900 border border-space-600 rounded px-2.5 py-1.5 font-mono text-xs text-space-100 placeholder:text-space-600 focus:outline-none focus:border-accent-cyan"
              />
            </div>
            <div>
              <label className="block text-[10px] font-mono font-bold text-space-400 uppercase mb-1">
                Stream Path
              </label>
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="/video"
                disabled={loading || isConnected}
                className="w-full bg-space-900 border border-space-600 rounded px-2.5 py-1.5 font-mono text-xs text-space-100 placeholder:text-space-600 focus:outline-none focus:border-accent-cyan"
              />
            </div>
          </div>

          {/* Presets */}
          <div className="flex items-center gap-1.5 pt-0.5">
            <span className="text-[9px] font-mono text-space-400">Presets:</span>
            <button
              type="button"
              disabled={loading || isConnected}
              onClick={() => handleApplyPreset('8080', '/video')}
              className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-space-700/60 hover:bg-space-700 text-space-300"
            >
              IP Webcam (:8080/video)
            </button>
            <button
              type="button"
              disabled={loading || isConnected}
              onClick={() => handleApplyPreset('4747', '/video')}
              className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-space-700/60 hover:bg-space-700 text-space-300"
            >
              DroidCam (:4747/video)
            </button>
          </div>
        </div>
      ) : (
        /* Complete URL Mode */
        <div>
          <label className="block text-[10px] font-mono font-bold text-space-400 uppercase mb-1 flex items-center justify-between">
            <span>Stream URL</span>
            <span className="text-[9px] text-space-400 lowercase">http:// or rtsp://</span>
          </label>
          <div className="relative">
            <input
              type="text"
              value={completeUrl}
              onChange={(e) => setCompleteUrl(e.target.value)}
              placeholder="http://192.168.1.105:8080/video"
              disabled={loading || isConnected}
              className="w-full bg-space-900 border border-space-600 rounded pl-7 pr-2.5 py-1.5 font-mono text-xs text-space-100 placeholder:text-space-600 focus:outline-none focus:border-accent-cyan"
            />
            <LinkIcon size={12} className="absolute left-2.5 top-2.5 text-space-400" />
          </div>
        </div>
      )}

      {/* Target Preview */}
      <div className="bg-space-900/70 border border-space-600/40 rounded px-2.5 py-1.5 font-mono text-[10px] text-space-400 flex items-center justify-between">
        <span className="truncate">URL: <span className="text-accent-cyan">{completeUrl || '—'}</span></span>
      </div>

      {/* Validation or API Error */}
      {(validationError || status.error) && (
        <div className="flex items-start gap-1.5 p-2 bg-status-critical/10 border border-status-critical/30 rounded text-status-critical font-mono text-xs">
          <AlertCircle size={14} className="shrink-0 mt-0.5" />
          <span>{validationError || status.error}</span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center gap-2 pt-1">
        {!isConnected ? (
          <button
            type="button"
            disabled={loading}
            onClick={handleConnect}
            className="flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded bg-accent-cyan hover:bg-accent-cyan/90 text-space-900 font-mono font-bold text-xs tracking-wider transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 size={13} className="animate-spin" />
                CONNECTING...
              </>
            ) : (
              <>
                <Play size={13} fill="currentColor" />
                CONNECT CAMERA
              </>
            )}
          </button>
        ) : (
          <button
            type="button"
            disabled={loading}
            onClick={onDisconnect}
            className="flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded bg-status-critical/20 hover:bg-status-critical/30 border border-status-critical/40 text-status-critical font-mono font-bold text-xs tracking-wider transition-all"
          >
            <Square size={13} fill="currentColor" />
            DISCONNECT CAMERA
          </button>
        )}
      </div>
    </div>
  );
}
