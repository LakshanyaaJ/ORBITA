import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Camera,
  Video,
  VideoOff,
  RotateCw,
  Sun,
  ShieldCheck,
  Smartphone,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import clsx from 'clsx';

export default function PhoneWebcam() {
  const [searchParams] = useSearchParams();
  const tokenFromUrl = searchParams.get('token') || searchParams.get('pair') || '';

  // Pairing & connection state
  const [pairingToken, setPairingToken] = useState(tokenFromUrl || '7429');
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'connecting' | 'live' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Camera settings
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [resolution, setResolution] = useState<{ width: number; height: number; label: string }>({
    width: 1280,
    height: 720,
    label: '720p',
  });
  const [targetFps, setTargetFps] = useState<number>(30);
  const [torchOn, setTorchOn] = useState(false);
  const [torchSupported, setTorchSupported] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [maxZoom, setMaxZoom] = useState(1);
  const [wakeLockActive, setWakeLockActive] = useState(false);

  // Telemetry metrics
  const [actualFps, setActualFps] = useState<number>(0);
  const [bitrateMbps, setBitrateMbps] = useState<number>(0);
  const [rttMs, setRttMs] = useState<number>(0);
  const [droppedFrames, setDroppedFrames] = useState<number>(0);

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const wakeLockRef = useRef<any>(null);
  const frameTimerRef = useRef<any>(null);
  const pingTimerRef = useRef<any>(null);
  const statsTimerRef = useRef<any>(null);

  const frameCounterRef = useRef<number>(0);
  const bytesCounterRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const isTransmittingRef = useRef<boolean>(false);

  // Determine WebSocket URL pointing to the Jetson / ORBITA server
  const getWsUrl = useCallback(() => {
    const loc = window.location;
    const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    // If running Vite dev port (5173/5174), point backend to 8000
    const port = loc.port === '5173' || loc.port === '5174' ? '8000' : (loc.port || '8000');
    return `${protocol}//${loc.hostname}:${port}/ws/phone_camera?token=${pairingToken}`;
  }, [pairingToken]);

  // Acquire Screen WakeLock to prevent screen dimming
  const requestWakeLock = useCallback(async () => {
    if ('wakeLock' in navigator) {
      try {
        wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
        setWakeLockActive(true);
        wakeLockRef.current.addEventListener('release', () => {
          setWakeLockActive(false);
        });
      } catch (e) {
        console.warn('WakeLock not granted', e);
      }
    }
  }, []);

  const releaseWakeLock = useCallback(() => {
    if (wakeLockRef.current) {
      wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
      setWakeLockActive(false);
    }
  }, []);

  // Inspect hardware track capabilities (torch, zoom)
  const inspectTrackCapabilities = useCallback((track: MediaStreamTrack) => {
    if ('getCapabilities' in track) {
      const caps = (track as any).getCapabilities();
      if (caps.torch) {
        setTorchSupported(true);
      } else {
        setTorchSupported(false);
      }
      if (caps.zoom) {
        setMaxZoom(caps.zoom.max || 1);
      } else {
        setMaxZoom(1);
      }
    }
  }, []);

  // Apply torch state
  const toggleTorch = async () => {
    if (!streamRef.current) return;
    const track = streamRef.current.getVideoTracks()[0];
    if (track && 'applyConstraints' in track) {
      try {
        const nextTorch = !torchOn;
        await (track as any).applyConstraints({
          advanced: [{ torch: nextTorch }],
        });
        setTorchOn(nextTorch);
      } catch (err) {
        console.warn('Torch constraint error', err);
      }
    }
  };

  // Apply hardware zoom
  const handleZoomChange = async (level: number) => {
    if (!streamRef.current) return;
    const track = streamRef.current.getVideoTracks()[0];
    if (track && 'applyConstraints' in track) {
      try {
        await (track as any).applyConstraints({
          advanced: [{ zoom: level }],
        });
        setZoomLevel(level);
      } catch (err) {
        console.warn('Zoom error', err);
      }
    }
  };

  // Start Camera Hardware
  const startCamera = async () => {
    setErrorMessage(null);
    setConnectionStatus('connecting');

    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      const constraints: MediaStreamConstraints = {
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
          frameRate: { ideal: targetFps, max: targetFps },
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const videoTrack = stream.getVideoTracks()[0];
      inspectTrackCapabilities(videoTrack);
      await requestWakeLock();

      // Connect WebSocket stream to Jetson
      connectWebSocket();
      setIsStreaming(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setConnectionStatus('error');
      setErrorMessage(
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access in browser settings.'
          : `Camera error: ${err.message || 'Unable to open camera stream.'}`
      );
    }
  };

  // Stop Camera & Streaming
  const stopCamera = () => {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
    if (pingTimerRef.current) clearInterval(pingTimerRef.current);
    if (statsTimerRef.current) clearInterval(statsTimerRef.current);

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    releaseWakeLock();
    setIsStreaming(false);
    setConnectionStatus('idle');
    setTorchOn(false);
  };

  // Connect WebSocket to Jetson Receiver
  const connectWebSocket = () => {
    const wsUrl = getWsUrl();
    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setConnectionStatus('live');
      setErrorMessage(null);

      // Send device info
      ws.send(
        JSON.stringify({
          type: 'device_info',
          userAgent: navigator.userAgent,
          facingMode,
          resolution: `${resolution.width}x${resolution.height}`,
          targetFps,
        })
      );

      // Start frame capture loop
      startTransmissionLoop(ws);

      // Periodic ping for RTT latency measurement
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        }
      }, 2000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong' && data.timestamp) {
          const rtt = Date.now() - data.timestamp;
          setRttMs(rtt);
        }
      } catch {}
    };

    ws.onerror = () => {
      setConnectionStatus('error');
      setErrorMessage('Failed to connect to ORBITA Jetson. Check Wi-Fi and pairing code.');
    };

    ws.onclose = () => {
      if (isStreaming) {
        setConnectionStatus('connecting');
        // Auto-reconnect after brief delay
        setTimeout(() => {
          if (isStreaming) connectWebSocket();
        }, 2000);
      }
    };

    socketRef.current = ws;
  };

  // Frame Capture & WebSocket Transmission Loop (<2ms encode)
  const startTransmissionLoop = (ws: WebSocket) => {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);

    const intervalMs = Math.round(1000 / targetFps);
    const canvas = canvasRef.current || document.createElement('canvas');
    canvasRef.current = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });

    frameTimerRef.current = setInterval(() => {
      if (!videoRef.current || !ctx || ws.readyState !== WebSocket.OPEN) return;
      if (isTransmittingRef.current) {
        setDroppedFrames((prev) => prev + 1);
        return; // Drop old frame: fresh frame > every frame
      }

      const video = videoRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      canvas.width = resolution.width;
      canvas.height = resolution.height;

      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      isTransmittingRef.current = true;
      // Fast JPEG compression (quality 0.70 is ideal for 720p low latency)
      canvas.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => {
              ws.send(buf);
              bytesCounterRef.current += buf.byteLength;
              frameCounterRef.current += 1;
              isTransmittingRef.current = false;
            });
          } else {
            isTransmittingRef.current = false;
          }
        },
        'image/jpeg',
        0.70
      );
    }, intervalMs);

    // Telemetry stats calculation timer (1 second interval)
    if (statsTimerRef.current) clearInterval(statsTimerRef.current);
    statsTimerRef.current = setInterval(() => {
      const now = performance.now();
      const elapsedSec = (now - lastFpsTimeRef.current) / 1000;
      if (elapsedSec >= 0.8) {
        const fps = Math.round((frameCounterRef.current / elapsedSec) * 10) / 10;
        const mbps = Math.round(((bytesCounterRef.current * 8) / (elapsedSec * 1_000_000)) * 100) / 100;

        setActualFps(fps);
        setBitrateMbps(mbps);

        frameCounterRef.current = 0;
        bytesCounterRef.current = 0;
        lastFpsTimeRef.current = now;
      }
    }, 1000);
  };

  // Flip camera front/rear
  const toggleFacingMode = async () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    if (isStreaming) {
      stopCamera();
      setTimeout(startCamera, 300);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="min-h-screen bg-space-950 text-space-100 flex flex-col font-sans select-none touch-manipulation">
      {/* Hidden canvas for fast frame serialization */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Mobile Top Header */}
      <header className="px-4 py-3 bg-space-900/90 border-b border-space-700/80 backdrop-blur flex items-center justify-between sticky top-0 z-30">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-accent-cyan shadow-[0_0_8px_#38bdf8]" />
          <span className="font-mono text-sm font-black tracking-wider text-space-100 uppercase">
            ORBITA CAM
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-space-800 text-space-300 border border-space-700">
            WEB
          </span>
        </div>

        {/* Live / Status Indicator Badge */}
        <div className="flex items-center gap-2">
          {connectionStatus === 'live' ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-status-success/20 text-status-success border border-status-success/40 text-[11px] font-mono font-bold tracking-widest animate-pulse">
              <span className="w-2 h-2 rounded-full bg-status-success" />
              LIVE
            </span>
          ) : connectionStatus === 'connecting' ? (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-status-warning/20 text-status-warning border border-status-warning/40 text-[11px] font-mono font-bold tracking-widest">
              <RefreshCw size={11} className="animate-spin" />
              CONNECTING
            </span>
          ) : (
            <span className="px-2.5 py-1 rounded bg-space-800 text-space-400 border border-space-700 text-[11px] font-mono font-bold tracking-widest">
              STANDBY
            </span>
          )}
        </div>
      </header>

      {/* Main Viewport Container */}
      <main className="flex-1 flex flex-col p-3 max-w-lg mx-auto w-full gap-3">
        {/* Error Notification Banner */}
        {errorMessage && (
          <div className="p-3 bg-status-critical/15 border border-status-critical/40 rounded flex items-start gap-2.5 text-xs text-status-critical font-mono">
            <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
            <div className="flex-1">{errorMessage}</div>
          </div>
        )}

        {/* Viewfinder Frame */}
        <div className="relative aspect-[4/3] bg-black rounded-lg border border-space-700 overflow-hidden flex items-center justify-center shadow-lg">
          <video
            ref={videoRef}
            playsInline
            autoPlay
            muted
            className={clsx(
              "w-full h-full object-cover",
              facingMode === 'user' && "scale-x-[-1]"
            )}
          />

          {/* Standby Viewfinder Overlay */}
          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-space-900/90 text-space-300">
              <Smartphone size={42} className="text-space-500 mb-3" />
              <div className="font-mono text-sm font-bold text-space-100 mb-1">PHONE WEBCAM READY</div>
              <p className="text-xs text-space-400 max-w-xs font-mono">
                No app required. Connects to ORBITA on Jetson Orin Nano via local Wi-Fi.
              </p>
            </div>
          )}

          {/* Live Viewfinder Crosshairs & HUD Overlays */}
          {isStreaming && (
            <>
              {/* Corner brackets */}
              <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-accent-cyan/70 pointer-events-none" />
              <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-accent-cyan/70 pointer-events-none" />
              <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-accent-cyan/70 pointer-events-none" />
              <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-accent-cyan/70 pointer-events-none" />

              {/* Resolution & FPS Badge */}
              <div className="absolute top-2 left-2 bg-space-950/70 backdrop-blur px-2 py-0.5 rounded text-[10px] font-mono text-space-300 border border-space-700 pointer-events-none">
                {resolution.label} · {targetFps} FPS · {facingMode === 'environment' ? 'REAR' : 'FRONT'}
              </div>

              {/* Torch & Flip Quick Controls Overlay */}
              <div className="absolute bottom-2 right-2 flex items-center gap-1.5">
                {torchSupported && (
                  <button
                    type="button"
                    onClick={toggleTorch}
                    className={clsx(
                      "p-2 rounded-full border backdrop-blur text-xs transition-colors",
                      torchOn
                        ? "bg-status-warning text-space-950 border-status-warning shadow-[0_0_8px_#eab308]"
                        : "bg-space-900/80 text-space-300 border-space-700"
                    )}
                    title="Toggle Flashlight"
                  >
                    <Sun size={15} />
                  </button>
                )}
                <button
                  type="button"
                  onClick={toggleFacingMode}
                  className="p-2 rounded-full bg-space-900/80 border border-space-700 text-space-300 backdrop-blur text-xs active:rotate-180 transition-transform"
                  title="Switch Front/Rear Camera"
                >
                  <RotateCw size={15} />
                </button>
              </div>
            </>
          )}
        </div>

        {/* Real-time Diagnostics Bar (Phase 10 & 11) */}
        <div className="grid grid-cols-4 gap-1.5 p-2 bg-space-900/90 border border-space-700 rounded-lg text-center font-mono text-xs">
          <div className="flex flex-col border-r border-space-700/60 pr-1">
            <span className="text-[9px] uppercase tracking-wider text-space-400">Stream</span>
            <span className="font-bold text-space-100">{isStreaming ? `${actualFps}` : '—'} <span className="text-[9px] font-normal text-space-400">FPS</span></span>
          </div>
          <div className="flex flex-col border-r border-space-700/60 px-1">
            <span className="text-[9px] uppercase tracking-wider text-space-400">Bitrate</span>
            <span className="font-bold text-space-100">{isStreaming ? `${bitrateMbps}` : '—'} <span className="text-[9px] font-normal text-space-400">M</span></span>
          </div>
          <div className="flex flex-col border-r border-space-700/60 px-1">
            <span className="text-[9px] uppercase tracking-wider text-space-400">LAN Ping</span>
            <span className="font-bold text-accent-cyan">{rttMs > 0 ? `${rttMs}` : '<10'} <span className="text-[9px] font-normal text-space-400">ms</span></span>
          </div>
          <div className="flex flex-col pl-1">
            <span className="text-[9px] uppercase tracking-wider text-space-400">Queue / Drop</span>
            <span className="font-bold text-status-success">1 <span className="text-[9px] font-normal text-space-400">| d:{droppedFrames}</span></span>
          </div>
        </div>

        {/* Hardware Controls & Configuration Accordion (Phase 7) */}
        <div className="bg-space-900 border border-space-700 rounded-lg p-3 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono font-bold text-space-300 uppercase tracking-wider pb-2 border-b border-space-700/60">
            <span>Camera Settings</span>
            <span className="text-[10px] text-accent-cyan lowercase">720p @ 30fps optimal</span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs font-mono">
            {/* Camera Switch */}
            <div>
              <label className="text-[10px] text-space-400 block mb-1">FACING</label>
              <button
                type="button"
                disabled={isStreaming}
                onClick={toggleFacingMode}
                className="w-full py-1.5 px-2 rounded bg-space-800 border border-space-700 text-space-200 text-left flex items-center justify-between disabled:opacity-50"
              >
                <span className="truncate">{facingMode === 'environment' ? 'Rear' : 'Front'}</span>
                <Camera size={12} className="text-space-400 flex-shrink-0" />
              </button>
            </div>

            {/* Resolution Selector */}
            <div>
              <label className="text-[10px] text-space-400 block mb-1">RESOLUTION</label>
              <select
                disabled={isStreaming}
                value={resolution.label}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === '480p') setResolution({ width: 640, height: 480, label: '480p' });
                  else if (val === '720p') setResolution({ width: 1280, height: 720, label: '720p' });
                  else if (val === '1080p') setResolution({ width: 1920, height: 1080, label: '1080p' });
                }}
                className="w-full py-1.5 px-1.5 rounded bg-space-800 border border-space-700 text-space-200 disabled:opacity-50 outline-none"
              >
                <option value="720p">720p</option>
                <option value="480p">480p</option>
                <option value="1080p">1080p</option>
              </select>
            </div>

            {/* Target FPS Selector */}
            <div>
              <label className="text-[10px] text-space-400 block mb-1">TARGET FPS</label>
              <select
                disabled={isStreaming}
                value={targetFps}
                onChange={(e) => setTargetFps(Number(e.target.value))}
                className="w-full py-1.5 px-1.5 rounded bg-space-800 border border-space-700 text-space-200 disabled:opacity-50 outline-none"
              >
                <option value={30}>30 FPS</option>
                <option value={24}>24 FPS</option>
                <option value={15}>15 FPS</option>
              </select>
            </div>
          </div>

          {/* Zoom Buttons (If Supported) */}
          {maxZoom > 1 && (
            <div>
              <div className="text-[10px] text-space-400 font-mono mb-1">HARDWARE ZOOM</div>
              <div className="grid grid-cols-3 gap-1.5">
                {[1, 2, Math.min(3, maxZoom)].map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    onClick={() => handleZoomChange(lvl)}
                    className={clsx(
                      "py-1 rounded font-mono text-xs border transition-colors",
                      zoomLevel === lvl
                        ? "bg-accent-cyan/20 border-accent-cyan text-accent-cyan font-bold"
                        : "bg-space-800 border-space-700 text-space-300"
                    )}
                  >
                    {lvl}x
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Pairing Code Input (Phase 5) */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] text-space-400 font-mono">PAIRING CODE</label>
              <span className="text-[9px] text-space-500 font-mono">From Jetson Dashboard</span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                disabled={isStreaming}
                value={pairingToken}
                onChange={(e) => setPairingToken(e.target.value.toUpperCase())}
                placeholder="4-digit code"
                maxLength={6}
                className="flex-1 py-1.5 px-3 rounded bg-space-800 border border-space-700 text-space-100 font-mono font-bold tracking-widest text-center disabled:opacity-50 outline-none focus:border-accent-cyan"
              />
            </div>
          </div>
        </div>

        {/* Start / Stop Stream Action Button */}
        <div className="pt-1">
          {!isStreaming ? (
            <button
              type="button"
              onClick={startCamera}
              className="w-full py-3.5 px-4 rounded-lg bg-accent-cyan hover:bg-sky-400 text-space-950 font-mono font-black text-sm tracking-wider uppercase transition-all shadow-[0_4px_16px_rgba(56,189,248,0.25)] flex items-center justify-center gap-2 active:scale-[0.98]"
            >
              <Video size={18} />
              START CAMERA STREAM
            </button>
          ) : (
            <button
              type="button"
              onClick={stopCamera}
              className="w-full py-3.5 px-4 rounded-lg bg-status-critical/20 hover:bg-status-critical/30 border border-status-critical text-status-critical font-mono font-black text-sm tracking-wider uppercase transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
            >
              <VideoOff size={18} />
              STOP CAMERA STREAM
            </button>
          )}
        </div>

        {/* Screen WakeLock status pill */}
        <div className="flex items-center justify-center gap-1.5 text-[10px] font-mono text-space-400 pt-1">
          <ShieldCheck size={12} className={wakeLockActive ? 'text-status-success' : 'text-space-500'} />
          <span>Screen WakeLock: {wakeLockActive ? 'Active (Display On)' : 'Inactive'}</span>
        </div>
      </main>

      {/* Footer */}
      <footer className="p-3 text-center border-t border-space-800/80 text-[10px] font-mono text-space-500">
        ORBITA MISSION CONTROL · WIRELESS PHONE WEBCAM
      </footer>
    </div>
  );
}
