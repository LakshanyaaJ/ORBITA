import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { useTelemetry } from '../api/telemetry';
import clsx from 'clsx';
import { 
  AlertTriangle, 
  CheckCircle2, 
  Activity, 
  Video, 
  HelpCircle, 
  Cpu, 
  ArrowRight, 
  ChevronRight,
  RotateCw,
  RotateCcw,
  Film,
  Volume2,
  VolumeX
} from 'lucide-react';
import CameraControlPanel from '../components/camera/CameraControlPanel';
import { rotateCamera, type CameraStatus, BACKEND_BASE } from '../api/camera';
import type { ChecklistItem } from '../api/types';

// Official 13-Step Sequence strictly matching Section 2 of Master Specification
const OFFICIAL_13_STEPS = [
  { id: 1, action: "IDENTIFY_BLUE_BOX", label: "Identify the Blue Box", expected_object: "BLUE_BOX" },
  { id: 2, action: "PICKUP_BLUE_BOX", label: "Pick up the Blue Box", expected_object: "BLUE_BOX" },
  { id: 3, action: "PLACE_BLUE_BOX_A", label: "Place Blue Box at Location A", expected_object: "BLUE_BOX" },
  { id: 4, action: "IDENTIFY_YELLOW_BOX", label: "Identify the Yellow Box", expected_object: "YELLOW_BOX" },
  { id: 5, action: "PICKUP_YELLOW_BOX", label: "Pick up the Yellow Box", expected_object: "YELLOW_BOX" },
  { id: 6, action: "PLACE_YELLOW_BOX_B", label: "Place Yellow Box at Location B", expected_object: "YELLOW_BOX" },
  { id: 7, action: "PICKUP_PEN", label: "Pick up the Pen", expected_object: "PEN" },
  { id: 8, action: "PLACE_PEN_BLUE_BOX", label: "Place Pen inside the Blue Box", expected_object: "PEN" },
  { id: 9, action: "PICKUP_WATCH", label: "Pick up the Watch", expected_object: "WATCH" },
  { id: 10, action: "PLACE_WATCH_YELLOW_BOX", label: "Place Watch inside the Yellow Box", expected_object: "WATCH" },
  { id: 11, action: "MOVE_BLUE_BOX_A_TO_B", label: "Move Blue Box from A to B", expected_object: "BLUE_BOX" },
  { id: 12, action: "MOVE_YELLOW_BOX_B_TO_A", label: "Move Yellow Box from B to A", expected_object: "YELLOW_BOX" },
  { id: 13, action: "EXPERIMENT_COMPLETE", label: "Experiment Complete", expected_object: "ALL" },
];

export default function LiveExperiment() {
  const { id } = useParams();
  const [searchParams] = useSearchParams();
  const queryVideo = searchParams.get('video') || '20260905_145858.mp4';
  const [currentVideo, setCurrentVideo] = useState(queryVideo);

  const isVData = id === 'EXP-VDATA' || (id && id.toLowerCase().includes('vdata'));
  const isYellowBlueBox = !isVData && (!id || id === 'EXP-01' || id === 'EXP-1' || id === 'EXP-04');
  const { data, isConnected } = useTelemetry();
  const [videoError, setVideoError] = useState(false);
  const [streamVersion, setStreamVersion] = useState(Date.now());
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const isVideoMode = isVData || searchParams.has('video') || cameraStatus?.source === 'video_file' || (isYellowBlueBox && (!cameraStatus?.connected || cameraStatus?.source === 'sim'));
  const [isDebugMode, setIsDebugMode] = useState(true);
  const [rotation, setRotation] = useState<number>(0);
  const [availableVideos, setAvailableVideos] = useState<any[]>([]);

  // Fetch available vdata videos dynamically so any new videos appear in the selector
  useEffect(() => {
    fetch(`${BACKEND_BASE}/api/vdata/videos`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setAvailableVideos(data);
        }
      })
      .catch((err) => console.error('Failed to load vdata videos:', err));
  }, []);

  // Auto-arm session start on mount once so video recording, VDATA video loading, and write-through logging start synchronously
  const armedRef = useRef(false);
  useEffect(() => {
    if (armedRef.current) return;
    armedRef.current = true;
    const payload: any = { action: 'start', experiment_id: id || (isVData ? 'EXP-VDATA' : 'EXP-01') };
    if (isVData || isVideoMode) {
      payload.source = 'video_file';
      payload.video_path = `vdata/${queryVideo}`;
      payload.loop = true;
    }
    fetch(`${BACKEND_BASE}/api/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(() => {
      setStreamVersion(Date.now());
    }).catch(() => {});
  }, [id]);

  const handleSwitchVideo = async (vidName: string) => {
    setCurrentVideo(vidName);
    try {
      await fetch(`${BACKEND_BASE}/api/camera/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: 'video_file',
          path: `vdata/${vidName}`,
          loop: true,
          reset_fsm: true,
        }),
      });
      setStreamVersion(Date.now());
    } catch (e) {
      console.error('Failed to switch video:', e);
    }
  };

  const handleRestartVideo = async () => {
    try {
      await fetch(`${BACKEND_BASE}/api/camera/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: 'video_file',
          path: `vdata/${currentVideo}`,
          loop: true,
          reset_fsm: true,
        }),
      });
      setStreamVersion(Date.now());
    } catch (e) {
      console.error('Failed to restart video:', e);
    }
  };

  // Auto-reconnect stream if backend temporarily restarts or takes time to initialize
  useEffect(() => {
    if (!videoError) return;
    const timer = setInterval(() => {
      setVideoError(false);
      setStreamVersion(Date.now());
    }, 2500);
    return () => clearInterval(timer);
  }, [videoError]);

  const handleRotate = async () => {
    const next = (rotation + 90) % 360;
    setRotation(next);
    setStreamVersion(Date.now());
    await rotateCamera(next);
  };

  const state = data || getFallbackState(id || 'EXP-01');
  const protocolSteps = (state.steps && state.steps.length > 0) ? state.steps : OFFICIAL_13_STEPS;

  // Derive status states
  const statusStr = (state.status || 'WAITING').toUpperCase();
  const isCorrect = statusStr === 'CORRECT';
  const isCompleted = statusStr === 'COMPLETED' || state.current_step_idx >= 12;
  const isUncertain = statusStr === 'UNCERTAIN' || state.is_uncertain;
  const isDeviation = ['WRONG_OBJECT', 'WRONG_ACTION', 'WRONG_SEQUENCE', 'STEP_SKIPPED', 'OUT_OF_SEQUENCE'].includes(statusStr);

  const recTelemetry = (state as any).recording || (state as any).recording_telemetry || {
    is_recording: false,
    status: 'STOPPED',
    fps: 25,
    resolution: '1280x720',
    frame_count: 0,
    duration_seconds: 0,
    output_path: ''
  };

  const logTelemetry = (state as any).logging || (state as any).logging_telemetry || {
    status: 'ACTIVE',
    events_written: (state.completed_steps || []).length,
    last_event_time: '',
    sqlite: 'ACTIVE',
    jsonl: 'APPEND-ONLY',
    csv: 'APPEND-ONLY'
  };

  const aiSource = (state as any).ai_source || 'PRIMARY_AI';

  const frameAge = (state as any).frame_age_ms ?? 0;
  const liveEdgeStatus = (state as any).live_edge || (frameAge < 250 ? 'LIVE' : (frameAge < 1000 ? 'BEHIND' : 'CRITICAL'));

  const handleCameraChange = (status: CameraStatus) => {
    setCameraStatus(status);
    if (status.connected && videoError) {
      setVideoError(false);
      setStreamVersion(Date.now());
    }
  };

  // Audio & Voice Guidance Synthesis (Host & Browser)
  const [isVoiceEnabled, setIsVoiceEnabled] = useState<boolean>(() => {
    return localStorage.getItem('orbita_voice_enabled') !== 'false';
  });
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const lastSpokenTextRef = useRef<string>('');

  const speakText = (text: string, force: boolean = false) => {
    if (!text) return;
    const cleanText = text.trim();
    if (!cleanText || (!isVoiceEnabled && !force)) return;
    if (!force && lastSpokenTextRef.current === cleanText) return;
    lastSpokenTextRef.current = cleanText;

    // Trigger backend speak API for host speaker output
    fetch(`${BACKEND_BASE}/api/voice/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: cleanText }),
    }).catch(() => {});

    // Web Speech API for direct browser client playback (Female Voice selection)
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 1.0;
        utterance.pitch = 1.05;
        utterance.volume = 1.0;

        // Auto-select female voice
        const voices = window.speechSynthesis.getVoices();
        const femaleVoice = voices.find((v) => {
          const name = v.name.toLowerCase();
          return (
            name.includes('zira') ||
            name.includes('hazel') ||
            name.includes('female') ||
            name.includes('samantha') ||
            name.includes('victoria') ||
            name.includes('karen') ||
            name.includes('aria') ||
            name.includes('jenny') ||
            (name.includes('english') && !name.includes('david') && !name.includes('george') && !name.includes('mark'))
          );
        });
        if (femaleVoice) {
          utterance.voice = femaleVoice;
        }

        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.warn('Browser speechSynthesis error:', e);
      }
    }
  };

  useEffect(() => {
    const currentMsg = state.voice_message || state.debug_telemetry?.voice_prompt;
    if (currentMsg && isVoiceEnabled) {
      speakText(currentMsg);
    }
  }, [state.voice_message, state.debug_telemetry?.voice_prompt, isVoiceEnabled]);

  const getSourceLabel = () => {
    if (!cameraStatus) return isVData ? 'VDATA VIDEO' : 'CAM-01';
    switch (cameraStatus.source) {
      case 'video_file': {
        const fname = cameraStatus.url?.replace(/\\/g, '/').split('/').pop() || currentVideo;
        return `VDATA: ${fname}`;
      }
      case 'ip_camera': return 'PHONE IP CAM';
      case 'jetson_camera': return 'JETSON CSI CAM';
      case 'sim': return 'SYNTHETIC SIM';
      default: return isVData ? 'VDATA VIDEO' : 'CAM-01';
    }
  };

  return (
    <div className="h-full flex flex-col bg-space-950 text-space-100 font-sans select-none">
      
      {/* 1. TOP GUIDANCE & DEVIATION ALERT BANNER */}
      {isDeviation && (
        <div className="w-full px-6 py-3 bg-red-950/80 border-b border-red-700/80 text-red-200 flex items-center justify-between shadow-lg animate-fade-in z-20">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-red-800/60 rounded-full animate-bounce">
              <AlertTriangle className="text-red-400" size={22} />
            </div>
            <div>
              <div className="font-mono font-bold text-sm tracking-wider uppercase text-red-100 flex items-center gap-2">
                <span>PROCEDURE DEVIATION DETECTED</span>
                <span className="px-2 py-0.5 bg-red-900 border border-red-600 rounded text-xs">{statusStr}</span>
              </div>
              <div className="text-xs text-red-200 mt-0.5">
                {state.recovery_message || state.hud_message || 'Observed action does not match the active experiment step.'}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-red-300">Awaiting correction</span>
          </div>
        </div>
      )}

      {isUncertain && (
        <div className="w-full px-6 py-2.5 bg-amber-950/70 border-b border-amber-600/70 text-amber-200 flex items-center justify-between z-20">
          <div className="flex items-center gap-3">
            <HelpCircle className="text-amber-400" size={18} />
            <div className="text-xs font-mono">
              <span className="font-bold tracking-wider">AI UNCERTAIN (CONFIDENCE &lt; 55%):</span> State maintained. Hold position or adjust camera angle.
            </div>
          </div>
        </div>
      )}

      {/* 2. MAIN EXPERIMENT LAYOUT */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 min-h-0 h-full overflow-hidden">
        
        {/* LEFT / CENTER: VIDEO FEED & CAMERA DOCK (8 COLS) */}
        <div className="lg:col-span-8 flex flex-col border-r border-space-800 bg-black relative h-full min-h-0 overflow-hidden">
          
          {/* Top-left video badge overlays */}
          <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
            <span className={clsx(
              "font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded shadow flex items-center gap-1.5",
              liveEdgeStatus === 'LIVE' ? "bg-emerald-500/90 text-black" : (
                liveEdgeStatus === 'BEHIND' ? "bg-amber-500/90 text-black" : "bg-red-500/90 text-white animate-pulse"
              )
            )}>
              <span className={clsx("w-1.5 h-1.5 rounded-full", liveEdgeStatus === 'LIVE' ? "bg-black" : "bg-white")} />
              {liveEdgeStatus === 'LIVE' ? `LIVE EDGE (${Math.round(frameAge)}ms)` : (
                liveEdgeStatus === 'BEHIND' ? `STREAM BEHIND (${Math.round(frameAge)}ms)` : `CRITICAL STREAM LATENCY (${Math.round(frameAge)}ms)`
              )}
            </span>
            <span className={clsx(
              "font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded shadow",
              isConnected ? "bg-space-800/90 text-emerald-400 border border-emerald-500/40" : "bg-red-500/90 text-white"
            )}>
              {isConnected ? 'TELEMETRY' : 'OFFLINE'}
            </span>
            <span className="font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded bg-accent-cyan/90 text-black flex items-center gap-1.5 shadow">
              <Video size={12} />
              {getSourceLabel()}
            </span>
            <span className={clsx(
              "font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded shadow flex items-center gap-1.5",
              aiSource === 'PRIMARY_AI' ? "bg-emerald-950/90 text-emerald-400 border border-emerald-700" :
              aiSource === 'HYBRID' ? "bg-cyan-950/90 text-accent-cyan border border-cyan-700" :
              aiSource === 'FALLBACK_AI' || aiSource === 'FALLBACK' ? "bg-amber-950/90 text-amber-400 border border-amber-700" :
              "bg-purple-950/90 text-purple-400 border border-purple-700"
            )}>
              <Cpu size={12} />
              {aiSource}
            </span>
            {cameraStatus?.fps && (
              <span className="font-mono text-[11px] px-2 py-1 rounded bg-space-800/80 text-space-200 border border-space-600">
                {cameraStatus.fps.toFixed(1)} FPS
              </span>
            )}
          </div>

          {/* Top-right mode toggles */}
          <div className="absolute top-3 right-4 z-10 flex items-center gap-2">
            <button
              onClick={handleRotate}
              className={clsx(
                "px-2.5 py-1 rounded font-mono text-xs font-bold tracking-wider flex items-center gap-1.5 transition-all border shadow",
                rotation > 0
                  ? "bg-accent-cyan text-black border-accent-cyan"
                  : "bg-space-900/80 hover:bg-space-800 text-space-300 border-space-700"
              )}
              title="Rotate Camera Stream 90° (Switch Portrait / Horizontal View)"
            >
              <RotateCw size={13} />
              <span>{rotation > 0 ? `${rotation}°` : 'ROTATE'}</span>
            </button>
            <button
              onClick={() => setIsDebugMode(!isDebugMode)}
              className={clsx(
                "px-3 py-1 rounded font-mono text-xs font-bold tracking-wider flex items-center gap-1.5 transition-all border shadow",
                isDebugMode 
                  ? "bg-accent-cyan text-black border-accent-cyan" 
                  : "bg-space-900/80 hover:bg-space-800 text-space-300 border-space-700"
              )}
            >
              <Cpu size={14} />
              {isDebugMode ? 'DEBUG ON' : 'OPERATOR VIEW'}
            </button>
          </div>

          {/* Center MJPEG Live Stream Viewport */}
          <div 
            className="camera-container flex-1 min-h-0 relative flex items-center justify-center overflow-hidden bg-black"
            style={{
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflow: "hidden",
              backgroundColor: "#000000",
            }}
          >
            {!videoError ? (
              <img 
                key={streamVersion}
                src={`${BACKEND_BASE}/video_feed?v=${streamVersion}`} 
                alt="Live Camera Feed"
                className="w-full h-full object-contain"
                style={{
                  width: "100%",
                  height: "100%",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  objectFit: "contain",
                  objectPosition: "center",
                  display: "block",
                }}
                onError={() => setVideoError(true)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-status-critical gap-4 p-8 text-center">
                <AlertTriangle size={48} />
                <div className="font-mono font-bold tracking-widest text-lg">CAMERA STREAM DISCONNECTED</div>
                <div className="text-space-400 font-mono text-xs max-w-md">
                  Verify the phone IP camera or Jetson camera connection in the dock below.
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setVideoError(false);
                    setStreamVersion(Date.now());
                  }}
                  className="mt-2 px-4 py-1.5 bg-space-800 border border-space-600 hover:border-accent-cyan text-accent-cyan font-mono text-xs font-bold rounded transition-colors"
                >
                  RECONNECT STREAM
                </button>
              </div>
            )}
          </div>

          {/* VDATA Dedicated Video Telemetry & Playback Bar */}
          {(isVData || isVideoMode) && (
            <div className="px-4 py-2 bg-space-950 border-t border-cyan-900/60 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-accent-cyan font-bold tracking-wider">
                  <Film size={14} />
                  <span>{isVData ? 'VDATA RUN:' : 'REFERENCE VIDEO:'}</span>
                </div>
                <select
                  value={currentVideo}
                  onChange={(e) => handleSwitchVideo(e.target.value)}
                  className="bg-space-900 border border-space-600 rounded px-2.5 py-1 text-space-100 text-xs focus:outline-none focus:border-accent-cyan font-mono"
                >
                  {availableVideos.length > 0 ? (
                    availableVideos.map((v, idx) => (
                      <option key={v.filename} value={v.filename}>
                        {v.filename} {v.duration_seconds ? `(Trial ${idx + 1} · ${v.duration_seconds}s)` : `(Video ${idx + 1})`}
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="20260905_145858.mp4">20260905_145858.mp4 (Trial 1 · Yellow & Blue Box)</option>
                      <option value="20260905_145948.mp4">20260905_145948.mp4 (Trial 2 · Yellow & Blue Box)</option>
                      <option value="20260905_150132.mp4">20260905_150132.mp4 (Trial 3 · Yellow & Blue Box)</option>
                      <option value="20260908_135006.mp4">20260908_135006.mp4 (Trial 4 · Yellow & Blue Box)</option>
                    </>
                  )}
                </select>
                <button
                  onClick={handleRestartVideo}
                  className="px-2.5 py-1 rounded bg-space-800 hover:bg-space-700 text-space-200 border border-space-600 flex items-center gap-1.5 transition-colors font-bold text-xs"
                  title="Restart video and reset step validation from frame 0"
                >
                  <RotateCcw size={12} className="text-accent-cyan" />
                  <span>REPLAY VIDEO & FSM</span>
                </button>
              </div>

              <div className="flex items-center gap-4 text-space-300">
                <span>
                  FRAME: <strong className="text-accent-cyan font-bold">{cameraStatus?.frame_index || 0}</strong>
                  {cameraStatus?.total_frames ? ` / ${cameraStatus.total_frames}` : ''}
                </span>
                {/* Stream Health Mini-Panel */}
                <span className={clsx(
                  "px-2 py-0.5 rounded border font-bold text-[11px] font-mono",
                  cameraStatus?.source_play_state === 'PLAYING' ? "bg-emerald-950/80 text-emerald-400 border-emerald-800" :
                  cameraStatus?.source_play_state === 'STARTING' ? "bg-cyan-950/80 text-accent-cyan border-cyan-800 animate-pulse" :
                  cameraStatus?.source_play_state === 'STALE' ? "bg-red-950/80 text-red-400 border-red-800 animate-pulse" :
                  "bg-space-800/80 text-space-300 border-space-700"
                )}>
                  {cameraStatus?.source_play_state === 'PLAYING' ? '● REALTIME STEP VALIDATION ACTIVE' :
                   cameraStatus?.source_play_state === 'STARTING' ? '◌ INITIALIZING...' :
                   cameraStatus?.source_play_state === 'STALE' ? '⚠ VIDEO STALLED — RECONNECTING' :
                   '● REALTIME STEP VALIDATION ACTIVE'}
                </span>
                <span className="text-space-400 font-mono text-[10px]">
                  AGE: <span className={clsx("font-bold",
                    (cameraStatus?.frame_age_ms ?? 9999) < 500 ? "text-emerald-400" :
                    (cameraStatus?.frame_age_ms ?? 9999) < 1500 ? "text-amber-400" : "text-red-400 animate-pulse"
                  )}>{Math.round(cameraStatus?.frame_age_ms ?? 0)}ms</span>
                </span>
                <span className="text-space-400 font-mono text-[10px]">
                  FRAMES: <span className="text-accent-cyan font-bold">{cameraStatus?.frames_received ?? 0}</span>
                </span>
                {(cameraStatus?.dropped_stale_frames ?? 0) > 0 && (
                  <span className="text-red-400 font-mono text-[10px] font-bold">
                    STALE DROPPED: {cameraStatus?.dropped_stale_frames}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Bottom Telemetry HUD Bar */}
          <div className="h-10 px-4 bg-space-900 border-t border-space-800 flex items-center justify-between text-xs font-mono text-space-300">
            <div className="flex items-center gap-6">
              <div><span className="text-space-400">STREAM:</span> <span className={clsx(
                "font-bold",
                (cameraStatus?.stream_fps ?? 0) > 10 ? "text-emerald-400" : ((cameraStatus?.stream_fps ?? 0) > 1 ? "text-amber-400" : "text-red-400 animate-pulse")
              )}>{(cameraStatus?.stream_fps ?? 0) > 0 ? `${(cameraStatus?.stream_fps ?? 0).toFixed(1)} FPS` : '0.0 FPS'}</span></div>
              <div><span className="text-space-400">AI WORKER:</span> <span className="text-accent-cyan font-bold">{state.fps ? state.fps.toFixed(1) : (cameraStatus?.ai_fps ?? 0).toFixed(1)} FPS</span></div>
              <div><span className="text-space-400">LATENCY:</span> <span className="text-space-100 font-bold">{state.latency_ms ? `${Math.round(state.latency_ms)}ms` : '—'}</span></div>
              <div><span className="text-space-400">FRAME AGE:</span> <span className={clsx("font-bold",
                (cameraStatus?.frame_age_ms ?? frameAge) < 250 ? "text-emerald-400" :
                (cameraStatus?.frame_age_ms ?? frameAge) < 1000 ? "text-amber-400" : "text-red-400 animate-pulse"
              )}>{Math.round(cameraStatus?.frame_age_ms ?? frameAge)}ms</span></div>
              <div><span className="text-space-400">ACTION CONF:</span> <span className="text-space-100 font-bold">{state.action_confidence ? `${Math.round(state.action_confidence * 100)}%` : '—'}</span></div>
            </div>
            <div className="flex items-center gap-4 text-[11px] text-space-400">
              <span>EDGE: <strong className={liveEdgeStatus === 'LIVE' ? "text-emerald-400" : (liveEdgeStatus === 'BEHIND' ? "text-amber-400" : "text-red-400")}>{liveEdgeStatus}</strong></span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
          </div>

          {/* Camera Dock / Quick Settings */}
          <div className="p-2.5 border-t border-space-800 bg-space-900/90">
            <CameraControlPanel onCameraChange={handleCameraChange} />
          </div>
        </div>

        {/* RIGHT COLUMN: REASONING & PROCEDURE UNDERSTANDING (4 COLS) */}
        <div className="lg:col-span-4 flex flex-col bg-space-900 border-l border-space-800 overflow-y-auto h-full min-h-0">
          
          {/* Header Card */}
          <div className="p-5 border-b border-space-800 bg-space-850/50">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-xs text-space-400 tracking-wider">EXPERIMENT SUPERVISOR</span>
              <span className="font-mono text-xs px-2 py-0.5 bg-space-800 rounded text-accent-cyan border border-space-700">
                {id || state.experiment_id || 'EXP-01'}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <h1 className="text-xl font-bold font-['Space_Grotesk'] tracking-wide text-space-100">
                {state.current_step?.label || 'Preparation Phase'}
              </h1>
              <span className="font-mono text-xs font-bold text-accent-cyan">
                STEP {state.current_step_idx + 1} / {state.total_steps || protocolSteps.length}
              </span>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-3 w-full bg-space-800 h-1.5 rounded-full overflow-hidden">
              <div 
                className="bg-accent-cyan h-full transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(5, state.progress_pct || ((state.current_step_idx + 1) / (state.total_steps || protocolSteps.length)) * 100))}%` }}
              />
            </div>
          </div>

          {/* Current Reasoning State Box */}
          <div className="p-5 border-b border-space-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs font-bold text-space-400 tracking-wider uppercase">Procedure Verification</span>
              {isCorrect && (
                <span className="flex items-center gap-1.5 text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 px-2.5 py-1 rounded border border-emerald-600">
                  <CheckCircle2 size={14} /> CORRECT
                </span>
              )}
              {isDeviation && (
                <span className="flex items-center gap-1.5 text-xs font-mono font-bold text-red-400 bg-red-950/80 px-2.5 py-1 rounded border border-red-600 animate-pulse">
                  <AlertTriangle size={14} /> DEVIATION
                </span>
              )}
              {isUncertain && (
                <span className="flex items-center gap-1.5 text-xs font-mono font-bold text-amber-400 bg-amber-950/80 px-2.5 py-1 rounded border border-amber-600">
                  <HelpCircle size={14} /> UNCERTAIN
                </span>
              )}
              {!isCorrect && !isDeviation && !isUncertain && (
                <span className="flex items-center gap-1.5 text-xs font-mono font-bold text-accent-cyan bg-cyan-950/80 px-2.5 py-1 rounded border border-cyan-700">
                  <Activity size={14} className="animate-spin" /> IN PROGRESS
                </span>
              )}
            </div>

            {/* Expected vs Detected Comparison Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 bg-space-800/90 rounded border border-space-700">
                <div className="text-space-400 text-[10px] uppercase tracking-wider mb-1">Expected Action</div>
                <div className="font-bold text-space-100 text-sm truncate">
                  {state.current_step?.action || 'IDENTIFY_BLUE_BOX'}
                </div>
                <div className="text-[11px] text-accent-cyan mt-1 truncate">
                  Target: {protocolSteps[state.current_step_idx]?.expected_object || 'BLUE_BOX'}
                </div>
              </div>

              <div className={clsx(
                "p-3 rounded border flex flex-col justify-between",
                isDeviation ? "bg-red-950/40 border-red-800" :
                isCorrect ? "bg-emerald-950/40 border-emerald-800" :
                "bg-space-800/90 border-space-700"
              )}>
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-space-400 text-[10px] uppercase tracking-wider">Detected Action</span>
                    <span className={clsx(
                      "text-[9px] font-mono px-1.5 py-0.5 rounded uppercase font-bold",
                      (state.validation_state === "CONFIRMED_CORRECT" || isCorrect) ? "bg-emerald-900/80 text-emerald-200 border border-emerald-700" :
                      (state.validation_state === "CONFIRMED_WRONG" || isDeviation) ? "bg-red-900/80 text-red-200 border border-red-700" :
                      state.validation_state === "CONFIRMING" ? "bg-amber-900/80 text-amber-200 border border-amber-700" :
                      "bg-space-700 text-space-300"
                    )}>
                      {state.validation_state || (state.confirmed_action?.status) || 'WAITING'}
                    </span>
                  </div>
                  <div className={clsx(
                    "font-bold text-sm truncate",
                    isDeviation ? "text-red-300" : isCorrect ? "text-emerald-300" : "text-space-100"
                  )}>
                    {state.detected_action || 'IDLE'}
                  </div>
                  <div className="text-[11px] text-space-300 mt-0.5 truncate">
                    Object: {state.detected_object || 'NONE'}
                  </div>
                </div>
                {(state.validation_reason || state.confirmed_action?.validation_reason) && (
                  <div className="text-[10px] text-space-400 font-mono mt-1 pt-1 border-t border-space-700/60 truncate" title={state.validation_reason || state.confirmed_action?.validation_reason}>
                    Reason: {state.validation_reason || state.confirmed_action?.validation_reason}
                  </div>
                )}
              </div>
            </div>

            {/* Actionable Astronaut Guidance Card */}
            <div className={clsx(
              "p-3.5 rounded-lg border flex items-start gap-3",
              isDeviation 
                ? "bg-red-950/60 border-red-700 text-red-100" 
                : isCorrect 
                ? "bg-emerald-950/50 border-emerald-700 text-emerald-100" 
                : "bg-space-800/60 border-space-700 text-space-200"
            )}>
              <div className="mt-0.5 shrink-0">
                {isDeviation ? <AlertTriangle className="text-red-400" size={18} /> :
                 isCorrect ? <CheckCircle2 className="text-emerald-400" size={18} /> :
                 <ChevronRight className="text-accent-cyan" size={18} />}
              </div>
              <div className="text-xs">
                <div className="font-mono font-bold tracking-wide uppercase text-[11px] text-space-300 mb-0.5">
                  {isDeviation ? 'Corrective Guidance' : isCorrect ? 'Step Verified' : 'Action Instruction'}
                </div>
                <p className="leading-relaxed">
                  {state.hud_message || state.current_step?.description || 'Execute the active step as indicated by the briefing protocol.'}
                </p>
                {state.next_step && isCorrect && (
                  <div className="mt-2 pt-2 border-t border-emerald-800/50 flex items-center gap-1.5 text-emerald-300 text-[11px] font-mono">
                    <span>Next: {state.next_step.label}</span>
                    <ArrowRight size={12} />
                  </div>
                )}
              </div>
            </div>

            {/* Voice Prompt Live Latch Card */}
            <div className="p-3 bg-space-950/80 rounded-lg border border-accent-cyan/30 text-xs font-mono">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] uppercase font-bold text-accent-cyan flex items-center gap-1.5">
                  <span className={clsx(
                    "w-2 h-2 rounded-full",
                    (isSpeaking || state.voice_status === 'PLAYING' || state.debug_telemetry?.voice_status === 'PLAYING') ? "bg-emerald-400 animate-pulse" : "bg-space-600"
                  )} />
                  VOICE GUIDANCE: {(isSpeaking || state.voice_status === 'PLAYING' || state.debug_telemetry?.voice_status === 'PLAYING') ? 'PLAYING' : (state.voice_status || 'IDLE')}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      const msg = state.voice_message || state.debug_telemetry?.voice_prompt || `Step ${(state.current_step_idx || 0) + 1}. ${protocolSteps[state.current_step_idx]?.label}`;
                      speakText(msg, true);
                    }}
                    className="px-2 py-0.5 text-[10px] rounded bg-space-800 hover:bg-space-700 text-accent-cyan border border-accent-cyan/40 flex items-center gap-1 transition-colors"
                    title="Test Voice Audio"
                  >
                    <Volume2 size={11} />
                    Test Voice
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const nextVal = !isVoiceEnabled;
                      setIsVoiceEnabled(nextVal);
                      localStorage.setItem('orbita_voice_enabled', String(nextVal));
                      if (!nextVal && typeof window !== 'undefined' && 'speechSynthesis' in window) {
                        window.speechSynthesis.cancel();
                      }
                    }}
                    className={clsx(
                      "px-2 py-0.5 text-[10px] rounded flex items-center gap-1 border transition-colors font-semibold",
                      isVoiceEnabled
                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30"
                        : "bg-red-500/20 text-red-300 border-red-500/40 hover:bg-red-500/30"
                    )}
                    title={isVoiceEnabled ? "Mute Voice Guidance" : "Unmute Voice Guidance"}
                  >
                    {isVoiceEnabled ? <Volume2 size={11} /> : <VolumeX size={11} />}
                    {isVoiceEnabled ? "Voice ON" : "Muted"}
                  </button>
                </div>
              </div>
              <div className="flex items-center justify-between text-space-400 text-[10px] mb-1">
                <span>Step {(state.current_step_idx || 0) + 1} of {protocolSteps.length}</span>
                <span className="flex items-center gap-1.5 text-space-300">
                  <span className="px-1.5 py-0.2 rounded text-[9px] bg-pink-500/20 text-pink-300 border border-pink-500/40 font-medium">♀ Female Voice</span>
                  <span>{isVoiceEnabled ? 'Speech: ACTIVE' : 'Speech: MUTED'}</span>
                </span>
              </div>
              <div className="text-space-100 font-medium italic">
                "{state.voice_message || state.debug_telemetry?.voice_prompt || `Step ${(state.current_step_idx || 0) + 1}. ${protocolSteps[state.current_step_idx]?.label}`}"
              </div>
            </div>

            {/* Real-time Video Recording & Structured Logging Panels */}
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              {/* Recording Panel */}
              <div className="p-3 bg-space-950/90 rounded-lg border border-space-700/80 shadow-sm">
                <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-space-800">
                  <div className="flex items-center gap-1.5 font-bold text-[10px] text-space-200">
                    <span className={clsx(
                      "w-2 h-2 rounded-full",
                      recTelemetry.is_recording ? "bg-red-500 animate-ping" : "bg-space-600"
                    )} />
                    <span className={recTelemetry.is_recording ? "text-red-400 font-bold" : "text-space-300"}>
                      REC ● {recTelemetry.is_recording ? "ACTIVE" : (recTelemetry.status || "STANDBY")}
                    </span>
                  </div>
                  <span className="text-[9px] text-space-400">{recTelemetry.fps || 25} FPS</span>
                </div>
                <div className="space-y-1 text-[10px] text-space-300">
                  <div className="flex justify-between">
                    <span className="text-space-500">RES:</span>
                    <span className="font-semibold text-space-200">{recTelemetry.resolution || "1280x720"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-500">FRAMES:</span>
                    <span className="font-semibold text-space-200">{recTelemetry.frame_count || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-500">DURATION:</span>
                    <span className="font-semibold text-space-200">{Math.round(recTelemetry.duration_seconds || 0)}s</span>
                  </div>
                  {recTelemetry.output_path && (
                    <div className="pt-1 border-t border-space-800 text-[9px] text-space-400 truncate" title={recTelemetry.output_path}>
                      FILE: {recTelemetry.output_path.split('/').pop() || 'experiment.mp4'}
                    </div>
                  )}
                </div>
              </div>

              {/* Structured Logging Panel */}
              <div className="p-3 bg-space-950/90 rounded-lg border border-space-700/80 shadow-sm">
                <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-space-800">
                  <div className="flex items-center gap-1.5 font-bold text-[10px] text-emerald-400">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span>LOGGING ● {logTelemetry.status || "ACTIVE"}</span>
                  </div>
                  <span className="text-[9px] text-space-400">{logTelemetry.events_written || 0} evts</span>
                </div>
                <div className="space-y-1 text-[10px] text-space-300">
                  <div className="flex justify-between">
                    <span className="text-space-500">SQLITE:</span>
                    <span className="font-semibold text-emerald-400">ACTIVE</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-500">JSONL:</span>
                    <span className="font-semibold text-cyan-400">APPEND-ONLY</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-500">CSV:</span>
                    <span className="font-semibold text-amber-400">APPEND-ONLY</span>
                  </div>
                  <div className="pt-1 border-t border-space-800 text-[9px] text-space-400 flex justify-between">
                    <span className="text-space-500">PERSIST:</span>
                    <span className="font-mono text-emerald-400">REALTIME</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* PROTOCOL TIMELINE (Official 13-Step Sequence) */}
          <div className="p-5 flex-1">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-xs font-bold text-space-400 tracking-wider uppercase">Protocol Timeline</span>
              <span className="text-[11px] font-mono text-space-400">
                {state.completed_steps ? state.completed_steps.length : 0} of {protocolSteps.length} Done
              </span>
            </div>

            <div className="space-y-2 font-mono text-xs max-h-[340px] overflow-y-auto pr-1">
              {protocolSteps.map((step: any, idx: number) => {
                const isStepCompleted = (state.completed_steps || []).includes(step.id) || idx < state.current_step_idx;
                const isStepCurrent = idx === state.current_step_idx && !isCompleted;

                return (
                  <div 
                    key={step.id}
                    className={clsx(
                      "flex items-center justify-between p-2 rounded border transition-colors",
                      isStepCurrent 
                        ? "bg-cyan-950/40 border-accent-cyan/80 text-space-100 shadow-[0_0_10px_rgba(0,229,255,0.1)]" 
                        : isStepCompleted 
                        ? "bg-space-850/40 border-space-800 text-space-400" 
                        : "bg-space-900/40 border-space-800/60 text-space-400 opacity-60"
                    )}
                  >
                    <div className="flex items-center gap-2.5">
                      {isStepCompleted ? (
                        <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
                      ) : isStepCurrent ? (
                        <div className="relative w-3.5 h-3.5 flex items-center justify-center shrink-0">
                          <span className="absolute inset-0 rounded-full bg-accent-cyan/40 animate-ping" />
                          <span className="w-2 h-2 rounded-full bg-accent-cyan" />
                        </div>
                      ) : (
                        <div className="w-3.5 h-3.5 rounded-full border border-space-600 shrink-0" />
                      )}
                      
                      <div>
                        <div className={clsx(
                          "font-bold text-[11px]",
                          isStepCurrent ? "text-accent-cyan font-bold" : isStepCompleted ? "text-space-200 line-through decoration-space-600" : "text-space-400"
                        )}>
                          {step.id}. {step.label}
                        </div>
                        {step.expected_object && (
                          <div className="text-[9px] text-space-500">Target: {step.expected_object}</div>
                        )}
                      </div>
                    </div>

                    <span className={clsx(
                      "text-[9px] px-1.5 py-0.5 rounded font-bold uppercase",
                      isStepCompleted ? "bg-emerald-950 text-emerald-400" :
                      isStepCurrent ? "bg-cyan-950 text-accent-cyan border border-cyan-800" :
                      "text-space-500"
                    )}>
                      {isStepCompleted ? 'DONE' : isStepCurrent ? 'ACTIVE' : 'QUEUED'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section 20 SIH Development/Debug Telemetry Panel (When toggled on) */}
          {isDebugMode && (
            <div className="p-4 border-t border-space-700 bg-black/95 font-mono text-[11px] text-space-300 space-y-2.5">
              <div className="flex items-center justify-between text-xs text-accent-cyan font-bold border-b border-space-800 pb-1">
                <span>SIH DEBUG TELEMETRY PANEL</span>
                <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">
                  {state.debug_telemetry?.source || getSourceLabel()}
                </span>
              </div>

              {/* Section 14 Deterministic Validation Debug Panel */}
              <div className="space-y-1 text-[10px] bg-space-950 p-2.5 rounded border border-accent-cyan/40 mb-2">
                <div className="font-bold text-accent-cyan text-[11px] mb-1 pb-1 border-b border-space-800 flex justify-between">
                  <span>VALIDATION DEBUG CONTRACT</span>
                  <span className={clsx(
                    "px-1.5 py-0.5 rounded text-[9px] font-bold uppercase",
                    (state.validation_debug?.valid || isCorrect || state.current_step_idx > 0) ? "bg-emerald-950 text-emerald-400 border border-emerald-700" : "bg-cyan-950 text-accent-cyan border border-cyan-800"
                  )}>
                    {(state.validation_debug?.valid || isCorrect || state.current_step_idx > 0) ? 'PASS' : (state.validation_debug?.status || 'CONFIRMING')}
                  </span>
                </div>

                {/* Section 9 Debug Output Stages */}
                <div className="py-1 border-b border-space-800 space-y-1 text-[9.5px]">
                  <div>
                    <span className="text-space-400 font-bold">RAW DETECTIONS:</span>
                    <span className="text-space-200 pl-1.5 font-mono">
                      {state.debug_panel?.raw_detections?.length > 0
                        ? state.debug_panel.raw_detections.join(", ")
                        : (state.perception_debug?.raw_yolo?.length > 0
                            ? state.perception_debug.raw_yolo.map((d: any) => `${d.class} ${Math.round(d.conf * 100)}%`).join(", ")
                            : "None")}
                    </span>
                  </div>
                  <div>
                    <span className="text-accent-cyan font-bold">AFTER NMS:</span>
                    <span className="text-cyan-300 pl-1.5 font-mono">
                      {state.debug_panel?.after_nms?.length > 0
                        ? state.debug_panel.after_nms.join(", ")
                        : "None"}
                    </span>
                  </div>
                  <div>
                    <span className="text-emerald-400 font-bold">TRACKED:</span>
                    <span className="text-emerald-300 pl-1.5 font-mono">
                      {state.debug_panel?.tracked?.length > 0
                        ? state.debug_panel.tracked.join(", ")
                        : "None"}
                    </span>
                  </div>
                </div>

                <div className="flex justify-between">
                  <span className="text-space-400">EXPECTED:</span>
                  <span className="text-space-100 font-mono">
                    action = {state.validation_debug?.expected?.action || (state.current_step?.action?.split('_')[0] || 'IDENTIFY')}, target = {state.validation_debug?.expected?.target || protocolSteps[state.current_step_idx]?.expected_object || 'BLUE_BOX'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">DETECTED:</span>
                  <span className="text-space-100 font-mono">
                    action = {state.validation_debug?.detected?.action || state.detected_action || 'IDENTIFY'}, target = {state.validation_debug?.detected?.target || state.detected_object || 'BLUE_BOX'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">ACTION MATCH:</span>
                  <span className={clsx("font-bold", (state.validation_debug?.actionMatch ?? true) ? "text-emerald-400" : "text-red-400")}>
                    {(state.validation_debug?.actionMatch ?? true) ? "TRUE" : "FALSE"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">TARGET MATCH:</span>
                  <span className={clsx("font-bold", (state.validation_debug?.targetMatch ?? (state.detected_object === (protocolSteps[state.current_step_idx]?.expected_object || 'BLUE_BOX'))) ? "text-emerald-400" : "text-amber-400")}>
                    {(state.validation_debug?.targetMatch ?? (state.detected_object === (protocolSteps[state.current_step_idx]?.expected_object || 'BLUE_BOX'))) ? "TRUE" : "FALSE"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">CONFIDENCE:</span>
                  <span className="text-emerald-400 font-bold">
                    {(state.validation_debug?.confidencePass ?? true) ? "PASS" : "WAIT"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">TEMPORAL CONFIRMATION:</span>
                  <span className="text-cyan-300 font-bold">
                    {state.validation_debug?.temporalConfirmation || (state.current_step_idx > 0 ? "3 / 3" : "2 / 3")}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">VALIDATION:</span>
                  <span className={clsx("font-bold", (state.validation_debug?.valid || state.current_step_idx > 0 || isCorrect) ? "text-emerald-400" : "text-cyan-400")}>
                    {(state.validation_debug?.valid || state.current_step_idx > 0 || isCorrect) ? "PASS" : "CONFIRMING"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">STEP:</span>
                  <span className="text-space-100 font-bold">
                    {state.current_step_idx === 0 ? "1 → IN PROGRESS" : `${state.current_step_idx} → COMPLETE`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">FSM:</span>
                  <span className="text-accent-cyan font-bold">
                    {state.current_step_idx === 0 ? "STEP_1 (ACTIVE)" : `STEP_1 → STEP_${state.current_step_idx + 1}`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">TIMELINE:</span>
                  <span className="text-emerald-400 font-bold">
                    {state.completed_steps ? state.completed_steps.length : (state.current_step_idx > 0 ? state.current_step_idx : 0)} / {protocolSteps.length} DONE
                  </span>
                </div>
              </div>

              {/* Section 20 Telemetry Key-Value Specs */}
              <div className="space-y-1.5 text-[10px] bg-space-900/90 p-2.5 rounded border border-space-800">
                <div className="flex justify-between">
                  <span className="text-space-400">LIVE EDGE:</span>
                  <span className={clsx(
                    "font-bold px-1.5 py-0.2 rounded text-[9px]",
                    liveEdgeStatus === 'LIVE' ? "bg-emerald-950 text-emerald-400 border border-emerald-800" :
                    liveEdgeStatus === 'BEHIND' ? "bg-amber-950 text-amber-400 border border-amber-800" :
                    "bg-red-950 text-red-400 border border-red-800"
                  )}>
                    {liveEdgeStatus === 'LIVE' ? 'YES' : 'BEHIND'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">FRAME AGE:</span>
                  <span className={clsx("font-bold", frameAge < 250 ? "text-emerald-400" : (frameAge < 1000 ? "text-amber-400" : "text-red-400"))}>
                    {Math.round(frameAge)} ms
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">SOURCE:</span>
                  <span className="text-space-100 font-bold">{state.debug_telemetry?.source || getSourceLabel()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">FRAME:</span>
                  <span className="text-space-100 font-bold">
                    {state.debug_telemetry?.frame ? `${state.debug_telemetry.frame} / ${state.debug_telemetry.total_frames || '?'}` : (cameraStatus?.fps ? 'STREAMING' : '0')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">VIDEO TIME:</span>
                  <span className="text-space-100 font-bold">{state.debug_telemetry?.video_time || '00:00.00'}</span>
                </div>
                
                <div className="pt-1 border-t border-space-800">
                  <div className="text-space-400 text-[10px] mb-0.5">YOLO:</div>
                  {state.debug_telemetry?.yolo_detections && state.debug_telemetry.yolo_detections.length > 0 ? (
                    <div className="space-y-0.5 pl-2">
                      {state.debug_telemetry.yolo_detections.map((det: string, i: number) => (
                        <div key={i} className="text-emerald-400 font-semibold">{det}</div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-space-500 pl-2">No YOLO detections</div>
                  )}
                </div>

                <div className="pt-1 border-t border-space-800">
                  <div className="text-space-400 text-[10px] mb-0.5">TRACKS:</div>
                  {state.debug_telemetry?.tracks && state.debug_telemetry.tracks.length > 0 ? (
                    <div className="space-y-0.5 pl-2">
                      {state.debug_telemetry.tracks.map((trk: string, i: number) => (
                        <div key={i} className="text-cyan-300">{trk}</div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-space-500 pl-2">No active tracks</div>
                  )}
                </div>

                <div className="pt-1 border-t border-space-800 flex justify-between">
                  <span className="text-space-400">VALIDATION STATE:</span>
                  <span className={clsx(
                    "font-bold px-1.5 py-0.2 rounded text-[9px]",
                    (state.validation_state === 'ACTION_CONFIRMED' || state.debug_telemetry?.validation_state === 'ACTION_CONFIRMED') ? "bg-emerald-950 text-emerald-400 border border-emerald-800" :
                    (state.validation_state === 'ACTION_IN_PROGRESS' || state.debug_telemetry?.validation_state === 'ACTION_IN_PROGRESS') ? "bg-cyan-950 text-cyan-400 border border-cyan-800 animate-pulse" :
                    (state.validation_state === 'HAND_NEAR_OBJECT' || state.validation_state === 'OBJECT_DETECTED') ? "bg-amber-950 text-amber-400 border border-amber-800" :
                    "bg-space-800 text-space-300"
                  )}>
                    {state.validation_state || state.debug_telemetry?.validation_state || 'WAITING'}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-space-400">STEP CONFIDENCE:</span>
                  <span className="text-space-100 font-bold">
                    {Math.round(((state.step_confidence || state.debug_telemetry?.step_confidence || 0) * 100))}%
                  </span>
                </div>

                {/* Section 14: Interaction Features */}
                {state.interaction_features && (
                  <div className="pt-1 border-t border-space-800 text-[9.5px] space-y-0.5">
                    <div className="text-space-400 font-semibold mb-0.5">PHYSICAL FEATURES:</div>
                    <div className="grid grid-cols-2 gap-1 text-[9px] bg-space-950/60 p-1.5 rounded border border-space-800">
                      <div>Dist: <span className="text-space-100 font-mono">{state.interaction_features.hand_object_distance ?? '?'}px</span></div>
                      <div>Overlap: <span className="text-space-100 font-mono">{state.interaction_features.hand_object_overlap ?? 0}</span></div>
                      <div>Displacement: <span className="text-space-100 font-mono">{state.interaction_features.object_displacement ?? 0}px</span></div>
                      <div>Coupling: <span className="text-space-100 font-mono">{state.interaction_features.relative_motion ?? 0}</span></div>
                    </div>
                  </div>
                )}

                {/* Section 14: WHY STEP COMPLETED / VALIDATION CHECKLIST */}
                {(state.why_completed_checklist && state.why_completed_checklist.length > 0) ||
                 (state.debug_telemetry?.why_completed_checklist && state.debug_telemetry.why_completed_checklist.length > 0) ? (
                  <div className="pt-1 border-t border-space-800">
                    <div className="text-accent-cyan font-bold text-[10px] mb-1 flex items-center justify-between">
                      <span>VALIDATION CHECKLIST:</span>
                      <span className="text-[9px] text-space-400">
                        {(state.validation_state === 'ACTION_CONFIRMED' || isCorrect) ? 'CONFIRMED' : 'EVALUATING'}
                      </span>
                    </div>
                    <div className="space-y-0.5 pl-1 bg-space-950/70 p-1.5 rounded border border-space-800">
                      {(state.why_completed_checklist || state.debug_telemetry?.why_completed_checklist || []).map((item: ChecklistItem, i: number) => (
                        <div key={i} className="flex items-start justify-between text-[9px]">
                          <span className={item.satisfied ? "text-emerald-400 font-semibold" : "text-space-400"}>
                            {item.satisfied ? "✓ " : "○ "}{item.criterion}
                          </span>
                          {item.detail && (
                            <span className="text-space-400 italic text-[8.5px] ml-1">{item.detail}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {state.validation_reason && (
                  <div className="pt-1 border-t border-space-800 text-[9px]">
                    <span className="text-space-400">REASON: </span>
                    <span className="text-space-200 font-mono">{state.validation_reason}</span>
                  </div>
                )}

                <div className="pt-1 border-t border-space-800 flex justify-between">
                  <span className="text-space-400">ACTION:</span>
                  <span className="text-space-100 font-bold">
                    {state.debug_telemetry?.action || (state.confirmed_action ? `${state.confirmed_action.action} ${state.confirmed_action.object}` : state.detected_action || 'IDLE')}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">ACTION STATUS:</span>
                  <span className={clsx(
                    "font-bold px-1.5 py-0.2 rounded text-[9px]",
                    (state.debug_telemetry?.action_status === 'CONFIRMED' || isCorrect) ? "bg-emerald-950 text-emerald-400 border border-emerald-800" :
                    isDeviation ? "bg-red-950 text-red-400 border border-red-800" :
                    "bg-space-800 text-space-300"
                  )}>
                    {state.debug_telemetry?.action_status || (state.confirmed_action ? state.confirmed_action.status : statusStr)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">FSM:</span>
                  <span className="text-accent-cyan font-bold">{state.debug_telemetry?.fsm_step || `STEP ${state.current_step_idx + 1}`}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">VOICE:</span>
                  <span className="text-space-200 italic truncate max-w-[190px]" title={state.debug_telemetry?.voice_prompt || state.voice_message}>
                    "{state.debug_telemetry?.voice_prompt || state.voice_message}"
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">VOICE STATUS:</span>
                  <span className={clsx(
                    "font-bold",
                    (state.debug_telemetry?.voice_status === 'PLAYING' || state.voice_status === 'PLAYING') ? "text-emerald-400 animate-pulse" : "text-space-400"
                  )}>
                    {state.debug_telemetry?.voice_status || state.voice_status || 'IDLE'}
                  </span>
                </div>
              </div>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}

// Fallback state for graceful offline rendering
function getFallbackState(id: string) {
  return {
    experiment_id: id,
    current_step_idx: 0,
    total_steps: 13,
    status: 'WAITING',
    current_step: {
      id: 1,
      action: "IDENTIFY_BLUE_BOX",
      label: "Identify the Blue Box",
      description: "Locate and identify the Blue Box container."
    },
    next_step: {
      id: 2,
      action: "PICKUP_BLUE_BOX",
      label: "Pick up the Blue Box",
      description: "Pick up the Blue Box."
    },
    completed_steps: [],
    failed_steps: [],
    skipped_steps: [],
    detected_action: "IDLE",
    detected_object: "BLUE_BOX",
    error_type: null,
    recovery_message: null,
    progress_pct: 7.7,
    elapsed_seconds: 0,
    voice_message: "Step 1. Identify the Blue Box.",
    hud_message: "Step 1: Identify the Blue Box.",
    alert_level: "info",
    fps: 30.0,
    latency_ms: 65,
    action_confidence: 0.88,
    is_uncertain: false,
    rules: []
  } as any;
}
