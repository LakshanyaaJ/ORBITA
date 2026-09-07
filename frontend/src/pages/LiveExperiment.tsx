import { useState } from 'react';
import { useParams } from 'react-router-dom';
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
  RotateCw
} from 'lucide-react';
import CameraControlPanel from '../components/camera/CameraControlPanel';
import { rotateCamera, type CameraStatus } from '../api/camera';

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
  const { data, isConnected } = useTelemetry();
  const [videoError, setVideoError] = useState(false);
  const [streamVersion, setStreamVersion] = useState(Date.now());
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [isDebugMode, setIsDebugMode] = useState(true);
  const [rotation, setRotation] = useState<number>(0);

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

  const handleCameraChange = (status: CameraStatus) => {
    setCameraStatus(status);
    if (status.connected && videoError) {
      setVideoError(false);
      setStreamVersion(Date.now());
    }
  };

  const getSourceLabel = () => {
    if (!cameraStatus) return 'CAM-01';
    switch (cameraStatus.source) {
      case 'ip_camera': return 'PHONE IP CAM';
      case 'jetson_camera': return 'JETSON CSI CAM';
      case 'sim': return 'SYNTHETIC SIM';
      default: return 'CAM-01';
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
      <div className="flex-1 grid grid-cols-1 xl:grid-cols-12 min-h-0 overflow-hidden">
        
        {/* LEFT / CENTER: VIDEO FEED & CAMERA DOCK (8 COLS) */}
        <div className="xl:col-span-8 flex flex-col border-r border-space-800 bg-black relative">
          
          {/* Top-left video badge overlays */}
          <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
            <span className={clsx(
              "font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded shadow",
              isConnected ? "bg-emerald-500/90 text-black" : "bg-red-500/90 text-white"
            )}>
              {isConnected ? 'LIVE TELEMETRY' : 'TELEMETRY OFFLINE'}
            </span>
            <span className="font-mono text-[11px] font-bold tracking-widest px-2 py-1 rounded bg-accent-cyan/90 text-black flex items-center gap-1.5 shadow">
              <Video size={12} />
              {getSourceLabel()}
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
            className="camera-container flex-1 relative flex items-center justify-center overflow-hidden bg-black"
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
                src={`http://localhost:8000/video_feed?v=${streamVersion}`} 
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

          {/* Bottom Telemetry HUD Bar */}
          <div className="h-10 px-4 bg-space-900 border-t border-space-800 flex items-center justify-between text-xs font-mono text-space-300">
            <div className="flex items-center gap-6">
              <div><span className="text-space-400">STREAM:</span> <span className="text-emerald-400 font-bold">{cameraStatus?.fps ? cameraStatus.fps.toFixed(1) : '30.0'} FPS</span></div>
              <div><span className="text-space-400">AI WORKER:</span> <span className="text-accent-cyan font-bold">{state.fps ? state.fps.toFixed(1) : '14.5'} FPS</span></div>
              <div><span className="text-space-400">LATENCY:</span> <span className="text-space-100 font-bold">{state.latency_ms ? `${Math.round(state.latency_ms)}ms` : '62ms'}</span></div>
              <div><span className="text-space-400">ACTION CONF:</span> <span className="text-space-100 font-bold">{state.action_confidence ? `${Math.round(state.action_confidence * 100)}%` : '88%'}</span></div>
            </div>
            <div className="flex items-center gap-4 text-[11px] text-space-400">
              <span>MODEL: <strong className="text-space-200">v1-production</strong></span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
          </div>

          {/* Camera Dock / Quick Settings */}
          <div className="p-2.5 border-t border-space-800 bg-space-900/90">
            <CameraControlPanel onCameraChange={handleCameraChange} />
          </div>
        </div>

        {/* RIGHT COLUMN: REASONING & PROCEDURE UNDERSTANDING (4 COLS) */}
        <div className="xl:col-span-4 flex flex-col bg-space-900 border-l border-space-800 overflow-y-auto">
          
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
                "p-3 rounded border",
                isDeviation ? "bg-red-950/40 border-red-800" :
                isCorrect ? "bg-emerald-950/40 border-emerald-800" :
                "bg-space-800/90 border-space-700"
              )}>
                <div className="text-space-400 text-[10px] uppercase tracking-wider mb-1">Detected Live</div>
                <div className={clsx(
                  "font-bold text-sm truncate",
                  isDeviation ? "text-red-300" : isCorrect ? "text-emerald-300" : "text-space-100"
                )}>
                  {state.detected_action || 'EVALUATING'}
                </div>
                <div className="text-[11px] text-space-300 mt-1 truncate">
                  Object: {state.detected_object || 'NONE'}
                </div>
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
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] uppercase font-bold text-accent-cyan flex items-center gap-1.5">
                  <span className={clsx(
                    "w-2 h-2 rounded-full",
                    (state.voice_status === 'PLAYING' || state.debug_telemetry?.voice_status === 'PLAYING') ? "bg-emerald-400 animate-pulse" : "bg-space-600"
                  )} />
                  VOICE GUIDANCE: {(state.voice_status || state.debug_telemetry?.voice_status || 'IDLE')}
                </span>
                <span className="text-[10px] text-space-400">Step {(state.current_step_idx || 0) + 1} of {protocolSteps.length}</span>
              </div>
              <div className="text-space-100 font-medium italic">
                "{state.voice_message || state.debug_telemetry?.voice_prompt || `Step ${state.current_step_idx + 1}. ${protocolSteps[state.current_step_idx]?.label}`}"
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

              {/* Section 20 Telemetry Key-Value Specs */}
              <div className="space-y-1.5 text-[10px] bg-space-900/90 p-2.5 rounded border border-space-800">
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
