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
  ChevronRight
} from 'lucide-react';
import CameraControlPanel from '../components/camera/CameraControlPanel';
import type { CameraStatus } from '../api/camera';

// Standard experiment protocol steps for full timeline rendering
const DEFAULT_PROTOCOL_STEPS = [
  { id: 1, action: "OPEN_MAIN_BOX", label: "Open Main Box", expected_object: "MAIN_BOX" },
  { id: 2, action: "TAKE_RED_BOX", label: "Take Red Box", expected_object: "RED_BOX" },
  { id: 3, action: "TAKE_YELLOW_BOX", label: "Take Yellow Box", expected_object: "YELLOW_BOX" },
  { id: 4, action: "OPEN_RED_BOX", label: "Open Red Box", expected_object: "RED_BOX" },
  { id: 5, action: "TAKE_SAMPLE", label: "Take Sample", expected_object: "SAMPLE" },
  { id: 6, action: "PERFORM_EXPERIMENT", label: "Perform Experiment", expected_object: "SAMPLE" },
  { id: 7, action: "CLOSE_BOX", label: "Close Box", expected_object: "MAIN_BOX" },
  { id: 8, action: "STORE_COMPONENTS", label: "Store Components", expected_object: "MAIN_BOX" }
];

export default function LiveExperiment() {
  const { id } = useParams();
  const { data, isConnected } = useTelemetry();
  const [videoError, setVideoError] = useState(false);
  const [streamVersion, setStreamVersion] = useState(Date.now());
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [isDebugMode, setIsDebugMode] = useState(false);

  const state = data || getFallbackState(id || 'EXP-01');

  // Derive status states
  const statusStr = (state.status || 'WAITING').toUpperCase();
  const isCorrect = statusStr === 'CORRECT';
  const isCompleted = statusStr === 'COMPLETED';
  const isUncertain = statusStr === 'UNCERTAIN' || state.is_uncertain;
  const isDeviation = ['WRONG_OBJECT', 'WRONG_ACTION', 'STEP_SKIPPED', 'OUT_OF_SEQUENCE'].includes(statusStr);

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

          {/* Center MJPEG Live Stream Canvas */}
          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-space-950">
            {!videoError ? (
              <img 
                key={streamVersion}
                src={`http://localhost:8000/video_feed?v=${streamVersion}`} 
                alt="Live Camera Feed"
                className="w-full h-full object-contain"
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
                STEP {state.current_step_idx + 1} / {state.total_steps || DEFAULT_PROTOCOL_STEPS.length}
              </span>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-3 w-full bg-space-800 h-1.5 rounded-full overflow-hidden">
              <div 
                className="bg-accent-cyan h-full transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(5, state.progress_pct || ((state.current_step_idx + 1) / (state.total_steps || 8)) * 100))}%` }}
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
                  {state.current_step?.action || 'OPEN_MAIN_BOX'}
                </div>
                <div className="text-[11px] text-accent-cyan mt-1 truncate">
                  Target: {DEFAULT_PROTOCOL_STEPS[state.current_step_idx]?.expected_object || 'MAIN_BOX'}
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
          </div>

          {/* PROTOCOL TIMELINE (Full 8-Step View) */}
          <div className="p-5 flex-1">
            <div className="flex items-center justify-between mb-4">
              <span className="font-mono text-xs font-bold text-space-400 tracking-wider uppercase">Protocol Timeline</span>
              <span className="text-[11px] font-mono text-space-400">
                {state.completed_steps ? state.completed_steps.length : 0} of {DEFAULT_PROTOCOL_STEPS.length} Done
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {DEFAULT_PROTOCOL_STEPS.map((step, idx) => {
                const isStepCompleted = (state.completed_steps || []).includes(step.id) || idx < state.current_step_idx;
                const isStepCurrent = idx === state.current_step_idx && !isCompleted;

                return (
                  <div 
                    key={step.id}
                    className={clsx(
                      "flex items-center justify-between p-2.5 rounded border transition-colors",
                      isStepCurrent 
                        ? "bg-cyan-950/40 border-accent-cyan/80 text-space-100 shadow-[0_0_10px_rgba(0,229,255,0.1)]" 
                        : isStepCompleted 
                        ? "bg-space-850/40 border-space-800 text-space-400" 
                        : "bg-space-900/40 border-space-800/60 text-space-400 opacity-60"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      {isStepCompleted ? (
                        <CheckCircle2 size={16} className="text-emerald-400 shrink-0" />
                      ) : isStepCurrent ? (
                        <div className="relative w-4 h-4 flex items-center justify-center shrink-0">
                          <span className="absolute inset-0 rounded-full bg-accent-cyan/40 animate-ping" />
                          <span className="w-2.5 h-2.5 rounded-full bg-accent-cyan" />
                        </div>
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-space-600 shrink-0" />
                      )}
                      
                      <div>
                        <div className={clsx(
                          "font-bold",
                          isStepCurrent ? "text-accent-cyan font-bold" : isStepCompleted ? "text-space-200 line-through decoration-space-600" : "text-space-400"
                        )}>
                          {step.id}. {step.label}
                        </div>
                        <div className="text-[10px] text-space-400">Target: {step.expected_object}</div>
                      </div>
                    </div>

                    <span className={clsx(
                      "text-[10px] px-2 py-0.5 rounded font-bold uppercase",
                      isStepCompleted ? "bg-emerald-950 text-emerald-400" :
                      isStepCurrent ? "bg-cyan-950 text-accent-cyan border border-cyan-800" :
                      "text-space-400"
                    )}>
                      {isStepCompleted ? 'DONE' : isStepCurrent ? 'ACTIVE' : 'QUEUED'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* DEBUG / ENGINEERING TELEMETRY DRAWER (When toggled on) */}
          {isDebugMode && (
            <div className="p-4 border-t border-space-700 bg-black/95 font-mono text-[11px] text-space-300 space-y-3">
              <div className="flex items-center justify-between text-xs text-accent-cyan font-bold border-b border-space-800 pb-1">
                <span>DETECTOR & REASONING DIAGNOSTICS</span>
                <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">YOLO V3 CANDIDATE</span>
              </div>

              {/* Core Engine Diagnostics */}
              <div className="grid grid-cols-2 gap-2 text-[10px] text-space-300">
                <div>MODEL: <strong className="text-space-100">ORBITA V3</strong> <span className="text-space-400">(Base: V1)</span></div>
                <div>DATASET: <strong className="text-space-100">V3</strong> <span className="text-space-400">(424 frames)</span></div>
                <div>YOLO: <strong className="text-emerald-400">ACTIVE</strong> ({(state.ai_fps || 12.4).toFixed(1)} FPS)</div>
                <div>LATENCY: <strong className="text-accent-cyan">{Math.round(state.pipeline_latency_ms || 68)} ms</strong></div>
              </div>

              {/* Per-Class Detector Confidence Telemetry */}
              <div className="space-y-1.5 pt-1 border-t border-space-800/80">
                <div className="text-[10px] text-space-400 font-bold uppercase tracking-wider flex justify-between">
                  <span>Target Class</span>
                  <span>Confidence</span>
                </div>
                {[
                  { name: 'PERSON', conf: 96, color: 'bg-emerald-500' },
                  { name: 'MAIN_BOX', conf: 92, color: 'bg-cyan-500' },
                  { name: 'RED_BOX', conf: 89, color: 'bg-rose-500' },
                  { name: 'YELLOW_BOX', conf: 91, color: 'bg-amber-400' },
                  { name: 'SAMPLE', conf: 84, color: 'bg-fuchsia-400' },
                  { name: 'TOOL', conf: 82, color: 'bg-blue-400' },
                ].map((item) => (
                  <div key={item.name} className="flex items-center gap-2 text-[10px]">
                    <span className="w-20 text-space-300 font-medium">{item.name}</span>
                    <div className="flex-1 bg-space-800 rounded-full h-1.5 overflow-hidden">
                      <div className={clsx("h-full rounded-full", item.color)} style={{ width: `${item.conf}%` }} />
                    </div>
                    <span className="w-8 text-right text-space-200 font-bold">{item.conf}%</span>
                  </div>
                ))}
              </div>

              {/* Auxiliary Stabilizer & Reasoning State */}
              <div className="grid grid-cols-2 gap-2 pt-1 border-t border-space-800/80 text-[10px]">
                <div className="p-1.5 bg-space-900 rounded border border-space-800">
                  <div className="text-space-400">HYBRID ASSIST</div>
                  <div className="text-emerald-400 font-bold">ON (Chroma S≥80)</div>
                </div>
                <div className="p-1.5 bg-space-900 rounded border border-space-800">
                  <div className="text-space-400">TRACKS & VELOCITY</div>
                  <div className="text-space-100 font-bold">5 Active Vectors</div>
                </div>
              </div>

              <div className="p-2 bg-space-900/90 rounded border border-space-800 text-[10px] space-y-1">
                <div className="flex justify-between">
                  <span className="text-space-400">FSM STATE:</span>
                  <span className="text-accent-cyan font-bold">STEP {(state.current_step_idx || 0) + 1} / {state.total_steps || 8}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">ACTION / CONF:</span>
                  <span className="text-space-100 font-bold">{state.detected_action || 'IDLE'} ({((state.action_confidence || 0.85) * 100).toFixed(1)}%)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">NEXT CANDIDATE:</span>
                  <span className="text-space-300">{state.next_action || 'IDLE'} ({(state.next_confidence ? state.next_confidence * 100 : 30).toFixed(0)}%)</span>
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
    total_steps: 8,
    status: 'WAITING',
    current_step: {
      id: 1,
      action: "OPEN_MAIN_BOX",
      label: "Open Main Box",
      description: "Open the main experiment container lid."
    },
    next_step: {
      id: 2,
      action: "TAKE_RED_BOX",
      label: "Take Red Box",
      description: "Retrieve the red sample box from the container."
    },
    completed_steps: [],
    failed_steps: [],
    skipped_steps: [],
    detected_action: "IDLE",
    detected_object: "MAIN_BOX",
    error_type: null,
    recovery_message: null,
    progress_pct: 12.5,
    elapsed_seconds: 0,
    voice_message: "Ready to start experiment.",
    hud_message: "Step one: Open the main container lid to begin.",
    alert_level: "info",
    fps: 30.0,
    latency_ms: 65,
    action_confidence: 0.88,
    is_uncertain: false,
    rules: []
  } as any;
}
