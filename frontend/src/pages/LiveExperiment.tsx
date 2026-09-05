import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useTelemetry } from '../api/telemetry';
import clsx from 'clsx';
import { AlertTriangle, CheckCircle, Activity, Video } from 'lucide-react';
import CameraControlPanel from '../components/camera/CameraControlPanel';
import type { CameraStatus } from '../api/camera';

export default function LiveExperiment() {
  const { id } = useParams();
  const { data, isConnected } = useTelemetry();
  const [videoError, setVideoError] = useState(false);
  const [streamVersion, setStreamVersion] = useState(Date.now());
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);

  // Fallback to simulated data if backend is not emitting or not connected
  // Normally we would use a true mock hook, but this keeps the component resilient.
  const state = data || getFallbackState(id || 'EXP-04');

  // Derive alert visual properties
  const isCritical = state.alert_level === 'error';
  const isWarning = state.alert_level === 'warning';
  const isSuccess = state.alert_level === 'success';

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
      case 'ip_camera':
        return 'PHONE IP CAM';
      case 'jetson_camera':
        return 'JETSON CAM';
      case 'sim':
        return 'SIM FEED';
      default:
        return 'CAM-01';
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Critical Alert Banner (if any) */}
      {(isCritical || isWarning) && (
        <div className={clsx(
          "w-full px-6 py-4 border-b flex items-center justify-between",
          isCritical ? "bg-status-critical/10 border-status-critical text-status-critical" : "bg-status-warning/10 border-status-warning text-status-warning"
        )}>
          <div className="flex items-center gap-4">
            <AlertTriangle size={24} />
            <div>
              <div className="font-mono font-bold tracking-widest">{state.hud_message}</div>
              <div className="text-sm mt-1">{state.recovery_message}</div>
            </div>
          </div>
          <button className={clsx(
            "px-6 py-2 rounded font-mono font-bold tracking-widest text-sm transition-colors border",
            isCritical ? "border-status-critical hover:bg-status-critical hover:text-space-900" : "border-status-warning hover:bg-status-warning hover:text-space-900"
          )}>
            ACKNOWLEDGE
          </button>
        </div>
      )}

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-1 xl:grid-cols-4 min-h-0">
        
        {/* Left Column: Video Feed & Camera Controls */}
        <div className="xl:col-span-3 flex flex-col border-r border-space-600 bg-black relative">
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
            <span className={clsx(
              "font-mono text-xs font-bold tracking-widest px-2 py-1 rounded",
              isConnected ? "bg-status-success/80 text-space-900" : "bg-status-critical/80 text-space-900"
            )}>
              {isConnected ? 'TELEMETRY LIVE' : 'TELEMETRY OFFLINE'}
            </span>
            <span className="font-mono text-xs font-bold tracking-widest px-2 py-1 rounded bg-accent-cyan/80 text-space-900 flex items-center gap-1.5">
              <Video size={12} />
              {getSourceLabel()}
            </span>
          </div>

          <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-space-900/60">
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
                <div className="font-mono font-bold tracking-widest text-xl">CAMERA CONNECTION LOST</div>
                <div className="text-space-400 font-mono text-sm max-w-md">
                  Unable to display camera feed. Verify camera connection settings below.
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setVideoError(false);
                    setStreamVersion(Date.now());
                  }}
                  className="mt-2 px-4 py-1.5 bg-space-800 border border-space-600 hover:border-accent-cyan text-accent-cyan font-mono text-xs font-bold rounded transition-colors"
                >
                  RETRY STREAM
                </button>
              </div>
            )}
          </div>

          {/* Integrated Camera Source & Settings Control Dock */}
          <div className="p-3 border-t border-space-600 bg-space-900/95">
            <CameraControlPanel onCameraChange={handleCameraChange} />
          </div>
        </div>

        {/* Right Column: Procedure & Action */}
        <div className="xl:col-span-1 flex flex-col bg-space-900 overflow-y-auto">
          
          {/* Current Action Card */}
          <div className="p-6 border-b border-space-600 flex-shrink-0">
            <div className="text-space-400 font-mono text-xs tracking-widest mb-4">
              STEP {state.current_step_idx + 1 < 10 ? `0${state.current_step_idx + 1}` : state.current_step_idx + 1} / {state.total_steps < 10 ? `0${state.total_steps}` : state.total_steps}
            </div>
            
            <h2 className="text-2xl font-bold text-space-100 mb-6">
              {state.current_step ? state.current_step.label.toUpperCase() : 'WAITING'}
            </h2>

            <div className="bg-space-800 rounded border border-space-600 p-4 space-y-3 font-mono text-sm">
              <div className="flex justify-between items-center text-space-400">
                <span>DETECTED ACTION</span>
                <span className="text-space-100">{state.detected_action || 'NONE'}</span>
              </div>
              <div className="flex justify-between items-center text-space-400">
                <span>DETECTED OBJECT</span>
                <span className="text-space-100">{state.detected_object || 'NONE'}</span>
              </div>
            </div>

            <div className="mt-6 flex items-center gap-3">
              {isSuccess ? <CheckCircle className="text-status-success animate-pulse" /> :
               isWarning ? <AlertTriangle className="text-status-warning" /> :
               isCritical ? <AlertTriangle className="text-status-critical" /> :
               <Activity className="text-accent-cyan animate-pulse" />}
              <span className={clsx(
                "font-mono font-bold tracking-widest text-sm",
                isSuccess ? "text-status-success" :
                isWarning ? "text-status-warning" :
                isCritical ? "text-status-critical" :
                "text-accent-cyan"
              )}>
                {state.hud_message || 'AI VERIFYING ACTION'}
              </span>
            </div>
          </div>

          {/* Step Tracker Rail */}
          <div className="p-6 flex-1">
            <div className="text-space-400 font-mono text-xs tracking-widest mb-6">PROTOCOL TIMELINE</div>
            
            <div className="space-y-6">
              {/* Previous */}
              {state.completed_steps.length > 0 && (
                <div className="flex items-start gap-4 opacity-50">
                  <div className="mt-1 flex-shrink-0 text-status-success"><CheckCircle size={18} /></div>
                  <div>
                    <div className="font-mono font-bold text-space-100">PREVIOUS STEP</div>
                    <div className="text-space-400 text-sm mt-1">Completed successfully</div>
                  </div>
                </div>
              )}

              {/* Current */}
              <div className="flex items-start gap-4">
                <div className="mt-1 flex-shrink-0 text-accent-cyan relative">
                  <div className="absolute inset-0 bg-accent-cyan rounded-full animate-ping opacity-20"></div>
                  <div className="w-4 h-4 rounded-full border-2 border-accent-cyan bg-space-900 mt-[2px] ml-[1px]"></div>
                </div>
                <div>
                  <div className="font-mono font-bold text-accent-cyan">CURRENT STEP</div>
                  <div className="text-space-100 mt-1 font-medium">{state.current_step?.label || 'In Progress...'}</div>
                </div>
              </div>

              {/* Next */}
              {state.next_step && (
                <div className="flex items-start gap-4 opacity-50">
                  <div className="mt-1 flex-shrink-0 text-space-400">
                    <div className="w-4 h-4 rounded-full border-2 border-space-400 bg-transparent mt-[2px] ml-[1px]"></div>
                  </div>
                  <div>
                    <div className="font-mono font-bold text-space-400">NEXT STEP</div>
                    <div className="text-space-400 text-sm mt-1">{state.next_step.label}</div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}

// Fallback data when backend is not running
function getFallbackState(id: string) {
  return {
    experiment_id: id,
    current_step_idx: 1,
    total_steps: 7,
    status: 'WAITING',
    current_step: {
      id: 2,
      action: "ADD_WATER",
      label: "Add Water",
      description: "Use syringe to add 50ml water."
    },
    next_step: {
      id: 3,
      action: "ADD_SEEDS",
      label: "Add Seeds",
      description: "Place 3 seeds into gel."
    },
    completed_steps: [1],
    failed_steps: [],
    skipped_steps: [],
    detected_action: "IDLE",
    detected_object: "",
    error_type: null,
    recovery_message: null,
    progress_pct: 14.2,
    elapsed_seconds: 45.2,
    voice_message: "Awaiting add water.",
    hud_message: "READY FOR ACTION",
    alert_level: "info",
    fps: 0,
    latency_ms: 0,
    rules: []
  } as any;
}
