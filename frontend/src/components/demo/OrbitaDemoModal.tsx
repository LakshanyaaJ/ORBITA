import { useState, useEffect } from 'react';
import { 
  Play, 
  CheckCircle2, 
  AlertTriangle, 
  Wifi, 
  WifiOff, 
  ShieldCheck, 
  FileText, 
  X,
  RefreshCw
} from 'lucide-react';
import clsx from 'clsx';
import { BACKEND_BASE } from '../../api/camera';

interface DemoStepInfo {
  step: number;
  title: string;
  category: 'SETUP' | 'CV_HMR' | 'HAR_FSM' | 'OFFLINE' | 'SYNC';
  description: string;
}

const DEMO_STEPS: DemoStepInfo[] = [
  { step: 1, title: "Load Experiment", category: 'SETUP', description: "Initialize EXP-01 protocol, step requirements, and safety rules." },
  { step: 2, title: "Start Experiment", category: 'SETUP', description: "Transition FSM to RUNNING state, arm recorders and logging." },
  { step: 3, title: "Start Simulated Video", category: 'SETUP', description: "Mount deterministic reference video stream (vdata/20260905_145858.mp4)." },
  { step: 4, title: "Detect Person (YOLO)", category: 'CV_HMR', description: "YOLOv8 person detector locates astronaut bounding box." },
  { step: 5, title: "HMR Processes Subject", category: 'CV_HMR', description: "Human Mesh Recovery estimates 3D body orientation and depth." },
  { step: 6, title: "Extract Pose Features", category: 'CV_HMR', description: "Extract 24 SMPL joint 3D spatial coordinates and kinematic angles." },
  { step: 7, title: "Temporal Model Processing", category: 'HAR_FSM', description: "30-frame temporal sliding window feature vector constructed." },
  { step: 8, title: "Activity Recognized (GRU)", category: 'HAR_FSM', description: "Temporal GRU classifies action: 'handling' (confidence 92.4%)." },
  { step: 9, title: "FSM Updates", category: 'HAR_FSM', description: "Procedural state machine advances to STEP 2 (PICKUP_BLUE_BOX)." },
  { step: 10, title: "Experiment Rule Triggers", category: 'HAR_FSM', description: "Rule R002 triggered: Required action verified with confidence > 85%." },
  { step: 11, title: "Ground Connection Lost", category: 'OFFLINE', description: "Simulating ground uplink loss. Platform enters autonomous offline mode." },
  { step: 12, title: "Local Edge Processing", category: 'OFFLINE', description: "Inference continues uninterrupted on local compute node." },
  { step: 13, title: "Experiment Completes", category: 'OFFLINE', description: "All procedural criteria met. FSM transitions to COMPLETED." },
  { step: 14, title: "Structured Result Generated", category: 'OFFLINE', description: "Section 20 machine-readable JSON generated with SHA-256 checksum." },
  { step: 15, title: "Result Stored Locally", category: 'OFFLINE', description: "Result, observations, and telemetry persisted to SQLite database." },
  { step: 16, title: "Connection Restored", category: 'SYNC', description: "Ground transmission link restored. Sync queue dispatcher awakens." },
  { step: 17, title: "Result Synchronized", category: 'SYNC', description: "Payload dispatched to ground station receiver via POST /api/sync." },
  { step: 18, title: "SYNCED (Ground ACK 200)", category: 'SYNC', description: "Ground validates SHA-256 checksum, checks duplicates, returns ACK 200." },
];

interface OrbitaDemoModalProps {
  isOpen: boolean;
  onClose: () => void;
  onViewResults?: () => void;
}

export default function OrbitaDemoModal({ isOpen, onClose, onViewResults }: OrbitaDemoModalProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [stepData, setStepData] = useState<any>(null);
  const [checksum, setChecksum] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Poll demo status while modal is open
  useEffect(() => {
    if (!isOpen) return;

    const checkStatus = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/api/demo/status`);
        if (res.ok) {
          const data = await res.json();
          setIsRunning(data.is_running);
          setCurrentStep(data.current_step);
          if (data.is_running) {
            setCompletedSteps(Array.from({ length: Math.max(0, data.current_step - 1) }, (_, i) => i + 1));
          } else if (data.current_step >= 18) {
            setCompletedSteps(Array.from({ length: 18 }, (_, i) => i + 1));
          }
          if (data.last_event) {
            setStepData(data.last_event.data);
            if (data.last_event.data?.checksum) {
              setChecksum(data.last_event.data.checksum);
            }
          }
        }
      } catch (err) {
        console.warn("Demo status poll failed:", err);
      }
    };

    checkStatus();
    const timer = setInterval(checkStatus, 750);
    return () => clearInterval(timer);
  }, [isOpen]);

  const handleStartDemo = async () => {
    setErrorMsg(null);
    setIsRunning(true);
    setCurrentStep(1);
    setCompletedSteps([]);
    try {
      const res = await fetch(`${BACKEND_BASE}/api/demo/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.message || 'Failed to start demo.');
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Network error running demo.');
      setIsRunning(false);
    }
  };

  if (!isOpen) return null;

  const isCompleted = currentStep >= 18 && !isRunning;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fade-in">
      <div className="bg-space-900 border border-accent-cyan/50 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden text-space-100 font-sans">
        
        {/* Header */}
        <div className="p-5 border-b border-space-700 bg-space-850 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-accent-cyan/10 border border-accent-cyan/40 flex items-center justify-center text-accent-cyan">
              <Play size={20} className={isRunning ? "animate-pulse" : ""} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold font-['Space_Grotesk'] tracking-wide text-space-100">
                  RUN ORBITA DEMO
                </h2>
                <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30 uppercase font-bold">
                  18-Step Deterministic Sequence
                </span>
              </div>
              <p className="text-xs text-space-400 mt-0.5">
                End-to-end trace: Video → YOLO → HMR 3D → GRU Activity → FSM → Offline Storage → Ground ACK
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {!isRunning && (
              <button
                type="button"
                onClick={handleStartDemo}
                className="px-4 py-2 bg-accent-cyan hover:bg-accent-cyan/80 text-space-950 font-mono font-bold text-xs rounded-lg flex items-center gap-2 transition-all shadow-md cursor-pointer"
              >
                <Play size={14} />
                <span>{isCompleted ? "RE-RUN DEMO" : "LAUNCH DEMO"}</span>
              </button>
            )}

            {isRunning && (
              <div className="px-3 py-1.5 bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded-lg text-xs font-mono font-bold flex items-center gap-2">
                <RefreshCw size={13} className="animate-spin" />
                <span>STEP {currentStep} / 18 IN PROGRESS</span>
              </div>
            )}

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 text-space-400 hover:text-space-100 hover:bg-space-800 rounded-lg transition-colors cursor-pointer"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Error Banner if any */}
        {errorMsg && (
          <div className="p-3 bg-red-950/80 border-b border-red-800 text-red-200 text-xs font-mono flex items-center gap-2">
            <AlertTriangle size={16} className="text-red-400 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Main Content: Steps Grid & Live Step Detail */}
        <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Steps Timeline (7 cols) */}
          <div className="lg:col-span-7 space-y-2">
            <div className="flex items-center justify-between pb-2 border-b border-space-800 text-xs font-mono text-space-400">
              <span>WORKFLOW EXECUTION PIPELINE</span>
              <span>{completedSteps.length} of 18 Complete</span>
            </div>

            <div className="space-y-1.5 max-h-[480px] overflow-y-auto pr-2">
              {DEMO_STEPS.map((item) => {
                const isStepDone = completedSteps.includes(item.step) || (isCompleted);
                const isStepActive = isRunning && currentStep === item.step;

                return (
                  <div
                    key={item.step}
                    className={clsx(
                      "p-2.5 rounded-lg border text-xs font-mono transition-all flex items-start justify-between gap-3",
                      isStepActive ? "bg-accent-cyan/15 border-accent-cyan text-space-100 shadow-[0_0_12px_rgba(0,229,255,0.15)]" :
                      isStepDone ? "bg-space-850/60 border-space-700/80 text-space-300" :
                      "bg-space-950/40 border-space-800/40 text-space-500 opacity-60"
                    )}
                  >
                    <div className="flex items-start gap-2.5">
                      <div className="mt-0.5 shrink-0">
                        {isStepDone ? (
                          <CheckCircle2 size={16} className="text-emerald-400" />
                        ) : isStepActive ? (
                          <div className="relative w-4 h-4 flex items-center justify-center">
                            <span className="absolute inset-0 rounded-full bg-accent-cyan/40 animate-ping" />
                            <span className="w-2.5 h-2.5 rounded-full bg-accent-cyan" />
                          </div>
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-space-700 text-[10px] flex items-center justify-center text-space-500">
                            {item.step}
                          </div>
                        )}
                      </div>

                      <div>
                        <div className="flex items-center gap-2">
                          <span className={clsx("font-bold", isStepActive ? "text-accent-cyan" : isStepDone ? "text-space-200" : "text-space-400")}>
                            {item.step}. {item.title}
                          </span>
                          <span className={clsx(
                            "text-[9px] px-1.5 py-0.2 rounded uppercase font-semibold",
                            item.category === 'CV_HMR' ? "bg-purple-950/80 text-purple-300 border border-purple-800" :
                            item.category === 'HAR_FSM' ? "bg-blue-950/80 text-blue-300 border border-blue-800" :
                            item.category === 'OFFLINE' ? "bg-amber-950/80 text-amber-300 border border-amber-800" :
                            item.category === 'SYNC' ? "bg-emerald-950/80 text-emerald-300 border border-emerald-800" :
                            "bg-space-800 text-space-400"
                          )}>
                            {item.category}
                          </span>
                        </div>
                        <p className="text-[11px] text-space-400 mt-0.5 leading-relaxed">
                          {item.description}
                        </p>
                      </div>
                    </div>

                    <div className="shrink-0 text-[10px] font-bold">
                      {isStepDone && <span className="text-emerald-400">PASSED</span>}
                      {isStepActive && <span className="text-accent-cyan animate-pulse">RUNNING</span>}
                      {!isStepDone && !isStepActive && <span className="text-space-600">WAITING</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Telemetry & Inspection Panel (5 cols) */}
          <div className="lg:col-span-5 flex flex-col space-y-4">
            <div className="p-4 bg-space-850 rounded-lg border border-space-700 font-mono text-xs space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-space-700">
                <span className="text-space-400 uppercase font-bold text-[11px]">DEMO RUNTIME TELEMETRY</span>
                <span className={clsx(
                  "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
                  isCompleted ? "bg-emerald-950 text-emerald-300 border border-emerald-700" :
                  isRunning ? "bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40 animate-pulse" :
                  "bg-space-800 text-space-400"
                )}>
                  {isCompleted ? "100% VERIFIED" : isRunning ? "RUNNING" : "STANDBY"}
                </span>
              </div>

              <div className="space-y-2 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-space-400">EXPERIMENT ID:</span>
                  <span className="text-space-100 font-bold">EXP-01</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">INPUT SOURCE:</span>
                  <span className="text-accent-cyan">vdata/20260905_145858.mp4</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">GROUND LINK:</span>
                  <span className={clsx("font-bold flex items-center gap-1", currentStep >= 11 && currentStep < 16 ? "text-red-400" : "text-emerald-400")}>
                    {currentStep >= 11 && currentStep < 16 ? <WifiOff size={12} /> : <Wifi size={12} />}
                    {currentStep >= 11 && currentStep < 16 ? "OFFLINE (Autonomous)" : "ONLINE (Linked)"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">HMR MESH (SMPL):</span>
                  <span className="text-purple-300">24 Joints · Kinematic</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">HAR MODEL:</span>
                  <span className="text-space-100">Temporal GRU (30 frames)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-space-400">STORAGE ENGINE:</span>
                  <span className="text-emerald-400 font-bold">SQLite (Local Edge)</span>
                </div>
              </div>

              {checksum && (
                <div className="pt-2 border-t border-space-700">
                  <div className="flex items-center justify-between text-[10px] text-space-400 mb-1">
                    <span className="flex items-center gap-1 text-emerald-400 font-bold">
                      <ShieldCheck size={13} />
                      SHA-256 CHECKSUM
                    </span>
                    <span className="text-emerald-300">GROUND ACKNOWLEDGED</span>
                  </div>
                  <div className="p-2 bg-space-950 rounded border border-space-800 text-[10px] text-emerald-300 font-mono break-all select-all">
                    {checksum}
                  </div>
                </div>
              )}
            </div>

            {/* Step payload details when available */}
            {stepData && (
              <div className="p-4 bg-space-950 rounded-lg border border-space-800 font-mono text-[11px] space-y-2 flex-1 overflow-hidden">
                <div className="text-[10px] text-space-400 font-bold uppercase pb-1 border-b border-space-800 flex items-center justify-between">
                  <span>STEP {currentStep} PAYLOAD LOG</span>
                  <span className="text-accent-cyan">JSON EVENT</span>
                </div>
                <pre className="text-[10px] text-space-300 overflow-x-auto max-h-[160px] p-2 bg-black/60 rounded">
                  {JSON.stringify(stepData, null, 2)}
                </pre>
              </div>
            )}

            {/* Completed Action Controls */}
            {isCompleted && (
              <div className="p-4 bg-emerald-950/40 rounded-lg border border-emerald-700/80 font-mono text-xs space-y-3 animate-fade-in">
                <div className="flex items-center gap-2 text-emerald-300 font-bold">
                  <CheckCircle2 size={18} />
                  <span>DEMO SEQUENCE COMPLETED SUCCESSFULLY</span>
                </div>
                <p className="text-[11px] text-emerald-200/80 leading-relaxed">
                  All 18 stages validated. Autonomous offline storage buffered the result, and ground synchronization confirmed SHA-256 integrity.
                </p>
                {onViewResults && (
                  <button
                    type="button"
                    onClick={() => {
                      onClose();
                      onViewResults();
                    }}
                    className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-black font-bold text-xs rounded-lg flex items-center justify-center gap-2 transition-colors cursor-pointer"
                  >
                    <FileText size={14} />
                    <span>VIEW SECTION 20 STRUCTURED RESULT</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-space-700 bg-space-850 flex items-center justify-between text-xs font-mono text-space-400">
          <div className="flex items-center gap-3">
            <span>TARGET: <strong className="text-space-200">NVIDIA Jetson Orin Nano</strong></span>
            <span>·</span>
            <span>FALLBACK: <strong className="text-space-200">Deterministic Kinematics</strong></span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 bg-space-800 hover:bg-space-700 text-space-200 rounded-lg transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
