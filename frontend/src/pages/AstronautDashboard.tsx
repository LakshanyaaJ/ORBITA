import React, { useState, useEffect } from 'react';
import { Navbar } from '../components/Navbar';
import {
  Play,
  Pause,
  RotateCcw,
  Square,
  Volume2,
  VolumeX,
  AlertTriangle,
  CheckCircle2,
  Camera,
  Activity,
  FileJson,
  FileSpreadsheet,
  ArrowRight,
} from 'lucide-react';

interface Step {
  id: number;
  label: string;
  expected_action: string;
  expected_object: string;
  status: 'COMPLETED' | 'IN_PROGRESS' | 'PENDING' | 'SKIPPED' | 'FAILED';
}

const DEFAULT_STEPS: Step[] = [
  { id: 1, label: 'Retrieve Main Container', expected_action: 'TAKE', expected_object: 'MAIN_BOX', status: 'COMPLETED' },
  { id: 2, label: 'Take out Red Box', expected_action: 'TAKE', expected_object: 'RED_BOX', status: 'IN_PROGRESS' },
  { id: 3, label: 'Place Red Box in Zone A', expected_action: 'PLACE', expected_object: 'RED_BOX', status: 'PENDING' },
  { id: 4, label: 'Retrieve Sampling Tool', expected_action: 'TAKE', expected_object: 'TOOL', status: 'PENDING' },
  { id: 5, label: 'Extract Sample Specimen', expected_action: 'PERFORM', expected_object: 'SAMPLE', status: 'PENDING' },
  { id: 6, label: 'Store Specimen in Yellow Box', expected_action: 'STORE', expected_object: 'YELLOW_BOX', status: 'PENDING' },
];

interface LogEvent {
  timestamp: string;
  step: number;
  status: string;
  confidence: number;
  event: string;
  severity: 'info' | 'warning' | 'critical';
}

export const AstronautDashboard: React.FC = () => {
  const [experimentStatus, setExperimentStatus] = useState<'RUNNING' | 'PAUSED' | 'ENDED'>('RUNNING');
  const [currentStepIdx, setCurrentStepIdx] = useState(1);
  const [steps] = useState<Step[]>(DEFAULT_STEPS);
  const [confidence, setConfidence] = useState(0.92);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [lastSpeech] = useState('"Next step: retrieve the red box."');
  
  const [logs, setLogs] = useState<LogEvent[]>([
    { timestamp: '10:15:32', step: 1, status: 'COMPLETED', confidence: 0.96, event: 'Step 1 completed: Retrieve Main Container', severity: 'info' },
    { timestamp: '10:16:05', step: 2, status: 'UNCERTAIN', confidence: 0.58, event: 'Low confidence interaction detected', severity: 'warning' },
    { timestamp: '10:16:12', step: 2, status: 'OUT_OF_SEQUENCE', confidence: 0.91, event: 'Out-of-sequence action detected: Yellow Box interaction', severity: 'critical' },
    { timestamp: '10:17:40', step: 2, status: 'DIFFICULTY', confidence: 0.88, event: 'Potential execution difficulty detected (Attempt 3)', severity: 'warning' },
  ]);

  // Connect to live backend WebSocket telemetry
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/live`;
    let ws: WebSocket | null = null;

    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.confidence !== undefined) setConfidence(data.confidence);
          if (data.current_step_idx !== undefined) setCurrentStepIdx(data.current_step_idx);
          if (data.status_message) {
            setLogs((prev) => [
              {
                timestamp: new Date().toLocaleTimeString(),
                step: data.current_step_idx ? data.current_step_idx + 1 : 2,
                status: data.fsm_status || 'INFO',
                confidence: data.confidence || 0.9,
                event: data.status_message,
                severity: data.fsm_status?.includes('WRONG') || data.fsm_status?.includes('OUT') ? 'critical' : 'info',
              },
              ...prev,
            ]);
          }
        } catch {
          // JSON parse fallback
        }
      };
    } catch {
      console.log('Using simulated offline telemetry');
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const handleStart = () => setExperimentStatus('RUNNING');
  const handlePause = () => setExperimentStatus('PAUSED');
  const handleResume = () => setExperimentStatus('RUNNING');
  const handleEnd = () => setExperimentStatus('ENDED');

  const exportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(logs, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `orbita_onboard_event_log_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const exportCSV = () => {
    const headers = ['Timestamp', 'Step', 'Status', 'Confidence', 'Event'];
    const rows = logs.map((l) => [l.timestamp, l.step, l.status, l.confidence, `"${l.event}"`]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', csvContent);
    downloadAnchor.setAttribute('download', `orbita_onboard_event_log_${Date.now()}.csv`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const currentStep = steps[currentStepIdx] || steps[1];
  const nextStep = steps[currentStepIdx + 1] || steps[2];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col">
      <Navbar telemetryConnected={true} fps={28} aiStatus="YOLO11 + GRU" activeModel="orbita_yolo11.pt" />

      {/* A. Header Section */}
      <div className="bg-slate-900 border-b border-slate-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <span className="px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider bg-cyan-950 text-cyan-400 border border-cyan-800">
                OFFLINE ONBOARD ASSISTANT
              </span>
              <h1 className="text-xl font-bold font-mono tracking-tight text-white">ORBITA — ONBOARD COPILOT</h1>
            </div>
            <p className="text-xs text-slate-400 mt-1">Autonomous Human Activity Recognition & Experiment Procedure Verification</p>
          </div>

          <div className="flex items-center space-x-3 flex-wrap gap-2">
            <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs flex items-center space-x-2">
              <span className="text-slate-400">Experiment:</span>
              <span className="font-mono text-cyan-300 font-bold">BAS-02</span>
            </div>

            <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs flex items-center space-x-2">
              <span className="text-slate-400">Status:</span>
              <span className={`font-bold ${experimentStatus === 'RUNNING' ? 'text-emerald-400' : 'text-amber-400'}`}>
                {experimentStatus}
              </span>
            </div>

            <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
              <span className="text-slate-400">AI:</span>
              <span className="font-bold text-emerald-400">ONLINE (YOLO11)</span>
            </div>

            <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs flex items-center space-x-2">
              <Camera className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">Camera:</span>
              <span className="font-bold text-cyan-300">CONNECTED</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 w-full">
        
        {/* Left/Center Column: Live Video & Step Guidance */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* B. Live Video Panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl relative group">
            <div className="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-semibold text-cyan-400">
                <Activity className="w-4 h-4" />
                <span>LIVE ONBOARD CAMERA STREAM — 1080p @ 28 FPS</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="px-2 py-0.5 text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800 rounded">
                  MODEL: ORBITA_YOLO11.PT
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
                  BYTETRACK ACTIVE
                </span>
              </div>
            </div>

            {/* Video Viewport */}
            <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
              <img
                src="/video_feed"
                alt="Live Telemetry Feed"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
              
              {/* Overlay simulation graphics */}
              <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-transparent to-slate-950/40 pointer-events-none p-6 flex flex-col justify-between">
                <div className="flex justify-between items-start">
                  <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/60 text-xs flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
                    <span className="font-mono text-white font-bold">REC ● 00:16:45</span>
                  </div>

                  <div className="bg-cyan-950/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-cyan-600/50 text-xs font-bold text-cyan-200">
                    STEP {currentStepIdx + 1} / {steps.length}: {currentStep.label}
                  </div>
                </div>

                {/* Bounding box overlay simulation */}
                <div className="absolute top-1/3 left-1/3 w-48 h-40 border-2 border-emerald-400 rounded-lg bg-emerald-500/10 p-2 flex flex-col justify-between pointer-events-none shadow-[0_0_15px_rgba(52,211,153,0.3)]">
                  <span className="bg-emerald-500 text-slate-950 px-2 py-0.5 text-[10px] font-bold rounded w-max">
                    RED_BOX 94.8% (YOLO11)
                  </span>
                  <span className="text-[10px] font-mono text-emerald-300">ID: #002 | VEL: 0.12 m/s</span>
                </div>

                <div className="flex justify-between items-end">
                  <div className="bg-slate-900/80 backdrop-blur-md p-2 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
                    <div>HANDS: MediaPipe Landmark Active</div>
                    <div>POSE: 17 Keypoints (YOLO11-Pose)</div>
                  </div>

                  <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-mono text-emerald-400">
                    CONFIDENCE: {(confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* H. Experiment Controls Bar */}
            <div className="p-4 bg-slate-900 border-t border-slate-800 flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center space-x-2">
                {experimentStatus !== 'RUNNING' ? (
                  <button
                    onClick={handleStart}
                    className="flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded-xl font-bold text-xs transition shadow-lg shadow-emerald-600/20"
                  >
                    <Play className="w-4 h-4" />
                    <span>START EXPERIMENT</span>
                  </button>
                ) : (
                  <button
                    onClick={handlePause}
                    className="flex items-center space-x-2 bg-amber-600 hover:bg-amber-500 text-white px-4 py-2 rounded-xl font-bold text-xs transition shadow-lg shadow-amber-600/20"
                  >
                    <Pause className="w-4 h-4" />
                    <span>PAUSE</span>
                  </button>
                )}

                <button
                  onClick={handleResume}
                  className="flex items-center space-x-2 bg-cyan-700 hover:bg-cyan-600 text-white px-4 py-2 rounded-xl font-bold text-xs transition"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span>RESUME</span>
                </button>

                <button
                  onClick={handleEnd}
                  className="flex items-center space-x-2 bg-rose-700 hover:bg-rose-600 text-white px-4 py-2 rounded-xl font-bold text-xs transition"
                >
                  <Square className="w-4 h-4" />
                  <span>END EXPERIMENT</span>
                </button>
              </div>

              <div className="text-xs text-slate-400 font-mono">
                SESSION ID: <span className="text-cyan-300 font-bold">EXP-20260911-002</span>
              </div>
            </div>
          </div>

          {/* Step Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden shadow-lg border-l-4 border-l-cyan-500">
              <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>CURRENT STEP</span>
                <span className="bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded text-[10px]">STEP {currentStepIdx + 1} OF 6</span>
              </div>

              <h2 className="text-lg font-bold text-white mb-2">{currentStep.label}</h2>
              
              <div className="space-y-2 mt-4 text-xs">
                <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-400">Action Status:</span>
                  <span className="px-2.5 py-0.5 rounded font-bold bg-amber-950 text-amber-300 border border-amber-800">
                    {currentStep.status}
                  </span>
                </div>

                <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-400">AI Confidence:</span>
                  <span className="font-mono font-bold text-emerald-400">{(confidence * 100).toFixed(0)}%</span>
                </div>

                <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-400">Expected Object:</span>
                  <span className="font-mono text-cyan-300 font-semibold">{currentStep.expected_object}</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative overflow-hidden shadow-lg border-l-4 border-l-purple-500">
              <div className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>SUGGESTED NEXT STEP</span>
                <ArrowRight className="w-4 h-4 text-purple-400" />
              </div>

              <h2 className="text-lg font-bold text-white mb-2">{nextStep ? nextStep.label : 'Procedure Complete'}</h2>
              
              <div className="space-y-2 mt-4 text-xs">
                <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-400">Planned Action:</span>
                  <span className="font-mono text-purple-300 font-semibold">{nextStep ? nextStep.expected_action : 'N/A'}</span>
                </div>

                <div className="flex justify-between items-center bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="text-slate-400">Target Object:</span>
                  <span className="font-mono text-purple-300 font-semibold">{nextStep ? nextStep.expected_object : 'N/A'}</span>
                </div>

                <p className="text-[11px] text-slate-400 italic pt-1">
                  PS Requirement: Autonomous guidance prepares next experiment phase.
                </p>
              </div>
            </div>
          </div>

          {/* Voice Assistance Panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-purple-500/10 rounded-xl border border-purple-500/30 text-purple-400">
                {ttsEnabled ? <Volume2 className="w-6 h-6 animate-pulse" /> : <VolumeX className="w-6 h-6 text-slate-500" />}
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="font-bold text-sm text-white">OFFLINE VOICE ASSISTANCE (pyttsx3)</h3>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                  <span className="text-xs text-emerald-400 font-semibold">Ready</span>
                </div>
                <p className="text-xs text-slate-400 font-mono mt-0.5">{lastSpeech}</p>
              </div>
            </div>

            <button
              onClick={() => setTtsEnabled(!ttsEnabled)}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-2 ${
                ttsEnabled
                  ? 'bg-purple-900/60 text-purple-200 border border-purple-700/60 hover:bg-purple-800/80'
                  : 'bg-slate-800 text-slate-400 border border-slate-700'
              }`}
            >
              <span>{ttsEnabled ? 'VOICE ON' : 'VOICE MUTED'}</span>
            </button>
          </div>
        </div>

        {/* Right Column: Progress Timeline, Alerts, System Health, Event Logs */}
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center justify-between">
              <span>PROCEDURE TIMELINE</span>
              <span className="text-cyan-400 font-mono">BAS-02</span>
            </h3>

            <div className="space-y-3">
              {steps.map((step, idx) => {
                const isCurrent = idx === currentStepIdx;
                const isCompleted = step.status === 'COMPLETED';

                return (
                  <div
                    key={step.id}
                    className={`flex items-start space-x-3 p-3 rounded-xl border transition-all ${
                      isCurrent
                        ? 'bg-cyan-950/60 border-cyan-600/70 shadow-md shadow-cyan-900/20'
                        : isCompleted
                        ? 'bg-slate-950/70 border-slate-800/80 text-slate-400'
                        : 'bg-slate-950/40 border-slate-800/40 text-slate-500'
                    }`}
                  >
                    <div className="mt-0.5">
                      {isCompleted ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : isCurrent ? (
                        <div className="w-4 h-4 rounded-full border-2 border-cyan-400 flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                        </div>
                      ) : (
                        <div className="w-4 h-4 rounded-full border border-slate-700" />
                      )}
                    </div>

                    <div className="flex-1 text-xs">
                      <div className="flex justify-between items-center">
                        <span className={`font-bold ${isCurrent ? 'text-white' : isCompleted ? 'text-slate-300' : 'text-slate-500'}`}>
                          Step {step.id}: {step.label}
                        </span>
                      </div>
                      <div className="text-[10px] font-mono text-slate-500 mt-0.5">
                        {step.expected_action} → {step.expected_object}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
              <span>REAL-TIME ALERTS</span>
              <span className="px-2 py-0.5 text-[10px] font-mono bg-rose-950 text-rose-300 border border-rose-800 rounded">
                LIVE
              </span>
            </h3>

            <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-xl border text-xs flex items-start space-x-2.5 ${
                    log.severity === 'critical'
                      ? 'bg-rose-950/40 border-rose-800/60 text-rose-200'
                      : log.severity === 'warning'
                      ? 'bg-amber-950/40 border-amber-800/60 text-amber-200'
                      : 'bg-slate-950 border-slate-800 text-slate-300'
                  }`}
                >
                  {log.severity === 'critical' || log.severity === 'warning' ? (
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <div className="flex justify-between items-center font-mono text-[10px] text-slate-400">
                      <span>{log.timestamp}</span>
                      <span>{(log.confidence * 100).toFixed(0)}% CONF</span>
                    </div>
                    <p className="mt-1 font-medium">{log.event}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
              LOCAL SYSTEM HEALTH (ONBOARD JETSON/PC)
            </h3>

            <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">CAMERA</div>
                <div className="text-emerald-400 font-bold mt-0.5">CONNECTED</div>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">AI PIPELINE</div>
                <div className="text-emerald-400 font-bold mt-0.5">RUNNING (YOLO11)</div>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">INFERENCE FPS</div>
                <div className="text-cyan-300 font-bold mt-0.5">28.4 FPS</div>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">HARDWARE</div>
                <div className="text-purple-300 font-bold mt-0.5">CUDA ACCEL</div>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">LOCAL STORAGE</div>
                <div className="text-slate-200 font-bold mt-0.5">72% FREE</div>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <div className="text-[10px] text-slate-400">STREAMING</div>
                <div className="text-emerald-400 font-bold mt-0.5">ACTIVE</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">LOCAL EVENT LOG</h3>
              <div className="flex space-x-2">
                <button
                  onClick={exportJSON}
                  className="flex items-center space-x-1 text-[11px] bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded-lg border border-slate-700 text-cyan-300 transition"
                >
                  <FileJson className="w-3.5 h-3.5" />
                  <span>JSON</span>
                </button>
                <button
                  onClick={exportCSV}
                  className="flex items-center space-x-1 text-[11px] bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded-lg border border-slate-700 text-emerald-300 transition"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>CSV</span>
                </button>
              </div>
            </div>

            <p className="text-[11px] text-slate-400">
              Export complete structured procedure telemetry logs for post-mission offline analysis.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};
