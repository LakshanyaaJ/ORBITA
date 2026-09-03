import React, { useState } from 'react';
import { Satellite, RotateCcw, Video, VideoOff, Download } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { ProcedureTimeline } from './components/ProcedureTimeline';
import { CurrentStep } from './components/CurrentStep';
import { AlertPanel } from './components/AlertPanel';
import { SystemStatusPanel } from './components/SystemStatus';
import { TelemetryBar } from './components/TelemetryBar';

const SCENARIOS = ['A', 'B', 'C'] as const;
type ScenarioId = typeof SCENARIOS[number];

const SCENARIO_LABELS: Record<ScenarioId, { label: string; desc: string; color: string }> = {
  A: { label: 'Scenario A', desc: 'Nominal Execution', color: '#22c55e' },
  B: { label: 'Scenario B', desc: 'Wrong Object', color: '#f59e0b' },
  C: { label: 'Scenario C', desc: 'Skipped Step', color: '#ef4444' },
};

function App() {
  const { state, connected, sendControl } = useWebSocket();
  const [activeScenario, setActiveScenario] = useState<ScenarioId>('A');
  const [isRecording, setIsRecording] = useState(false);

  const handleScenario = async (s: ScenarioId) => {
    setActiveScenario(s);
    await sendControl('scenario', { scenario: s });
  };

  const handleReset = () => sendControl('reset');

  const handleRecording = async () => {
    if (isRecording) {
      await sendControl('stop_recording');
    } else {
      await sendControl('start_recording');
    }
    setIsRecording(!isRecording);
  };

  const handleExport = () => sendControl('export_log');

  return (
    <div className="min-h-screen bg-[#030811] text-white flex flex-col font-mono select-none overflow-hidden">
      {/* ── Top Navigation Bar ─────────────────────────────── */}
      <header className="flex items-center px-5 py-3 bg-[#060e1a] border-b border-[#1e3a5f] gap-4 shrink-0">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <Satellite size={22} className="text-[#38bdf8]" />
            <span
              className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-[#22c55e] border border-[#060e1a]"
              style={{ boxShadow: '0 0 6px #22c55e' }}
            />
          </div>
          <div>
            <h1 className="text-base font-black tracking-[0.25em] text-white leading-none">ORBITA</h1>
            <p className="text-[9px] text-[#334155] tracking-widest uppercase leading-none">
              Offline AI Experiment Copilot · SIH 26174
            </p>
          </div>
        </div>

        {/* Scenario Controls */}
        <div className="flex items-center gap-1.5 ml-6">
          <span className="text-[10px] text-[#334155] uppercase tracking-widest mr-1">Scenario:</span>
          {SCENARIOS.map((s) => {
            const cfg = SCENARIO_LABELS[s];
            const isActive = activeScenario === s;
            return (
              <button
                key={s}
                onClick={() => handleScenario(s)}
                className="text-[10px] font-bold px-3 py-1.5 rounded transition-all duration-200"
                style={{
                  color: isActive ? '#000' : cfg.color,
                  background: isActive ? cfg.color : 'transparent',
                  border: `1px solid ${cfg.color}${isActive ? '' : '66'}`,
                }}
              >
                {s} · {cfg.desc}
              </button>
            );
          })}
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 ml-auto">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded border border-[#1e3a5f] text-[#94a3b8] hover:border-[#38bdf8] hover:text-[#38bdf8] transition-all"
          >
            <RotateCcw size={11} /> RESET
          </button>
          <button
            onClick={handleRecording}
            className="flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded border transition-all"
            style={{
              borderColor: isRecording ? '#ef4444' : '#1e3a5f',
              color: isRecording ? '#ef4444' : '#94a3b8',
            }}
          >
            {isRecording ? <VideoOff size={11} /> : <Video size={11} />}
            {isRecording ? 'STOP REC' : 'RECORD'}
            {isRecording && (
              <span className="w-1.5 h-1.5 rounded-full bg-[#ef4444] animate-pulse" />
            )}
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 text-[10px] px-3 py-1.5 rounded border border-[#1e3a5f] text-[#94a3b8] hover:border-[#22c55e] hover:text-[#22c55e] transition-all"
          >
            <Download size={11} /> EXPORT LOG
          </button>

          {/* Connection indicator */}
          <div className="flex items-center gap-1.5 ml-2 px-2 py-1 rounded border border-[#1e3a5f]">
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: connected ? '#22c55e' : '#ef4444',
                boxShadow: connected ? '0 0 6px #22c55e' : 'none',
              }}
            />
            <span className={`text-[10px] font-bold ${connected ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
              {connected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* ── Main Content Grid ───────────────────────────────── */}
      <main className="flex-1 grid grid-cols-[1fr_340px] gap-0 overflow-hidden">
        {/* Left column: Video + Alert */}
        <div className="flex flex-col gap-0 border-r border-[#1e3a5f] overflow-hidden">
          {/* Live Video Feed */}
          <div className="relative bg-black flex-1 flex items-center justify-center overflow-hidden">
            <img
              src={`http://${window.location.hostname}:8000/video_feed`}
              alt="ORBITA Live Feed"
              className="w-full h-full object-contain"
              style={{ maxHeight: 'calc(100vh - 200px)' }}
            />
            {/* Video overlays */}
            <div className="absolute top-3 left-3 flex items-center gap-2">
              <span className="text-[10px] font-bold text-white bg-black/60 px-2 py-1 rounded">
                LIVE · {state.fps.toFixed(1)} FPS
              </span>
              {isRecording && (
                <span className="text-[10px] font-bold text-white bg-[#ef4444]/80 px-2 py-1 rounded flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                  REC
                </span>
              )}
            </div>
            <div className="absolute top-3 right-3">
              <span
                className="text-[10px] font-bold px-2 py-1 rounded"
                style={{
                  color: state.alert_level === 'error' ? '#ef4444' : state.alert_level === 'success' ? '#22c55e' : state.alert_level === 'warning' ? '#f59e0b' : '#94a3b8',
                  background: 'rgba(0,0,0,0.6)',
                }}
              >
                {state.hud_message.slice(0, 50)}
              </span>
            </div>
          </div>

          {/* Alert Panel below video */}
          <div className="p-4 border-t border-[#1e3a5f] shrink-0">
            <AlertPanel state={state} />
          </div>
        </div>

        {/* Right column: Procedure + Step + Status */}
        <div className="flex flex-col overflow-y-auto p-4 gap-4 bg-[#060e1a]">
          <CurrentStep state={state} />
          <ProcedureTimeline state={state} />
          <SystemStatusPanel connected={connected} />
        </div>
      </main>

      {/* ── Telemetry Bar ────────────────────────────────────── */}
      <TelemetryBar state={state} />
    </div>
  );
}

export default App;
