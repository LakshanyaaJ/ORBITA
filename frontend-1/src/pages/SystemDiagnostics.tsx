import { useState, useEffect } from 'react';
import { Camera, Brain, Database, Mic, Video } from 'lucide-react';

export default function SystemDiagnostics() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    // In a real implementation, this would fetch from /api/status periodically
    // Using mock data for UI scaffolding
    setStatus({
      ai_engine: "ONLINE",
      camera: "ONLINE",
      tts: "ONLINE",
      recording: "OFF",
      stream: "ON",
      storage: "OK",
      mode: "webcam",
      scenario: "A",
      pipeline_fps: 29.5,
      cpu_pct: 42.1,
      mem_pct: 38.4,
      ws_clients: 1
    });
  }, []);

  if (!status) return null;

  return (
    <div className="p-8 max-w-6xl mx-auto h-full flex flex-col">
      <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100 mb-8 border-b border-space-600 pb-4">
        SYSTEM DIAGNOSTICS
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 flex-1 overflow-y-auto">
        <section>
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mb-4">PIPELINE STATUS</h2>
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 space-y-6">
            <DiagnosticRow icon={<Camera />} label="CAMERA MODULE" state={status.camera} />
            <DiagnosticRow icon={<Brain />} label="AI INFERENCE ENGINE" state={status.ai_engine} />
            <DiagnosticRow icon={<Database />} label="STORAGE SUBSYSTEM" state={status.storage} />
            <DiagnosticRow icon={<Mic />} label="VOICE ENGINE" state={status.tts} />
            <DiagnosticRow icon={<Video />} label="VIDEO STREAM" state={status.stream} />
          </div>
        </section>

        <section>
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mb-4">HARDWARE TELEMETRY</h2>
          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="CPU LOAD" value={`${status.cpu_pct}%`} />
            <MetricCard label="MEMORY" value={`${status.mem_pct}%`} />
            <MetricCard label="INFERENCE FPS" value={`${status.pipeline_fps}`} />
            <MetricCard label="WS CLIENTS" value={`${status.ws_clients}`} />
          </div>

          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mt-8 mb-4">SYSTEM CONFIG</h2>
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 font-mono text-sm space-y-4 text-space-100">
            <div className="flex justify-between">
              <span className="text-space-400">OPERATION MODE</span>
              <span className="uppercase">{status.mode}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-space-400">ACTIVE SCENARIO</span>
              <span className="uppercase">{status.scenario}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-space-400">MODEL VERSION</span>
              <span>ORBITA-HAR-v1.2</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function DiagnosticRow({ icon, label, state }: { icon: React.ReactNode, label: string, state: string }) {
  const isOk = state === 'ONLINE' || state === 'ON' || state === 'OK';
  
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3 text-space-100">
        <span className="text-space-400">{icon}</span>
        <span className="font-mono tracking-widest text-sm">{label}</span>
      </div>
      <span className={`font-mono text-sm font-bold tracking-widest ${isOk ? 'text-status-success' : 'text-status-warning'}`}>
        {state}
      </span>
    </div>
  );
}

function MetricCard({ label, value }: { label: string, value: string }) {
  return (
    <div className="bg-space-800 border border-space-600 rounded-lg p-6">
      <div className="text-space-400 font-mono text-xs tracking-widest mb-2">{label}</div>
      <div className="text-3xl font-mono text-space-100">{value}</div>
    </div>
  );
}
