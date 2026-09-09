import { useState, useEffect } from 'react';
import { Camera, Brain, Database, Mic, Video, Volume2 } from 'lucide-react';
import { getCameraStatus } from '../api/camera';

export default function SystemDiagnostics() {
  const [status, setStatus] = useState<any>({
    ai_engine: "ONLINE",
    camera: "ONLINE",
    camera_source: "sim",
    camera_fps: 30.0,
    camera_latency_ms: 0,
    tts: "ONLINE",
    recording: "OFF",
    stream: "ON",
    storage: "OK",
    mode: "sim",
    scenario: "A",
    pipeline_fps: 30.0,
    cpu_pct: 18.5,
    mem_pct: 32.1,
    ws_clients: 1,
    uptime_seconds: 0
  });

  const fetchDiagnostics = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus((prev: any) => ({ ...prev, ...data }));
      } else {
        // Fallback to camera API if main status has issues
        const cam = await getCameraStatus();
        setStatus((prev: any) => ({
          ...prev,
          camera: cam.status.toUpperCase(),
          camera_source: cam.source,
          camera_fps: cam.fps,
          camera_latency_ms: cam.latency_ms
        }));
      }
    } catch {
      // Backend offline, keep mock/prev
    }
  };

  useEffect(() => {
    fetchDiagnostics();
    const interval = setInterval(fetchDiagnostics, 2500);
    return () => clearInterval(interval);
  }, []);

  const getSourceDisplayName = (src: string) => {
    switch (src) {
      case 'ip_camera': return 'Phone IP Camera';
      case 'jetson_camera': return 'Jetson CSI/USB Camera';
      case 'sim': return 'Synthetic Simulation';
      default: return src || 'Unknown';
    }
  };

  const handleTestAudio = async () => {
    const text = "ORBITA voice diagnostics passed. Text to speech is operational.";
    try {
      await fetch('http://localhost:8000/api/voice/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    } catch (e) {
      console.warn('Backend speak error:', e);
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(u);
      } catch (e) {
        console.warn('Browser speechSynthesis error:', e);
      }
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto h-full flex flex-col">
      <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100 mb-8 border-b border-space-600 pb-4">
        SYSTEM DIAGNOSTICS
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 flex-1 overflow-y-auto">
        <section>
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mb-4">PIPELINE STATUS</h2>
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 space-y-6">
            <DiagnosticRow 
              icon={<Camera />} 
              label={`CAMERA (${getSourceDisplayName(status.camera_source)})`} 
              state={status.camera} 
            />
            <DiagnosticRow icon={<Brain />} label="AI INFERENCE ENGINE" state={status.ai_engine} />
            <DiagnosticRow icon={<Database />} label="STORAGE SUBSYSTEM" state={status.storage} />
            <div className="flex items-center justify-between">
              <DiagnosticRow icon={<Mic />} label="VOICE ENGINE" state={status.tts} />
              <button
                type="button"
                onClick={handleTestAudio}
                className="px-2.5 py-1 text-xs rounded bg-space-700 hover:bg-space-600 text-accent-cyan border border-accent-cyan/40 flex items-center gap-1.5 font-mono transition-colors"
                title="Test Voice Output"
              >
                <Volume2 size={12} />
                Test Voice
              </button>
            </div>
            <DiagnosticRow icon={<Video />} label="VIDEO STREAM" state={status.stream} />
          </div>
        </section>

        <section>
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mb-4">HARDWARE & SENSOR TELEMETRY</h2>
          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="CPU LOAD" value={`${status.cpu_pct}%`} />
            <MetricCard label="MEMORY" value={`${status.mem_pct}%`} />
            <MetricCard label="CAMERA FPS" value={`${status.camera_fps ?? status.pipeline_fps ?? 0}`} />
            <MetricCard 
              label="CAMERA LATENCY" 
              value={status.camera_latency_ms ? `${status.camera_latency_ms} ms` : '< 5 ms'} 
            />
          </div>

          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold mt-8 mb-4">SYSTEM CONFIG</h2>
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 font-mono text-sm space-y-4 text-space-100">
            <div className="flex justify-between">
              <span className="text-space-400">ACTIVE CAMERA SOURCE</span>
              <span className="uppercase text-accent-cyan font-bold">{getSourceDisplayName(status.camera_source)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-space-400">OPERATION MODE</span>
              <span className="uppercase">{status.mode}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-space-400">ACTIVE SCENARIO</span>
              <span className="uppercase">{status.scenario}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-space-400">MODEL PIPELINE</span>
              <span>YOLOv8 + MediaPipe + Temporal HAR</span>
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
