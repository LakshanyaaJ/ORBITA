import React, { useEffect, useState } from 'react';
import { Activity, Camera, Mic, Circle, Wifi, HardDrive } from 'lucide-react';
import type { SystemStatus as SystemStatusType } from '../types/orbita';

interface Props {
  connected: boolean;
}

const StatusDot: React.FC<{ status: string }> = ({ status }) => {
  const isOn = status === 'ONLINE' || status === 'ON' || status === 'OK';
  return (
    <span
      className="inline-block w-2 h-2 rounded-full"
      style={{
        background: isOn ? '#22c55e' : '#475569',
        boxShadow: isOn ? '0 0 6px #22c55e' : 'none',
      }}
    />
  );
};

export const SystemStatusPanel: React.FC<Props> = ({ connected }) => {
  const [status, setStatus] = useState<SystemStatusType | null>(null);

  useEffect(() => {
    const fetch_status = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok) setStatus(await res.json());
      } catch { /* ignore */ }
    };
    fetch_status();
    const interval = setInterval(fetch_status, 2000);
    return () => clearInterval(interval);
  }, []);

  const items = [
    { label: 'AI ENGINE', value: status?.ai_engine || 'LOADING', icon: Activity },
    { label: 'CAMERA', value: status?.camera || '—', icon: Camera },
    { label: 'TTS VOICE', value: status?.tts || '—', icon: Mic },
    { label: 'RECORDING', value: status?.recording || '—', icon: Circle },
    { label: 'STREAM', value: status?.stream || '—', icon: Wifi },
    { label: 'STORAGE', value: status?.storage || '—', icon: HardDrive },
  ];

  return (
    <div className="bg-[#0b1120] border border-[#1e3a5f] rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[#4fc3f7] text-xs font-semibold tracking-widest uppercase">
          System Status
        </h3>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${connected ? 'text-[#22c55e] bg-[#052e16]' : 'text-[#ef4444] bg-[#2d0a0a]'}`}>
          WS: {connected ? 'LIVE' : 'RECONNECTING'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {items.map(({ label, value, icon: Icon }) => (
          <div key={label} className="flex items-center gap-2 py-1">
            <Icon size={11} className="text-[#334155]" />
            <span className="text-[#334155] text-[10px] font-mono">{label}</span>
            <div className="ml-auto flex items-center gap-1.5">
              <StatusDot status={value} />
              <span className={`text-[10px] font-mono font-bold ${value === 'ONLINE' || value === 'ON' || value === 'OK' ? 'text-[#22c55e]' : 'text-[#475569]'}`}>
                {value}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Resource meters */}
      {status && (
        <div className="mt-3 pt-3 border-t border-[#1e3a5f] space-y-2">
          <div>
            <div className="flex justify-between text-[10px] text-[#475569] mb-1">
              <span>CPU</span><span>{status.cpu_pct.toFixed(0)}%</span>
            </div>
            <div className="h-1 bg-[#0f2032] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${status.cpu_pct}%`,
                  background: status.cpu_pct > 80 ? '#ef4444' : '#0ea5e9',
                }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-[10px] text-[#475569] mb-1">
              <span>RAM</span><span>{status.mem_pct.toFixed(0)}%</span>
            </div>
            <div className="h-1 bg-[#0f2032] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all bg-[#a855f7]"
                style={{ width: `${status.mem_pct}%` }}
              />
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-[#475569]">
            <span>MODE: {status.mode?.toUpperCase()}</span>
            <span>SCENARIO: {status.scenario}</span>
          </div>
        </div>
      )}
    </div>
  );
};
