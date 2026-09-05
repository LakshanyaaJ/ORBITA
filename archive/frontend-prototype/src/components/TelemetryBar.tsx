import React from 'react';
import { TelemetryState } from '../types/orbita';

interface Props {
  state: TelemetryState;
}

export const TelemetryBar: React.FC<Props> = ({ state }) => {
  const { fps, latency_ms, action_confidence, elapsed_seconds } = state;

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const metrics = [
    { label: 'FPS', value: fps.toFixed(1), unit: '', color: fps > 20 ? '#22c55e' : fps > 10 ? '#f59e0b' : '#ef4444' },
    { label: 'LATENCY', value: latency_ms.toFixed(1), unit: 'ms', color: latency_ms < 50 ? '#22c55e' : '#f59e0b' },
    { label: 'CONFIDENCE', value: (action_confidence * 100).toFixed(0), unit: '%', color: action_confidence > 0.7 ? '#22c55e' : action_confidence > 0.55 ? '#f59e0b' : '#ef4444' },
    { label: 'ELAPSED', value: formatElapsed(elapsed_seconds), unit: '', color: '#94a3b8' },
  ];

  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-[#060e1a] border-t border-[#1e3a5f]">
      <span className="text-[10px] text-[#334155] font-mono tracking-widest uppercase">Telemetry</span>
      <div className="flex items-center gap-6 ml-2">
        {metrics.map(({ label, value, unit, color }) => (
          <div key={label} className="flex items-baseline gap-1">
            <span className="text-[10px] text-[#334155] font-mono">{label}:</span>
            <span className="text-xs font-mono font-bold" style={{ color }}>
              {value}{unit}
            </span>
          </div>
        ))}
      </div>
      <div className="ml-auto text-[10px] text-[#1e3a5f] font-mono">
        EXP: {state.experiment_id}
      </div>
    </div>
  );
};
