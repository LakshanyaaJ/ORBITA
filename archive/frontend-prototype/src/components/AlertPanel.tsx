import React from 'react';
import { AlertTriangle, CheckCircle, Info, XCircle } from 'lucide-react';
import { TelemetryState } from '../types/orbita';

interface Props {
  state: TelemetryState;
}

const isError = (status: string) =>
  ['WRONG_OBJECT', 'WRONG_ACTION', 'STEP_SKIPPED', 'OUT_OF_SEQUENCE'].includes(status);

export const AlertPanel: React.FC<Props> = ({ state }) => {
  const { status, error_type, recovery_message, voice_message, detected_action, detected_object, current_step } = state;

  const hasError = isError(status);

  if (!hasError && status !== 'UNCERTAIN') {
    return (
      <div className="bg-[#0b1120] border border-[#1e3a5f] rounded-xl p-4">
        <div className="flex items-center gap-2 mb-2">
          <Info size={14} className="text-[#38bdf8]" />
          <h3 className="text-[#4fc3f7] text-xs font-semibold tracking-widest uppercase">
            Alert Panel
          </h3>
        </div>
        <p className="text-[#334155] text-xs font-mono">
          {status === 'COMPLETED' ? '✓ Experiment complete — no alerts.' : 'No alerts — procedure nominal.'}
        </p>
        {voice_message && (
          <div className="mt-2 p-2 bg-[#061020] rounded-lg border border-[#1e3a5f]">
            <p className="text-[#475569] text-[10px] font-mono uppercase tracking-widest mb-0.5">Voice</p>
            <p className="text-[#64748b] text-xs italic">"{voice_message}"</p>
          </div>
        )}
      </div>
    );
  }

  const severityMap: Record<string, { icon: React.FC<{ size: number; className: string }>, color: string, bg: string }> = {
    WRONG_OBJECT:    { icon: XCircle, color: '#ef4444', bg: '#2d0a0a' },
    WRONG_ACTION:    { icon: XCircle, color: '#ef4444', bg: '#2d0a0a' },
    STEP_SKIPPED:    { icon: AlertTriangle, color: '#f59e0b', bg: '#1c1100' },
    OUT_OF_SEQUENCE: { icon: AlertTriangle, color: '#f59e0b', bg: '#1c1100' },
    UNCERTAIN:       { icon: Info, color: '#38bdf8', bg: '#001f3f' },
  };

  const sev = severityMap[status] || severityMap.UNCERTAIN;
  const Icon = sev.icon;

  return (
    <div
      className="bg-[#0b1120] rounded-xl p-4 border animate-pulse-once"
      style={{ borderColor: sev.color + '66', boxShadow: `0 0 20px ${sev.color}22` }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span style={{ color: sev.color }}>
          <Icon size={16} className="" />
        </span>
        <h3 className="text-xs font-semibold tracking-widest uppercase" style={{ color: sev.color }}>
          Procedure Alert
        </h3>
        <span
          className="ml-auto text-[10px] font-mono font-bold px-2 py-0.5 rounded"
          style={{ color: sev.color, background: sev.bg }}
        >
          {error_type?.replace('_', ' ') || status.replace(/_/g, ' ')}
        </span>
      </div>

      {/* Expected vs Detected */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="p-2 rounded-lg bg-[#061020] border border-[#22c55e22]">
          <p className="text-[10px] text-[#22c55e] font-mono uppercase tracking-widest mb-1">Expected</p>
          <p className="text-white font-bold text-sm font-mono">
            {current_step?.action?.replace(/_/g, ' ') || '—'}
          </p>
        </div>
        <div className="p-2 rounded-lg bg-[#061020] border border-[#ef444422]">
          <p className="text-[10px] text-[#ef4444] font-mono uppercase tracking-widest mb-1">Detected</p>
          <p className="text-white font-bold text-sm font-mono">
            {detected_action} {detected_object}
          </p>
        </div>
      </div>

      {/* Recovery */}
      {recovery_message && (
        <div className="p-2 rounded-lg border" style={{ background: sev.bg, borderColor: sev.color + '44' }}>
          <p className="text-[10px] font-mono uppercase tracking-widest mb-1" style={{ color: sev.color }}>
            Recovery
          </p>
          <p className="text-white text-xs">{recovery_message}</p>
        </div>
      )}

      {/* Voice message */}
      {voice_message && (
        <div className="mt-2 p-2 bg-[#061020] rounded-lg border border-[#1e3a5f]">
          <p className="text-[#475569] text-[10px] font-mono uppercase tracking-widest mb-0.5">🔊 Voice</p>
          <p className="text-[#94a3b8] text-xs italic">"{voice_message}"</p>
        </div>
      )}
    </div>
  );
};
