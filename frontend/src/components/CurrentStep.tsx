import React from 'react';
import { TelemetryState, FSMStatus } from '../types/orbita';

interface Props {
  state: TelemetryState;
}

const STATUS_CONFIG: Record<FSMStatus, { label: string; color: string; bg: string; glow: string }> = {
  WAITING:          { label: 'AWAITING', color: '#94a3b8', bg: '#1e293b', glow: '' },
  CORRECT:          { label: 'COMPLETED', color: '#22c55e', bg: '#052e16', glow: 'shadow-[0_0_20px_#22c55e44]' },
  WRONG_OBJECT:     { label: 'WRONG OBJECT', color: '#ef4444', bg: '#2d0a0a', glow: 'shadow-[0_0_20px_#ef444444]' },
  WRONG_ACTION:     { label: 'WRONG ACTION', color: '#ef4444', bg: '#2d0a0a', glow: 'shadow-[0_0_20px_#ef444444]' },
  STEP_SKIPPED:     { label: 'STEP SKIPPED', color: '#f59e0b', bg: '#1c1100', glow: 'shadow-[0_0_20px_#f59e0b44]' },
  OUT_OF_SEQUENCE:  { label: 'OUT OF SEQUENCE', color: '#f59e0b', bg: '#1c1100', glow: 'shadow-[0_0_20px_#f59e0b44]' },
  REPEATED_ACTION:  { label: 'ALREADY DONE', color: '#a855f7', bg: '#1a0030', glow: '' },
  UNCERTAIN:        { label: 'UNCERTAIN', color: '#38bdf8', bg: '#001f3f', glow: '' },
  COMPLETED:        { label: 'EXPERIMENT COMPLETE', color: '#22c55e', bg: '#052e16', glow: 'shadow-[0_0_30px_#22c55e66]' },
};

export const CurrentStep: React.FC<Props> = ({ state }) => {
  const { status, current_step, next_step, action_confidence } = state;
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.WAITING;

  return (
    <div className="space-y-3">
      {/* Current Step */}
      <div
        className={`bg-[#0b1120] border border-[#1e3a5f] rounded-xl p-4 transition-all duration-300 ${cfg.glow}`}
        style={{ borderColor: cfg.color + '44' }}
      >
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-[#4fc3f7] text-xs font-semibold tracking-widest uppercase">
            Current Step
          </h3>
          <span
            className="text-[10px] font-mono font-bold px-2 py-0.5 rounded"
            style={{ color: cfg.color, background: cfg.bg, border: `1px solid ${cfg.color}44` }}
          >
            {cfg.label}
          </span>
        </div>

        {current_step ? (
          <>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-3xl font-mono font-black text-white">
                {String(current_step.id).padStart(2, '0')}
              </span>
              <span className="text-lg font-bold text-white uppercase tracking-wide">
                {current_step.label}
              </span>
            </div>
            <p className="text-[#475569] text-xs font-mono">
              ACTION: <span className="text-[#94a3b8]">{current_step.action}</span>
            </p>
          </>
        ) : status === 'COMPLETED' ? (
          <p className="text-[#22c55e] font-bold text-lg">✓ All Steps Complete</p>
        ) : (
          <p className="text-[#475569] font-mono text-sm">Initializing...</p>
        )}

        {/* Confidence bar */}
        {action_confidence > 0 && (
          <div className="mt-3">
            <div className="flex justify-between text-[10px] text-[#475569] mb-1">
              <span>ACTION CONFIDENCE</span>
              <span style={{ color: action_confidence > 0.7 ? '#22c55e' : action_confidence > 0.55 ? '#f59e0b' : '#ef4444' }}>
                {(action_confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-1 bg-[#0f2032] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${action_confidence * 100}%`,
                  background: action_confidence > 0.7 ? '#22c55e' : action_confidence > 0.55 ? '#f59e0b' : '#ef4444',
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Next Step */}
      {next_step && (
        <div className="bg-[#0b1120] border border-[#1e3a5f] rounded-xl p-4">
          <h3 className="text-[#475569] text-xs font-semibold tracking-widest uppercase mb-2">
            Next Action
          </h3>
          <div className="flex items-baseline gap-2">
            <span className="text-[#1e3a5f] text-2xl font-mono font-black">
              {String(next_step.id).padStart(2, '0')}
            </span>
            <span className="text-[#334155] font-bold uppercase tracking-wide">
              {next_step.label}
            </span>
          </div>
          <p className="text-[#334155] text-xs font-mono mt-1">
            {next_step.action}
          </p>
        </div>
      )}
    </div>
  );
};
