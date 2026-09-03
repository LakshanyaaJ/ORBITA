import React from 'react';
import { TelemetryState } from '../types/orbita';

interface Props {
  state: TelemetryState;
}

const ALL_STEPS = [
  'Open Main Box',
  'Take Red Box',
  'Take Yellow Box',
  'Open Red Box',
  'Take Sample',
  'Perform Experiment',
  'Close Box',
  'Store Components',
];

export const ProcedureTimeline: React.FC<Props> = ({ state }) => {
  const { completed_steps, failed_steps, skipped_steps, current_step_idx, total_steps } = state;

  return (
    <div className="bg-[#0b1120] border border-[#1e3a5f] rounded-xl p-4">
      <h3 className="text-[#4fc3f7] text-xs font-semibold tracking-widest uppercase mb-3">
        Procedure Timeline
      </h3>
      <div className="space-y-1.5">
        {ALL_STEPS.slice(0, total_steps || 8).map((label, idx) => {
          const stepId = idx + 1;
          const isDone = completed_steps.includes(stepId);
          const isFailed = failed_steps.includes(stepId);
          const isSkipped = skipped_steps.includes(stepId);
          const isCurrent = current_step_idx === idx && !isDone;

          let icon = '○';
          let colorClass = 'text-[#334155]';
          let bgClass = '';

          if (isDone) {
            icon = '✓';
            colorClass = 'text-[#22c55e]';
          } else if (isSkipped) {
            icon = '⚠';
            colorClass = 'text-[#f59e0b]';
          } else if (isFailed) {
            icon = '✗';
            colorClass = 'text-[#ef4444]';
          } else if (isCurrent) {
            icon = '→';
            colorClass = 'text-[#38bdf8]';
            bgClass = 'bg-[#0f2744]';
          }

          return (
            <div
              key={stepId}
              className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg transition-all ${bgClass}`}
            >
              <span className={`text-sm font-mono font-bold w-4 ${colorClass}`}>{icon}</span>
              <span className={`text-xs font-mono ${isCurrent ? 'text-white' : isDone ? 'text-[#94a3b8]' : 'text-[#475569]'}`}>
                {String(stepId).padStart(2, '0')} {label}
              </span>
              {isCurrent && (
                <span className="ml-auto text-[10px] text-[#38bdf8] font-mono animate-pulse">
                  IN PROGRESS
                </span>
              )}
              {isDone && (
                <span className="ml-auto text-[10px] text-[#22c55e] font-mono">
                  DONE
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="mt-4">
        <div className="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>PROGRESS</span>
          <span>{Math.round(state.progress_pct)}%</span>
        </div>
        <div className="h-1.5 bg-[#0f2032] rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[#0ea5e9] to-[#22c55e] rounded-full transition-all duration-500"
            style={{ width: `${state.progress_pct}%` }}
          />
        </div>
      </div>
    </div>
  );
};
