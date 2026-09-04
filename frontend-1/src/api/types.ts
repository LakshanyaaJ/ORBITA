export interface ExperimentStep {
  id: number;
  action: string;
  label: string;
  description: string;
}

export interface FSMState {
  experiment_id: string;
  current_step_idx: number;
  total_steps: number;
  status: 'WAITING' | 'CORRECT' | 'WRONG_OBJECT' | 'WRONG_ACTION' | 'STEP_SKIPPED' | 'OUT_OF_SEQUENCE' | 'REPEATED_ACTION' | 'UNCERTAIN' | 'COMPLETED';
  current_step: ExperimentStep | null;
  next_step: ExperimentStep | null;
  completed_steps: number[];
  failed_steps: number[];
  skipped_steps: number[];
  detected_action: string;
  detected_object: string;
  error_type: string | null;
  recovery_message: string | null;
  progress_pct: number;
  elapsed_seconds: number;
}

export interface ValidationResult extends FSMState {
  voice_message: string;
  hud_message: string;
  alert_level: 'info' | 'warning' | 'error' | 'success';
  fps: number;
  latency_ms: number;
  rules: any[];
  action_confidence?: number;
  next_action?: string;
  next_confidence?: number;
  is_uncertain?: boolean;
}
