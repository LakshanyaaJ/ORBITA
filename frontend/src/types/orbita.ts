/** ORBITA Type Definitions */

export type AlertLevel = 'info' | 'warning' | 'error' | 'success';

export type FSMStatus =
  | 'WAITING'
  | 'CORRECT'
  | 'WRONG_OBJECT'
  | 'WRONG_ACTION'
  | 'STEP_SKIPPED'
  | 'OUT_OF_SEQUENCE'
  | 'REPEATED_ACTION'
  | 'UNCERTAIN'
  | 'COMPLETED';

export interface ExperimentStep {
  id: number;
  action: string;
  label: string;
  description?: string;
}

export interface TelemetryState {
  experiment_id: string;
  current_step_idx: number;
  total_steps: number;
  status: FSMStatus;
  current_step: ExperimentStep | null;
  next_step: ExperimentStep | null;
  completed_steps: number[];
  failed_steps: number[];
  skipped_steps: number[];
  detected_action: string;
  detected_object: string;
  error_type: string | null;
  recovery_message: string | null;
  voice_message: string;
  hud_message: string;
  alert_level: AlertLevel;
  action_confidence: number;
  next_action: string;
  next_confidence: number;
  is_uncertain: boolean;
  progress_pct: number;
  elapsed_seconds: number;
  fps: number;
  latency_ms: number;
  rules: Array<{ id: string; result: string; message: string }>;
}

export interface SystemStatus {
  ai_engine: string;
  camera: string;
  tts: string;
  recording: string;
  stream: string;
  storage: string;
  mode: string;
  scenario: string;
  pipeline_fps: number;
  cpu_pct: number;
  mem_pct: number;
  ws_clients: number;
}
