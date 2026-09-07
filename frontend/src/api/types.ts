export interface ExperimentStep {
  id: number;
  action: string;
  label: string;
  description: string;
  expected_action?: string;
  expected_object?: string;
  expected_target?: string;
}

export interface DebugTelemetry {
  source: string;
  frame: number;
  total_frames: number;
  video_time: string;
  yolo_detections: string[];
  tracks: string[];
  action: string;
  action_status: string;
  fsm_step: string;
  fsm_step_number: number;
  voice_prompt: string;
  voice_status: string;
}

export interface ConfirmedActionInfo {
  action: string;
  object: string;
  target?: string;
  status: string;
  confidence: number;
}

export interface FSMState {
  experiment_id: string;
  current_step_idx: number;
  total_steps: number;
  status: 'WAITING' | 'CORRECT' | 'WRONG_OBJECT' | 'WRONG_ACTION' | 'WRONG_SEQUENCE' | 'STEP_SKIPPED' | 'OUT_OF_SEQUENCE' | 'REPEATED_ACTION' | 'UNCERTAIN' | 'COMPLETED';
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
  voice_status?: string;
  hud_message: string;
  alert_level: 'info' | 'warning' | 'error' | 'success';
  fps: number;
  latency_ms: number;
  rules: any[];
  action_confidence?: number;
  next_action?: string;
  next_confidence?: number;
  is_uncertain?: boolean;
  steps?: ExperimentStep[];
  confirmed_action?: ConfirmedActionInfo;
  debug_telemetry?: DebugTelemetry;
}
