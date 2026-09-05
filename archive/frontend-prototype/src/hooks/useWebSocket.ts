import { useState, useEffect, useRef, useCallback } from 'react';
import { TelemetryState } from '../types/orbita';

const WS_URL = `ws://${window.location.hostname}:8000/ws/telemetry`;

const DEFAULT_STATE: TelemetryState = {
  experiment_id: 'EXP00000',
  current_step_idx: 0,
  total_steps: 8,
  status: 'WAITING',
  current_step: null,
  next_step: null,
  completed_steps: [],
  failed_steps: [],
  skipped_steps: [],
  detected_action: 'IDLE',
  detected_object: '',
  error_type: null,
  recovery_message: null,
  voice_message: '',
  hud_message: 'CONNECTING...',
  alert_level: 'info',
  action_confidence: 0,
  next_action: '',
  next_confidence: 0,
  is_uncertain: false,
  progress_pct: 0,
  elapsed_seconds: 0,
  fps: 0,
  latency_ms: 0,
  rules: [],
};

export function useWebSocket() {
  const [state, setState] = useState<TelemetryState>(DEFAULT_STATE);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data) as Partial<TelemetryState>;
        setState(prev => ({ ...prev, ...data }));
      } catch {
        /* ignore parse errors */
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 2 seconds
      reconnectTimer.current = setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendControl = useCallback(async (action: string, params: Record<string, unknown> = {}) => {
    try {
      await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...params }),
      });
    } catch (e) {
      console.error('Control request failed:', e);
    }
  }, []);

  return { state, connected, sendControl };
}
