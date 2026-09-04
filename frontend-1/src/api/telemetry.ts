import { useState, useEffect, useRef } from 'react';
import type { ValidationResult } from './types';

// The default backend port is 8000
const WS_URL = 'ws://localhost:8000/ws/telemetry';

export function useTelemetry() {
  const [data, setData] = useState<ValidationResult | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws.current = new WebSocket(WS_URL);

      ws.current.onopen = () => {
        setIsConnected(true);
        console.log('Connected to ORBITA telemetry');
      };

      ws.current.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          setData(payload);
        } catch (e) {
          console.error('Failed to parse telemetry data', e);
        }
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        console.log('Disconnected from ORBITA telemetry. Reconnecting...');
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws.current?.close();
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  return { data, isConnected };
}
