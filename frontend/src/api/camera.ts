/**
 * ORBITA Camera API Client
 */

export interface CameraStatus {
  connected: boolean;
  source: 'sim' | 'jetson_camera' | 'ip_camera' | 'disconnected';
  url?: string;
  device_index?: number;
  fps: number;
  latency_ms: number;
  status: 'connected' | 'connecting' | 'reconnecting' | 'disconnected' | 'error';
  error?: string | null;
}

export interface ConnectCameraParams {
  source: 'jetson_camera' | 'ip_camera' | 'sim';
  url?: string;
  ip?: string;
  port?: number | string;
  path?: string;
  device_index?: number;
  timeout_sec?: number;
}

const BACKEND_BASE = 'http://localhost:8000';

export async function getCameraStatus(): Promise<CameraStatus> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/status`);
    if (!res.ok) {
      throw new Error(`Status HTTP error: ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    return {
      connected: false,
      source: 'disconnected',
      fps: 0,
      latency_ms: 0,
      status: 'disconnected',
      error: err?.message || 'Failed to fetch camera status',
    };
  }
}

export async function connectCamera(params: ConnectCameraParams): Promise<{ success: boolean; error?: string }> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/connect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await res.json();
    if (!res.ok) {
      return { success: false, error: data.error || 'Failed to connect camera' };
    }
    return { success: true };
  } catch (err: any) {
    return { success: false, error: err?.message || 'Network connection failed' };
  }
}

export async function disconnectCamera(): Promise<{ success: boolean; error?: string }> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/disconnect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) {
      const data = await res.json();
      return { success: false, error: data.error || 'Failed to disconnect camera' };
    }
    return { success: true };
  } catch (err: any) {
    return { success: false, error: err?.message || 'Network error on disconnect' };
  }
}
