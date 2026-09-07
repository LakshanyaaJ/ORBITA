/**
 * ORBITA Camera API Client
 */

export interface CameraStatus {
  connected: boolean;
  source: 'sim' | 'jetson_camera' | 'ip_camera' | 'phone_webcam' | 'disconnected';
  url?: string;
  device_index?: number;
  fps: number;
  stream_fps?: number;
  ai_fps?: number;
  latency_ms: number;
  status: 'connected' | 'connecting' | 'reconnecting' | 'disconnected' | 'waiting' | 'error';
  error?: string | null;
  rotation?: number;
}

export interface CameraDiagnostics {
  camera_fps: number;
  stream_fps: number;
  ai_fps: number;
  pipeline_latency_ms: number;
  camera_latency_ms: number;
  encode_latency_ms: number;
  dropped_frames_pct: number;
  buffer_size: number;
  resolution: string;
  source: string;
  status: string;
  active_clients: number;
  cpu_pct: number;
  mem_pct: number;
}

export interface AvailableNetworkInterface {
  interface: string;
  ip: string;
  is_default: boolean;
}

export interface PhonePairingInfo {
  pairing_token: string;
  lan_ip: string;
  port: number;
  https_port?: number;
  connection_url: string;
  https_url?: string;
  http_url?: string;
  available_ips?: AvailableNetworkInterface[];
  connected: boolean;
  fps: number;
  latency_ms: number;
  bitrate_mbps: number;
  resolution: string;
  device_info: Record<string, any>;
}

export interface ConnectCameraParams {
  source: 'jetson_camera' | 'ip_camera' | 'phone_webcam' | 'sim';
  url?: string;
  ip?: string;
  port?: number | string;
  path?: string;
  device_index?: number;
  timeout_sec?: number;
}

export const BACKEND_BASE =
  typeof window !== 'undefined' && window.location.hostname
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : 'http://localhost:8000';

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
      stream_fps: 0,
      ai_fps: 0,
      latency_ms: 0,
      status: 'disconnected',
      error: err?.message || 'Failed to fetch camera status',
    };
  }
}

export async function getCameraDiagnostics(): Promise<CameraDiagnostics | null> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/diagnostics`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function getPhonePairingInfo(): Promise<PhonePairingInfo | null> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/phone_pairing`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function refreshPhonePairingToken(): Promise<PhonePairingInfo | null> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/phone_pairing/refresh`, {
      method: 'POST',
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function connectCamera(params: ConnectCameraParams): Promise<{ success: boolean; error?: string; pairing?: PhonePairingInfo }> {
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
    return { success: true, pairing: data.pairing };
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

export async function rotateCamera(degrees: number = 90): Promise<{ success: boolean; rotation: number }> {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/camera/rotate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ degrees }),
    });
    if (!res.ok) return { success: false, rotation: 0 };
    return await res.json();
  } catch {
    return { success: false, rotation: 0 };
  }
}

