import { Activity, Clock, Cpu, Video } from 'lucide-react';
import type { CameraStatus } from '../../api/camera';

interface CameraMetricsProps {
  status: CameraStatus;
  className?: string;
}

export default function CameraMetrics({ status, className = '' }: CameraMetricsProps) {
  const getSourceLabel = () => {
    switch (status.source) {
      case 'ip_camera':
        return 'Phone IP Camera';
      case 'jetson_camera':
        return 'Jetson Camera';
      case 'sim':
        return 'Experiment Simulator';
      default:
        return 'Standby';
    }
  };

  const streamFps = status.stream_fps !== undefined ? status.stream_fps : status.fps;
  const aiFps = status.ai_fps !== undefined ? status.ai_fps : 15.0;
  const latency = status.connected && status.source === 'ip_camera' ? Math.round(status.latency_ms) : 5;
  const isUltraLowLatency = latency <= 120;

  return (
    <div className={`grid grid-cols-4 gap-2 bg-space-800/80 border border-space-600 rounded p-2.5 font-mono text-xs ${className}`}>
      {/* Source */}
      <div className="flex flex-col gap-0.5 border-r border-space-600/50 pr-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Video size={11} className="text-accent-cyan" />
          Source
        </span>
        <span className="text-space-100 font-semibold truncate text-[11px]" title={status.url || getSourceLabel()}>
          {getSourceLabel()}
        </span>
      </div>

      {/* Stream FPS */}
      <div className="flex flex-col gap-0.5 border-r border-space-600/50 px-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Activity size={11} className="text-status-success" />
          Stream FPS
        </span>
        <span className="text-space-100 font-semibold text-[11px]">
          {status.connected ? `${streamFps.toFixed(1)}` : '0.0'}
        </span>
      </div>

      {/* AI FPS */}
      <div className="flex flex-col gap-0.5 border-r border-space-600/50 px-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Cpu size={11} className="text-accent-cyan" />
          AI Rate
        </span>
        <span className="text-space-100 font-semibold text-[11px]">
          {status.connected ? `${aiFps.toFixed(1)} FPS` : '0.0'}
        </span>
      </div>

      {/* Latency & Buffer */}
      <div className="flex flex-col gap-0.5 pl-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Clock size={11} className={isUltraLowLatency ? 'text-status-success' : 'text-status-warning'} />
          Latency / Buf
        </span>
        <span className={`font-semibold text-[11px] ${isUltraLowLatency ? 'text-status-success' : 'text-status-warning'}`}>
          {status.connected ? `${latency} ms` : '—'} <span className="text-[10px] text-space-400 font-normal">| B:1</span>
        </span>
      </div>
    </div>
  );
}
