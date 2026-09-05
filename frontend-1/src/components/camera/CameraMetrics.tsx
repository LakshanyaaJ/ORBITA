import { Activity, Clock, Video } from 'lucide-react';
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

  return (
    <div className={`grid grid-cols-3 gap-2 bg-space-800/80 border border-space-600 rounded p-2.5 font-mono text-xs ${className}`}>
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

      {/* FPS */}
      <div className="flex flex-col gap-0.5 border-r border-space-600/50 px-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Activity size={11} className="text-status-success" />
          FPS
        </span>
        <span className="text-space-100 font-semibold text-[11px]">
          {status.connected ? `${status.fps.toFixed(1)}` : '0.0'}
        </span>
      </div>

      {/* Latency */}
      <div className="flex flex-col gap-0.5 pl-2">
        <span className="text-[10px] text-space-400 font-bold uppercase tracking-widest flex items-center gap-1">
          <Clock size={11} className="text-status-warning" />
          Latency
        </span>
        <span className="text-space-100 font-semibold text-[11px]">
          {status.connected && status.source === 'ip_camera' ? `${status.latency_ms.toFixed(0)} ms` : '< 5 ms'}
        </span>
      </div>
    </div>
  );
}
