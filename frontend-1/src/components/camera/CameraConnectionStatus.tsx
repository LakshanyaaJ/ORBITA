import clsx from 'clsx';

interface CameraConnectionStatusProps {
  status: 'connected' | 'connecting' | 'reconnecting' | 'disconnected' | 'error';
  error?: string | null;
  className?: string;
}

export default function CameraConnectionStatus({
  status,
  error,
  className = '',
}: CameraConnectionStatusProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          dotClass: 'bg-status-success shadow-[0_0_8px_rgba(76,175,125,0.6)]',
          textClass: 'text-status-success',
          label: 'Connected',
        };
      case 'connecting':
        return {
          dotClass: 'bg-status-warning animate-pulse shadow-[0_0_8px_rgba(214,168,79,0.6)]',
          textClass: 'text-status-warning',
          label: 'Connecting...',
        };
      case 'reconnecting':
        return {
          dotClass: 'bg-status-warning animate-pulse shadow-[0_0_8px_rgba(214,168,79,0.6)]',
          textClass: 'text-status-warning',
          label: 'Reconnecting...',
        };
      case 'error':
        return {
          dotClass: 'bg-status-critical shadow-[0_0_8px_rgba(217,101,101,0.6)]',
          textClass: 'text-status-critical',
          label: 'Connection Error',
        };
      case 'disconnected':
      default:
        return {
          dotClass: 'bg-space-400',
          textClass: 'text-space-400',
          label: 'Disconnected',
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div className={clsx("flex items-center gap-2 font-mono text-xs font-bold tracking-wider", className)}>
      <span className={clsx("w-2 h-2 rounded-full inline-block transition-colors", config.dotClass)} />
      <span className={config.textClass}>{config.label}</span>
      {error && status === 'error' && (
        <span className="text-[10px] text-space-400 font-normal truncate max-w-[220px]" title={error}>
          ({error})
        </span>
      )}
    </div>
  );
}
