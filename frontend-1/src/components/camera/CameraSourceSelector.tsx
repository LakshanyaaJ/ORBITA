import { Smartphone, Video, Sparkles } from 'lucide-react';
import clsx from 'clsx';

export type CameraSourceType = 'jetson_camera' | 'ip_camera' | 'sim';

interface CameraSourceSelectorProps {
  selectedSource: CameraSourceType;
  onChange: (source: CameraSourceType) => void;
  disabled?: boolean;
}

export default function CameraSourceSelector({
  selectedSource,
  onChange,
  disabled = false,
}: CameraSourceSelectorProps) {
  const options: { id: CameraSourceType; label: string; icon: typeof Video }[] = [
    { id: 'jetson_camera', label: 'Jetson Camera', icon: Video },
    { id: 'ip_camera', label: 'Phone IP Camera', icon: Smartphone },
    { id: 'sim', label: 'Simulation', icon: Sparkles },
  ];

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[11px] font-mono font-bold tracking-widest text-space-400 uppercase">
        Camera Source
      </div>
      <div className="grid grid-cols-3 gap-1.5 p-1 bg-space-900 rounded border border-space-600">
        {options.map((opt) => {
          const Icon = opt.icon;
          const isSelected = selectedSource === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              disabled={disabled}
              onClick={() => onChange(opt.id)}
              className={clsx(
                "flex items-center justify-center gap-1.5 py-1.5 px-2 rounded text-xs font-mono font-bold transition-all",
                isSelected
                  ? "bg-space-700 text-accent-cyan shadow border border-space-600"
                  : "text-space-400 hover:text-space-100 hover:bg-space-800/60",
                disabled && "opacity-50 cursor-not-allowed"
              )}
            >
              <Icon size={14} />
              <span className="truncate">{opt.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
