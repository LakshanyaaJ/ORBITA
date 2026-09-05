import { useState } from 'react';
import { Link } from 'react-router-dom';

interface GoogleAppsButtonProps {
  className?: string;
  to?: string;
}

export default function GoogleAppsButton({
  className = '',
  to = '/mission-control',
}: GoogleAppsButtonProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative inline-flex items-center">
      <Link
        to={to}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        aria-label="Launch ORBITA Mission Control"
        title="ORBITA Mission Control"
        className={`group relative flex items-center justify-center w-8 h-8 rounded-md text-space-400 hover:text-accent-cyan hover:bg-space-700 active:bg-space-600 transition-all duration-150 border border-transparent hover:border-space-600 focus:outline-none focus:ring-2 focus:ring-accent-cyan/30 ${className}`}
      >
        {/* 3x3 Google Apps / Waffle Icon */}
        <svg
          className="w-4.5 h-4.5 transition-transform duration-200 group-hover:scale-110"
          viewBox="0 0 24 24"
          fill="currentColor"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Row 1 */}
          <circle cx="5" cy="5" r="2" />
          <circle cx="12" cy="5" r="2" />
          <circle cx="19" cy="5" r="2" />
          {/* Row 2 */}
          <circle cx="5" cy="12" r="2" />
          <circle cx="12" cy="12" r="2" />
          <circle cx="19" cy="12" r="2" />
          {/* Row 3 */}
          <circle cx="5" cy="19" r="2" />
          <circle cx="12" cy="19" r="2" />
          <circle cx="19" cy="19" r="2" />
        </svg>
      </Link>

      {/* Floating Tooltip */}
      {showTooltip && (
        <div className="absolute right-0 top-full mt-2 z-50 pointer-events-none whitespace-nowrap px-3 py-1.5 rounded-md bg-space-800 border border-space-600 shadow-md text-xs font-mono text-space-100 flex items-center gap-2 animate-in fade-in zoom-in-95 duration-100">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse"></span>
          <span className="font-medium text-space-100">ORBITA Mission Control</span>
          <span className="text-space-400 text-[10px]">(/mission-control)</span>
        </div>
      )}
    </div>
  );
}
