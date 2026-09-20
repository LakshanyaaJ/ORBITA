import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Rocket, Satellite, Activity, Cpu } from 'lucide-react';

interface NavbarProps {
  telemetryConnected?: boolean;
  fps?: number;
  aiStatus?: string;
  activeModel?: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  telemetryConnected = true,
  fps = 28,
  aiStatus = 'YOLO11 + GRU',
  activeModel = 'orbita_yolo11.pt',
}) => {
  const location = useLocation();
  const isGround = location.pathname.startsWith('/ground');

  return (
    <header className="bg-slate-950 border-b border-slate-800 text-slate-100 sticky top-0 z-50 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand & Identity */}
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30 text-cyan-400">
              <Rocket className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-mono text-xl font-bold tracking-wider text-white">ORBITA</span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-cyan-900/60 text-cyan-300 border border-cyan-700/50">
                  v2.0 YOLO11
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium">Onboard Experiment Copilot & Ground Telemetry</p>
            </div>
          </div>

          {/* Dual Dashboard Mode Switches */}
          <div className="flex items-center bg-slate-900/90 p-1.5 rounded-xl border border-slate-800 space-x-2">
            <NavLink
              to="/onboard"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
                  isActive || (!isGround && location.pathname === '/')
                    ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Rocket className="w-4 h-4" />
              <span>Astronaut / Onboard</span>
            </NavLink>

            <NavLink
              to="/ground"
              className={({ isActive }) =>
                `flex items-center space-x-2 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
                  isActive
                    ? 'bg-amber-600 text-white shadow-md shadow-amber-600/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`
              }
            >
              <Satellite className="w-4 h-4" />
              <span>Earth / Ground Control</span>
            </NavLink>
          </div>

          {/* Telemetry & System Status Indicators */}
          <div className="hidden md:flex items-center space-x-4">
            <div className="flex items-center space-x-2 bg-slate-900/70 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">AI:</span>
              <span className="font-mono text-cyan-300 font-semibold">{aiStatus}</span>
            </div>

            <div className="flex items-center space-x-2 bg-slate-900/70 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
              <Activity className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-slate-400">FPS:</span>
              <span className="font-mono text-emerald-400 font-bold">{fps}</span>
            </div>

            <div className="flex items-center space-x-2 bg-slate-900/70 px-3 py-1.5 rounded-lg border border-slate-800 text-xs" title={`Active Weights: ${activeModel}`}>
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  telemetryConnected ? 'bg-emerald-500 animate-ping' : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300 font-semibold">
                {telemetryConnected ? 'OFFLINE EDGE READY' : 'DISCONNECTED'}
              </span>
            </div>
          </div>

        </div>
      </div>
    </header>
  );
};
