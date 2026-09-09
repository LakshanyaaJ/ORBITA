import { Outlet, useNavigate, useLocation, Link } from 'react-router-dom';
import { Activity, Cpu, Settings, ArrowLeft, Radio, Film } from 'lucide-react';
import clsx from 'clsx';

import GoogleAppsButton from './GoogleAppsButton';

export default function PageShell() {
  const navigate = useNavigate();
  const location = useLocation();

  const isLive = location.pathname.includes('/live');

  return (
    <div className={clsx(
      "bg-space-900 text-space-100 flex flex-col font-sans",
      isLive ? "h-screen overflow-hidden" : "min-h-screen"
    )}>
      {/* Top Navigation Bar - Clean Mission Control Light Aerospace Style */}
      <header className="h-16 border-b border-space-600 flex items-center justify-between px-6 shrink-0 bg-space-800 shadow-[0_1px_2px_rgba(0,0,0,0.03)] z-30">
        <div className="flex items-center gap-6">
          {/* ORBITA Brand with Mission Control Mark */}
          <Link to="/mission" className="flex items-center gap-3 group focus:outline-none">
            <div className="w-6 h-7 relative flex flex-col justify-between py-1">
              <span className="block h-1 w-3.5 bg-accent-cyan rounded-xs transform -skew-x-[24deg] transition-all group-hover:w-4.5" />
              <span className="block h-1 w-5 bg-accent-cyan rounded-xs transform -skew-x-[24deg]" />
              <span className="block h-1 w-3.5 bg-accent-cyan rounded-xs transform -skew-x-[24deg] transition-all group-hover:w-4.5" />
            </div>
            <div>
              <div className="font-['Space_Grotesk'] text-lg font-bold tracking-[0.14em] text-space-100 leading-none">
                ORBITA
              </div>
              <div className="text-[8px] font-bold tracking-[0.12em] text-space-400 leading-none mt-1 uppercase">
                AI Activity & Procedure Verification
              </div>
            </div>
          </Link>
          
          {!isLive && (
            <nav className="flex items-center gap-1.5 ml-6 pl-6 border-l border-space-600">
              <NavButton 
                active={location.pathname === '/mission' || location.pathname.includes('/experiments')} 
                onClick={() => navigate('/mission')}
                icon={<Activity size={16} strokeWidth={2} />}
                label="MISSION"
              />
              <NavButton 
                active={location.pathname.includes('/log')} 
                onClick={() => navigate('/experiments/EXP-01/log')}
                icon={<Settings size={16} strokeWidth={2} />}
                label="LOG"
              />
              <NavButton 
                active={location.pathname === '/system'} 
                onClick={() => navigate('/system')}
                icon={<Cpu size={16} strokeWidth={2} />}
                label="SYSTEM"
              />
              <NavButton 
                active={location.pathname === '/vdata' || location.pathname === '/dataset'} 
                onClick={() => navigate('/vdata')}
                icon={<Film size={16} strokeWidth={2} />}
                label="VDATA / DATASET"
              />
              <NavButton 
                active={location.pathname === '/mission-control'} 
                onClick={() => navigate('/mission-control')}
                icon={<Radio size={16} strokeWidth={2} />}
                label="MISSION CONTROL"
              />
            </nav>
          )}

          {isLive && (
            <button 
              onClick={() => {
                if (window.confirm("Are you sure you want to exit the live experiment?")) {
                  navigate('/mission');
                }
              }}
              className="flex items-center gap-2 text-space-400 hover:text-status-critical transition-colors uppercase font-bold text-xs tracking-wider ml-4 px-3 py-1.5 rounded-md hover:bg-red-50"
            >
              <ArrowLeft size={16} />
              EXIT EXPERIMENT
            </button>
          )}
        </div>

        <div className="flex items-center gap-4">
          {/* Status Indicators */}
          <div className="flex items-center gap-3 text-xs font-mono font-semibold tracking-wider">
            <StatusIndicator label="AI" active={true} color="bg-status-success" />
            <StatusIndicator label="REC" active={isLive} color={isLive ? "bg-status-warning animate-pulse" : "bg-space-400"} />
            <StatusIndicator label="STREAM" active={true} color="bg-status-success" />
          </div>

          <div className="w-px h-5 bg-space-600 mx-1"></div>
          
          {/* Clock */}
          <div className="font-mono text-space-400 text-xs bg-space-700 px-2.5 py-1 rounded border border-space-600">
            {new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })} UTC
          </div>

          <div className="w-px h-5 bg-space-600 mx-1"></div>

          {/* Google Apps Icon Launcher */}
          <GoogleAppsButton />
        </div>
      </header>

      {/* Main Content Area */}
      <main className={clsx("flex-1 relative bg-space-900 min-h-0", isLive ? "overflow-hidden" : "overflow-y-auto")}>
        <Outlet />
      </main>
    </div>
  );
}

function NavButton({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-3.5 py-1.5 rounded-md font-bold tracking-wider text-xs transition-all cursor-pointer font-mono",
        active 
          ? "bg-[#eaf2f7] text-[#2f6f9f] border border-[#cfdbe3] shadow-xs" 
          : "text-space-400 hover:bg-space-700 hover:text-space-100"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function StatusIndicator({ label, active, color }: { label: string, active: boolean, color: string }) {
  return (
    <div className={clsx(
      "flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] border",
      active ? "border-space-600 bg-space-800 text-space-100" : "border-transparent text-space-400"
    )}>
      <div className={clsx("w-2 h-2 rounded-full", color)}></div>
      <span>{label}</span>
    </div>
  );
}
