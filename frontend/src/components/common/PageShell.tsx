import { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation, Link } from 'react-router-dom';
import { Activity, Cpu, Settings, ArrowLeft, Radio, Film, Play, Wifi, WifiOff } from 'lucide-react';
import clsx from 'clsx';

import GoogleAppsButton from './GoogleAppsButton';
import OrbitaDemoModal from '../demo/OrbitaDemoModal';
import StructuredResultModal from '../results/StructuredResultModal';
import { BACKEND_BASE } from '../../api/camera';

export default function PageShell() {
  const navigate = useNavigate();
  const location = useLocation();

  const isLive = location.pathname.includes('/live');

  const [isGroundOnline, setIsGroundOnline] = useState<boolean>(true);
  const [pendingSync, setPendingSync] = useState<number>(0);
  const [isDemoModalOpen, setIsDemoModalOpen] = useState<boolean>(false);
  const [isResultModalOpen, setIsResultModalOpen] = useState<boolean>(false);

  // Poll sync and ground link status
  useEffect(() => {
    const fetchSync = async () => {
      try {
        const res = await fetch(`${BACKEND_BASE}/api/sync/status`);
        if (res.ok) {
          const data = await res.json();
          setIsGroundOnline(data.ground_link_online ?? true);
          setPendingSync(data.pending_count ?? 0);
        }
      } catch (err) {
        // Backend offline or unreachable
      }
    };
    fetchSync();
    const interval = setInterval(fetchSync, 2000);
    return () => clearInterval(interval);
  }, []);

  const toggleGroundLink = async () => {
    const nextState = !isGroundOnline;
    setIsGroundOnline(nextState);
    try {
      await fetch(`${BACKEND_BASE}/api/sync/ground_link`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ online: nextState }),
      });
    } catch (err) {
      console.warn("Failed to toggle ground link:", err);
    }
  };

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
                onClick={() => navigate('/experiments/EXP-MICROBE/log')}
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
              className="flex items-center gap-2 text-space-400 hover:text-status-critical transition-colors uppercase font-bold text-xs tracking-wider ml-4 px-3 py-1.5 rounded-md hover:bg-red-50 cursor-pointer"
            >
              <ArrowLeft size={16} />
              EXIT EXPERIMENT
            </button>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* RUN ORBITA DEMO Trigger Button */}
          <button
            type="button"
            onClick={() => setIsDemoModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-accent-cyan hover:bg-accent-cyan/80 text-space-950 text-xs font-mono font-bold tracking-wider shadow-sm transition-all cursor-pointer"
            title="Launch 18-step deterministic ORBITA demo"
          >
            <Play size={13} />
            <span>RUN ORBITA DEMO</span>
          </button>

          {/* Ground Link Status & Simulation Toggle */}
          <button
            type="button"
            onClick={toggleGroundLink}
            className={clsx(
              "flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-mono font-bold border transition-colors cursor-pointer",
              isGroundOnline 
                ? "bg-emerald-950/60 border-emerald-700/80 text-emerald-300 hover:bg-emerald-900/60" 
                : "bg-red-950/70 border-red-700/80 text-red-300 hover:bg-red-900/70"
            )}
            title="Click to toggle Ground Link online/offline simulation"
          >
            {isGroundOnline ? <Wifi size={14} className="text-emerald-400" /> : <WifiOff size={14} className="text-red-400 animate-pulse" />}
            <span>GROUND: {isGroundOnline ? "ONLINE" : "OFFLINE"}</span>
            {pendingSync > 0 && (
              <span className="px-1.5 py-0.2 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40">
                {pendingSync} PENDING
              </span>
            )}
          </button>

          {/* Status Indicators */}
          <div className="flex items-center gap-2 text-xs font-mono font-semibold tracking-wider">
            <StatusIndicator label="AI" active={true} color="bg-status-success" />
            <StatusIndicator label="REC" active={isLive} color={isLive ? "bg-status-warning animate-pulse" : "bg-space-400"} />
            <StatusIndicator label="STREAM" active={true} color="bg-status-success" />
          </div>

          <div className="w-px h-5 bg-space-600 mx-0.5"></div>
          
          {/* Clock */}
          <div className="font-mono text-space-400 text-xs bg-space-700 px-2 py-1 rounded border border-space-600">
            {new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })} UTC
          </div>

          <div className="w-px h-5 bg-space-600 mx-0.5"></div>

          {/* Google Apps Icon Launcher */}
          <GoogleAppsButton />
        </div>
      </header>

      {/* Main Content Area */}
      <main className={clsx("flex-1 relative bg-space-900 min-h-0", isLive ? "overflow-hidden" : "overflow-y-auto")}>
        <Outlet />
      </main>

      {/* Global Modals */}
      <OrbitaDemoModal
        isOpen={isDemoModalOpen}
        onClose={() => setIsDemoModalOpen(false)}
        onViewResults={() => setIsResultModalOpen(true)}
      />
      <StructuredResultModal
        isOpen={isResultModalOpen}
        onClose={() => setIsResultModalOpen(false)}
      />
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
