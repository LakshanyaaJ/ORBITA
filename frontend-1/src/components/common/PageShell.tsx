import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Activity, Cpu, Settings, ArrowLeft } from 'lucide-react';
import clsx from 'clsx';

export default function PageShell() {
  const navigate = useNavigate();
  const location = useLocation();

  const isLive = location.pathname.includes('/live');

  return (
    <div className="min-h-screen bg-space-900 text-space-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="h-16 border-b border-space-600 flex items-center justify-between px-6 shrink-0 bg-space-800">
        <div className="flex items-center gap-6">
          <div className="font-mono text-xl font-bold tracking-widest text-space-100">
            ORBITA
          </div>
          
          {!isLive && (
            <nav className="flex items-center gap-2 ml-8">
              <NavButton 
                active={location.pathname === '/mission' || location.pathname.includes('/experiments')} 
                onClick={() => navigate('/mission')}
                icon={<Activity size={18} />}
                label="MISSION"
              />
              <NavButton 
                active={location.pathname.includes('/log')} 
                onClick={() => navigate('/experiments/EXP-04/log')} // Placeholder
                icon={<Settings size={18} />}
                label="LOG"
              />
              <NavButton 
                active={location.pathname === '/system'} 
                onClick={() => navigate('/system')}
                icon={<Cpu size={18} />}
                label="SYSTEM"
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
              className="flex items-center gap-2 text-space-400 hover:text-status-critical transition-colors uppercase font-bold text-sm tracking-widest ml-4"
            >
              <ArrowLeft size={18} />
              EXIT EXPERIMENT
            </button>
          )}
        </div>

        <div className="flex items-center gap-6">
          {/* Status Indicators */}
          <div className="flex items-center gap-4 text-xs font-mono font-bold tracking-widest">
            <StatusIndicator label="AI" active={true} color="bg-status-success" />
            <StatusIndicator label="REC" active={isLive} color={isLive ? "bg-status-warning" : "bg-space-600"} />
            <StatusIndicator label="STREAM" active={true} color="bg-status-success" />
          </div>

          <div className="w-px h-6 bg-space-600 mx-2"></div>
          
          {/* Clock */}
          <div className="font-mono text-space-400 text-sm">
            {new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden relative">
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
        "flex items-center gap-2 px-4 py-2 rounded-md font-bold tracking-wider text-sm transition-colors",
        active 
          ? "bg-space-700 text-accent-cyan" 
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
    <div className={clsx("flex items-center gap-2", active ? "text-space-100" : "text-space-400")}>
      <div className={clsx("w-2.5 h-2.5 rounded-full", color)}></div>
      {label}
    </div>
  );
}
