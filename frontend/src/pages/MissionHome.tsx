import { useNavigate } from 'react-router-dom';
import { Activity, CheckCircle, Clock } from 'lucide-react';
import clsx from 'clsx';

export default function MissionHome() {
  const navigate = useNavigate();

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center gap-4 mb-12">
        <div className="w-12 h-12 rounded-full bg-accent-cyan/10 border border-accent-cyan flex items-center justify-center text-accent-cyan">
          <Activity size={24} />
        </div>
        <div>
          <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100">SYSTEM READY</h1>
          <p className="text-space-400 font-mono text-sm tracking-wider">ASTRONAUT AUTHORIZED • ORBITA ACTIVE</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Assigned Experiments */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2">ASSIGNED EXPERIMENTS</h2>
          
          <div className="space-y-4">
            <ExperimentCard 
              id="EXP-04" 
              name="SEED GERMINATION" 
              steps={7} 
              status="READY"
              onClick={() => navigate('/experiments/EXP-04/briefing')}
            />
            <ExperimentCard 
              id="EXP-07" 
              name="SAMPLE ANALYSIS" 
              steps={5} 
              status="PENDING"
              onClick={() => {}}
            />
          </div>
          
          <button 
            onClick={() => navigate('/experiments')}
            className="mt-4 text-accent-cyan hover:text-space-100 font-mono text-sm tracking-widest transition-colors flex items-center gap-2"
          >
            VIEW ALL EXPERIMENTS &rarr;
          </button>
        </div>

        {/* Right Column: Operational Readiness */}
        <div className="space-y-6">
          <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2">OPERATIONAL READINESS</h2>
          
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 space-y-4 font-mono text-sm">
            <ReadinessRow label="CAMERA" status="READY" />
            <ReadinessRow label="STORAGE" status="READY" />
            <ReadinessRow label="RECORDING" status="READY" />
            <ReadinessRow label="AI ENGINE" status="READY" />
          </div>

          <div className="mt-8 border border-space-600 rounded-lg p-6 flex flex-col items-center justify-center opacity-30 pointer-events-none">
            {/* Subtle aerospace graphic */}
            <svg viewBox="0 0 100 100" className="w-24 h-24 stroke-space-400 fill-none" strokeWidth="1">
              <rect x="20" y="20" width="60" height="60" rx="5" />
              <circle cx="50" cy="50" r="20" />
              <line x1="50" y1="10" x2="50" y2="20" />
              <line x1="50" y1="80" x2="50" y2="90" />
              <line x1="10" y1="50" x2="20" y2="50" />
              <line x1="80" y1="50" x2="90" y2="50" />
            </svg>
            <span className="mt-4 font-mono text-xs tracking-widest">PAYLOAD SYS</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ExperimentCard({ id, name, steps, status, onClick }: { id: string, name: string, steps: number, status: string, onClick: () => void }) {
  const isReady = status === 'READY';
  
  return (
    <div 
      onClick={isReady ? onClick : undefined}
      className={clsx(
        "bg-space-800 border rounded-lg p-6 flex items-center justify-between transition-all",
        isReady 
          ? "border-space-600 hover:border-accent-cyan cursor-pointer group" 
          : "border-space-600/50 opacity-60 cursor-not-allowed"
      )}
    >
      <div className="flex gap-6 items-center">
        <div className={clsx(
          "font-mono text-xl font-bold tracking-widest",
          isReady ? "text-accent-cyan group-hover:text-space-100 transition-colors" : "text-space-400"
        )}>
          {id}
        </div>
        <div>
          <h3 className="font-bold text-lg text-space-100">{name}</h3>
          <p className="font-mono text-space-400 text-sm mt-1">{steps < 10 ? `0${steps}` : steps} STEPS</p>
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        {isReady ? <CheckCircle size={18} className="text-status-success" /> : <Clock size={18} className="text-space-400" />}
        <span className={clsx(
          "font-mono text-sm tracking-widest font-bold",
          isReady ? "text-status-success" : "text-space-400"
        )}>
          {status}
        </span>
      </div>
    </div>
  );
}

function ReadinessRow({ label, status }: { label: string, status: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-space-400">{label}</span>
      <span className="text-status-success font-bold">{status}</span>
    </div>
  );
}
