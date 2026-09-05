import { useNavigate } from 'react-router-dom';
import { 
  CheckCircle, 
  Clock, 
  Cpu, 
  Camera, 
  HardDrive, 
  Video, 
  Radio, 
  ArrowRight, 
  ExternalLink 
} from 'lucide-react';
import clsx from 'clsx';

export default function MissionHome() {
  const navigate = useNavigate();

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Mission Context Header matching Mission Control */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-space-600">
        <div>
          <div className="font-mono text-[10px] font-bold tracking-[0.16em] text-space-400 uppercase mb-1 flex items-center gap-2">
            <span>MISSION ALPHA-01</span>
            <span className="text-space-600">/</span>
            <span>ACTIVE OPERATION</span>
            <span className="inline-flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-full bg-[#eafaf1] text-[#2e7d58] text-[10px] font-mono border border-[#c1e8d4]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2e7d58] animate-pulse" />
              SYSTEM ONLINE
            </span>
          </div>
          <h1 className="text-3xl font-bold font-['Space_Grotesk'] tracking-tight text-space-100">
            Mission Overview
          </h1>
          <p className="text-space-400 text-sm mt-1">
            Real-time astronaut activity and procedure verification.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-space-800 border border-space-600 rounded-md text-xs font-mono text-space-400 shadow-xs">
            <span className="w-2 h-2 rounded-full bg-status-success"></span>
            <span>Last sync: {new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
          </div>
          <button
            onClick={() => navigate('/mission-control')}
            className="flex items-center gap-2 px-3.5 py-1.5 bg-space-800 hover:bg-space-700 text-accent-cyan border border-space-600 hover:border-accent-cyan rounded-md text-xs font-mono font-bold tracking-wider transition-all shadow-xs"
          >
            <Radio size={14} />
            <span>MISSION CONTROL VIEW</span>
            <ExternalLink size={12} />
          </button>
        </div>
      </div>

      {/* 4-Metric KPI Summary Row matching Mission Control */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard 
          label="CURRENT ACTIVITY" 
          value="Tool Pickup" 
          sub="ASTRONAUT-01" 
          tag="ACT-04" 
        />
        <MetricCard 
          label="AI CONFIDENCE" 
          value="97.4%" 
          sub="Stable recognition" 
          tone="primary" 
        />
        <MetricCard 
          label="PROCEDURE" 
          value="04 / 08" 
          sub="Emergency equipment" 
        />
        <MetricCard 
          label="STATUS" 
          value="Verified" 
          sub="No safety interlock" 
          tone="success" 
        />
      </div>

      {/* Main Two-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Assigned Experiments */}
        <div className="lg:col-span-2 space-y-5">
          <div className="flex items-center justify-between pb-2 border-b border-space-600">
            <h2 className="font-mono text-xs font-bold tracking-[0.14em] text-space-400 uppercase">
              ASSIGNED EXPERIMENTS
            </h2>
            <span className="font-mono text-[11px] text-space-400">
              2 PROTOCOLS LOADED
            </span>
          </div>
          
          <div className="space-y-4">
            <ExperimentCard 
              id="EXP-04" 
              name="SEED GERMINATION" 
              payload="VEG-03 PAYLOAD"
              steps={7} 
              status="READY"
              description="Biological growth substrate hydration, canister seal verification, and lighting cycle activation."
              onClick={() => navigate('/experiments/EXP-04/briefing')}
            />
            <ExperimentCard 
              id="EXP-07" 
              name="SAMPLE ANALYSIS" 
              payload="BIO-CHAMBER 02"
              steps={5} 
              status="PENDING"
              description="Centrifuge extraction, reagent vial placement, and spectral optical validation."
              onClick={() => {}}
            />
          </div>
          
          <div className="pt-2 flex items-center justify-between">
            <button 
              onClick={() => navigate('/experiments')}
              className="text-accent-cyan hover:text-accent-cyan/80 font-mono text-xs font-bold tracking-widest transition-colors flex items-center gap-2 group cursor-pointer"
            >
              <span>VIEW ALL EXPERIMENTS</span>
              <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
            </button>
            <span className="font-mono text-xs text-space-400">
              ORBITA V1.0 · AUTONOMOUS COPILOT
            </span>
          </div>
        </div>

        {/* Right Column: Operational Readiness */}
        <div className="space-y-5">
          <div className="flex items-center justify-between pb-2 border-b border-space-600">
            <h2 className="font-mono text-xs font-bold tracking-[0.14em] text-space-400 uppercase">
              OPERATIONAL READINESS
            </h2>
            <span className="font-mono text-[11px] text-[#2e7d58] font-bold">
              ALL SYSTEMS GO
            </span>
          </div>
          
          <div className="bg-space-800 border border-space-600 rounded-lg p-5 space-y-3.5 font-mono text-xs shadow-xs">
            <ReadinessRow icon={Camera} label="CAMERA SUBSYSTEM" status="READY" detail="30 FPS · Low Latency" />
            <ReadinessRow icon={HardDrive} label="TELEMETRY STORAGE" status="READY" detail="SQLite · orbita.db" />
            <ReadinessRow icon={Video} label="REALTIME STREAMING" status="READY" detail="MJPEG · WebSockets" />
            <ReadinessRow icon={Cpu} label="EDGE AI ENGINE" status="READY" detail="YOLOv8 + Spatial HAR" />
          </div>

          {/* Aerospace Blueprint Graphic Widget */}
          <div className="bg-space-800 border border-space-600 rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden shadow-xs">
            <div className="absolute inset-0 bg-[radial-gradient(#2f6f9f_1px,transparent_1px)] [background-size:16px_16px] opacity-10 pointer-events-none" />
            
            <svg viewBox="0 0 100 100" className="w-24 h-24 stroke-accent-cyan fill-none relative z-10" strokeWidth="1.2">
              <rect x="20" y="20" width="60" height="60" rx="6" strokeDasharray="3 3" className="stroke-space-400" />
              <circle cx="50" cy="50" r="22" strokeWidth="1.5" className="stroke-accent-cyan" />
              <circle cx="50" cy="50" r="8" className="fill-accent-cyan/10 stroke-accent-cyan" />
              <line x1="50" y1="8" x2="50" y2="24" className="stroke-accent-cyan" />
              <line x1="50" y1="76" x2="50" y2="92" className="stroke-accent-cyan" />
              <line x1="8" y1="50" x2="24" y2="50" className="stroke-accent-cyan" />
              <line x1="76" y1="50" x2="92" y2="50" className="stroke-accent-cyan" />
            </svg>
            
            <div className="mt-3 text-center relative z-10">
              <span className="font-mono text-xs tracking-[0.18em] font-bold text-space-100 block">
                PAYLOAD SYS · EVA-01
              </span>
              <span className="font-mono text-[10px] text-space-400 mt-0.5 block tracking-wider">
                JETSON ORIN NANO EDGE TELEMETRY
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, sub, tag, tone }: { label: string, value: string, sub: string, tag?: string, tone?: 'primary' | 'success' }) {
  return (
    <div className="bg-space-800 border border-space-600 rounded-lg p-4 shadow-xs">
      <div className="flex items-center justify-between text-[10px] font-mono font-bold tracking-[0.14em] text-space-400 uppercase">
        <span>{label}</span>
        {tag && <span className="text-[9px] px-1.5 py-0.5 rounded bg-space-700 text-space-400">{tag}</span>}
      </div>
      <div className={clsx(
        "font-['Space_Grotesk'] text-2xl font-bold mt-2 tracking-tight",
        tone === 'primary' ? "text-accent-cyan" : tone === 'success' ? "text-status-success" : "text-space-100"
      )}>
        {value}
      </div>
      <div className="text-xs text-space-400 mt-1 font-sans">
        {sub}
      </div>
    </div>
  );
}

function ExperimentCard({ 
  id, 
  name, 
  payload, 
  steps, 
  status, 
  description, 
  onClick 
}: { 
  id: string, 
  name: string, 
  payload: string, 
  steps: number, 
  status: string, 
  description: string, 
  onClick: () => void 
}) {
  const isReady = status === 'READY';
  
  return (
    <div 
      onClick={isReady ? onClick : undefined}
      className={clsx(
        "bg-space-800 border rounded-lg p-5 transition-all shadow-xs",
        isReady 
          ? "border-space-600 hover:border-accent-cyan hover:shadow-md cursor-pointer group" 
          : "border-space-600/70 opacity-65 cursor-not-allowed"
      )}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-space-600/60">
        <div className="flex items-center gap-3">
          <span className={clsx(
            "font-mono text-xs font-bold tracking-wider px-2.5 py-1 rounded",
            isReady 
              ? "bg-[#eaf2f7] text-[#2f6f9f] group-hover:bg-[#2f6f9f] group-hover:text-white transition-colors" 
              : "bg-space-700 text-space-400"
          )}>
            {id}
          </span>
          <div>
            <h3 className="font-['Space_Grotesk'] font-bold text-base text-space-100 group-hover:text-accent-cyan transition-colors">
              {name}
            </h3>
            <span className="font-mono text-[11px] text-space-400 tracking-wider">
              {payload} · {steps < 10 ? `0${steps}` : steps} STEPS
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {isReady ? (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#eafaf1] text-[#2e7d58] border border-[#c1e8d4] font-mono text-xs font-bold tracking-wider">
              <CheckCircle size={14} />
              READY
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#fef8eb] text-[#a06822] border border-[#fae4ba] font-mono text-xs font-bold tracking-wider">
              <Clock size={14} />
              PENDING
            </span>
          )}
        </div>
      </div>

      <p className="text-xs text-space-400 mt-3 line-clamp-2 font-sans">
        {description}
      </p>

      {isReady && (
        <div className="mt-4 pt-3 border-t border-space-600/40 flex items-center justify-between text-xs font-mono">
          <span className="text-space-400">Next Action: Step 01 Hydration</span>
          <span className="text-accent-cyan font-bold group-hover:translate-x-0.5 transition-transform flex items-center gap-1">
            START EXPERIMENT &rarr;
          </span>
        </div>
      )}
    </div>
  );
}

function ReadinessRow({ icon: Icon, label, status, detail }: { icon: any, label: string, status: string, detail?: string }) {
  return (
    <div className="flex justify-between items-center py-1">
      <div className="flex items-center gap-2.5">
        <Icon size={15} className="text-space-400" />
        <div>
          <span className="text-space-100 font-medium block">{label}</span>
          {detail && <span className="text-[10px] text-space-400 block -mt-0.5">{detail}</span>}
        </div>
      </div>
      <span className="inline-flex items-center gap-1 text-[#2e7d58] font-bold text-[11px] bg-[#eafaf1] px-2 py-0.5 rounded border border-[#c1e8d4]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#2e7d58]" />
        {status}
      </span>
    </div>
  );
}
