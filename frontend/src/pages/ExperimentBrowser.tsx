import { useNavigate } from 'react-router-dom';

export default function ExperimentBrowser() {
  const navigate = useNavigate();

  return (
    <div className="p-8 max-w-6xl mx-auto h-full flex flex-col">
      <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100 mb-8 border-b border-space-600 pb-4">
        EXPERIMENT CATALOG
      </h1>

      <div className="flex-1 overflow-y-auto pr-4 space-y-4">
        {/* EXP-VDATA */}
        <div className="bg-space-800 border border-space-600 hover:border-accent-cyan rounded-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-colors shadow-lg">
          <div className="flex-1">
            <div className="flex items-center gap-4 mb-2">
              <span className="font-mono text-xl font-bold tracking-widest text-accent-cyan">EXP-VDATA</span>
              <span className="bg-cyan-500/20 text-accent-cyan border border-accent-cyan/40 px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                REFERENCE RUN
              </span>
              <span className="bg-status-success/20 text-status-success px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                READY
              </span>
            </div>
            <h3 className="font-bold text-2xl text-space-100 mb-2">BLUE AND YELLOW BOX VDATA</h3>
            <p className="text-space-400 text-sm max-w-2xl leading-relaxed">
              Step-by-step AI copilot validation over real reference video telemetry (vdata/). Plays recorded experiment trials while YOLO object detection, hand tracking, and FSM validate all 13 physical procedure steps in real time.
            </p>
          </div>
          
          <div className="flex items-center gap-8 border-l border-space-600 pl-8">
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">PROTOCOL</div>
              <div className="text-space-100 font-bold">13 STEPS</div>
            </div>
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">SOURCE</div>
              <div className="text-accent-cyan font-bold">vdata/ MP4</div>
            </div>
          </div>

          <div className="flex flex-col gap-3 ml-8">
            <button 
              onClick={() => navigate('/experiments/EXP-VDATA/briefing')}
              className="px-6 py-3 bg-accent-cyan text-space-900 font-mono font-bold tracking-widest rounded-md hover:bg-accent-cyan/90 transition-colors shadow-md"
            >
              START
            </button>
            <button 
              onClick={() => navigate('/experiments/EXP-VDATA/briefing')}
              className="px-6 py-3 bg-space-700 text-space-100 font-mono font-bold tracking-widest rounded-md hover:bg-space-600 transition-colors"
            >
              BRIEFING
            </button>
          </div>
        </div>

        {/* EXP-MICROBE */}
        <div className="bg-space-800 border border-space-600 hover:border-emerald-500/80 rounded-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-colors shadow-lg">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              <span className="font-mono text-xl font-bold tracking-widest text-emerald-400">EXP-MICROBE</span>
              <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                BIOLOGICAL RESEARCH
              </span>
              <span className="bg-purple-500/20 text-purple-300 border border-purple-500/40 px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                ISRO–AXIOM-4 CONTEXT
              </span>
              <span className="bg-status-success/20 text-status-success px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                READY
              </span>
            </div>
            <h3 className="font-bold text-2xl text-space-100 mb-2">MICROBIAL EXPERIMENT IN MICROGRAVITY</h3>
            <p className="text-space-400 text-sm max-w-2xl leading-relaxed">
              Ground-based demonstration of an observation and monitoring workflow inspired by space-related microbial biological research under computer vision copilot guidance.
            </p>
          </div>
          
          <div className="flex items-center gap-8 border-l border-space-600 pl-8">
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">PROTOCOL</div>
              <div className="text-space-100 font-bold">07 STEPS</div>
            </div>
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">MODE</div>
              <div className="text-emerald-400 font-bold">DEMO</div>
            </div>
          </div>

          <div className="flex flex-col gap-3 ml-8">
            <button 
              onClick={() => navigate('/experiments/EXP-MICROBE/briefing')}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 text-space-950 font-mono font-bold tracking-widest rounded-md transition-colors shadow-md"
            >
              START
            </button>
            <button 
              onClick={() => navigate('/experiments/EXP-MICROBE/briefing')}
              className="px-6 py-3 bg-space-700 text-space-100 font-mono font-bold tracking-widest rounded-md hover:bg-space-600 transition-colors"
            >
              BRIEFING
            </button>
          </div>
        </div>

        {/* EXP-02 */}
        <div className="bg-space-800 border border-space-600 hover:border-purple-500/80 rounded-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-colors shadow-lg">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              <span className="font-mono text-xl font-bold tracking-widest text-purple-400">EXP-02</span>
              <span className="bg-purple-500/20 text-purple-300 border border-purple-500/40 px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                SPECTRAL ANALYSIS
              </span>
              <span className="bg-status-success/20 text-status-success px-2 py-0.5 rounded font-mono text-xs font-bold tracking-widest">
                READY
              </span>
            </div>
            <h3 className="font-bold text-2xl text-space-100 mb-2">SAMPLE ANALYSIS</h3>
            <p className="text-space-400 text-sm max-w-2xl leading-relaxed">
              Sample preparation, reagent vial placement, and spectral optical validation under computer vision copilot tracking.
            </p>
          </div>
          
          <div className="flex items-center gap-8 border-l border-space-600 pl-8">
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">PROTOCOL</div>
              <div className="text-space-100 font-bold">02 STEPS</div>
            </div>
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">MODE</div>
              <div className="text-purple-400 font-bold">INTERACTIVE</div>
            </div>
          </div>

          <div className="flex flex-col gap-3 ml-8">
            <button 
              onClick={() => navigate('/experiments/EXP-02/briefing')}
              className="px-6 py-3 bg-purple-500 hover:bg-purple-400 text-space-950 font-mono font-bold tracking-widest rounded-md transition-colors shadow-md"
            >
              START
            </button>
            <button 
              onClick={() => navigate('/experiments/EXP-02/briefing')}
              className="px-6 py-3 bg-space-700 text-space-100 font-mono font-bold tracking-widest rounded-md hover:bg-space-600 transition-colors"
            >
              BRIEFING
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
