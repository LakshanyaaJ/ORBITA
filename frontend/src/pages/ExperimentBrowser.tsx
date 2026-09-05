import { useNavigate } from 'react-router-dom';

export default function ExperimentBrowser() {
  const navigate = useNavigate();

  return (
    <div className="p-8 max-w-6xl mx-auto h-full flex flex-col">
      <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100 mb-8 border-b border-space-600 pb-4">
        EXPERIMENT CATALOG
      </h1>

      <div className="flex-1 overflow-y-auto pr-4 space-y-4">
        {/* EXP-04 */}
        <div className="bg-space-800 border border-space-600 hover:border-accent-cyan rounded-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 transition-colors">
          <div className="flex-1">
            <div className="flex items-center gap-4 mb-2">
              <span className="font-mono text-xl font-bold tracking-widest text-accent-cyan">EXP-04</span>
              <span className="bg-status-success/20 text-status-success px-2 py-1 rounded font-mono text-xs font-bold tracking-widest">READY</span>
            </div>
            <h3 className="font-bold text-2xl text-space-100 mb-2">SEED GERMINATION</h3>
            <p className="text-space-400 text-sm max-w-2xl">
              Study seed growth under microgravity conditions. Validates the viability of on-board agriculture using standard nutrient gel containers.
            </p>
          </div>
          
          <div className="flex items-center gap-8 border-l border-space-600 pl-8">
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">PROTOCOL</div>
              <div className="text-space-100 font-bold">07 STEPS</div>
            </div>
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">CAMERA</div>
              <div className="text-status-success font-bold">READY</div>
            </div>
          </div>

          <div className="flex flex-col gap-3 ml-8">
            <button 
              onClick={() => navigate('/experiments/EXP-04/briefing')}
              className="px-6 py-3 bg-accent-cyan text-space-900 font-mono font-bold tracking-widest rounded-md hover:bg-accent-cyan/90 transition-colors"
            >
              START
            </button>
            <button className="px-6 py-3 bg-space-700 text-space-100 font-mono font-bold tracking-widest rounded-md hover:bg-space-600 transition-colors">
              VIEW
            </button>
          </div>
        </div>

        {/* EXP-07 */}
        <div className="bg-space-800 border border-space-600/50 rounded-lg p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 opacity-75">
          <div className="flex-1">
            <div className="flex items-center gap-4 mb-2">
              <span className="font-mono text-xl font-bold tracking-widest text-space-400">EXP-07</span>
              <span className="bg-space-600 text-space-400 px-2 py-1 rounded font-mono text-xs font-bold tracking-widest">PENDING</span>
            </div>
            <h3 className="font-bold text-2xl text-space-100 mb-2">SAMPLE ANALYSIS</h3>
            <p className="text-space-400 text-sm max-w-2xl">
              Routine chemical analysis of atmospheric samples using standard spectroscopy interfaces.
            </p>
          </div>
          
          <div className="flex items-center gap-8 border-l border-space-600 pl-8">
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">PROTOCOL</div>
              <div className="text-space-100 font-bold">05 STEPS</div>
            </div>
            <div className="text-right font-mono">
              <div className="text-space-400 text-xs tracking-widest">CAMERA</div>
              <div className="text-space-400 font-bold">OFFLINE</div>
            </div>
          </div>

          <div className="flex flex-col gap-3 ml-8">
            <button disabled className="px-6 py-3 bg-space-700/50 text-space-400 font-mono font-bold tracking-widest rounded-md cursor-not-allowed">
              START
            </button>
            <button className="px-6 py-3 bg-space-700 text-space-100 font-mono font-bold tracking-widest rounded-md hover:bg-space-600 transition-colors">
              VIEW
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
