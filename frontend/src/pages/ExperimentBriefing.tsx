import { useNavigate, useParams } from 'react-router-dom';
import { Play, Check, AlertTriangle } from 'lucide-react';


export default function ExperimentBriefing() {
  const navigate = useNavigate();
  const { id } = useParams();

  // EXP-01 (Yellow & Blue Box) vs EXP-02 (Sample Analysis) briefing
  const isYellowBlueBox = !id || id === 'EXP-01' || id === 'EXP-1' || id === 'EXP-04';
  const isSampleAnalysis = id === 'EXP-02' || id === 'EXP-2' || id === 'EXP-07';

  return (
    <div className="p-8 max-w-4xl mx-auto flex flex-col h-full">
      <div className="mb-8">
        <h1 className="text-3xl font-bold font-mono tracking-widest text-space-100 mb-2">
          {isYellowBlueBox ? 'YELLOW AND BLUE BOX' : (isSampleAnalysis ? 'SAMPLE ANALYSIS' : 'UNKNOWN EXPERIMENT')}
        </h1>
        <div className="font-mono text-accent-cyan tracking-widest">{id || 'EXP-01'}</div>
      </div>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-12 overflow-y-auto">
        <div className="space-y-8">
          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              OBJECTIVE
            </h2>
            <p className="text-space-100 leading-relaxed text-sm">
              Autonomous identification, manipulation, and placement validation of Yellow Box and Blue Box physical assets. Validates procedural order, object classification, and spatial placement under computer vision copilot guidance.
            </p>
          </section>

          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              PROTOCOL
            </h2>
            <div className="font-mono text-2xl font-bold text-space-100 mb-4">
              02 STEPS
            </div>
            <ul className="space-y-3 font-mono text-sm">
              <li className="flex items-center gap-3 text-space-100">
                <span className="text-space-400">01</span> IDENTIFY BLUE BOX
              </li>
              <li className="flex items-center gap-3 text-space-100">
                <span className="text-space-400">02</span> PICK UP / IDENTIFY YELLOW BOX
              </li>
              <li className="flex items-center gap-3 text-space-400">...</li>
            </ul>
          </section>
        </div>

        <div className="space-y-8">
          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              REQUIRED EQUIPMENT
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <EquipmentItem name="BLUE BOX" />
              <EquipmentItem name="YELLOW BOX" />
              <EquipmentItem name="LOCATION A" />
              <EquipmentItem name="LOCATION B" />
            </div>
          </section>

          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              SYSTEM READINESS
            </h2>
            <div className="space-y-3 font-mono text-sm bg-space-800 p-4 rounded-lg border border-space-600">
              <div className="flex justify-between items-center">
                <span className="text-space-400">CAMERA (CAM-01)</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-space-400">RECORDING</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-space-400">AI ENGINE</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
            </div>
          </section>

          <div className="pt-8">
            <button 
              onClick={() => navigate(`/experiments/${id}/live`)}
              className="w-full h-16 bg-accent-cyan text-space-900 font-mono font-bold tracking-widest rounded-md hover:bg-accent-cyan/90 transition-colors flex items-center justify-center gap-3 text-lg"
            >
              <Play size={24} fill="currentColor" />
              START EXPERIMENT
            </button>
            <p className="text-center font-mono text-xs text-status-warning mt-4 flex items-center justify-center gap-2">
              <AlertTriangle size={14} /> Ensure physical workspace is clear before starting.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function EquipmentItem({ name }: { name: string }) {
  return (
    <div className="bg-space-800 border border-space-600 p-3 rounded text-center font-mono text-xs tracking-widest text-space-100">
      {name}
    </div>
  );
}
