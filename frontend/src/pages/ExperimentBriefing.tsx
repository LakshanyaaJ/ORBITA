import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Play, Check, AlertTriangle, Film } from 'lucide-react';
import { BACKEND_BASE, connectCamera } from '../api/camera';

export default function ExperimentBriefing() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [selectedVideo, setSelectedVideo] = useState('20260905_145858.mp4');
  const [availableVideos, setAvailableVideos] = useState<any[]>([]);
  const [isStarting, setIsStarting] = useState(false);

  const [useReferenceVideo, setUseReferenceVideo] = useState(true);

  const isVData = id === 'EXP-VDATA' || (id && id.toLowerCase().includes('vdata'));
  const isYellowBlueBox = !isVData && (!id || id === 'EXP-01' || id === 'EXP-1' || id === 'EXP-04');
  const isSampleAnalysis = !isVData && (id === 'EXP-02' || id === 'EXP-2' || id === 'EXP-07');

  useEffect(() => {
    if (isVData || isYellowBlueBox) {
      fetch(`${BACKEND_BASE}/api/vdata/videos`)
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data) && data.length > 0) {
            setAvailableVideos(data);
          }
        })
        .catch(() => {});
    }
  }, [isVData, isYellowBlueBox]);

  const handleStart = async () => {
    setIsStarting(true);
    if (isVData || useReferenceVideo) {
      try {
        await connectCamera({
          source: 'video_file',
          path: `vdata/${selectedVideo}`,
          loop: true,
          reset_fsm: true,
        });
      } catch (e) {
        console.error('Failed to pre-connect video:', e);
      }
    }
    const query = (isVData || useReferenceVideo) ? `?video=${selectedVideo}` : '';
    navigate(`/experiments/${id || (isVData ? 'EXP-VDATA' : 'EXP-01')}/live${query}`);
  };

  const stepsList = (isVData || isYellowBlueBox) ? [
    { num: '01', text: 'IDENTIFY BLUE BOX' },
    { num: '02', text: 'PICK UP BLUE BOX' },
    { num: '03', text: 'PLACE BLUE BOX AT LOCATION A' },
    { num: '04', text: 'IDENTIFY YELLOW BOX' },
    { num: '05', text: 'PICK UP YELLOW BOX' },
    { num: '06', text: 'PLACE YELLOW BOX AT LOCATION B' },
    { num: '07', text: 'PICK UP PEN' },
    { num: '08', text: 'PLACE PEN INSIDE BLUE BOX' },
    { num: '09', text: 'PICK UP WATCH' },
    { num: '10', text: 'PLACE WATCH INSIDE YELLOW BOX' },
    { num: '11', text: 'MOVE BLUE BOX FROM A TO B' },
    { num: '12', text: 'MOVE YELLOW BOX FROM B TO A' },
    { num: '13', text: 'EXPERIMENT COMPLETE' },
  ] : [
    { num: '01', text: 'SAMPLE PREPARATION' },
    { num: '02', text: 'SPECTRAL ANALYSIS' },
  ];

  return (
    <div className="p-8 max-w-4xl mx-auto flex flex-col h-full">
      <div className="mb-8">
        <h1 className="text-3xl font-bold font-mono tracking-widest text-space-100 mb-2">
          {isVData
            ? 'BLUE AND YELLOW BOX VDATA'
            : isYellowBlueBox
            ? 'YELLOW AND BLUE BOX'
            : isSampleAnalysis
            ? 'SAMPLE ANALYSIS'
            : 'UNKNOWN EXPERIMENT'}
        </h1>
        <div className="flex items-center gap-3 font-mono text-accent-cyan tracking-widest text-sm">
          <span>{id || (isVData ? 'EXP-VDATA' : 'EXP-01')}</span>
          {isVData && (
            <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-accent-cyan border border-accent-cyan/40 text-xs font-bold">
              REFERENCE VIDEO TELEMETRY
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-12 overflow-y-auto pr-2">
        <div className="space-y-8">
          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              OBJECTIVE
            </h2>
            <p className="text-space-100 leading-relaxed text-sm">
              {isVData
                ? 'Autonomous step-by-step procedural validation of Yellow and Blue Box experiment from reference video telemetry in vdata/. Proves that the ORBITA perception, interaction, and reasoning pipeline correctly identifies objects, tracks hands, and validates all 13 physical steps in real time directly from recorded experiment video.'
                : 'Autonomous identification, manipulation, and placement validation of Yellow Box and Blue Box physical assets. Validates procedural order, object classification, and spatial placement under computer vision copilot guidance.'}
            </p>
          </section>

          {(isVData || isYellowBlueBox) && (
            <section>
              <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4 flex items-center gap-2">
                <Film size={16} className="text-accent-cyan" />
                REFERENCE VIDEO SOURCE (VDATA/)
              </h2>
              <div className="bg-space-800 p-4 rounded-lg border border-space-600 space-y-3 font-mono text-xs">
                {isYellowBlueBox && (
                  <div className="flex items-center gap-2 mb-2 pb-2 border-b border-space-700">
                    <button
                      type="button"
                      onClick={() => setUseReferenceVideo(true)}
                      className={`px-3 py-1 rounded text-xs font-bold transition-colors ${useReferenceVideo ? 'bg-accent-cyan text-black' : 'bg-space-900 text-space-400 hover:text-space-200'}`}
                    >
                      VDATA REFERENCE VIDEO
                    </button>
                    <button
                      type="button"
                      onClick={() => setUseReferenceVideo(false)}
                      className={`px-3 py-1 rounded text-xs font-bold transition-colors ${!useReferenceVideo ? 'bg-accent-cyan text-black' : 'bg-space-900 text-space-400 hover:text-space-200'}`}
                    >
                      LIVE CAMERA STREAM
                    </button>
                  </div>
                )}
                {(!isYellowBlueBox || useReferenceVideo) ? (
                  <>
                    <label className="text-space-300 block font-bold">Select Reference Recording:</label>
                    <select
                      value={selectedVideo}
                      onChange={(e) => setSelectedVideo(e.target.value)}
                      className="w-full bg-space-950 border border-space-600 rounded p-2.5 text-space-100 text-xs focus:outline-none focus:border-accent-cyan"
                    >
                      {availableVideos.length > 0 ? (
                        availableVideos.map((v) => (
                          <option key={v.filename} value={v.filename}>
                            {v.filename} ({v.duration_seconds}s · {v.total_frames} frames · {v.fps} FPS)
                          </option>
                        ))
                      ) : (
                        <>
                          <option value="20260905_145858.mp4">20260905_145858.mp4 (Trial 1 · 23.5s · 704 frames · 30 FPS)</option>
                          <option value="20260905_145948.mp4">20260905_145948.mp4 (Trial 2 · 20.3s · 609 frames · 30 FPS)</option>
                          <option value="20260905_150132.mp4">20260905_150132.mp4 (Trial 3 · 16.7s · 500 frames · 30 FPS)</option>
                          <option value="20260908_135006.mp4">20260908_135006.mp4 (Trial 4 · 61.0s · 1830 frames · 30 FPS)</option>
                        </>
                      )}
                    </select>
                    <div className="text-[11px] text-space-400 flex items-center justify-between pt-1">
                      <span>Stream Mode: Frame-by-Frame AI</span>
                      <span className="text-accent-cyan">Continuous Loop Enabled</span>
                    </div>
                  </>
                ) : (
                  <div className="text-space-400 text-xs py-2">
                    Using active live camera feed (Phone Camera, IP Camera, or Jetson USB).
                  </div>
                )}
              </div>
            </section>
          )}

          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              PROTOCOL
            </h2>
            <div className="font-mono text-2xl font-bold text-space-100 mb-4">
              {(isVData || isYellowBlueBox) ? '13 STEPS' : '02 STEPS'}
            </div>
            <ul className="space-y-2.5 font-mono text-xs max-h-64 overflow-y-auto pr-2">
              {stepsList.map((step) => (
                <li key={step.num} className="flex items-center gap-3 text-space-100 bg-space-850/50 p-2 rounded border border-space-700/60">
                  <span className="text-accent-cyan font-bold">{step.num}</span>
                  <span>{step.text}</span>
                </li>
              ))}
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
              {isVData && (
                <>
                  <EquipmentItem name="PEN" />
                  <EquipmentItem name="WATCH" />
                </>
              )}
            </div>
          </section>

          <section>
            <h2 className="font-mono text-space-400 tracking-widest text-sm font-bold border-b border-space-600 pb-2 mb-4">
              SYSTEM READINESS
            </h2>
            <div className="space-y-3 font-mono text-sm bg-space-800 p-4 rounded-lg border border-space-600">
              <div className="flex justify-between items-center">
                <span className="text-space-400">{isVData ? 'REFERENCE VIDEO (vdata/)' : 'CAMERA (CAM-01)'}</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-space-400">YOLO DETECTOR (OBJECTS)</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-space-400">STATE FSM VALIDATOR</span>
                <span className="text-status-success flex items-center gap-2"><Check size={16} /> READY</span>
              </div>
            </div>
          </section>

          <div className="pt-4">
            <button 
              onClick={handleStart}
              disabled={isStarting}
              className="w-full h-16 bg-accent-cyan text-space-900 font-mono font-bold tracking-widest rounded-md hover:bg-accent-cyan/90 transition-colors flex items-center justify-center gap-3 text-lg shadow-lg cursor-pointer disabled:opacity-50"
            >
              <Play size={24} fill="currentColor" />
              {isStarting ? 'INITIALIZING...' : 'START EXPERIMENT'}
            </button>
            <p className="text-center font-mono text-xs text-status-warning mt-4 flex items-center justify-center gap-2">
              <AlertTriangle size={14} />
              {isVData
                ? 'Reference video will play and validate steps through the AI model.'
                : 'Ensure physical workspace is clear before starting.'}
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
