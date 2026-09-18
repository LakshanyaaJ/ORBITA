import { useState } from 'react';
import { Activity, Box, Info, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

export interface HMRJoint {
  name: string;
  index: number;
  x: number;
  y: number;
  z: number;
  confidence: number;
}

export interface HMRData {
  mesh_id?: string;
  person_id?: number;
  num_joints?: number;
  joints_3d?: HMRJoint[];
  global_orientation?: [number, number, number];
  camera_translation?: [number, number, number];
  is_fallback?: boolean;
  model_name?: string;
  source_type?: string;
  timestamp?: number;
}

// Standard SMPL 24-joint kinematic bones (parent -> child)
const SMPL_BONES: [number, number][] = [
  [0, 1], [0, 2], [0, 3],       // Pelvis -> L_Hip, R_Hip, Spine1
  [1, 4], [2, 5], [3, 6],       // L_Hip->L_Knee, R_Hip->R_Knee, Spine1->Spine2
  [4, 7], [5, 8], [6, 9],       // L_Knee->L_Ankle, R_Knee->R_Ankle, Spine2->Spine3
  [7, 10], [8, 11], [9, 12],    // L_Ankle->L_Foot, R_Ankle->R_Foot, Spine3->Neck
  [9, 13], [9, 14],             // Spine3 -> L_Collar, R_Collar
  [12, 15],                     // Neck -> Head
  [13, 16], [14, 17],           // Collar -> Shoulder
  [16, 18], [17, 19],           // Shoulder -> Elbow
  [18, 20], [19, 21],           // Elbow -> Wrist
  [20, 22], [21, 23],           // Wrist -> Hand
];

export default function HmrMeshViewer({ hmrData }: { hmrData?: HMRData | null }) {
  const [selectedJoint, setSelectedJoint] = useState<HMRJoint | null>(null);
  const [viewAngle, setViewAngle] = useState<'FRONT' | 'TOP' | 'SIDE'>('FRONT');

  const joints = hmrData?.joints_3d || [];
  const isFallback = hmrData?.is_fallback ?? true;
  const modelName = hmrData?.model_name || "Deterministic 3D Kinematic Lifter";
  const orient = hmrData?.global_orientation || [0.0, 0.0, 0.0];
  const trans = hmrData?.camera_translation || [0.0, 0.0, 2.5];

  // Coordinate projection for SVG 3D viewport (200x200 canvas centered at 100, 100)
  const projectPoint = (j: HMRJoint) => {
    let px = j.x;
    let py = j.y;
    if (viewAngle === 'TOP') {
      px = j.x;
      py = j.z;
    } else if (viewAngle === 'SIDE') {
      px = j.z;
      py = j.y;
    }
    // Scale and center: joints are typically in meters [-0.8, 0.8]
    const svgX = 100 + px * 85;
    const svgY = 100 - py * 85; // invert Y for SVG standard coordinates
    return { x: svgX, y: svgY };
  };

  return (
    <div className="bg-space-950/90 rounded-lg border border-space-700 p-3.5 font-mono text-xs shadow-md space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-space-800">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded bg-purple-950 text-purple-300 border border-purple-800">
            <Box size={14} />
          </div>
          <div>
            <div className="font-bold text-space-100 text-[11px] flex items-center gap-1.5">
              <span>HMR 3D MESH RECOVERY</span>
              <span className="text-[9px] text-purple-400 font-semibold">SMPL-24</span>
            </div>
            <div className="text-[9px] text-space-400">
              Human Mesh Recovery / 4D-Humans Telemetry
            </div>
          </div>
        </div>

        {/* Model & Fallback Badge */}
        <div className="flex items-center gap-2">
          {isFallback ? (
            <span 
              className="px-2 py-0.5 rounded text-[9px] font-bold bg-amber-950/80 text-amber-300 border border-amber-700/80 flex items-center gap-1"
              title="Deterministic Kinematic Lifter active. Neural 4D-Humans weights not bundled."
            >
              <AlertTriangle size={11} className="text-amber-400" />
              <span>HMR_PROTOTYPE_FALLBACK</span>
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-purple-950/80 text-purple-300 border border-purple-700 flex items-center gap-1">
              <Activity size={11} />
              <span>4D_HUMANS_NEURAL</span>
            </span>
          )}
        </div>
      </div>

      {/* Main 3D Skeleton SVG Wireframe & Telemetry Grid */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
        {/* SVG Kinematic Viewport (5 cols) */}
        <div className="md:col-span-5 relative flex flex-col items-center bg-black/60 rounded-lg border border-space-800 p-2">
          <div className="absolute top-2 left-2 flex items-center gap-1 z-10">
            {(['FRONT', 'SIDE', 'TOP'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewAngle(mode)}
                className={clsx(
                  "px-1.5 py-0.2 rounded text-[8.5px] font-bold transition-colors cursor-pointer",
                  viewAngle === mode ? "bg-purple-600 text-white" : "bg-space-800 text-space-400 hover:text-space-200"
                )}
              >
                {mode}
              </button>
            ))}
          </div>

          <svg viewBox="0 0 200 200" className="w-40 h-40 overflow-visible">
            {/* Grid crosshair */}
            <line x1="100" y1="10" x2="100" y2="190" stroke="#1f293d" strokeWidth="0.8" strokeDasharray="2 2" />
            <line x1="10" y1="100" x2="190" y2="100" stroke="#1f293d" strokeWidth="0.8" strokeDasharray="2 2" />

            {/* Bones (Lines connecting joints) */}
            {joints.length >= 24 && SMPL_BONES.map(([pIdx, cIdx], i) => {
              const p = joints[pIdx];
              const c = joints[cIdx];
              if (!p || !c) return null;
              const ptA = projectPoint(p);
              const ptB = projectPoint(c);
              return (
                <line
                  key={i}
                  x1={ptA.x}
                  y1={ptA.y}
                  x2={ptB.x}
                  y2={ptB.y}
                  stroke="#a855f7"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  opacity="0.85"
                />
              );
            })}

            {/* Joints (Circles) */}
            {joints.map((j) => {
              const pt = projectPoint(j);
              const isSelected = selectedJoint?.index === j.index;
              const isHand = j.index === 20 || j.index === 21 || j.index === 22 || j.index === 23;
              const isHead = j.index === 15;

              return (
                <g key={j.index} onClick={() => setSelectedJoint(j)} className="cursor-pointer">
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isSelected ? 4.5 : isHand ? 3.5 : isHead ? 4.0 : 2.5}
                    fill={isSelected ? "#00e5ff" : isHand ? "#34d399" : isHead ? "#fbbf24" : "#c084fc"}
                    stroke="#000"
                    strokeWidth="0.8"
                    className="transition-all hover:scale-125"
                  />
                  {isSelected && (
                    <circle cx={pt.x} cy={pt.y} r="7" fill="none" stroke="#00e5ff" strokeWidth="1" className="animate-ping" />
                  )}
                </g>
              );
            })}
          </svg>

          <div className="text-[9px] text-space-400 mt-1 flex items-center justify-between w-full px-1">
            <span>24 SMPL 3D JOINTS</span>
            <span className="text-purple-400">{joints.length > 0 ? "TRACKING ACTIVE" : "SYNTHETIC IDLE"}</span>
          </div>
        </div>

        {/* Telemetry Metrics & Joint Inspector (7 cols) */}
        <div className="md:col-span-7 space-y-2 text-[10px]">
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 bg-space-900 rounded border border-space-800">
              <div className="text-space-400 uppercase tracking-wider text-[9px] mb-0.5">3D Orientation (deg)</div>
              <div className="font-bold text-space-100 flex items-center gap-1.5">
                <span>Y: {(orient[0] * 57.3).toFixed(1)}°</span>
                <span>P: {(orient[1] * 57.3).toFixed(1)}°</span>
                <span>R: {(orient[2] * 57.3).toFixed(1)}°</span>
              </div>
            </div>

            <div className="p-2 bg-space-900 rounded border border-space-800">
              <div className="text-space-400 uppercase tracking-wider text-[9px] mb-0.5">Camera Translation (m)</div>
              <div className="font-bold text-space-100 flex items-center gap-1.5">
                <span>Tx: {trans[0].toFixed(2)}</span>
                <span>Ty: {trans[1].toFixed(2)}</span>
                <span>Tz: {trans[2].toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Selected or Default Joint Data */}
          <div className="p-2 bg-space-900 rounded border border-space-800">
            <div className="flex items-center justify-between text-[9px] text-space-400 pb-1 mb-1 border-b border-space-800">
              <span>JOINT INSPECTOR: <strong className="text-space-100">{selectedJoint ? `${selectedJoint.name} (#${selectedJoint.index})` : "Pelvis (#0)"}</strong></span>
              <span className="text-emerald-400">CONF: {selectedJoint ? `${Math.round(selectedJoint.confidence * 100)}%` : "95%"}</span>
            </div>
            {selectedJoint ? (
              <div className="grid grid-cols-3 gap-1 font-mono text-[9px] text-space-200">
                <div>X: <span className="text-accent-cyan">{selectedJoint.x.toFixed(3)}m</span></div>
                <div>Y: <span className="text-accent-cyan">{selectedJoint.y.toFixed(3)}m</span></div>
                <div>Z: <span className="text-accent-cyan">{selectedJoint.z.toFixed(3)}m</span></div>
              </div>
            ) : joints.length > 0 ? (
              <div className="grid grid-cols-3 gap-1 font-mono text-[9px] text-space-200">
                <div>X: <span className="text-accent-cyan">{joints[0].x.toFixed(3)}m</span></div>
                <div>Y: <span className="text-accent-cyan">{joints[0].y.toFixed(3)}m</span></div>
                <div>Z: <span className="text-accent-cyan">{joints[0].z.toFixed(3)}m</span></div>
              </div>
            ) : (
              <div className="text-space-500 text-[9px]">Awaiting 3D pose stream...</div>
            )}
          </div>

          {/* Model Status Notice */}
          <div className="p-2 bg-space-900/60 rounded border border-space-800 text-[9px] text-space-400 flex items-start gap-1.5">
            <Info size={12} className="text-purple-400 shrink-0 mt-0.5" />
            <span>
              Engine: <strong className="text-space-200">{modelName}</strong>. 
              {isFallback && " Provides deterministic kinematic joint lifting for zero-dependency edge offline execution."}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
