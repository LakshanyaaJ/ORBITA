import { Film } from 'lucide-react';
import DatasetControlPanel from '../components/dataset/DatasetControlPanel';

export default function VdataDatasetPage() {
  return (
    <div className="p-8 max-w-7xl mx-auto h-full flex flex-col space-y-6 overflow-y-auto">
      <div className="border-b border-space-600 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <Film className="w-6 h-6 text-accent-cyan" />
            <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100">
              VDATA & DATASET MANAGEMENT
            </h1>
          </div>
          <p className="text-space-400 text-xs font-mono mt-1">
            Local Video Library (vdata/) • Frame-by-Frame Inspector • Quality Gating • Synthetic Frame Extraction
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-space-400">STORAGE:</span>
          <span className="font-mono text-xs font-bold px-2.5 py-1 rounded bg-[#eafaf1] text-[#2e7d58] border border-[#c1e8d4] flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#2e7d58]" />
            vdata/ CONNECTED
          </span>
        </div>
      </div>

      <div className="flex-1">
        <DatasetControlPanel />
      </div>
    </div>
  );
}
