import { useState, useEffect } from 'react';
import { Database, Film, Play, RefreshCw, Layers, ShieldCheck } from 'lucide-react';
import VdataVideoInspector from './VdataVideoInspector';

interface DatasetManifest {
  dataset_version: string;
  total_videos: number;
  total_extracted_frames: number;
  annotated_frames: number;
  approved_samples: number;
  pending_review: number;
  rejected_samples: number;
  split_counts: { train: number; val: number; test: number };
}

interface ProductionModels {
  yolo_detector?: { version: string; metrics: any; is_production: boolean };
  har_gru?: { version: string; metrics: any; is_production: boolean };
}

interface CandidateItem {
  run_id: string;
  experiment_id: string;
  status: string;
  recorded_at: string;
  duration_seconds: number;
  samples_count: number;
  quality_score: number;
  is_acceptable: boolean;
  rejection_reasons: string[];
}

export default function DatasetControlPanel() {
  const [datasetStatus, setDatasetStatus] = useState<{
    stage: string;
    manifest: DatasetManifest | null;
    ingestion_report: any;
    production_models: ProductionModels;
    review_queue_counts: { pending: number; approved: number; rejected: number };
  } | null>(null);

  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/dataset/status');
      if (res.ok) {
        const data = await res.json();
        setDatasetStatus(data);
      }
      const candRes = await fetch('http://localhost:8000/api/dataset/candidates');
      if (candRes.ok) {
        const candData = await candRes.json();
        setCandidates(candData);
      }
    } catch {
      // Backend offline or reconnecting
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleIngest = async () => {
    setLoading(true);
    setActionMessage({ type: 'info', text: 'Scanning vdata/ and extracting quality-filtered frames...' });
    try {
      const res = await fetch('http://localhost:8000/api/dataset/ingest', { method: 'POST' });
      const data = await res.json();
      setActionMessage({ type: 'success', text: `Ingestion complete: ${data.manifest?.total_extracted_frames || 0} frames extracted.` });
      fetchStatus();
    } catch (e: any) {
      setActionMessage({ type: 'error', text: `Ingestion failed: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  const handlePreAnnotate = async () => {
    setLoading(true);
    setActionMessage({ type: 'info', text: 'Running assisted YOLO pre-annotation on frames...' });
    try {
      const res = await fetch('http://localhost:8000/api/dataset/pre_annotate', { method: 'POST' });
      const data = await res.json();
      setActionMessage({ type: 'success', text: `Pre-annotated ${data.frames_annotated} frames (${data.boxes_generated} boxes).` });
      fetchStatus();
    } catch (e: any) {
      setActionMessage({ type: 'error', text: `Pre-annotation failed: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async (modelType: 'har' | 'yolo') => {
    setLoading(true);
    setActionMessage({ type: 'info', text: `Executing controlled retraining for ${modelType.toUpperCase()}...` });
    try {
      const res = await fetch('http://localhost:8000/api/training/retrain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelType, epochs: modelType === 'har' ? 10 : 3 }),
      });
      const data = await res.json();
      if (data.status === 'ANNOTATIONS_REQUIRED') {
        setActionMessage({ type: 'error', text: `⚠ Safety Gate: ${data.message}` });
      } else if (data.status === 'TRAINING_COMPLETE') {
        setActionMessage({ type: 'success', text: `✓ Retraining complete: ${data.version} (${data.message})` });
      } else {
        setActionMessage({ type: 'info', text: data.message || 'Retraining finished.' });
      }
      fetchStatus();
    } catch (e: any) {
      setActionMessage({ type: 'error', text: `Retraining request failed: ${e.message}` });
    } finally {
      setLoading(false);
    }
  };

  const handleReviewCandidate = async (runId: string, action: 'approve' | 'reject') => {
    try {
      const res = await fetch(`http://localhost:8000/api/dataset/candidates/${runId}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, reviewer: 'mission_specialist', target_split: 'train' }),
      });
      if (res.ok) {
        setActionMessage({ type: 'success', text: `Candidate ${runId} ${action}d successfully.` });
        fetchStatus();
      }
    } catch (e: any) {
      setActionMessage({ type: 'error', text: `Review action failed: ${e.message}` });
    }
  };

  const manifest = datasetStatus?.manifest;
  const prodHar = datasetStatus?.production_models?.har_gru;
  const prodYolo = datasetStatus?.production_models?.yolo_detector;

  return (
    <div className="bg-[#121720] border border-[#232f42] rounded-xl p-6 text-white space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#232f42] pb-4">
        <div className="flex items-center space-x-3">
          <Database className="w-6 h-6 text-cyan-400" />
          <div>
            <h2 className="text-lg font-bold tracking-wide">ORBITA Closed-Loop Dataset & Training System</h2>
            <p className="text-xs text-slate-400">Reference Video Ingestion • Quality Gating • Candidate Review Queue • Model Governance</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-slate-400">STAGE:</span>
          <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-cyan-950 text-cyan-300 border border-cyan-700">
            {datasetStatus?.stage || 'INITIALIZING'}
          </span>
        </div>
      </div>

      {/* Action Notification Banner */}
      {actionMessage && (
        <div className={`p-3 rounded-lg text-xs font-mono flex items-center justify-between ${
          actionMessage.type === 'success' ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-700' :
          actionMessage.type === 'error' ? 'bg-rose-950/80 text-rose-300 border border-rose-700' :
          'bg-cyan-950/80 text-cyan-300 border border-cyan-700'
        }`}>
          <span>{actionMessage.text}</span>
          <button onClick={() => setActionMessage(null)} className="ml-3 hover:opacity-80">×</button>
        </div>
      )}

      {/* Top Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Ref Videos</div>
          <div className="text-xl font-bold text-cyan-400 mt-1">{manifest?.total_videos ?? 3}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">vdata/ catalog</div>
        </div>

        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Extracted Frames</div>
          <div className="text-xl font-bold text-white mt-1">{manifest?.total_extracted_frames ?? 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            {manifest?.split_counts ? `${manifest.split_counts.train}T / ${manifest.split_counts.val}V / ${manifest.split_counts.test}Te` : 'Adaptive 2.5 FPS'}
          </div>
        </div>

        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Annotated Frames</div>
          <div className="text-xl font-bold text-amber-400 mt-1">{manifest?.annotated_frames ?? 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">YOLO .txt labels</div>
        </div>

        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Approved Samples</div>
          <div className="text-xl font-bold text-emerald-400 mt-1">{manifest?.approved_samples ?? 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Merged to training</div>
        </div>

        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Pending Review</div>
          <div className="text-xl font-bold text-blue-400 mt-1">{manifest?.pending_review ?? 0}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Review queue</div>
        </div>

        <div className="bg-[#0b0f15] p-3 rounded-lg border border-[#1b2433]">
          <div className="text-[10px] text-slate-400 uppercase font-mono">Active Production</div>
          <div className="text-xs font-bold text-purple-300 mt-1 truncate">
            {prodHar?.version || prodYolo?.version || 'orbita_har_gru_v1'}
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5 truncate">
            {prodYolo?.version ? `${prodYolo.version} • HAR` : 'YOLOv8n + GRU'}
          </div>
        </div>
      </div>

      {/* Action Toolbar */}
      <div className="bg-[#0b0f15] p-4 rounded-lg border border-[#1b2433] flex flex-wrap items-center gap-3">
        <span className="text-xs font-mono text-slate-400 mr-2">PIPELINE CONTROLS:</span>
        <button
          onClick={handleIngest}
          disabled={loading}
          className="px-3 py-1.5 bg-[#1a2332] hover:bg-[#253247] border border-cyan-800/60 rounded text-xs text-cyan-300 font-medium flex items-center space-x-1.5 transition disabled:opacity-50"
        >
          <Film className="w-3.5 h-3.5" />
          <span>Ingest & Extract Frames</span>
        </button>

        <button
          onClick={handlePreAnnotate}
          disabled={loading}
          className="px-3 py-1.5 bg-[#1a2332] hover:bg-[#253247] border border-amber-800/60 rounded text-xs text-amber-300 font-medium flex items-center space-x-1.5 transition disabled:opacity-50"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Run Pre-Annotation</span>
        </button>

        <button
          onClick={() => handleRetrain('har')}
          disabled={loading}
          className="px-3 py-1.5 bg-[#1a2332] hover:bg-[#253247] border border-emerald-800/60 rounded text-xs text-emerald-300 font-medium flex items-center space-x-1.5 transition disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5" />
          <span>Retrain Temporal HAR (GRU)</span>
        </button>

        <button
          onClick={() => handleRetrain('yolo')}
          disabled={loading}
          className="px-3 py-1.5 bg-[#1a2332] hover:bg-[#253247] border border-purple-800/60 rounded text-xs text-purple-300 font-medium flex items-center space-x-1.5 transition disabled:opacity-50"
        >
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Retrain YOLO Detector</span>
        </button>

        <button
          onClick={fetchStatus}
          disabled={loading}
          className="px-2.5 py-1.5 bg-[#1a2332] hover:bg-[#253247] rounded text-xs text-slate-300 transition"
          title="Refresh telemetry"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Human Review Queue */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-200 flex items-center space-x-2">
            <span>Human Review Queue (Candidate Real Runs)</span>
            <span className="px-2 py-0.5 text-[10px] bg-slate-800 rounded-full text-slate-300 font-mono">
              {candidates.length} candidates
            </span>
          </h3>
          <span className="text-[11px] text-slate-400">
            Rule: Only approved samples enter future training datasets.
          </span>
        </div>

        {candidates.length === 0 ? (
          <div className="p-6 bg-[#0b0f15] border border-dashed border-[#1f293d] rounded-lg text-center text-xs text-slate-500">
            No pending candidate runs in the review queue. Successful live experiments will appear here for quality review.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border border-[#1b2433] rounded-lg overflow-hidden">
              <thead className="bg-[#0b0f15] text-slate-400 uppercase font-mono border-b border-[#1b2433]">
                <tr>
                  <th className="p-3">Run ID</th>
                  <th className="p-3">Duration</th>
                  <th className="p-3">Quality Score</th>
                  <th className="p-3">Status</th>
                  <th className="p-3 text-right">Review Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1b2433] bg-[#0e131b]">
                {candidates.map((cand) => (
                  <tr key={cand.run_id} className="hover:bg-[#131b26]">
                    <td className="p-3 font-mono text-cyan-300">{cand.run_id}</td>
                    <td className="p-3 text-slate-300">{cand.duration_seconds.toFixed(1)}s ({cand.samples_count} frames)</td>
                    <td className="p-3">
                      <span className={`font-mono font-bold ${cand.quality_score >= 70 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {cand.quality_score}%
                      </span>
                    </td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-mono ${
                        cand.status === 'approved' ? 'bg-emerald-950 text-emerald-300 border border-emerald-700' :
                        cand.status === 'rejected' ? 'bg-rose-950 text-rose-300 border border-rose-700' :
                        'bg-blue-950 text-blue-300 border border-blue-700'
                      }`}>
                        {cand.status}
                      </span>
                    </td>
                    <td className="p-3 text-right space-x-2">
                      {cand.status === 'pending' && (
                        <>
                          <button
                            onClick={() => handleReviewCandidate(cand.run_id, 'approve')}
                            className="px-2.5 py-1 bg-emerald-900/60 hover:bg-emerald-800 text-emerald-200 rounded text-[11px] font-medium"
                          >
                            Approve
                          </button>
                          <button
                            onClick={() => handleReviewCandidate(cand.run_id, 'reject')}
                            className="px-2.5 py-1 bg-rose-900/60 hover:bg-rose-800 text-rose-200 rounded text-[11px] font-medium"
                          >
                            Reject
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Reference Video YOLO Detection Visualizer */}
      <VdataVideoInspector />
    </div>
  );
}
