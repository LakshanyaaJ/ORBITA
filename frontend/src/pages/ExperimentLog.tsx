import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import clsx from 'clsx';
import { 
  Download, 
  CheckCircle2, 
  AlertTriangle, 
  PauseCircle, 
  PlayCircle, 
  Volume2, 
  ShieldCheck, 
  Clock, 
  Activity, 
  RotateCw,
  Server
} from 'lucide-react';
import { BACKEND_BASE } from '../api/camera';

const CATALOG_EXPERIMENTS = [
  { id: 'EXP-MICROBE', name: 'EXP-MICROBE', label: 'Microbial Microgravity', steps: '07 STEPS' },
  { id: 'EXP-VDATA', name: 'EXP-VDATA', label: 'Blue & Yellow Box VDATA', steps: '13 STEPS' },
  { id: 'EXP-02', name: 'EXP-02', label: 'Sample Analysis', steps: '02 STEPS' },
];

interface LogEvent {
  id?: number | string;
  timestamp: string;
  elapsed_time?: number;
  elapsed_seconds?: number;
  step_number: number;
  total_steps: number;
  expected_action?: string;
  step_name?: string;
  detected_action?: string;
  target_object?: string;
  detected_object?: string;
  status: string;
  validation_status?: string;
  procedure_state?: string;
  confidence?: number;
  event_type?: string;
  voice_guidance?: string;
  deviation_reason?: string;
  source?: string;
  execution_id?: string;
  metadata?: Record<string, any>;
  data?: Record<string, any>;
}

interface LogData {
  experiment_id: string;
  experiment_name: string;
  execution_id?: string;
  protocol_steps?: number;
  total_steps: number;
  completed_steps: number;
  validated_count?: number;
  unexpected_count?: number;
  deviation_count?: number;
  current_state?: string;
  execution_mode?: string;
  sync_status?: string;
  start_time: string;
  end_time?: string;
  duration_seconds: number;
  checksum?: string;
  integrity_hash?: string;
  integrity_checksum?: string;
  events: LogEvent[];
}

export default function ExperimentLog() {
  const { id } = useParams();
  const navigate = useNavigate();
  const currentExpId = id || 'EXP-MICROBE';

  const [logData, setLogData] = useState<LogData | null>(null);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [availableRuns, setAvailableRuns] = useState<any[]>([]);
  const [filter, setFilter] = useState<'ALL' | 'VALIDATED' | 'UNEXPECTED' | 'DEVIATIONS' | 'VOICE' | 'FAILED'>('ALL');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Fetch available runs for this experiment
  useEffect(() => {
    fetch(`${BACKEND_BASE}/api/experiments/${currentExpId}/runs`)
      .then((res) => res.json())
      .then((runs) => {
        if (Array.isArray(runs)) {
          setAvailableRuns(runs);
        }
      })
      .catch(() => {});
  }, [currentExpId]);

  // Fetch structured log data
  const fetchLogs = () => {
    const query = selectedRun ? `?execution_id=${encodeURIComponent(selectedRun)}` : '';
    fetch(`${BACKEND_BASE}/api/experiments/${currentExpId}/logs${query}`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.events) {
          setLogData(data);
        }
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load experiment logs:', err);
        setIsLoading(false);
      });
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [currentExpId, selectedRun]);

  const handleExportJSON = () => {
    if (!logData) return;
    const jsonStr = JSON.stringify(logData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentExpId}_${logData.execution_id}_structured_log.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatTime = (ts: string | number) => {
    if (!ts) return '--:--:--';
    try {
      if (typeof ts === 'number') {
        const d = new Date(ts * 1000);
        return d.toLocaleTimeString([], { hour12: false });
      }
      const d = new Date(ts);
      if (isNaN(d.getTime())) return String(ts).slice(11, 19) || '--:--:--';
      return d.toLocaleTimeString([], { hour12: false });
    } catch {
      return '--:--:--';
    }
  };

  const getFilteredEvents = () => {
    if (!logData || !logData.events) return [];
    return logData.events.filter((ev) => {
      const typeUpper = (ev.event_type || '').toUpperCase();
      const statusUpper = (ev.status || ev.validation_status || '').toUpperCase();

      if (filter === 'ALL') return true;
      if (filter === 'VALIDATED') return typeUpper === 'STEP_VALIDATED' || statusUpper === 'VALIDATED' || statusUpper === 'SUCCESS';
      if (filter === 'UNEXPECTED') return typeUpper === 'PROCEDURE_DEVIATION' || statusUpper === 'UNEXPECTED' || statusUpper === 'DEVIATION';
      if (filter === 'DEVIATIONS') return typeUpper === 'PROCEDURE_DEVIATION' || statusUpper === 'UNEXPECTED' || statusUpper === 'DEVIATION';
      if (filter === 'VOICE') return typeUpper === 'VOICE_GUIDANCE' || Boolean(ev.voice_guidance);
      if (filter === 'FAILED') return statusUpper === 'FAILED' || statusUpper === 'ERROR';
      return true;
    });
  };

  const filteredEvents = getFilteredEvents();

  return (
    <div className="p-6 max-w-7xl mx-auto h-full flex flex-col font-sans select-none overflow-y-auto">
      
      {/* 0. EXPERIMENT SELECTOR TABS */}
      <div className="flex items-center gap-2 mb-4 p-2 bg-space-850 rounded-lg border border-space-700 flex-wrap">
        <span className="font-mono text-xs font-bold text-space-400 uppercase tracking-wider px-2">
          EXPERIMENT CATALOG LOGS:
        </span>
        {CATALOG_EXPERIMENTS.map((exp) => {
          const isActive = currentExpId.toUpperCase() === exp.id;
          return (
            <button
              key={exp.id}
              onClick={() => navigate(`/experiments/${exp.id}/log`)}
              className={clsx(
                "px-3.5 py-1.5 rounded-md font-mono text-xs font-bold tracking-wider transition-all flex items-center gap-2 border cursor-pointer",
                isActive
                  ? "bg-accent-cyan text-space-950 border-accent-cyan shadow-sm"
                  : "bg-space-800 text-space-300 border-space-600 hover:bg-space-700 hover:text-space-100"
              )}
            >
              <span>{exp.name}</span>
              <span className="text-[10px] opacity-75">({exp.label})</span>
            </button>
          );
        })}
      </div>

      {/* 1. TOP HEADER & METADATA BAR */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-space-600 pb-4 mb-6 gap-4">
        <div>
          <div className="flex items-center gap-3 flex-wrap mb-1">
            <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100">
              EXPERIMENT LOG
            </h1>
            <span className="px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40">
              {logData?.experiment_id || currentExpId}
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-space-800 text-space-300 border border-space-600 flex items-center gap-1.5">
              <Server size={12} className="text-accent-cyan" />
              MODE: {logData?.execution_mode || 'OFFLINE'}
            </span>
            <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border flex items-center gap-1.5 ${
              (logData?.sync_status || '').includes('SYNCED')
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
            }`}>
              <Activity size={12} />
              {logData?.sync_status || 'LOCAL ✓'}
            </span>
          </div>
          <h2 className="text-lg font-bold text-space-200">
            {logData?.experiment_name || 'Loading Experiment...'}
          </h2>
          <div className="flex items-center gap-4 text-xs font-mono text-space-400 mt-1">
            <span>Execution ID: <strong className="text-space-200">{logData?.execution_id || 'RUN-INITIALIZING'}</strong></span>
            {logData?.checksum && (
              <span className="flex items-center gap-1 text-emerald-400" title={`SHA-256 Checksum: ${logData.checksum}`}>
                <ShieldCheck size={13} />
                {logData.integrity_hash}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {availableRuns.length > 1 && (
            <select
              value={selectedRun}
              onChange={(e) => setSelectedRun(e.target.value)}
              className="bg-space-900 border border-space-600 text-space-200 text-xs font-mono rounded px-3 py-2 focus:outline-none focus:border-accent-cyan"
            >
              <option value="">Latest Active Run</option>
              {availableRuns.map((r) => (
                <option key={r.execution_id} value={r.execution_id}>
                  {r.execution_id} ({r.event_count} events)
                </option>
              ))}
            </select>
          )}

          <button
            onClick={fetchLogs}
            className="p-2 bg-space-800 border border-space-600 hover:border-space-400 rounded text-space-300 hover:text-space-100 transition-colors"
            title="Refresh Logs"
          >
            <RotateCw size={16} />
          </button>

          <button
            onClick={handleExportJSON}
            className="flex items-center gap-2 px-4 py-2 bg-accent-cyan text-space-950 font-mono font-bold tracking-widest text-xs rounded hover:bg-accent-cyan/90 transition-colors shadow-md"
          >
            <Download size={15} />
            EXPORT LOG (JSON)
          </button>
        </div>
      </div>

      {/* 2. DYNAMIC SUMMARY METRICS CARDS */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6 font-mono">
        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">PROTOCOL PROGRESS</div>
          <div className="text-xl font-bold text-space-100">
            {logData?.completed_steps ?? 0} <span className="text-space-400 text-sm">/ {logData?.protocol_steps ?? '--'}</span>
          </div>
          <div className="text-[10px] text-accent-cyan mt-1">STEPS COMPLETED</div>
        </div>

        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">VALIDATED ACTIONS</div>
          <div className="text-xl font-bold text-emerald-400 flex items-center gap-1.5">
            <CheckCircle2 size={18} />
            {logData?.validated_count ?? 0}
          </div>
          <div className="text-[10px] text-emerald-300/80 mt-1">CONFIRMED</div>
        </div>

        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">UNEXPECTED / HELD</div>
          <div className="text-xl font-bold text-amber-400 flex items-center gap-1.5">
            <AlertTriangle size={18} />
            {logData?.unexpected_count ?? 0}
          </div>
          <div className="text-[10px] text-amber-300/80 mt-1">PROCEDURE DEVIATIONS</div>
        </div>

        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">CURRENT STATE</div>
          <div className="text-base font-bold text-accent-cyan truncate">
            {logData?.current_state || 'READY'}
          </div>
          <div className="text-[10px] text-space-400 mt-1">FSM VALIDATION</div>
        </div>

        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">STARTED AT</div>
          <div className="text-xs font-bold text-space-200 mt-1">
            {formatTime(logData?.start_time || '')}
          </div>
          <div className="text-[10px] text-space-400 mt-1">UTC TIMESTAMP</div>
        </div>

        <div className="bg-space-800 p-3.5 rounded-lg border border-space-600">
          <div className="text-[10px] text-space-400 tracking-wider uppercase mb-1">DURATION</div>
          <div className="text-xl font-bold text-space-100 flex items-center gap-1.5">
            <Clock size={16} className="text-space-400" />
            {logData?.duration_seconds ?? 0}s
          </div>
          <div className="text-[10px] text-space-400 mt-1">ELAPSED TIME</div>
        </div>
      </div>

      {/* 3. LIGHTWEIGHT FILTER BAR */}
      <div className="flex items-center justify-between bg-space-900 p-2 rounded-lg border border-space-700 mb-4 font-mono text-xs flex-wrap gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-space-400 text-[11px] px-2 font-bold uppercase">Filter Events:</span>
          {(['ALL', 'VALIDATED', 'UNEXPECTED', 'DEVIATIONS', 'VOICE', 'FAILED'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded transition-colors font-bold ${
                filter === f
                  ? 'bg-accent-cyan text-space-950 shadow'
                  : 'bg-space-800 text-space-300 hover:bg-space-700 hover:text-space-100'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <div className="text-space-400 text-[11px] pr-2">
          Showing {filteredEvents.length} of {logData?.events?.length || 0} events
        </div>
      </div>

      {/* 4. CHRONOLOGICAL EVENT TIMELINE TABLE */}
      <div className="flex-1 bg-space-900 border border-space-700 rounded-lg overflow-hidden flex flex-col font-mono text-xs">
        <div className="grid grid-cols-12 bg-space-850 p-3 border-b border-space-700 text-space-400 font-bold uppercase tracking-wider text-[11px]">
          <div className="col-span-2 flex items-center gap-1">TIME / ELAPSED</div>
          <div className="col-span-1 text-center">STEP</div>
          <div className="col-span-2">EXPECTED ACTION</div>
          <div className="col-span-2">DETECTED ACTION</div>
          <div className="col-span-2">TARGET OBJECT</div>
          <div className="col-span-2">STATUS</div>
          <div className="col-span-1 text-right">CONF.</div>
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-space-800">
          {isLoading ? (
            <div className="p-8 text-center text-space-400 flex flex-col items-center justify-center gap-2">
              <RotateCw className="animate-spin text-accent-cyan" size={24} />
              <span>Loading structured experiment logs...</span>
            </div>
          ) : filteredEvents.length === 0 ? (
            <div className="p-12 text-center text-space-400 flex flex-col items-center justify-center gap-3">
              <Activity className="text-space-600" size={32} />
              <div className="font-bold text-space-300">No matching log events recorded.</div>
              <p className="text-xs max-w-md text-space-400 leading-relaxed">
                Start or run an experiment trial to generate real-time structured execution telemetry. Events will automatically stream into this chronological timeline.
              </p>
            </div>
          ) : (
            filteredEvents.map((ev, idx) => {
              const evType = (ev.event_type || '').toUpperCase();
              const statusStr = (ev.status || ev.validation_status || 'PENDING').toUpperCase();
              const confVal = ev.confidence !== undefined && ev.confidence !== null ? ev.confidence : null;

              const isVal = evType === 'STEP_VALIDATED' || statusStr === 'VALIDATED' || statusStr === 'SUCCESS';
              const isDev = evType === 'PROCEDURE_DEVIATION' || statusStr === 'UNEXPECTED' || statusStr === 'DEVIATION';
              const isHeld = evType === 'STEP_HELD' || statusStr === 'HELD' || statusStr === 'UNCERTAIN';
              const isVoice = evType === 'VOICE_GUIDANCE' || Boolean(ev.voice_guidance);
              const isStart = evType === 'EXPERIMENT_STARTED' || evType === 'EXPERIMENT_COMPLETED';

              return (
                <div
                  key={ev.id || idx}
                  className={`grid grid-cols-12 p-3 items-center hover:bg-space-800/80 transition-colors ${
                    isDev ? 'bg-red-950/20' : isVal ? 'bg-emerald-950/10' : isVoice ? 'bg-cyan-950/10' : ''
                  }`}
                >
                  {/* TIME / ELAPSED */}
                  <div className="col-span-2 flex flex-col">
                    <span className="text-space-100 font-bold">{formatTime(ev.timestamp)}</span>
                    <span className="text-[10px] text-space-400">
                      +{typeof ev.elapsed_seconds === 'number' ? ev.elapsed_seconds.toFixed(1) : (ev.elapsed_time ? Number(ev.elapsed_time).toFixed(1) : '0.0')}s
                    </span>
                  </div>

                  {/* STEP */}
                  <div className="col-span-1 text-center font-bold">
                    <span className="px-1.5 py-0.5 rounded bg-space-800 border border-space-700 text-space-200">
                      {String(ev.step_number || 1).padStart(2, '0')}/{ev.total_steps || logData?.protocol_steps || '--'}
                    </span>
                  </div>

                  {/* EXPECTED ACTION */}
                  <div className="col-span-2 truncate pr-2 font-bold text-space-200" title={ev.expected_action || ev.step_name}>
                    {ev.expected_action || ev.step_name || '--'}
                  </div>

                  {/* DETECTED ACTION */}
                  <div className="col-span-2 truncate pr-2 text-space-300" title={ev.detected_action}>
                    {ev.detected_action || '--'}
                  </div>

                  {/* TARGET OBJECT */}
                  <div className="col-span-2 truncate pr-2 text-accent-cyan/90 font-semibold" title={ev.target_object || ev.detected_object}>
                    {ev.target_object || ev.detected_object || '--'}
                  </div>

                  {/* STATUS PILL */}
                  <div className="col-span-2 flex flex-col gap-0.5">
                    {isVal && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 w-fit">
                        <CheckCircle2 size={13} />
                        VALIDATED
                      </span>
                    )}
                    {isDev && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold bg-red-500/20 text-red-300 border border-red-500/40 w-fit">
                        <AlertTriangle size={13} />
                        UNEXPECTED
                      </span>
                    )}
                    {isHeld && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 w-fit">
                        <PauseCircle size={13} />
                        HELD
                      </span>
                    )}
                    {isVoice && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold bg-cyan-500/20 text-accent-cyan border border-accent-cyan/40 w-fit">
                        <Volume2 size={13} />
                        VOICE
                      </span>
                    )}
                    {isStart && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40 w-fit">
                        <PlayCircle size={13} />
                        {evType}
                      </span>
                    )}
                    {ev.deviation_reason && (
                      <span className="text-[10px] text-red-300/80 truncate leading-tight mt-0.5" title={ev.deviation_reason}>
                        Reason: {ev.deviation_reason}
                      </span>
                    )}
                    {ev.voice_guidance && (
                      <span className="text-[10px] text-accent-cyan/80 truncate leading-tight mt-0.5" title={ev.voice_guidance}>
                        "{ev.voice_guidance}"
                      </span>
                    )}
                  </div>

                  {/* CONFIDENCE */}
                  <div className="col-span-1 text-right font-bold text-space-300">
                    {confVal !== null && confVal > 0 ? (confVal > 1.0 ? confVal.toFixed(2) : confVal.toFixed(2)) : '--'}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
