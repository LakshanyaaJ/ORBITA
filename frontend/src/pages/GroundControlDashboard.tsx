import React, { useState } from 'react';
import { Navbar } from '../components/Navbar';
import {
  Satellite,
  Play,
  Filter,
  Video,
  Upload,
  CheckSquare,
  Send,
  ChevronRight,
  Download,
  Eye,
} from 'lucide-react';

interface Incident {
  id: string;
  type: string;
  timestamp: string;
  expected: string;
  observed: string;
  confidence: number;
  evidenceRange: string;
  videoClip: string;
  severity: 'high' | 'medium' | 'low';
}

const SAMPLE_INCIDENTS: Incident[] = [
  {
    id: 'INC-004',
    type: 'OUT-OF-SEQUENCE',
    timestamp: '10:21:34',
    expected: 'Retrieve Red Box',
    observed: 'Yellow Box Interaction',
    confidence: 0.91,
    evidenceRange: '10:21:34 → 10:21:41',
    videoClip: '20260905_145858',
    severity: 'high',
  },
  {
    id: 'INC-003',
    type: 'POTENTIAL_DIFFICULTY',
    timestamp: '10:17:40',
    expected: 'Open Red Box',
    observed: 'Incomplete Action (Attempt 3)',
    confidence: 0.88,
    evidenceRange: '10:17:30 → 10:17:45',
    videoClip: '20260905_145858',
    severity: 'medium',
  },
  {
    id: 'INC-002',
    type: 'LOW_CONFIDENCE',
    timestamp: '10:16:05',
    expected: 'Take out Red Box',
    observed: 'Partial Occlusion',
    confidence: 0.58,
    evidenceRange: '10:16:00 → 10:16:10',
    videoClip: '20260905_145858',
    severity: 'low',
  },
];

export const GroundControlDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'OVERVIEW' | 'INCIDENTS' | 'TIMELINE' | 'ARCHIVE' | 'MANAGEMENT'>('OVERVIEW');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(SAMPLE_INCIDENTS[0]);
  const [playingEvidence, setPlayingEvidence] = useState(false);
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  // Deployment workflow state
  const [procedureApproved, setProcedureApproved] = useState(false);
  const [deployed, setDeployed] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col">
      <Navbar telemetryConnected={true} fps={28} aiStatus="YOLO11 + GRU" activeModel="orbita_yolo11.pt" />

      {/* Sub-Header & Ground Station Navigation Tabs */}
      <div className="bg-slate-900 border-b border-slate-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <span className="px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider bg-amber-950 text-amber-400 border border-amber-800">
                MISSION CONTROL GROUND STATION
              </span>
              <h1 className="text-xl font-bold font-mono tracking-tight text-white">EARTH GROUND CONTROL</h1>
            </div>
            <p className="text-xs text-slate-400 mt-1">Ground Station Monitoring, Incident Evidence Review & Procedure Governance</p>
          </div>

          {/* Sub-Navigation Tabs */}
          <div className="flex items-center bg-slate-950 p-1.5 rounded-xl border border-slate-800 space-x-1 flex-wrap gap-1">
            <button
              onClick={() => setActiveTab('OVERVIEW')}
              className={`px-3 py-1.5 rounded-lg font-semibold text-xs transition ${
                activeTab === 'OVERVIEW' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('INCIDENTS')}
              className={`px-3 py-1.5 rounded-lg font-semibold text-xs transition ${
                activeTab === 'INCIDENTS' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Incident Review ({SAMPLE_INCIDENTS.length})
            </button>
            <button
              onClick={() => setActiveTab('TIMELINE')}
              className={`px-3 py-1.5 rounded-lg font-semibold text-xs transition ${
                activeTab === 'TIMELINE' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Event Timeline
            </button>
            <button
              onClick={() => setActiveTab('ARCHIVE')}
              className={`px-3 py-1.5 rounded-lg font-semibold text-xs transition ${
                activeTab === 'ARCHIVE' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Experiment Archive
            </button>
            <button
              onClick={() => setActiveTab('MANAGEMENT')}
              className={`px-3 py-1.5 rounded-lg font-semibold text-xs transition ${
                activeTab === 'MANAGEMENT' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Experiment Management
            </button>
          </div>
        </div>
      </div>

      {/* Main Tab Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full">
        
        {/* VIEW 1: MISSION OVERVIEW */}
        {activeTab === 'OVERVIEW' && (
          <div className="space-y-6">
            
            {/* Top Mission Status Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 font-mono text-xs">
              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">CONNECTION</div>
                <div className="text-emerald-400 font-bold mt-1 flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <span>CONNECTED</span>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">EXPERIMENT</div>
                <div className="text-amber-300 font-bold mt-1">BAS-02</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">PROGRESS</div>
                <div className="text-cyan-300 font-bold mt-1">4 / 6 STEPS</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">CURRENT STEP</div>
                <div className="text-white font-bold mt-1 truncate">Retrieve Tool</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">ACTIVE ALERTS</div>
                <div className="text-rose-400 font-bold mt-1">2 UNRESOLVED</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">AI STATUS</div>
                <div className="text-emerald-400 font-bold mt-1">HEALTHY</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">CAMERA</div>
                <div className="text-emerald-400 font-bold mt-1">HEALTHY</div>
              </div>

              <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                <div className="text-[10px] text-slate-400">LAST TELEMETRY</div>
                <div className="text-slate-300 font-bold mt-1">10:21:34 UTC</div>
              </div>
            </div>

            {/* Main Overview Grid: Live Monitoring Stream & Action Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Compressed Near-Live Stream View */}
              <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                <div className="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex justify-between items-center text-xs">
                  <div className="flex items-center space-x-2 text-amber-400 font-semibold">
                    <Satellite className="w-4 h-4" />
                    <span>GROUND TELEMETRY RECEPTION — COMPRESSED NEAR-LIVE FEED</span>
                  </div>
                  <span className="font-mono text-slate-400 text-[10px]">LATENCY: 140ms</span>
                </div>

                <div className="relative aspect-video bg-black flex items-center justify-center">
                  <img src="/video_feed" alt="Ground Telemetry Stream" className="w-full h-full object-contain" />
                  
                  <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-amber-600/40 text-xs font-mono text-amber-300">
                    EXPECTED: Retrieve Red Box
                  </div>

                  <div className="absolute bottom-4 right-4 bg-rose-950/90 backdrop-blur-md px-3.5 py-2 rounded-lg border border-rose-600/60 text-xs font-bold text-rose-200">
                    ALERT: OUT OF SEQUENCE (91% CONF)
                  </div>
                </div>
              </div>

              {/* Observed vs Expected Action Breakdown */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-amber-400 border-b border-slate-800 pb-2">
                  ACTION DISCREPANCY ANALYSIS
                </h3>

                <div className="space-y-3 text-xs">
                  <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                    <div className="text-slate-400 text-[10px] uppercase font-mono">EXPECTED PROCEDURE ACTION</div>
                    <div className="text-sm font-bold text-emerald-400 mt-1">Retrieve Red Box</div>
                    <div className="text-[11px] text-slate-500 font-mono mt-0.5">Action: TAKE | Object: RED_BOX</div>
                  </div>

                  <div className="bg-rose-950/40 p-3.5 rounded-xl border border-rose-800/60">
                    <div className="text-rose-400 text-[10px] uppercase font-mono">OBSERVED AI DETECTION</div>
                    <div className="text-sm font-bold text-rose-200 mt-1">Yellow Box Interaction</div>
                    <div className="text-[11px] text-rose-300 font-mono mt-0.5">Action: TAKE | Object: YELLOW_BOX</div>
                  </div>

                  <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                    <div className="text-slate-400 text-[10px] uppercase font-mono">CONFIDENCE SCORE</div>
                    <div className="text-lg font-bold font-mono text-cyan-300 mt-1">91.0%</div>
                  </div>

                  <div className="bg-amber-950/40 p-3.5 rounded-xl border border-amber-800/60 text-amber-200">
                    <div className="font-bold">STATUS: OUT OF SEQUENCE</div>
                    <p className="text-[11px] mt-1 text-amber-300">
                      Astronaut interacted with Yellow Box prior to completing Step 2 (Red Box).
                    </p>
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* VIEW 2: INCIDENT REVIEW WITH TIMESTAMPED EVIDENCE PLAYER */}
        {activeTab === 'INCIDENTS' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Incident List */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                FLAGGED MISSION INCIDENTS
              </h3>

              {SAMPLE_INCIDENTS.map((inc) => (
                <div
                  key={inc.id}
                  onClick={() => setSelectedIncident(inc)}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    selectedIncident?.id === inc.id
                      ? 'bg-amber-950/50 border-amber-600 shadow-lg'
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-mono font-bold text-amber-400">{inc.id}</span>
                    <span className="text-slate-400 text-[10px] font-mono">{inc.timestamp}</span>
                  </div>

                  <h4 className="text-sm font-bold text-white mt-1">{inc.type}</h4>
                  
                  <div className="text-xs text-slate-400 mt-2 space-y-0.5 font-mono">
                    <div>Expected: {inc.expected}</div>
                    <div>Observed: {inc.observed}</div>
                  </div>

                  <div className="mt-3 flex justify-between items-center pt-2 border-t border-slate-800 text-[11px]">
                    <span className="font-mono text-cyan-300">CONF: {(inc.confidence * 100).toFixed(0)}%</span>
                    <span className="text-amber-400 font-bold flex items-center space-x-1">
                      <span>VIEW EVIDENCE</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Dedicated Incident Evidence Viewer */}
            <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              {selectedIncident ? (
                <div>
                  <div className="flex justify-between items-start border-b border-slate-800 pb-4">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800">
                          {selectedIncident.type}
                        </span>
                        <span className="font-mono text-slate-400 text-xs">{selectedIncident.timestamp} UTC</span>
                      </div>
                      <h2 className="text-lg font-bold text-white mt-1">INCIDENT {selectedIncident.id} ANALYSIS</h2>
                    </div>

                    <div className="text-right font-mono text-xs">
                      <div className="text-slate-400">EVIDENCE TIMEFRAME</div>
                      <div className="text-cyan-300 font-bold">{selectedIncident.evidenceRange}</div>
                    </div>
                  </div>

                  {/* Video Evidence Player */}
                  <div className="mt-4 relative aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
                    {playingEvidence ? (
                      <video
                        src={`/api/evidence/${selectedIncident.videoClip}`}
                        controls
                        autoPlay
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-center space-y-3">
                        <Video className="w-12 h-12 text-amber-500 mx-auto animate-bounce" />
                        <p className="text-xs text-slate-400">Recorded Video Evidence Segment Available</p>
                        <button
                          onClick={() => setPlayingEvidence(true)}
                          className="bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs px-5 py-2.5 rounded-xl transition shadow-lg shadow-amber-600/30 flex items-center space-x-2 mx-auto"
                        >
                          <Play className="w-4 h-4 fill-white" />
                          <span>[ VIEW EVIDENCE ]</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Detailed Analysis Breakdown */}
                  <div className="mt-5 grid grid-cols-2 gap-4 text-xs font-mono">
                    <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                      <div className="text-slate-400">EXPECTED PROCEDURE</div>
                      <div className="text-emerald-400 font-bold text-sm mt-1">{selectedIncident.expected}</div>
                    </div>

                    <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                      <div className="text-slate-400">AI OBSERVED ACTION</div>
                      <div className="text-rose-400 font-bold text-sm mt-1">{selectedIncident.observed}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-slate-500">Select an incident to review evidence clip.</div>
              )}
            </div>

          </div>
        )}

        {/* VIEW 3: EVENT TIMELINE WITH FILTERING */}
        {activeTab === 'TIMELINE' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            
            {/* Filter Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="flex items-center space-x-2">
                <Filter className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-bold text-white uppercase tracking-wider">FILTER TIMELINE</span>
              </div>

              <div className="flex items-center space-x-3 flex-wrap gap-2 text-xs">
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-amber-500"
                >
                  <option value="ALL">All Severities</option>
                  <option value="HIGH">High Severity</option>
                  <option value="MEDIUM">Medium Severity</option>
                  <option value="LOW">Low Severity</option>
                </select>

                <input
                  type="text"
                  placeholder="Search event keywords..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-amber-500"
                />
              </div>
            </div>

            {/* Timeline Stream */}
            <div className="space-y-3">
              {[
                { time: '10:14:50', step: 'Step 1', event: 'Step 1 completed: Retrieve Main Container', status: '✓ PASSED', type: 'info', sev: 'LOW' },
                { time: '10:15:10', step: 'Step 2', event: 'Step 2 initiated: Take out Red Box', status: 'IN PROGRESS', type: 'info', sev: 'LOW' },
                { time: '10:16:05', step: 'Step 2', event: 'Low confidence interaction detected (58%)', status: '⚠ UNCERTAIN', type: 'warning', sev: 'LOW' },
                { time: '10:17:20', step: 'Step 2', event: 'Attempt 2 incomplete interaction', status: '⚠ INCOMPLETE', type: 'warning', sev: 'MEDIUM' },
                { time: '10:21:34', step: 'Step 2', event: 'Out-of-sequence interaction with Yellow Box', status: '🔴 DEVIATION', type: 'critical', sev: 'HIGH' },
                { time: '10:29:42', step: 'Step 2', event: 'Potential execution difficulty detected (Attempt 3)', status: '🟠 DIFFICULTY', type: 'warning', sev: 'MEDIUM' },
              ]
                .filter((item) => (severityFilter === 'ALL' ? true : item.sev === severityFilter))
                .filter((item) => (searchQuery ? item.event.toLowerCase().includes(searchQuery.toLowerCase()) : true))
                .map((item, index) => (
                  <div key={index} className="flex items-center space-x-4 p-3.5 bg-slate-950 rounded-xl border border-slate-800 text-xs">
                    <span className="font-mono text-slate-400 text-xs w-20">{item.time}</span>
                    <span className="font-mono font-bold text-amber-300 w-16">{item.step}</span>
                    <span className="flex-1 font-medium text-slate-200">{item.event}</span>
                    <span className={`font-mono font-bold px-2.5 py-1 rounded ${
                      item.type === 'critical' ? 'bg-rose-950 text-rose-300 border border-rose-800' : 'bg-slate-800 text-slate-300'
                    }`}>
                      {item.status}
                    </span>
                  </div>
                ))}
            </div>

          </div>
        )}

        {/* VIEW 4: EXPERIMENT ARCHIVE */}
        {activeTab === 'ARCHIVE' && (
          <div className="space-y-6">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">COMPLETED EXPERIMENT SESSIONS</h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {['BAS-01', 'BAS-02', 'BAS-03'].map((expId, idx) => (
                <div key={expId} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-lg font-bold text-amber-400">{expId}</span>
                    <span className="px-2 py-0.5 text-[10px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
                      ARCHIVED
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 font-medium">Onboard BAS Scientific Procedure Monitoring</p>
                  
                  <div className="text-xs font-mono text-slate-400 space-y-1 pt-2 border-t border-slate-800">
                    <div>Date: 2026-09-05</div>
                    <div>Total Duration: 14m 22s</div>
                    <div>Accuracy: {(94.2 + idx).toFixed(1)}%</div>
                  </div>

                  <div className="pt-2 flex space-x-2">
                    <button className="flex-1 bg-slate-800 hover:bg-slate-700 text-xs font-bold py-2 rounded-xl text-white transition flex items-center justify-center space-x-1">
                      <Eye className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Review Video</span>
                    </button>
                    <button className="bg-slate-800 hover:bg-slate-700 text-xs p-2 rounded-xl text-emerald-400 transition">
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* VIEW 5: EXPERIMENT MANAGEMENT & DEPLOYMENT */}
        {activeTab === 'MANAGEMENT' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div>
              <h2 className="text-lg font-bold text-white">EXPERIMENT & AI MODEL GOVERNANCE</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Define procedures, upload reference videos, validate YOLO11 model adaptation, and approve edge deployment.
              </p>
            </div>

            {/* Stage Pipeline Visualization */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-center text-xs font-mono">
              {['Upload', 'Dataset Prep', 'Training', 'Validation', 'Human Approval', 'Deployment'].map((stage, idx) => (
                <div
                  key={stage}
                  className={`p-3 rounded-xl border ${
                    idx < 4
                      ? 'bg-emerald-950/50 border-emerald-800 text-emerald-300'
                      : idx === 4 && procedureApproved
                      ? 'bg-emerald-950/50 border-emerald-800 text-emerald-300'
                      : idx === 5 && deployed
                      ? 'bg-cyan-950/50 border-cyan-800 text-cyan-300'
                      : 'bg-slate-950 border-slate-800 text-slate-500'
                  }`}
                >
                  <div className="text-[10px] text-slate-400 font-bold">STAGE {idx + 1}</div>
                  <div className="font-bold mt-1">{stage}</div>
                </div>
              ))}
            </div>

            {/* Procedure Definition & Status */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-800">
              
              <div className="space-y-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-amber-400">PROCEDURE BAS-02 STATUS</h3>
                
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs space-y-2 font-mono">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Procedure Version:</span>
                    <span className="text-white font-bold">v1.3</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Dataset Status:</span>
                    <span className="text-emerald-400 font-bold">READY (v3 split)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">YOLO11 Model:</span>
                    <span className="text-emerald-400 font-bold">VALIDATED (mAP50: 86.5%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Procedure Specification:</span>
                    <span className="text-emerald-400 font-bold">VALIDATED (6 STEPS)</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={() => setProcedureApproved(true)}
                    disabled={procedureApproved}
                    className={`flex-1 font-bold text-xs py-3 rounded-xl transition flex items-center justify-center space-x-2 ${
                      procedureApproved
                        ? 'bg-slate-800 text-slate-500 border border-slate-700'
                        : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30'
                    }`}
                  >
                    <CheckSquare className="w-4 h-4" />
                    <span>{procedureApproved ? 'APPROVED ✓' : '[ APPROVE PROCEDURE ]'}</span>
                  </button>

                  <button
                    onClick={() => setDeployed(true)}
                    disabled={!procedureApproved || deployed}
                    className={`flex-1 font-bold text-xs py-3 rounded-xl transition flex items-center justify-center space-x-2 ${
                      deployed
                        ? 'bg-cyan-900 text-cyan-300 border border-cyan-700'
                        : procedureApproved
                        ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-lg shadow-amber-600/30'
                        : 'bg-slate-800 text-slate-600 border border-slate-800'
                    }`}
                  >
                    <Send className="w-4 h-4" />
                    <span>{deployed ? 'DEPLOYED TO ONBOARD ✓' : '[ DEPLOY TO ONBOARD ]'}</span>
                  </button>
                </div>
              </div>

              {/* Reference Video Uploader Simulation */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-amber-400">UPLOAD REFERENCE EXPERIMENT VIDEO</h3>
                
                <div className="border-2 border-dashed border-slate-700 hover:border-amber-500/60 rounded-2xl p-8 text-center bg-slate-950/60 transition cursor-pointer">
                  <Upload className="w-8 h-8 text-amber-400 mx-auto mb-2" />
                  <p className="text-xs font-bold text-slate-200">Drag & drop reference procedure MP4 video here</p>
                  <p className="text-[11px] text-slate-500 mt-1">Upload triggers Dataset Preparation → Adaptation → Validation</p>
                </div>
              </div>

            </div>

          </div>
        )}

      </main>
    </div>
  );
};
