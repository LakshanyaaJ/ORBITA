import { useParams } from 'react-router-dom';
import { Download, CheckCircle, AlertTriangle } from 'lucide-react';

export default function ExperimentLog() {
  const { id } = useParams();
  
  // Dummy data representing logs
  const logs = [
    { time: '14:21:02', step: '01', name: 'PREPARE CONTAINER', status: 'COMPLETED', type: 'success' },
    { time: '14:22:18', step: '02', name: 'ADD WATER', status: 'COMPLETED', type: 'success' },
    { time: '14:24:43', step: '03', name: 'ADD SEEDS', status: 'COMPLETED', type: 'success' },
    { time: '14:26:10', step: '04', name: 'CLOSE CONTAINER', status: 'UNEXPECTED ACTION', type: 'warning' },
    { time: '14:26:18', step: '04', name: 'CLOSE CONTAINER', status: 'COMPLETED', type: 'success' },
  ];

  return (
    <div className="p-8 max-w-4xl mx-auto h-full flex flex-col">
      <div className="flex items-center justify-between border-b border-space-600 pb-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold font-mono tracking-widest text-space-100 mb-1">
            EXPERIMENT LOG
          </h1>
          <div className="font-mono text-accent-cyan tracking-widest">{id || 'EXP-04'}</div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-space-800 border border-space-600 hover:border-space-400 rounded transition-colors text-sm font-mono tracking-widest">
          <Download size={16} />
          EXPORT LOG
        </button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4">
        {logs.map((log, i) => (
          <div key={i} className="flex flex-col sm:flex-row sm:items-center gap-4 bg-space-800 border border-space-600 p-4 rounded-lg font-mono">
            <div className="text-space-400 text-sm">{log.time}</div>
            <div className="flex-1">
              <div className="text-xs text-space-400 tracking-widest mb-1">STEP {log.step}</div>
              <div className="text-space-100 font-bold">{log.name}</div>
            </div>
            <div className={`flex items-center gap-2 font-bold tracking-widest text-sm ${
              log.type === 'success' ? 'text-status-success' : 'text-status-warning'
            }`}>
              {log.type === 'success' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
              {log.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
