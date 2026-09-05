import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ScanFace, Fingerprint, Lock } from 'lucide-react';
import GoogleAppsButton from '../components/common/GoogleAppsButton';


export default function AuthPage() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState<'IDLE' | 'SCANNING' | 'VERIFIED' | 'FAILED'>('IDLE');

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (phase === 'SCANNING') {
      timer = setTimeout(() => {
        setPhase('VERIFIED');
      }, 2500);
    } else if (phase === 'VERIFIED') {
      timer = setTimeout(() => {
        navigate('/mission');
      }, 1500);
    }
    return () => clearTimeout(timer);
  }, [phase, navigate]);

  return (
    <div className="min-h-screen bg-space-900 flex flex-col items-center justify-center relative overflow-hidden font-sans text-space-100">
      {/* Top-Right Mission Control Launcher */}
      <div className="absolute top-4 right-6 z-20 flex items-center gap-2">
        <GoogleAppsButton />
      </div>
      {/* Background Technical Illustration */}
      <div className="absolute inset-0 opacity-5 pointer-events-none flex items-center justify-center">
        <svg viewBox="0 0 800 800" className="w-[800px] h-[800px] text-accent-cyan">
          <circle cx="400" cy="400" r="300" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 8" />
          <circle cx="400" cy="400" r="250" fill="none" stroke="currentColor" strokeWidth="1" />
          <line x1="100" y1="400" x2="700" y2="400" stroke="currentColor" strokeWidth="1" strokeDasharray="2 4" />
          <line x1="400" y1="100" x2="400" y2="700" stroke="currentColor" strokeWidth="1" strokeDasharray="2 4" />
        </svg>
      </div>

      <div className="z-10 flex flex-col items-center">
        <h1 className="text-4xl font-bold tracking-widest font-mono mb-2">ORBITA</h1>
        <p className="text-space-400 tracking-[0.2em] text-sm mb-16 uppercase">Onboard Experiment Operations</p>

        {/* Camera / Scan Area */}
        <div className="relative w-72 h-72 mb-12">
          {/* Frame Corners */}
          <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-space-400"></div>
          <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-space-400"></div>
          <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-space-400"></div>
          <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-space-400"></div>

          <div className="absolute inset-0 flex items-center justify-center">
            {phase === 'IDLE' && (
              <button 
                onClick={() => setPhase('SCANNING')}
                className="w-48 h-48 rounded-full bg-space-800 border border-space-600 flex flex-col items-center justify-center gap-4 hover:border-accent-cyan hover:text-accent-cyan transition-colors"
              >
                <ScanFace size={48} strokeWidth={1.5} />
                <span className="font-mono text-sm tracking-widest">START SCAN</span>
              </button>
            )}

            {phase === 'SCANNING' && (
              <div className="w-48 h-48 rounded-full border-2 border-accent-cyan flex flex-col items-center justify-center gap-4 relative">
                <div className="absolute inset-0 rounded-full border-t-2 border-transparent border-r-accent-cyan animate-spin"></div>
                <ScanFace size={48} className="text-accent-cyan" strokeWidth={1.5} />
                <span className="font-mono text-sm tracking-widest text-accent-cyan animate-pulse">VERIFYING...</span>
              </div>
            )}

            {phase === 'VERIFIED' && (
              <div className="w-48 h-48 rounded-full bg-status-success/10 border-2 border-status-success flex flex-col items-center justify-center gap-4 text-status-success">
                <ScanFace size={48} strokeWidth={1.5} />
                <span className="font-mono text-sm tracking-widest">AUTHORIZED</span>
              </div>
            )}

            {phase === 'FAILED' && (
              <div className="w-48 h-48 rounded-full bg-status-critical/10 border-2 border-status-critical flex flex-col items-center justify-center gap-4 text-status-critical cursor-pointer" onClick={() => setPhase('IDLE')}>
                <ScanFace size={48} strokeWidth={1.5} />
                <span className="font-mono text-sm tracking-widest">REJECTED</span>
              </div>
            )}
          </div>
        </div>

        {/* Fallbacks */}
        <div className="flex gap-8 text-space-400">
          <button className="flex flex-col items-center gap-2 hover:text-space-100 transition-colors">
            <Fingerprint size={24} />
            <span className="font-mono text-xs tracking-wider">FINGERPRINT</span>
          </button>
          <button className="flex flex-col items-center gap-2 hover:text-space-100 transition-colors">
            <Lock size={24} />
            <span className="font-mono text-xs tracking-wider">CREDENTIALS</span>
          </button>
        </div>
      </div>
    </div>
  );
}
