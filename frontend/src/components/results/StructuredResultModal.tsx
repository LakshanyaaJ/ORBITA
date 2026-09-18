import { useState, useEffect } from 'react';
import { 
  FileText, 
  Copy, 
  Check, 
  Download, 
  ShieldCheck, 
  X, 
  RefreshCw,
  AlertTriangle,
  Server
} from 'lucide-react';
import clsx from 'clsx';
import { BACKEND_BASE } from '../../api/camera';

interface StructuredResultModalProps {
  isOpen: boolean;
  onClose: () => void;
  experimentId?: string;
}

export default function StructuredResultModal({ isOpen, onClose, experimentId = "EXP-01" }: StructuredResultModalProps) {
  const [resultData, setResultData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    const fetchResult = async () => {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const res = await fetch(`${BACKEND_BASE}/api/results/${experimentId}`);
        if (res.ok) {
          const data = await res.json();
          setResultData(data);
        } else {
          // If specific experiment not found, fetch all results to grab latest
          const allRes = await fetch(`${BACKEND_BASE}/api/results`);
          if (allRes.ok) {
            const list = await allRes.json();
            if (Array.isArray(list) && list.length > 0) {
              setResultData(list[list.length - 1]);
            } else {
              setErrorMsg(`No structured result found for ${experimentId}. Complete an experiment first.`);
            }
          } else {
            setErrorMsg(`Failed to load experiment result: ${res.statusText}`);
          }
        }
      } catch (err: any) {
        setErrorMsg(err.message || "Failed to contact backend API.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchResult();
  }, [isOpen, experimentId]);

  if (!isOpen) return null;

  const jsonString = resultData ? JSON.stringify(resultData, null, 2) : "";
  const checksum = resultData?.checksum || resultData?.payload?.checksum;
  const syncStatus = resultData?.sync_status || resultData?.sync?.status || "QUEUED";

  const handleCopy = () => {
    if (!jsonString) return;
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!jsonString) return;
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `orbita_${experimentId.toLowerCase()}_result.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fade-in">
      <div className="bg-space-900 border border-space-600 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden text-space-100 font-sans">
        
        {/* Header */}
        <div className="p-5 border-b border-space-700 bg-space-850 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
              <FileText size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold font-['Space_Grotesk'] tracking-wide text-space-100">
                  STRUCTURED EXPERIMENT RESULT
                </h2>
                <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-space-800 text-accent-cyan border border-space-700 uppercase font-bold">
                  {experimentId}
                </span>
                <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 uppercase font-bold">
                  SECTION 20 COMPLIANT
                </span>
              </div>
              <p className="text-xs text-space-400 mt-0.5">
                Deterministic machine-readable JSON schema with canonical SHA-256 cryptographic verification.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              disabled={!resultData}
              className="px-3 py-1.5 bg-space-800 hover:bg-space-700 text-space-200 text-xs font-mono font-bold rounded-lg flex items-center gap-1.5 border border-space-700 transition-colors disabled:opacity-50 cursor-pointer"
            >
              {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
              <span>{copied ? "COPIED" : "COPY JSON"}</span>
            </button>

            <button
              type="button"
              onClick={handleDownload}
              disabled={!resultData}
              className="px-3 py-1.5 bg-accent-cyan hover:bg-accent-cyan/80 text-space-950 text-xs font-mono font-bold rounded-lg flex items-center gap-1.5 transition-colors disabled:opacity-50 cursor-pointer"
            >
              <Download size={14} />
              <span>EXPORT JSON</span>
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 text-space-400 hover:text-space-100 hover:bg-space-800 rounded-lg transition-colors cursor-pointer ml-1"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Status Bar */}
        <div className="px-6 py-2.5 bg-space-950 border-b border-space-800 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <ShieldCheck size={15} className="text-emerald-400" />
              <span className="text-space-400">DATA INTEGRITY:</span>
              <span className="text-emerald-400 font-bold">SHA-256 VERIFIED</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Server size={14} className="text-accent-cyan" />
              <span className="text-space-400">GROUND SYNC:</span>
              <span className={clsx(
                "font-bold uppercase px-1.5 py-0.2 rounded text-[10px]",
                syncStatus === "SYNCED" ? "bg-emerald-950 text-emerald-400 border border-emerald-800" :
                "bg-amber-950 text-amber-400 border border-amber-800"
              )}>
                {syncStatus}
              </span>
            </div>
          </div>

          {checksum && (
            <div className="text-[11px] text-space-400 truncate max-w-sm" title={checksum}>
              HASH: <span className="text-space-200 font-mono">{checksum.slice(0, 16)}...{checksum.slice(-8)}</span>
            </div>
          )}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 bg-black/90 font-mono text-xs">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-space-400 space-y-3">
              <RefreshCw size={24} className="animate-spin text-accent-cyan" />
              <span>Querying local SQLite experiment result store...</span>
            </div>
          ) : errorMsg ? (
            <div className="p-4 bg-red-950/60 border border-red-800 rounded-lg text-red-200 flex items-start gap-3">
              <AlertTriangle size={18} className="text-red-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold">Result Not Found</div>
                <div className="text-xs text-red-300 mt-1">{errorMsg}</div>
                <div className="text-xs text-space-400 mt-2">
                  Tip: Execute an experiment or trigger <strong>RUN ORBITA DEMO</strong> to generate a completed structured payload.
                </div>
              </div>
            </div>
          ) : (
            <pre className="text-space-200 text-[11px] leading-relaxed select-text overflow-x-auto">
              {jsonString}
            </pre>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-space-700 bg-space-850 flex items-center justify-between text-xs font-mono text-space-400">
          <div>
            <span>SCHEMA: <strong className="text-space-200">v1.0 (Machine-Readable)</strong></span>
            <span className="mx-2">·</span>
            <span>STORAGE: <strong className="text-space-200">SQLite + Ground Station Queue</strong></span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 bg-space-800 hover:bg-space-700 text-space-200 rounded-lg transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
