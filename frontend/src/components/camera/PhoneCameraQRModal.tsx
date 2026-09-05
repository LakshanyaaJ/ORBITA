import { useState, useEffect, useRef } from 'react';
import QRCode from 'qrcode';
import {
  Smartphone,
  X,
  RefreshCw,
  Copy,
  Check,
  Wifi,
  ExternalLink,
  ShieldCheck,
} from 'lucide-react';
import {
  getPhonePairingInfo,
  refreshPhonePairingToken,
  type PhonePairingInfo,
} from '../../api/camera';

interface PhoneCameraQRModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConnected?: () => void;
}

export default function PhoneCameraQRModal({
  isOpen,
  onClose,
  onConnected,
}: PhoneCameraQRModalProps) {
  const [pairingInfo, setPairingInfo] = useState<PhonePairingInfo | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const pollTimerRef = useRef<any>(null);

  const fetchPairing = async () => {
    try {
      const info = await getPhonePairingInfo();
      if (info) {
        setPairingInfo(info);
        setFetchError(null);
        setLoading(false);
        if (info.connected) {
          onConnected?.();
        }
        try {
          const url = await QRCode.toDataURL(info.connection_url, {
            width: 240,
            margin: 1.5,
            color: {
              dark: '#030712',
              light: '#ffffff',
            },
          });
          setQrDataUrl(url);
        } catch (e) {
          console.error('QR code generation error', e);
        }
      } else {
        setFetchError('Unable to connect to ORBITA server on port 8000');
        setLoading(false);
      }
    } catch (err: any) {
      setFetchError(err?.message || 'Failed to reach backend');
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    const info = await refreshPhonePairingToken();
    if (info) {
      setPairingInfo(info);
      const url = await QRCode.toDataURL(info.connection_url, {
        width: 240,
        margin: 1.5,
        color: {
          dark: '#030712',
          light: '#ffffff',
        },
      });
      setQrDataUrl(url);
    }
    setRefreshing(false);
  };

  const copyUrl = () => {
    if (pairingInfo?.connection_url) {
      navigator.clipboard.writeText(pairingInfo.connection_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchPairing();
      pollTimerRef.current = setInterval(fetchPairing, 1800);
    } else {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    }
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-md bg-space-900 border border-space-600 rounded-xl shadow-2xl flex flex-col font-sans overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-space-700/80 flex items-center justify-between bg-space-800/40">
          <div className="flex items-center gap-2.5">
            <Smartphone size={18} className="text-accent-cyan" />
            <h3 className="font-mono text-sm font-bold tracking-wider text-space-100 uppercase">
              Connect Phone Webcam
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded text-space-400 hover:text-space-100 hover:bg-space-700/60 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 flex flex-col items-center text-center space-y-4">
          <p className="text-xs text-space-300 font-mono">
            Scan this QR code with your iPhone or Android camera to start streaming directly from your browser.
          </p>

          {/* QR Code Card */}
          <div className="p-3 bg-white rounded-lg shadow-inner flex items-center justify-center border border-space-500/30 relative">
            {qrDataUrl ? (
              <img
                src={qrDataUrl}
                alt="Phone Webcam Pairing QR"
                className="w-56 h-56 object-contain rounded"
              />
            ) : fetchError ? (
              <div className="w-56 h-56 flex flex-col items-center justify-center text-rose-600 font-mono text-xs p-3 text-center gap-2">
                <span className="text-red-500 font-bold">{fetchError}</span>
                <button
                  type="button"
                  onClick={fetchPairing}
                  className="px-3 py-1.5 bg-space-800 text-space-100 rounded text-xs font-bold hover:bg-space-700 transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="w-56 h-56 flex flex-col items-center justify-center text-space-900 font-mono text-xs gap-2">
                <RefreshCw size={20} className="animate-spin text-space-600" />
                <span>Generating QR...</span>
              </div>
            )}

            {/* Connected Overlay Badge */}
            {pairingInfo?.connected && (
              <div className="absolute inset-0 bg-space-950/85 backdrop-blur-sm rounded flex flex-col items-center justify-center text-status-success gap-2">
                <ShieldCheck size={40} className="animate-bounce" />
                <span className="font-mono font-bold text-sm tracking-widest uppercase">
                  PHONE CONNECTED!
                </span>
                <span className="text-xs text-space-300 font-mono">Streaming 720p @ 30 FPS</span>
              </div>
            )}
          </div>

          {/* 4-Digit Pairing Code Pill */}
          <div className="w-full bg-space-800 border border-space-700 rounded-lg p-3 flex items-center justify-between">
            <div className="text-left">
              <span className="text-[10px] text-space-400 font-mono uppercase block">
                PAIRING CODE
              </span>
              <span className="font-mono text-2xl font-black text-space-100 tracking-widest">
                {pairingInfo?.pairing_token || '----'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRefresh}
                disabled={refreshing}
                title="Regenerate Code"
                className="p-2 rounded bg-space-700 hover:bg-space-600 text-space-300 hover:text-accent-cyan text-xs font-mono transition-colors"
              >
                <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
              </button>
              <button
                type="button"
                onClick={copyUrl}
                className="py-1.5 px-3 rounded bg-accent-cyan/15 hover:bg-accent-cyan/25 border border-accent-cyan/40 text-accent-cyan font-mono text-xs font-bold flex items-center gap-1.5 transition-colors"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
                <span>{copied ? 'COPIED' : 'COPY URL'}</span>
              </button>
            </div>
          </div>

          {/* Direct URL text */}
          <div className="w-full text-left font-mono text-[11px] text-space-400 bg-space-950/50 p-2.5 rounded border border-space-800 break-all flex items-center justify-between gap-2">
            <span className="truncate">{pairingInfo?.connection_url || 'http://...'}</span>
            <a
              href={pairingInfo?.connection_url}
              target="_blank"
              rel="noreferrer"
              title="Open directly in new tab"
              className="text-accent-cyan hover:underline flex-shrink-0 flex items-center gap-1"
            >
              <span>Test</span>
              <ExternalLink size={12} />
            </a>
          </div>

          {/* Instructions checklist */}
          <div className="w-full text-left space-y-1.5 font-mono text-[11px] text-space-300 bg-space-800/30 p-3 rounded border border-space-700/60">
            <div className="font-bold text-space-200 text-xs mb-1">QUICK INSTRUCTIONS:</div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">1</span>
              <span>Ensure Phone and Jetson are on the same Wi-Fi.</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">2</span>
              <span>Scan QR code with phone camera (Safari/Chrome).</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">3</span>
              <span>Tap <b>START CAMERA STREAM</b>. Live feed begins instantly.</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 bg-space-800/50 border-t border-space-700/60 flex items-center justify-between text-xs font-mono">
          <span className="flex items-center gap-1.5 text-space-400">
            <Wifi size={13} className="text-status-success" />
            <span>LAN IP: {pairingInfo?.lan_ip || '127.0.0.1'}</span>
          </span>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded bg-space-700 hover:bg-space-600 text-space-200 font-bold transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
