import { useState, useEffect, useRef, useMemo } from 'react';
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
  Lock,
  Globe,
  AlertTriangle,
  Edit3,
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
  const [selectedIp, setSelectedIp] = useState<string>('');
  const [isCustomIp, setIsCustomIp] = useState<boolean>(false);
  const [customIpInput, setCustomIpInput] = useState<string>('');
  const [protocol, setProtocol] = useState<'https' | 'http'>('https');
  const [qrDataUrl, setQrDataUrl] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const pollTimerRef = useRef<any>(null);

  // Compute the active IP to use
  const activeIp = useMemo(() => {
    if (isCustomIp && customIpInput.trim()) {
      return customIpInput.trim();
    }
    return selectedIp || pairingInfo?.lan_ip || '127.0.0.1';
  }, [isCustomIp, customIpInput, selectedIp, pairingInfo]);

  // Compute the live target URL for phone browser
  const activeUrl = useMemo(() => {
    const token = pairingInfo?.pairing_token || '----';
    const port = protocol === 'https' ? (pairingInfo?.https_port || 8443) : (pairingInfo?.port || 8000);
    return `${protocol}://${activeIp}:${port}/cam?token=${token}`;
  }, [protocol, activeIp, pairingInfo]);

  // Regenerate QR code only when activeUrl changes
  useEffect(() => {
    if (!activeUrl || !isOpen) return;

    QRCode.toDataURL(activeUrl, {
      width: 260,
      margin: 2,
      color: {
        dark: '#030712',
        light: '#ffffff',
      },
    })
      .then((url) => setQrDataUrl(url))
      .catch((err) => console.error('QR code generation error:', err));
  }, [activeUrl, isOpen]);

  // Fetch pairing info on mount and poll for connection status
  const fetchPairing = async (isFirstLoad = false) => {
    try {
      const info = await getPhonePairingInfo();
      if (info) {
        setPairingInfo(info);
        setFetchError(null);
        setLoading(false);

        // On first load, select the default LAN IP
        if (isFirstLoad && !selectedIp) {
          setSelectedIp(info.lan_ip || '127.0.0.1');
        }

        if (info.connected) {
          onConnected?.();
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
    }
    setRefreshing(false);
  };

  const copyUrl = () => {
    if (activeUrl) {
      navigator.clipboard.writeText(activeUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchPairing(true);
      pollTimerRef.current = setInterval(() => fetchPairing(false), 2000);
    } else {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    }
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const availableIps = pairingInfo?.available_ips || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-space-950/85 backdrop-blur-md animate-in fade-in duration-150 overflow-y-auto">
      <div className="w-full max-w-lg bg-space-900 border border-space-600 rounded-2xl shadow-2xl flex flex-col font-sans overflow-hidden my-8">
        {/* Header */}
        <div className="px-6 py-4 border-b border-space-700/80 flex items-center justify-between bg-space-800/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30">
              <Smartphone size={20} />
            </div>
            <div>
              <h3 className="font-mono text-sm font-bold tracking-wider text-space-100 uppercase">
                Connect Phone Webcam
              </h3>
              <p className="text-[11px] text-space-400 font-mono">
                Wireless ultra-low latency camera pairing
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-space-400 hover:text-space-100 hover:bg-space-700/60 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 flex flex-col items-center text-center space-y-4 max-h-[80vh] overflow-y-auto">
          {/* Protocol Switcher (HTTPS vs HTTP) */}
          <div className="w-full flex items-center justify-center gap-2 p-1 bg-space-950/80 rounded-xl border border-space-800">
            <button
              type="button"
              onClick={() => setProtocol('https')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-mono font-bold transition-all ${
                protocol === 'https'
                  ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40 shadow-sm'
                  : 'text-space-400 hover:text-space-200'
              }`}
            >
              <Lock size={13} />
              <span>HTTPS (Port 8443) - Recommended</span>
            </button>
            <button
              type="button"
              onClick={() => setProtocol('http')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-mono font-bold transition-all ${
                protocol === 'http'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                  : 'text-space-400 hover:text-space-200'
              }`}
            >
              <Globe size={13} />
              <span>HTTP (Port 8000)</span>
            </button>
          </div>

          {/* QR Code Card */}
          <div className="p-3.5 bg-white rounded-2xl shadow-xl flex items-center justify-center border-2 border-space-500/30 relative">
            {qrDataUrl ? (
              <img
                src={qrDataUrl}
                alt="Phone Webcam Pairing QR"
                className="w-56 h-56 object-contain rounded-lg"
              />
            ) : fetchError ? (
              <div className="w-56 h-56 flex flex-col items-center justify-center text-rose-600 font-mono text-xs p-3 text-center gap-2">
                <span className="text-red-500 font-bold">{fetchError}</span>
                <button
                  type="button"
                  onClick={() => fetchPairing(true)}
                  className="px-3 py-1.5 bg-space-800 text-space-100 rounded text-xs font-bold hover:bg-space-700 transition-colors"
                >
                  Retry
                </button>
              </div>
            ) : loading || !qrDataUrl ? (
              <div className="w-56 h-56 flex flex-col items-center justify-center text-space-900 font-mono text-xs gap-2">
                <RefreshCw size={24} className="animate-spin text-space-600" />
                <span>Generating Pairing QR...</span>
              </div>
            ) : null}

            {/* Connected Overlay Badge */}
            {pairingInfo?.connected && (
              <div className="absolute inset-0 bg-space-950/90 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center text-status-success gap-2 animate-in fade-in">
                <ShieldCheck size={44} className="animate-bounce" />
                <span className="font-mono font-bold text-sm tracking-widest uppercase">
                  PHONE CONNECTED!
                </span>
                <span className="text-xs text-space-300 font-mono">
                  Streaming {pairingInfo.resolution} @ {Math.round(pairingInfo.fps)} FPS
                </span>
              </div>
            )}
          </div>

          {/* 4-Digit Pairing Code Pill */}
          <div className="w-full bg-space-800/80 border border-space-700 rounded-xl p-3 flex items-center justify-between">
            <div className="text-left">
              <span className="text-[10px] text-space-400 font-mono uppercase tracking-wider block">
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
                className="p-2.5 rounded-lg bg-space-700 hover:bg-space-600 text-space-300 hover:text-accent-cyan text-xs font-mono transition-colors"
              >
                <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
              </button>
              <button
                type="button"
                onClick={copyUrl}
                className="py-2 px-3.5 rounded-lg bg-accent-cyan/15 hover:bg-accent-cyan/25 border border-accent-cyan/40 text-accent-cyan font-mono text-xs font-bold flex items-center gap-1.5 transition-colors"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
                <span>{copied ? 'COPIED' : 'COPY URL'}</span>
              </button>
            </div>
          </div>

          {/* IP Address & Network Interface Selector */}
          <div className="w-full text-left font-mono text-xs bg-space-950/60 p-3.5 rounded-xl border border-space-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-space-400 uppercase tracking-wider flex items-center gap-1.5">
                <Wifi size={12} className="text-status-success" />
                Network IP Address
              </span>
              <button
                type="button"
                onClick={() => {
                  setIsCustomIp(!isCustomIp);
                  if (!isCustomIp && !customIpInput) {
                    setCustomIpInput(activeIp);
                  }
                }}
                className="text-[10px] text-accent-cyan hover:underline flex items-center gap-1 font-sans"
              >
                <Edit3 size={11} />
                <span>{isCustomIp ? 'Use Auto IP' : 'Enter Custom IP'}</span>
              </button>
            </div>

            {isCustomIp ? (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customIpInput}
                  onChange={(e) => setCustomIpInput(e.target.value)}
                  placeholder="e.g. 192.168.137.1"
                  className="flex-1 bg-space-900 border border-space-700 rounded-lg px-3 py-1.5 text-xs text-space-100 font-mono focus:outline-none focus:border-accent-cyan"
                />
              </div>
            ) : availableIps.length > 0 ? (
              <select
                value={selectedIp}
                onChange={(e) => setSelectedIp(e.target.value)}
                className="w-full bg-space-900 border border-space-700 rounded-lg px-3 py-1.5 text-xs text-space-100 font-mono focus:outline-none focus:border-accent-cyan cursor-pointer"
              >
                {availableIps.map((iface) => (
                  <option key={iface.ip} value={iface.ip}>
                    {iface.interface}: {iface.ip} {iface.is_default ? '(Default)' : ''}
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-space-200 font-mono text-xs px-2 py-1 bg-space-900 rounded border border-space-800">
                {activeIp}
              </div>
            )}
          </div>

          {/* Direct URL Preview */}
          <div className="w-full text-left font-mono text-[11px] text-space-400 bg-space-950/50 p-3 rounded-xl border border-space-800 break-all flex items-center justify-between gap-2">
            <span className="truncate text-space-300">{activeUrl}</span>
            <a
              href={activeUrl}
              target="_blank"
              rel="noreferrer"
              title="Open directly in browser tab"
              className="text-accent-cyan hover:underline flex-shrink-0 flex items-center gap-1 text-xs font-bold"
            >
              <span>Test</span>
              <ExternalLink size={12} />
            </a>
          </div>

          {/* SSL Notice for Mobile Safari / Chrome */}
          {protocol === 'https' && (
            <div className="w-full text-left font-mono text-[11px] text-amber-200/90 bg-amber-950/30 border border-amber-700/40 p-3 rounded-xl flex items-start gap-2.5">
              <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-amber-300 block mb-0.5">Phone Browser Security Notice:</span>
                Since ORBITA runs locally on your Wi-Fi, your phone browser will show a self-signed certificate warning (e.g. <i>"Connection Not Private"</i>).
                Tap <b className="text-white">Advanced</b> ➔ <b className="text-white">Proceed to IP (unsafe)</b> to start the live camera.
              </div>
            </div>
          )}

          {/* Quick Instructions */}
          <div className="w-full text-left space-y-1.5 font-mono text-[11px] text-space-300 bg-space-800/30 p-3.5 rounded-xl border border-space-700/60">
            <div className="font-bold text-space-200 text-xs mb-1">QUICK INSTRUCTIONS:</div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">1</span>
              <span>Connect your phone to the same Wi-Fi or hotspot.</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">2</span>
              <span>Scan QR code with phone camera (Safari / Chrome).</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-space-700 text-accent-cyan flex items-center justify-center text-[10px] font-bold">3</span>
              <span>Tap <b>START CAMERA STREAM</b>. Live feed starts in landscape!</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-space-800/50 border-t border-space-700/60 flex items-center justify-between text-xs font-mono">
          <span className="flex items-center gap-1.5 text-space-400">
            <Wifi size={13} className="text-status-success" />
            <span>Active IP: {activeIp}</span>
          </span>

          <button
            type="button"
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-space-700 hover:bg-space-600 text-space-100 font-bold transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
