import { useState, useEffect } from 'react';
import { Eye, Film, Play, Pause, SkipForward, SkipBack, Cpu, Layers, CheckCircle2 } from 'lucide-react';

interface VdataVideo {
  filename: string;
  path: string;
  duration_seconds: number;
  total_frames: number;
  fps: number;
  width: number;
  height: number;
  resolution: string;
  size_mb: number;
}

interface DetectionItem {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  source: string;
}

interface InspectFrameResult {
  filename: string;
  frame_idx: number;
  total_frames: number;
  latency_ms: number;
  model: string;
  detections_count: number;
  detections: DetectionItem[];
  image_data: string;
}

const CLASS_COLOR_MAP: Record<string, { border: string; bg: string; text: string }> = {
  PERSON: { border: 'border-emerald-500', bg: 'bg-emerald-500/20', text: 'text-emerald-400' },
  MAIN_BOX: { border: 'border-sky-500', bg: 'bg-sky-500/20', text: 'text-sky-400' },
  RED_BOX: { border: 'border-red-500', bg: 'bg-red-500/20', text: 'text-red-400' },
  YELLOW_BOX: { border: 'border-amber-400', bg: 'bg-amber-400/20', text: 'text-amber-300' },
  SAMPLE: { border: 'border-purple-500', bg: 'bg-purple-500/20', text: 'text-purple-400' },
  TOOL: { border: 'border-cyan-500', bg: 'bg-cyan-500/20', text: 'text-cyan-400' },
};

export default function VdataVideoInspector() {
  const [videos, setVideos] = useState<VdataVideo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [mode, setMode] = useState<'stream' | 'scrub'>('stream');
  const [isPlayingStream, setIsPlayingStream] = useState<boolean>(true);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [frameData, setFrameData] = useState<InspectFrameResult | null>(null);
  const [loadingFrame, setLoadingFrame] = useState<boolean>(false);
  const [streamKey, setStreamKey] = useState<number>(Date.now());

  // Fetch reference videos
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/vdata/videos');
        if (res.ok) {
          const data: VdataVideo[] = await res.json();
          setVideos(data);
          if (data.length > 0 && !selectedVideo) {
            setSelectedVideo(data[0].filename);
          }
        }
      } catch (err) {
        console.error('Failed to load vdata videos:', err);
      }
    };
    fetchVideos();
  }, []);

  // Fetch frame when scrubbing
  useEffect(() => {
    if (mode !== 'scrub' || !selectedVideo) return;

    let isMounted = true;
    const inspectFrame = async () => {
      setLoadingFrame(true);
      try {
        const res = await fetch('http://localhost:8000/api/vdata/inspect_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: selectedVideo, frame_idx: currentFrameIdx }),
        });
        if (res.ok && isMounted) {
          const data = await res.json();
          setFrameData(data);
        }
      } catch (err) {
        console.error('Failed to inspect frame:', err);
      } finally {
        if (isMounted) setLoadingFrame(false);
      }
    };

    const timeout = setTimeout(inspectFrame, 150);
    return () => {
      isMounted = false;
      clearTimeout(timeout);
    };
  }, [mode, selectedVideo, currentFrameIdx]);

  const activeVideoMeta = videos.find((v) => v.filename === selectedVideo);
  const totalFrames = activeVideoMeta?.total_frames || 100;

  const handleVideoSelect = (fname: string) => {
    setSelectedVideo(fname);
    setCurrentFrameIdx(0);
    setStreamKey(Date.now());
  };

  return (
    <div className="bg-space-900 border border-space-600 rounded-xl p-6 mt-8 shadow-2xl font-mono text-space-100">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 mb-6 border-b border-space-700 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-accent-cyan" />
            <h3 className="text-lg font-bold tracking-widest text-space-100">
              REFERENCE VIDEO YOLO INSPECTOR
            </h3>
          </div>
          <p className="text-xs text-space-400 mt-1">
            Observe real-time YOLO object detection & bounding boxes across reference videos in <span className="text-accent-cyan">vdata/</span>
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center bg-space-800 p-1 rounded-lg border border-space-600 self-start md:self-auto">
          <button
            onClick={() => setMode('stream')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-bold transition-colors ${
              mode === 'stream'
                ? 'bg-accent-cyan text-space-900 shadow-md'
                : 'text-space-300 hover:text-white'
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            Live Detection Stream
          </button>
          <button
            onClick={() => setMode('scrub')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-bold transition-colors ${
              mode === 'scrub'
                ? 'bg-accent-cyan text-space-900 shadow-md'
                : 'text-space-300 hover:text-white'
            }`}
          >
            <Film className="w-3.5 h-3.5" />
            Frame-by-Frame Scrubber
          </button>
        </div>
      </div>

      {/* Video Selector Pills */}
      <div className="mb-6">
        <label className="text-xs text-space-400 block mb-2 tracking-wider">
          SELECT REFERENCE VIDEO FROM VDATA:
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {videos.map((vid) => {
            const isSelected = vid.filename === selectedVideo;
            return (
              <button
                key={vid.filename}
                onClick={() => handleVideoSelect(vid.filename)}
                className={`text-left p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'border-accent-cyan bg-accent-cyan/10 shadow-lg shadow-accent-cyan/5'
                    : 'border-space-700 bg-space-800/80 hover:border-space-500'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-xs truncate max-w-[160px] text-white">
                    {vid.filename}
                  </span>
                  {isSelected && <CheckCircle2 className="w-4 h-4 text-accent-cyan shrink-0" />}
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] text-space-400">
                  <span className="bg-space-900/80 px-1.5 py-0.5 rounded border border-space-700">
                    {vid.resolution}
                  </span>
                  <span className="bg-space-900/80 px-1.5 py-0.5 rounded border border-space-700">
                    {vid.duration_seconds}s
                  </span>
                  <span className="bg-space-900/80 px-1.5 py-0.5 rounded border border-space-700">
                    {vid.size_mb} MB
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Viewport & Detection Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player Display */}
        <div className="lg:col-span-2 flex flex-col bg-space-950 border border-space-700 rounded-lg overflow-hidden shadow-inner">
          <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
            {mode === 'stream' && selectedVideo ? (
              isPlayingStream ? (
                <img
                  key={`${selectedVideo}-${streamKey}`}
                  src={`http://localhost:8000/api/vdata/stream/${selectedVideo}?t=${streamKey}`}
                  alt="YOLO Reference Stream"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-space-400 gap-2">
                  <Pause className="w-10 h-10" />
                  <span className="text-xs">Stream Paused</span>
                </div>
              )
            ) : mode === 'scrub' ? (
              frameData?.image_data ? (
                <img
                  src={frameData.image_data}
                  alt={`Frame ${currentFrameIdx}`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-space-400 gap-2">
                  <Film className="w-8 h-8 animate-pulse text-accent-cyan" />
                  <span className="text-xs">Loading frame {currentFrameIdx}...</span>
                </div>
              )
            ) : (
              <div className="text-space-500 text-xs">No video selected</div>
            )}

            {/* Overlaid Badges */}
            <div className="absolute top-3 left-3 bg-space-900/90 backdrop-blur-md border border-space-600 px-2.5 py-1 rounded text-[11px] font-bold flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              YOLO ACTIVE DETECTION
            </div>

            <div className="absolute top-3 right-3 bg-space-900/90 backdrop-blur-md border border-space-600 px-2.5 py-1 rounded text-[11px] text-space-300">
              {mode === 'stream' ? 'LIVE CONTINUOUS LOOP' : `FRAME ${currentFrameIdx} / ${totalFrames}`}
            </div>
          </div>

          {/* Controls Bar */}
          <div className="p-4 bg-space-900 border-t border-space-700 flex flex-col gap-3">
            {mode === 'stream' ? (
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setIsPlayingStream(!isPlayingStream)}
                  className="flex items-center gap-2 px-4 py-2 bg-space-800 hover:bg-space-700 border border-space-600 rounded text-xs font-bold transition-all text-white"
                >
                  {isPlayingStream ? <Pause className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-emerald-400" />}
                  {isPlayingStream ? 'PAUSE STREAM' : 'RESUME STREAM'}
                </button>
                <button
                  onClick={() => setStreamKey(Date.now())}
                  className="text-xs text-space-400 hover:text-accent-cyan transition-colors underline"
                >
                  Restart Loop
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setCurrentFrameIdx((prev) => Math.max(0, prev - 10))}
                    className="p-1.5 bg-space-800 hover:bg-space-700 border border-space-600 rounded text-xs"
                    title="Back 10 frames"
                  >
                    <SkipBack className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setCurrentFrameIdx((prev) => Math.max(0, prev - 1))}
                    className="px-2.5 py-1.5 bg-space-800 hover:bg-space-700 border border-space-600 rounded text-xs font-bold"
                  >
                    -1
                  </button>
                  <input
                    type="range"
                    min={0}
                    max={totalFrames - 1}
                    value={currentFrameIdx}
                    onChange={(e) => setCurrentFrameIdx(Number(e.target.value))}
                    className="flex-1 accent-accent-cyan cursor-pointer"
                  />
                  <button
                    onClick={() => setCurrentFrameIdx((prev) => Math.min(totalFrames - 1, prev + 1))}
                    className="px-2.5 py-1.5 bg-space-800 hover:bg-space-700 border border-space-600 rounded text-xs font-bold"
                  >
                    +1
                  </button>
                  <button
                    onClick={() => setCurrentFrameIdx((prev) => Math.min(totalFrames - 1, prev + 10))}
                    className="p-1.5 bg-space-800 hover:bg-space-700 border border-space-600 rounded text-xs"
                    title="Forward 10 frames"
                  >
                    <SkipForward className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex justify-between text-[11px] text-space-400">
                  <span>Frame: {currentFrameIdx}</span>
                  <span>{loadingFrame ? 'Evaluating YOLO inference...' : `Inference: ${frameData?.latency_ms || 0}ms`}</span>
                  <span>Total: {totalFrames}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Detections & Class Legend Sidebar */}
        <div className="flex flex-col gap-4">
          {/* Active Model Status Card */}
          <div className="bg-space-800 border border-space-700 rounded-lg p-4">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="text-space-400 flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-accent-cyan" /> ACTIVE DETECTOR
              </span>
              <span className="text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                ONLINE
              </span>
            </div>
            <div className="text-sm font-bold text-white truncate">
              {frameData?.model || 'models/yolov8n.pt'}
            </div>
            <div className="mt-2 text-[11px] text-space-400 flex justify-between">
              <span>Target Source:</span>
              <span className="text-accent-cyan font-semibold">{selectedVideo}</span>
            </div>
          </div>

          {/* Real-Time Detected Objects List */}
          <div className="bg-space-800 border border-space-700 rounded-lg p-4 flex-1 flex flex-col">
            <div className="flex items-center justify-between pb-3 border-b border-space-700 mb-3">
              <span className="text-xs font-bold tracking-wider text-space-200 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-accent-cyan" /> DETECTED ENTITIES
              </span>
              <span className="text-xs bg-space-700 px-2 py-0.5 rounded-full font-bold text-accent-cyan">
                {mode === 'scrub' ? frameData?.detections_count || 0 : 'Active'}
              </span>
            </div>

            {mode === 'scrub' && frameData?.detections ? (
              frameData.detections.length === 0 ? (
                <div className="text-xs text-space-400 py-6 text-center italic">
                  No objects detected above confidence threshold in this frame.
                </div>
              ) : (
                <div className="space-y-2.5 overflow-y-auto max-h-56 pr-1">
                  {frameData.detections.map((det, idx) => {
                    const style = CLASS_COLOR_MAP[det.class_name] || {
                      border: 'border-space-500',
                      bg: 'bg-space-700',
                      text: 'text-white',
                    };
                    return (
                      <div
                        key={idx}
                        className={`p-2.5 rounded border ${style.border} ${style.bg} flex flex-col gap-1`}
                      >
                        <div className="flex justify-between items-center">
                          <span className={`font-bold text-xs ${style.text}`}>
                            {det.class_name}
                          </span>
                          <span className="text-[11px] font-bold text-white">
                            {(det.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-space-900 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="h-full bg-current opacity-80"
                            style={{ width: `${det.confidence * 100}%` }}
                          />
                        </div>
                        <div className="text-[10px] text-space-300 flex justify-between mt-0.5">
                          <span>Box: [{det.bbox.join(', ')}]</span>
                          <span className="uppercase text-[9px] opacity-70">via {det.source}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )
            ) : (
              <div className="text-xs text-space-300 space-y-3 py-2">
                <p>
                  Observing live YOLO bounding box stream.
                </p>
                <div className="p-3 bg-space-900/60 rounded border border-space-700 text-[11px] space-y-1.5">
                  <div className="flex justify-between">
                    <span className="text-space-400">Stream Transport:</span>
                    <span className="text-emerald-400 font-bold">MJPEG 30 FPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-400">Detection Layer:</span>
                    <span>YOLOv8 + Chroma</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-space-400">HUD Overlay:</span>
                    <span>Active (<span className="text-accent-cyan">Boxes + Centroids + Labels</span>)</span>
                  </div>
                </div>
              </div>
            )}

            {/* Color Legend */}
            <div className="mt-auto pt-4 border-t border-space-700">
              <span className="text-[10px] text-space-400 block mb-2 tracking-wider">
                ONTOLOGY COLOR KEY:
              </span>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                {Object.entries(CLASS_COLOR_MAP).map(([cls, style]) => (
                  <div key={cls} className="flex items-center gap-1.5">
                    <span className={`w-2.5 h-2.5 rounded-full border ${style.border} ${style.bg}`} />
                    <span className={style.text}>{cls}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
