import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../mission-control.css";
import CameraControlPanel from "../../components/camera/CameraControlPanel";
import { BACKEND_BASE, getCameraStatus, connectCamera } from "../../api/camera";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Bell,
  Check,
  ChevronRight,
  CircleGauge,
  Cpu,
  Eye,
  House,
  LayoutGrid,
  Menu,
  Monitor,
  Radio,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Thermometer,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type NavItem = {
  label: string;
  icon: typeof House;
};

type EventTone = "warning" | "verified" | "info" | "critical";

type EventItem = {
  time: string;
  title: string;
  step: string;
  tone: EventTone;
};

const navItems: NavItem[] = [
  { label: "Dashboard", icon: House },
  { label: "Live Monitoring", icon: Monitor },
  { label: "Procedure", icon: ShieldCheck },
  { label: "Alerts", icon: Bell },
  { label: "Analytics", icon: BarChart3 },
  { label: "Mission", icon: Radio },
  { label: "Settings", icon: Settings2 },
];

const baseEvents: EventItem[] = [
  { time: "10:42:18", title: "Procedure deviation", step: "Step 04", tone: "warning" },
  { time: "10:41:52", title: "Tool pickup verified", step: "Step 03", tone: "verified" },
  { time: "10:40:12", title: "Procedure started", step: "Step 01", tone: "info" },
];

const procedureSteps = [
  { label: "Verify equipment", state: "completed" },
  { label: "Open container", state: "completed" },
  { label: "Retrieve tool", state: "completed" },
  { label: "Position tool", state: "current" },
  { label: "Activate mechanism", state: "pending" },
  { label: "Verify activation", state: "pending" },
  { label: "Secure equipment", state: "pending" },
  { label: "Complete procedure", state: "pending" },
];

const complianceData = [
  { day: "01", value: 91 },
  { day: "02", value: 93 },
  { day: "03", value: 92 },
  { day: "04", value: 95 },
  { day: "05", value: 94 },
  { day: "06", value: 96 },
  { day: "07", value: 94.6 },
];

const confidenceData = [
  { time: "10:36", value: 94 },
  { time: "10:37", value: 95 },
  { time: "10:38", value: 96 },
  { time: "10:39", value: 95 },
  { time: "10:40", value: 97 },
  { time: "10:41", value: 96 },
  { time: "10:42", value: 97.4 },
];

function StatusPill({ tone, children }: { tone: EventTone | "online"; children: React.ReactNode }) {
  return <span className={`status-pill status-${tone}`}>{children}</span>;
}

function SectionHeading({ eyebrow, title, action }: { eyebrow?: string; title: string; action?: React.ReactNode }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

function MetricBlock({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail?: string; tone?: string }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{label}</div>
      <div className={`metric-value metric-${tone}`}>{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  );
}

function TelemetryCard({ icon: Icon, label, value, detail, tone }: { icon: typeof Cpu; label: string; value: string; detail: string; tone: string }) {
  return (
    <div className="telemetry-card">
      <div className={`telemetry-icon ${tone}`}><Icon size={16} strokeWidth={1.8} /></div>
      <div className="telemetry-copy">
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
      <span className="telemetry-detail">{detail}</span>
    </div>
  );
}

function CameraFeed({ 
  demoMode, 
  confidence,
  onExpand,
  isExpanded = false
}: { 
  demoMode: boolean; 
  confidence: number;
  onExpand?: () => void;
  isExpanded?: boolean;
}) {
  const [streamError, setStreamError] = useState(false);
  const [streamVersion, setStreamVersion] = useState(Date.now());
  const [cameraSource, setCameraSource] = useState<string>("sim");
  const [cameraFps, setCameraFps] = useState<number>(30);

  useEffect(() => {
    const pollStatus = () => {
      fetch(`${BACKEND_BASE}/api/camera/status`)
        .then((r) => r.json())
        .then((data) => {
          if (data.source) setCameraSource(data.source);
          if (data.fps) setCameraFps(data.fps);
          if (streamError && data.connected) {
            setStreamError(false);
            setStreamVersion(Date.now());
          }
        })
        .catch(() => {});
    };
    pollStatus();
    const interval = setInterval(pollStatus, 2500);
    return () => clearInterval(interval);
  }, [streamError]);

  useEffect(() => {
    setStreamError(false);
    setStreamVersion(Date.now());
  }, [demoMode]);

  const sourceName = cameraSource === "ip_camera" 
    ? "PHONE IP STREAM" 
    : cameraSource === "jetson_camera" 
    ? "JETSON CSI STREAM" 
    : demoMode 
    ? "SIMULATION FEED" 
    : "LIVE FEED";

  return (
    <div className="camera-frame">
      <div 
        className="camera-surface"
        style={isExpanded ? { height: "460px", position: "relative", overflow: "hidden", borderRadius: "4px" } : { height: "357px", position: "relative", overflow: "hidden", borderRadius: "4px" }}
      >
        {/* Real Live Camera MJPEG Feed from ORBITA backend */}
        {!streamError ? (
          <img
            key={streamVersion}
            src={`${BACKEND_BASE}/video_feed?v=${streamVersion}`}
            alt="ORBITA Live Camera Stream"
            style={{
              width: "100%",
              height: "100%",
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              objectPosition: "center",
              display: "block",
              backgroundColor: "#000000",
              position: "absolute",
              inset: 0,
              zIndex: 1,
            }}
            onError={() => setStreamError(true)}
          />
        ) : (
          <div
            style={{
              position: "absolute",
              inset: 0,
              zIndex: 2,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "#16252e",
              color: "#dfe5e8",
              padding: "20px",
              textAlign: "center",
            }}
          >
            <AlertTriangle size={36} style={{ color: "#b8772a", marginBottom: "8px" }} />
            <strong style={{ fontFamily: "Space Grotesk", fontSize: "15px", letterSpacing: "0.04em" }}>
              CAMERA STREAM OFFLINE
            </strong>
            <p style={{ fontSize: "11px", color: "#8b989f", marginTop: "4px", maxWidth: "300px" }}>
              Connecting to backend stream at http://localhost:8000/video_feed. Verify the backend server is running.
            </p>
            <button
              onClick={() => {
                setStreamError(false);
                setStreamVersion(Date.now());
              }}
              style={{
                marginTop: "12px",
                padding: "6px 14px",
                background: "#2f6f9f",
                color: "#ffffff",
                border: "none",
                borderRadius: "4px",
                fontSize: "11px",
                fontFamily: "DM Mono",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              RETRY CONNECTION
            </button>
          </div>
        )}

        {/* Lightweight semi-transparent telemetry HUD overlay */}
        <div style={{ position: "absolute", inset: 0, zIndex: 4, pointerEvents: "none" }}>
          <div className="camera-topline">
            <span style={{ background: "rgba(13,27,34,0.7)", padding: "3px 8px", borderRadius: "3px", backdropFilter: "blur(4px)" }}>
              CAM-02 / MODULE B
            </span>
            <span style={{ background: "rgba(13,27,34,0.7)", padding: "3px 8px", borderRadius: "3px", backdropFilter: "blur(4px)" }}>
              {sourceName}
            </span>
          </div>

          <div className="camera-readout">
            <span style={{ background: "rgba(13,27,34,0.7)", padding: "3px 8px", borderRadius: "3px", backdropFilter: "blur(4px)" }}>
              <b className="live-dot" /> LIVE STREAM
            </span>
            <span style={{ background: "rgba(13,27,34,0.7)", padding: "3px 8px", borderRadius: "3px", backdropFilter: "blur(4px)" }}>
              {cameraFps ? `${Math.round(cameraFps)} FPS` : "30 FPS"}
            </span>
            <span style={{ background: "rgba(13,27,34,0.7)", padding: "3px 8px", borderRadius: "3px", backdropFilter: "blur(4px)" }}>
              AI INFERENCE ACTIVE
            </span>
          </div>
        </div>
      </div>

      <div className="camera-footer">
        <div className="feed-subject">
          <span className="avatar-chip">A1</span>
          <div>
            <strong>ASTRONAUT-01</strong>
            <span>Active procedure execution</span>
          </div>
        </div>
        <div className="feed-confidence">
          <span>AI CONFIDENCE</span>
          <strong>{confidence.toFixed(1)}%</strong>
        </div>
        {onExpand && (
          <button className="feed-expand" onClick={onExpand} aria-label="Expand camera feed" style={{ cursor: "pointer" }}>
            <Eye size={15} /> Expand
          </button>
        )}
      </div>
    </div>
  );
}

function ProcedureTimeline() {
  return (
    <div className="procedure-timeline">
      <div className="timeline-line" />
      {procedureSteps.map((step, index) => (
        <div className={`procedure-step ${step.state}`} key={step.label}>
          <div className="step-marker">{step.state === "completed" ? <Check size={12} /> : step.state === "current" ? <span /> : null}</div>
          <div className="step-content"><span>0{index + 1}</span><strong>{step.label}</strong></div>
          {step.state === "current" && <span className="current-label">CURRENT</span>}
        </div>
      ))}
    </div>
  );
}

function EventsList({ events = baseEvents }: { events?: EventItem[] }) {
  return (
    <div className="events-list">
      {events.map((event) => (
        <div className="event-row" key={`${event.time}-${event.title}`}>
          <span className="event-time">{event.time}</span>
          <div className="event-name"><strong>{event.title}</strong><span>{event.step}</span></div>
          <StatusPill tone={event.tone}>{event.tone.toUpperCase()}</StatusPill>
        </div>
      ))}
    </div>
  );
}

function Overview({ 
  demoMode, 
  confidence, 
  onViewAlerts,
  onExpandCamera
}: { 
  demoMode: boolean; 
  confidence: number; 
  onViewAlerts: () => void;
  onExpandCamera?: () => void;
}) {
  return (
    <>
      <div className="page-intro">
        <div><div className="eyebrow">MISSION ALPHA-01 / ACTIVE OPERATION</div><h1>Mission Overview</h1><p>Real-time astronaut activity and procedure verification.</p></div>
        <div className="intro-actions"><button className="quiet-button" onClick={onViewAlerts}><Bell size={15} /> 1 active alert</button><span className="last-sync"><span className="sync-dot" /> Last sync 10:42:18</span></div>
      </div>

      <section className="metric-strip" aria-label="Mission metrics">
        <MetricBlock label="Current activity" value="Tool pickup" detail="ASTRONAUT-01" />
        <MetricBlock label="AI confidence" value={`${confidence.toFixed(1)}%`} detail="Stable recognition" tone="blue" />
        <MetricBlock label="Procedure" value="04 / 08" detail="Emergency equipment" />
        <MetricBlock label="Status" value="Verified" detail="No safety interlock" tone="green" />
      </section>

      <div className="dashboard-grid">
        <section className="panel live-panel">
          <SectionHeading eyebrow="CAMERA 02 / MODULE B" title="Live Activity" action={<span className="feed-status"><span className="live-dot" /> {demoMode ? "DEMO STREAM" : "LIVE STREAM"}</span>} />
          <CameraFeed demoMode={demoMode} confidence={confidence} onExpand={onExpandCamera} />
        </section>

        <section className="panel activity-panel">
          <SectionHeading eyebrow="RECOGNITION ENGINE" title="Activity Recognition" action={<span className="panel-code">ACT-04</span>} />
          <div className="activity-primary"><span>Current activity</span><strong>Picking up tool</strong><div className="confidence-line"><div className="confidence-label"><span>Confidence</span><b>{confidence.toFixed(1)}%</b></div><div className="confidence-track"><div style={{ width: `${confidence}%` }} /></div></div></div>
          <div className="activity-context"><div><span>Previous</span><strong>Reaching</strong></div><ChevronRight size={16} /><div><span>Expected next</span><strong>Position tool</strong></div></div>
          <div className="insight-callout"><Zap size={15} /><p><strong>Recognition stable.</strong> Hand-to-object alignment detected within expected range.</p></div>
        </section>

        <section className="panel procedure-panel">
          <SectionHeading eyebrow="PROCEDURE 04 / 08" title="Procedure Verification" action={<StatusPill tone="verified">VERIFIED</StatusPill>} />
          <div className="procedure-summary"><div><span>Current step</span><strong>Position tool</strong></div><div className="procedure-score"><span>COMPLIANCE</span><strong>94.6%</strong></div></div>
          <ProcedureTimeline />
        </section>

        <section className="panel verification-panel">
          <SectionHeading eyebrow="AI DECISION SUPPORT" title="Action Verification" action={<span className="panel-code">VER-91.3</span>} />
          <div className="comparison-grid"><div className="comparison-cell expected"><span>Expected</span><strong>Position tool in docking area</strong><small>Procedure definition · Step 04</small></div><div className="comparison-cell detected"><span>Detected</span><strong>Positioning tool</strong><small>Vision + pose model · 91.3%</small></div></div>
          <div className="warning-band"><div className="warning-icon"><AlertTriangle size={15} /></div><div><strong>Spatial deviation detected</strong><span>Tool is 4.2 cm outside the docking tolerance.</span></div><b>91.3%</b></div>
        </section>

        <section className="panel events-panel">
          <SectionHeading eyebrow="MISSION LOG" title="Recent Events" action={<button className="text-button" onClick={onViewAlerts}>View all <ChevronRight size={14} /></button>} />
          <EventsList />
        </section>
      </div>
    </>
  );
}

function LiveMonitoringPage({ 
  demoMode, 
  confidence 
}: { 
  demoMode: boolean; 
  confidence: number 
}) {
  return (
    <div className="page-content">
      <div className="page-intro">
        <div>
          <div className="eyebrow">MISSION ALPHA-01 / REAL-TIME STREAM</div>
          <h1>Live Monitoring &amp; Camera Feed</h1>
          <p>Real-time edge video pipeline with YOLOv8 object detection, 17-keypoint skeleton pose estimation, and source controls.</p>
        </div>
        <div className="intro-actions">
          <span className="last-sync">
            <span className="sync-dot" /> STREAM LIVE: 30 FPS · MJPEG
          </span>
        </div>
      </div>

      <div className="dashboard-grid" style={{ gridTemplateColumns: "minmax(0, 1.7fr) minmax(340px, 1fr)" }}>
        {/* Left Column: Camera Feed & Camera Source Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <section className="panel live-panel">
            <SectionHeading 
              eyebrow="CAMERA STREAM · ULTRA-LOW LATENCY" 
              title="Live Video Feed" 
              action={<span className="feed-status"><span className="live-dot" /> {demoMode ? "DEMO STREAM" : "LIVE STREAM"}</span>} 
            />
            <CameraFeed demoMode={demoMode} confidence={confidence} isExpanded={true} />
          </section>

          {/* Integrated Camera Source & Settings Control Dock */}
          <section className="panel" style={{ padding: "18px 22px" }}>
            <SectionHeading 
              eyebrow="SOURCE SELECTION &amp; CONTROLS" 
              title="Camera Hardware &amp; Phone IP Stream" 
              action={<span className="panel-code">CSI / USB / IP / SIM</span>} 
            />
            <div style={{ marginTop: "12px" }}>
              <CameraControlPanel />
            </div>
          </section>
        </div>

        {/* Right Column: AI Inference, Action Recognition, and Procedure Verification */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <section className="panel activity-panel">
            <SectionHeading eyebrow="RECOGNITION ENGINE" title="Real-Time Detection" action={<span className="panel-code">ACT-04</span>} />
            <div className="activity-primary">
              <span>Current activity</span>
              <strong>Picking up tool</strong>
              <div className="confidence-line">
                <div className="confidence-label">
                  <span>Confidence</span>
                  <b>{confidence.toFixed(1)}%</b>
                </div>
                <div className="confidence-track">
                  <div style={{ width: `${confidence}%` }} />
                </div>
              </div>
            </div>
            <div className="activity-context">
              <div><span>Target Object</span><strong>Tool / Red Box</strong></div>
              <ChevronRight size={16} />
              <div><span>Hand-Object Distance</span><strong>2.4 cm</strong></div>
            </div>
            <div className="insight-callout">
              <Zap size={15} />
              <p><strong>YOLOv8 + Spatial HAR Active.</strong> Wrist keypoints tracking proximity to target payload container.</p>
            </div>
          </section>

          <section className="panel procedure-panel">
            <SectionHeading eyebrow="PROCEDURE 04 / 08" title="Active Step Tracking" action={<StatusPill tone="verified">VERIFIED</StatusPill>} />
            <div className="procedure-summary">
              <div><span>Current step</span><strong>Position tool</strong></div>
              <div className="procedure-score"><span>COMPLIANCE</span><strong>94.6%</strong></div>
            </div>
            <ProcedureTimeline />
          </section>

          <section className="panel">
            <SectionHeading eyebrow="SYSTEM METRICS" title="Hardware Pipeline" action={<span className="panel-code">NVIDIA JETSON</span>} />
            <div style={{ padding: "0 20px 20px" }}>
              <div className="telemetry-stack">
                <TelemetryCard icon={Monitor} label="Video Latency" value="12.5 ms" detail="Drop-stale buffer" tone="green" />
                <TelemetryCard icon={Zap} label="Inference Rate" value="28.4 FPS" detail="YOLOv8n-pose" tone="blue" />
                <TelemetryCard icon={Cpu} label="Compute Load" value="20.9%" detail="Jetson Orin Nano" tone="blue" />
                <TelemetryCard icon={ShieldCheck} label="Safety Interlock" value="CLEAR" detail="Nominal workspace" tone="green" />
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

function AnalyticsPage() {
  return (
    <div className="page-content">
      <div className="page-intro"><div><div className="eyebrow">MISSION ALPHA-01 / PERFORMANCE</div><h1>Analytics</h1><p>Operational performance and model confidence over the current mission window.</p></div><button className="quiet-button"><SlidersHorizontal size={15} /> Date range · 7 days</button></div>
      <div className="analytics-kpis"><MetricBlock label="Procedure compliance" value="94.6%" detail="+2.1% vs prior mission" tone="green" /><MetricBlock label="Recognition accuracy" value="97.2%" detail="+0.8% vs prior mission" tone="blue" /><MetricBlock label="Completed procedures" value="128" detail="6 in current shift" /><MetricBlock label="Deviations" value="7" detail="1 requires review" tone="amber" /></div>
      <div className="chart-grid"><section className="panel chart-panel wide"><SectionHeading eyebrow="PROCEDURE QUALITY" title="Procedure compliance" action={<span className="chart-legend"><i className="legend-blue" /> Compliance</span>} /><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={complianceData}><defs><linearGradient id="complianceFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2f6f9f" stopOpacity={0.16} /><stop offset="100%" stopColor="#2f6f9f" stopOpacity={0} /></linearGradient></defs><CartesianGrid vertical={false} stroke="#e6ebef" /><XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#7a8792", fontSize: 11 }} /><YAxis domain={[88, 100]} tickLine={false} axisLine={false} tick={{ fill: "#7a8792", fontSize: 11 }} tickFormatter={(v) => `${v}%`} /><Tooltip contentStyle={{ border: "1px solid #dce3e8", borderRadius: 8, boxShadow: "0 8px 24px rgba(32, 51, 67, .08)", fontSize: 12 }} /><Area type="monotone" dataKey="value" stroke="#2f6f9f" strokeWidth={2} fill="url(#complianceFill)" /></AreaChart></ResponsiveContainer></div></section><section className="panel chart-panel"><SectionHeading eyebrow="MODEL HEALTH" title="AI confidence" action={<span className="chart-legend"><i className="legend-green" /> Confidence</span>} /><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={confidenceData}><CartesianGrid vertical={false} stroke="#e6ebef" /><XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: "#7a8792", fontSize: 10 }} /><YAxis domain={[90, 100]} tickLine={false} axisLine={false} tick={{ fill: "#7a8792", fontSize: 11 }} tickFormatter={(v) => `${v}%`} /><Tooltip contentStyle={{ border: "1px solid #dce3e8", borderRadius: 8, fontSize: 12 }} /><Line type="monotone" dataKey="value" stroke="#4d8b73" strokeWidth={2} dot={{ r: 2, fill: "#4d8b73" }} /></LineChart></ResponsiveContainer></div></section></div>
      <div className="insight-row"><div className="insight-icon"><CircleGauge size={18} /></div><div><strong>System trend is within mission tolerances.</strong><p>Model confidence has remained above 94% for the last 42 minutes. Procedure compliance improved after the Step 03 tool pickup verification.</p></div></div>
    </div>
  );
}

function MissionPage() {
  return <div className="page-content"><div className="page-intro"><div><div className="eyebrow">MISSION OPERATIONS / BRIEF</div><h1>Mission Alpha-01</h1><p>Current mission context and astronaut assignment.</p></div><StatusPill tone="online">SYSTEM ONLINE</StatusPill></div><div className="mission-overview"><div className="mission-identity"><div className="mission-orbit-mark"><span>01</span><div /></div><div><span className="eyebrow">ACTIVE MISSION</span><h2>ALPHA-01</h2><p>Emergency Equipment Deployment</p></div></div><div className="mission-stat"><span>Astronaut</span><strong>ASTRONAUT-01</strong><small>Commander · EVA Unit 01</small></div><div className="mission-stat"><span>Mission status</span><strong className="text-green">ACTIVE</strong><small>Nominal communication link</small></div><div className="mission-stat"><span>Mission duration</span><strong className="mono">04:32:18</strong><small>Started 06:10:00 UTC</small></div><div className="mission-stat"><span>Overall compliance</span><strong className="text-blue">94.6%</strong><small>128 actions verified</small></div></div><div className="mission-grid"><section className="panel"><SectionHeading eyebrow="CURRENT ASSIGNMENT" title="Procedure brief" /><div className="brief-list"><div><span>Objective</span><strong>Deploy emergency equipment from external storage.</strong></div><div><span>Active step</span><strong>Position tool in docking area</strong></div><div><span>Expected completion</span><strong>10:48:00 mission time</strong></div><div><span>Safety state</span><StatusPill tone="verified">NOMINAL</StatusPill></div></div></section><section className="panel"><SectionHeading eyebrow="MISSION HEALTH" title="System health" /><div className="telemetry-stack"><TelemetryCard icon={Monitor} label="Camera" value="Online" detail="28 FPS" tone="green" /><TelemetryCard icon={Zap} label="AI Engine" value="Online" detail="97.2% acc." tone="blue" /><TelemetryCard icon={Cpu} label="GPU" value="61%" detail="Jetson Orin" tone="blue" /><TelemetryCard icon={Thermometer} label="Temperature" value="52°C" detail="Within range" tone="green" /></div></section></div></div>;
}

function AlertsPage() {
  const allEvents: EventItem[] = [{ time: "10:42:18", title: "Procedure deviation", step: "Step 04 · Spatial deviation detected", tone: "critical" }, ...baseEvents.slice(1), { time: "10:38:44", title: "AI engine reconnected", step: "System telemetry", tone: "info" }, { time: "10:36:12", title: "Camera calibration complete", step: "CAM-02 / Module B", tone: "verified" }];
  return <div className="page-content"><div className="page-intro"><div><div className="eyebrow">MISSION LOG / EVENT STREAM</div><h1>Alerts & Events</h1><p>System notifications, procedure deviations, and verification history.</p></div><button className="quiet-button"><Bell size={15} /> Mark all reviewed</button></div><div className="alert-summary"><div className="alert-summary-item critical"><AlertTriangle size={17} /><div><span>Critical</span><strong>1</strong></div></div><div className="alert-summary-item warning"><AlertTriangle size={17} /><div><span>Warnings</span><strong>1</strong></div></div><div className="alert-summary-item verified"><Check size={17} /><div><span>Verified events</span><strong>16</strong></div></div><div className="alert-summary-item"><Activity size={17} /><div><span>Events today</span><strong>18</strong></div></div></div><section className="panel alerts-table-panel"><SectionHeading eyebrow="10:30 — 10:42 UTC" title="Recent events" action={<span className="panel-code">18 TOTAL</span>} /><EventsList events={allEvents} /></section></div>;
}

function ProcedurePage() {
  return <div className="page-content"><div className="page-intro"><div><div className="eyebrow">PROCEDURE LIBRARY / ACTIVE RUN</div><h1>Procedure Verification</h1><p>Emergency Equipment Deployment · Procedure ALPHA-EED-08.</p></div><StatusPill tone="verified">STEP 04 OF 08</StatusPill></div><div className="procedure-page-grid"><section className="panel procedure-detail-panel"><SectionHeading eyebrow="ACTIVE PROCEDURE" title="Emergency Equipment Deployment" action={<span className="panel-code">ALPHA-EED-08</span>} /><div className="large-progress"><div className="large-progress-top"><span>Mission completion</span><strong>50%</strong></div><div className="large-progress-track"><div style={{ width: "50%" }} /></div></div><ProcedureTimeline /></section><section className="panel"><SectionHeading eyebrow="CURRENT CHECK" title="Action verification" /><div className="check-detail"><div className="check-badge"><AlertTriangle size={20} /></div><span className="eyebrow">STEP 04 · WARNING</span><h3>Position tool</h3><p>Tool detected in the docking workflow, but spatial deviation is outside the expected tolerance.</p><div className="check-metrics"><div><span>Deviation</span><strong>4.2 cm</strong></div><div><span>Confidence</span><strong>91.3%</strong></div><div><span>Next action</span><strong>Reposition</strong></div></div><button className="primary-button">View event details <ChevronRight size={15} /></button></div></section></div></div>;
}

function SettingsPage({ 
  demoMode, 
  onToggleDemoMode, 
  switchingSource 
}: { 
  demoMode: boolean; 
  onToggleDemoMode: () => void; 
  switchingSource?: boolean; 
}) {
  return (
    <div className="page-content">
      <div className="page-intro">
        <div>
          <div className="eyebrow">SYSTEM CONFIGURATION</div>
          <h1>Settings</h1>
          <p>Configure mission display, data source, and operational preferences.</p>
        </div>
      </div>
      <div className="settings-grid">
        <section className="panel settings-panel">
          <SectionHeading eyebrow="DATA SOURCE" title="Monitoring mode" />
          <div className="setting-row">
            <div>
              <strong>Demo mode (Synthetic Simulation)</strong>
              <span>
                {demoMode 
                  ? "Synthetic simulation mode active. Emulating procedure stream." 
                  : "Live camera mode active. Streaming from physical webcam or mobile IP stream."}
              </span>
            </div>
            <button 
              className={`switch ${demoMode ? "on" : ""}`} 
              onClick={onToggleDemoMode} 
              disabled={switchingSource}
              aria-label="Toggle demo mode"
            >
              <span />
            </button>
          </div>
          <div className="setting-row">
            <div>
              <strong>WebSocket endpoint</strong>
              <span className="mono">ws://localhost:8000/ws/telemetry</span>
            </div>
            <span className="connection-state">
              <span className="sync-dot" /> Ready
            </span>
          </div>
        </section>
        <section className="panel settings-panel">
          <SectionHeading eyebrow="DISPLAY" title="Operator preferences" />
          <div className="setting-row">
            <div>
              <strong>Compact telemetry</strong>
              <span>Prioritize information density for 1366px displays.</span>
            </div>
            <button className="switch on" aria-label="Compact telemetry enabled">
              <span />
            </button>
          </div>
          <div className="setting-row">
            <div>
              <strong>Alert sound</strong>
              <span>Play a subtle tone for critical deviations only.</span>
            </div>
            <button className="switch on" aria-label="Alert sound enabled">
              <span />
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}

export default function Home() {
  const navigate = useNavigate();
  const [activeNav, setActiveNav] = useState("Dashboard");
  const [demoMode, setDemoMode] = useState(true);
  const [switchingSource, setSwitchingSource] = useState(false);
  const [, setSourceNotice] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(97.4);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Sync initial mode with backend camera source
  useEffect(() => {
    getCameraStatus()
      .then((status) => {
        if (status && status.source) {
          setDemoMode(status.source === "sim");
        }
      })
      .catch(() => {});
  }, []);

  // Real function: toggles between Simulation (DEMO) and Live Camera Hardware
  const handleToggleDataSource = async () => {
    if (switchingSource) return;
    setSwitchingSource(true);
    setSourceNotice(null);

    const willBeDemo = !demoMode;
    try {
      if (willBeDemo) {
        const res = await connectCamera({ source: "sim" });
        if (res.success) {
          setDemoMode(true);
          setSourceNotice("Switched to Synthetic Simulation Feed");
        } else {
          setSourceNotice(`Error: ${res.error}`);
        }
      } else {
        let res = await connectCamera({ source: "jetson_camera", device_index: 0 });
        if (res.success) {
          setDemoMode(false);
          setSourceNotice("Live Camera Connected (Device Index 0)");
        } else {
          const ipRes = await connectCamera({ source: "ip_camera" });
          if (ipRes.success) {
            setDemoMode(false);
            setSourceNotice("Connected to Phone IP Camera");
          } else {
            setDemoMode(false);
            setSourceNotice("Live Mode active. Connect Phone Cam or USB Camera.");
          }
        }
      }
    } catch (err: any) {
      setSourceNotice(err?.message || "Failed to switch camera mode");
    } finally {
      setSwitchingSource(false);
    }
  };

  useEffect(() => {
    if (!demoMode) return;
    const timer = window.setInterval(() => setConfidence((value) => value >= 98.1 ? 96.8 : Number((value + 0.2).toFixed(1))), 4500);
    return () => window.clearInterval(timer);
  }, [demoMode]);

  const timeLabel = useMemo(() => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  }, [confidence]);

  const content = activeNav === "Dashboard"
    ? <Overview demoMode={demoMode} confidence={confidence} onViewAlerts={() => setActiveNav("Alerts")} onExpandCamera={() => setActiveNav("Live Monitoring")} />
    : activeNav === "Live Monitoring"
      ? <LiveMonitoringPage demoMode={demoMode} confidence={confidence} />
      : activeNav === "Analytics" ? <AnalyticsPage />
        : activeNav === "Mission" ? <MissionPage />
          : activeNav === "Alerts" ? <AlertsPage />
            : activeNav === "Procedure" ? <ProcedurePage />
              : <SettingsPage demoMode={demoMode} onToggleDemoMode={handleToggleDataSource} switchingSource={switchingSource} />;

  return (
    <div className="orbita-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand-block"><div className="brand-mark"><span /><span /><span /></div><div><div className="brand-name">ORBITA</div><div className="brand-subtitle">AI ACTIVITY &amp; PROCEDURE<br />VERIFICATION</div></div><button className="sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X size={16} /></button></div>
        <div className="mission-selector"><div className="mission-selector-top"><span>ACTIVE MISSION</span><span className="selector-dot" /></div><strong>Alpha-01</strong><span>ASTRONAUT-01 · EVA UNIT 01</span></div>
        <nav className="sidebar-nav" aria-label="Primary navigation">
          <div className="nav-label">MISSION CONTROL</div>
          {navItems.map(({ label, icon: Icon }) => <button key={label} className={`nav-item ${activeNav === label ? "active" : ""}`} onClick={() => { setActiveNav(label); setSidebarOpen(false); }}><Icon size={17} strokeWidth={1.8} /><span>{label}</span>{label === "Alerts" && <span className="nav-badge">1</span>}</button>)}
          <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px solid #e2e8eb' }}>
            <div className="nav-label">EXPERIMENT COPILOT</div>
            <button
              className="nav-item"
              onClick={() => navigate('/mission')}
              style={{ color: '#2f6f9f', fontWeight: 600 }}
              title="Switch to Astronaut HUD / Live Experiment"
            >
              <ArrowLeft size={16} strokeWidth={2} />
              <span>Astronaut HUD</span>
            </button>
          </div>
        </nav>
        <div className="sidebar-bottom"><div className="system-mini"><span className="online-ring"><span /></span><div><strong>System online</strong><span>All services nominal</span></div></div><div className="operator-row"><div className="operator-avatar">MC</div><div><strong>Mission Control</strong><span>Operator console</span></div><MoreIcon /></div></div>
      </aside>
      {sidebarOpen && <button className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-label="Close navigation overlay" />}
      <main className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="topbar-context"><span className="topbar-kicker">MISSION</span><strong>ALPHA-01</strong><span className="topbar-divider" /><span className="topbar-status"><span className="sync-dot" /> SYSTEM ONLINE</span></div>
          <div className="topbar-right">
            <div className="mode-control">
              <span className="mode-label">DATA SOURCE</span>
              <button 
                className={`mode-toggle ${demoMode ? "demo" : "live"}`} 
                onClick={handleToggleDataSource}
                disabled={switchingSource}
                title={demoMode ? "Current: Simulation Feed (DEMO). Click to switch to Live Camera Hardware" : "Current: Live Camera Stream. Click to switch to Simulation Feed (DEMO)"}
                style={{ cursor: switchingSource ? 'wait' : 'pointer', opacity: switchingSource ? 0.7 : 1 }}
              >
                <span className="toggle-knob" />
                <span>{switchingSource ? "..." : demoMode ? "DEMO" : "LIVE"}</span>
              </button>
            </div>
            <div className="topbar-time mono">{timeLabel}<span>UTC</span></div>
            <button className="icon-button" aria-label="Open notifications" onClick={() => setActiveNav("Alerts")}><Bell size={17} /><i /></button>
            <button
              onClick={() => navigate('/mission')}
              className="quiet-button"
              title="Switch to Astronaut Operations"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: '#2f6f9f' }}
            >
              <LayoutGrid size={15} />
              <span>Astronaut HUD</span>
            </button>
          </div>
        </header>
        <div className="page-scroller"><div className="page-wrap">{content}</div></div>
        <footer className="app-footer"><span>ORBITA MISSION CONTROL <b>·</b> v0.8.4</span><span>WEBSOCKET <b className="footer-online">● CONNECTED</b><b>·</b> LAST SYNC 10:42:18 UTC</span></footer>
      </main>
    </div>
  );
}

function MoreIcon() {
  return <span className="more-icon"><i /><i /><i /></span>;
}
