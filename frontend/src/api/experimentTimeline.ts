export interface DemoEvent {
  id: string;
  time: number;          // Video timestamp (seconds) when event triggers
  endTime?: number;       // Video timestamp (seconds) when event completes
  step: number;          // Procedure step index (1-based)
  type: string;          // Action category (e.g., 'researcher_entered', 'sample_pickup', 'sample_transfer', 'sample_placed', 'experiment_complete')
  title: string;
  description: string;
  confidence: number;
  object?: string;
  zone?: string;
  from?: string;
  to?: string;
  hud_message?: string;
  guidance?: string;
  detected_objects?: Array<{
    label: string;
    bbox: [number, number, number, number]; // [top, left, width, height] in percentages (0-100)
    confidence: number;
  }>;
}

export interface ProcedureStep {
  id: number;
  label: string;
  expected_action: string;
  expected_object: string;
  description: string;
}

// 5-Step Sample Placement & Transfer Procedure (Primary Hackathon Demo Spec)
export const SAMPLE_PROCEDURE_STEPS: ProcedureStep[] = [
  {
    id: 1,
    label: "Researcher entered",
    expected_action: "ZONE_ENTRY",
    expected_object: "RESEARCHER",
    description: "Researcher entered the experiment zone and prepared workspace."
  },
  {
    id: 2,
    label: "Pick up Sample A",
    expected_action: "PICKUP",
    expected_object: "Sample A",
    description: "Researcher picked up Sample A from primary container."
  },
  {
    id: 3,
    label: "Transfer Sample A to Zone B",
    expected_action: "TRANSFER",
    expected_object: "Sample A",
    description: "Transfer Sample A across workspace boundaries toward Zone B."
  },
  {
    id: 4,
    label: "Place Sample A",
    expected_action: "PLACEMENT",
    expected_object: "Sample A",
    description: "Securely place Sample A inside target Zone B slot."
  },
  {
    id: 5,
    label: "Apparatus interaction & Complete",
    expected_action: "VERIFICATION",
    expected_object: "APPARATUS",
    description: "Verify apparatus seal and finalize procedural run."
  }
];

// Official 13-Step Procedure Sequence
export const OFFICIAL_13_STEPS_PROCEDURE: ProcedureStep[] = [
  { id: 1, label: "Identify the Blue Box", expected_action: "IDENTIFY", expected_object: "BLUE_BOX", description: "Detect and localize Blue Box in workspace." },
  { id: 2, label: "Pick up the Blue Box", expected_action: "PICKUP", expected_object: "BLUE_BOX", description: "Grasp and lift Blue Box from initial position." },
  { id: 3, label: "Place Blue Box at Location A", expected_action: "PLACE", expected_object: "BLUE_BOX", description: "Position Blue Box accurately at Location A." },
  { id: 4, label: "Identify the Yellow Box", expected_action: "IDENTIFY", expected_object: "YELLOW_BOX", description: "Detect and verify Yellow Box in workspace." },
  { id: 5, label: "Pick up the Yellow Box", expected_action: "PICKUP", expected_object: "YELLOW_BOX", description: "Grasp and lift Yellow Box." },
  { id: 6, label: "Place Yellow Box at Location B", expected_action: "PLACE", expected_object: "YELLOW_BOX", description: "Position Yellow Box at Location B." },
  { id: 7, label: "Pick up the Pen", expected_action: "PICKUP", expected_object: "PEN", description: "Locate and pick up Pen tool." },
  { id: 8, label: "Place Pen inside the Blue Box", expected_action: "PLACE", expected_object: "PEN", description: "Insert Pen tool inside Blue Box container." },
  { id: 9, label: "Pick up the Watch", expected_action: "PICKUP", expected_object: "WATCH", description: "Locate and pick up Watch." },
  { id: 10, label: "Place Watch inside the Yellow Box", expected_action: "PLACE", expected_object: "WATCH", description: "Place Watch inside Yellow Box container." },
  { id: 11, label: "Move Blue Box from A to B", expected_action: "MOVE", expected_object: "BLUE_BOX", description: "Transfer Blue Box from Location A to Location B." },
  { id: 12, label: "Move Yellow Box from B to A", expected_action: "MOVE", expected_object: "YELLOW_BOX", description: "Transfer Yellow Box from Location B to Location A." },
  { id: 13, label: "Experiment Complete", expected_action: "COMPLETE", expected_object: "ALL", description: "All procedural steps validated successfully." },
];

// 7-Step Microbial Experiment Demonstration Sequence (EXP-MICROBE)
export const MICROBIAL_PROCEDURE_STEPS: ProcedureStep[] = [
  { id: 1, label: "Prepare Experiment Setup", expected_action: "SETUP", expected_object: "WORK_SURFACE", description: "Set up and sanitize work surface and experiment container." },
  { id: 2, label: "Prepare Microbial Sample", expected_action: "PREPARE", expected_object: "SAMPLE_CONTAINER", description: "Retrieve and inspect microbial sample container." },
  { id: 3, label: "Transfer Sample", expected_action: "TRANSFER", expected_object: "MICROBIAL_SAMPLE", description: "Transfer microbial sample into designated experiment chamber." },
  { id: 4, label: "Secure Experiment Container", expected_action: "SECURE", expected_object: "EXPERIMENT_CONTAINER", description: "Seal and lock experiment container enclosure." },
  { id: 5, label: "Begin Observation", expected_action: "OBSERVE", expected_object: "OBSERVATION_EQUIPMENT", description: "Initiate optical recording and observation phase." },
  { id: 6, label: "Record Observation", expected_action: "RECORD", expected_object: "TELEMETRY", description: "Record observation timestamps, telemetry, and visual logs." },
  { id: 7, label: "Complete Experiment", expected_action: "COMPLETE", expected_object: "ALL", description: "Finalize experiment protocol and package data result." },
];

export const EXP_MICROBE_METADATA = {
  id: "EXP-MICROBE",
  name: "Microbial Experiment in Microgravity",
  category: "Biological Research",
  context: "ISRO–Axiom-4 Biological Research",
  mode: "Ground-Based Demonstration",
  description: "Ground-based demonstration of an observation and monitoring workflow inspired by space-related microbial biological research.",
  disclaimer: "College demonstration for workflow visualization only. This setup does not reproduce the microgravity environment or constitute the official ISRO/Axiom-4 experimental protocol.",
  objectives: [
    "Observe microbial experiment activities.",
    "Monitor the sequence of experimental actions.",
    "Demonstrate automated observation and procedure monitoring.",
    "Record experiment events and timestamps.",
    "Demonstrate how ORBITA can support biological experiment workflows relevant to future space research."
  ],
  equipment: [
    { name: "EXPERIMENT CONTAINER", supported: true },
    { name: "SAMPLE CONTAINER", supported: true },
    { name: "MICROBIAL SAMPLE", supported: true },
    { name: "TRANSFER TOOL / PIPETTE", supported: false, status: "visual_validation: unavailable" },
    { name: "OBSERVATION EQUIPMENT", supported: true },
    { name: "WORK SURFACE", supported: true }
  ]
};

// Deterministic Event Timelines tied to exact video timestamps (seconds)
export const DEFAULT_DEMO_EVENTS: DemoEvent[] = [
  {
    id: "evt_1",
    time: 5.0,
    endTime: 11.5,
    step: 1,
    type: "researcher_entered",
    title: "Researcher entered experiment zone",
    description: "Researcher entered the experiment zone and initialized safety boundary.",
    confidence: 0.97,
    object: "Researcher",
    zone: "Zone A",
    hud_message: "Researcher present in camera field of view. Ready to begin sample processing.",
    guidance: "Proceed with Step 2: Pick up Sample A.",
    detected_objects: [
      { label: "RESEARCHER", bbox: [15, 25, 55, 60], confidence: 0.97 },
      { label: "Zone A", bbox: [65, 10, 25, 35], confidence: 0.94 }
    ]
  },
  {
    id: "evt_2",
    time: 11.5,
    endTime: 17.0,
    step: 2,
    type: "sample_pickup",
    title: "Sample A picked up",
    description: "Researcher picked up Sample A from primary container.",
    confidence: 0.91,
    object: "Sample A",
    zone: "Zone A",
    hud_message: "Sample A grasp detected. Hold stable for boundary transfer.",
    guidance: "Proceed with Step 3: Transfer Sample A to Zone B.",
    detected_objects: [
      { label: "RESEARCHER", bbox: [15, 25, 55, 60], confidence: 0.96 },
      { label: "Sample A (HELD)", bbox: [45, 40, 18, 22], confidence: 0.91 },
      { label: "Zone A", bbox: [65, 10, 25, 35], confidence: 0.95 }
    ]
  },
  {
    id: "evt_3",
    time: 17.0,
    endTime: 23.5,
    step: 3,
    type: "sample_transfer",
    title: "Sample A transferred A → B",
    description: "Sample A transferred smoothly across workspace from Zone A to Zone B.",
    confidence: 0.94,
    object: "Sample A",
    from: "Zone A",
    to: "Zone B",
    hud_message: "Transfer vector validated (Zone A → Zone B). Trajectory within safe limits.",
    guidance: "Proceed with Step 4: Place Sample A in Zone B.",
    detected_objects: [
      { label: "RESEARCHER", bbox: [15, 25, 55, 60], confidence: 0.95 },
      { label: "Sample A (IN_TRANSFER)", bbox: [50, 48, 18, 22], confidence: 0.94 },
      { label: "Zone B", bbox: [65, 55, 25, 35], confidence: 0.93 }
    ]
  },
  {
    id: "evt_4",
    time: 23.5,
    endTime: 23.46,
    step: 4,
    type: "sample_placed",
    title: "Sample A placed in Zone B",
    description: "Sample A placed securely inside Zone B target location.",
    confidence: 0.95,
    object: "Sample A",
    zone: "Zone B",
    hud_message: "Sample A placement verified in Zone B slot. All procedure constraints satisfied.",
    guidance: "Experiment complete. Generating Section 20 structured result.",
    detected_objects: [
      { label: "RESEARCHER", bbox: [15, 25, 55, 60], confidence: 0.98 },
      { label: "Sample A (PLACED)", bbox: [68, 62, 18, 22], confidence: 0.95 },
      { label: "Zone B (OCCUPIED)", bbox: [65, 55, 25, 35], confidence: 0.96 }
    ]
  }
];

export function getEventsForVideo(duration: number, events: DemoEvent[] = DEFAULT_DEMO_EVENTS): DemoEvent[] {
  if (duration <= 0) return events;
  if (Math.abs(duration - 23.5) < 3.0) return events;

  const scale = duration / 23.5;
  return events.map(evt => ({
    ...evt,
    time: Math.round(evt.time * scale * 10) / 10,
    endTime: evt.endTime ? Math.round(evt.endTime * scale * 10) / 10 : undefined,
  }));
}

export interface SynchronizedExperimentState {
  currentTime: number;
  duration: number;
  activeStepIndex: number;         // 0-indexed (0 to 4)
  activeStepNumber: number;        // 1-indexed (1 to 5)
  fsmStatus: 'INITIALIZED' | 'RESEARCHER_READY' | 'SAMPLE_PICKUP' | 'SAMPLE_TRANSFER' | 'SAMPLE_PLACEMENT' | 'EXPERIMENT_COMPLETE';
  completedEvents: DemoEvent[];
  activeEvent: DemoEvent | null;
  pendingEvents: DemoEvent[];
  detectedObjects: Array<{
    label: string;
    bbox: [number, number, number, number];
    confidence: number;
  }>;
  hudMessage: string;
  guidanceMessage: string;
  confidence: number;
  isComplete: boolean;
  totalEventsCount: number;
  completedEventsCount: number;
}

/**
 * IDEMPOTENT Experiment State Calculation Function
 * Reconstructs the exact experiment state from `currentTime` strictly.
 */
export function getExperimentStateAtTime(
  currentTime: number,
  duration: number = 23.46,
  rawEvents: DemoEvent[] = DEFAULT_DEMO_EVENTS
): SynchronizedExperimentState {
  const events = getEventsForVideo(duration, rawEvents);
  
  const completedEvents = events.filter(e => e.time <= currentTime);
  const pendingEvents = events.filter(e => e.time > currentTime);
  const activeEvent = completedEvents.length > 0 ? completedEvents[completedEvents.length - 1] : null;

  const isAtEnd = duration > 0 && currentTime >= duration - 0.5;

  let activeStepIndex = 0;
  let fsmStatus: SynchronizedExperimentState['fsmStatus'] = 'INITIALIZED';

  if (!activeEvent || currentTime < events[0].time) {
    activeStepIndex = 0;
    fsmStatus = 'INITIALIZED';
  } else if (activeEvent.step === 1) {
    activeStepIndex = 1;
    fsmStatus = 'RESEARCHER_READY';
  } else if (activeEvent.step === 2) {
    activeStepIndex = 2;
    fsmStatus = 'SAMPLE_PICKUP';
  } else if (activeEvent.step === 3) {
    activeStepIndex = 3;
    fsmStatus = 'SAMPLE_TRANSFER';
  } else if (activeEvent.step === 4 || isAtEnd) {
    activeStepIndex = 4;
    fsmStatus = isAtEnd || completedEvents.length === events.length ? 'EXPERIMENT_COMPLETE' : 'SAMPLE_PLACEMENT';
  }

  const isComplete = fsmStatus === 'EXPERIMENT_COMPLETE' || (completedEvents.length === events.length && isAtEnd);

  const fallbackBbox: [number, number, number, number] = [20, 20, 60, 60];
  const detectedObjects: Array<{
    label: string;
    bbox: [number, number, number, number];
    confidence: number;
  }> = activeEvent && activeEvent.detected_objects ? activeEvent.detected_objects : [
    { label: "EXPERIMENT ZONE", bbox: fallbackBbox, confidence: 0.95 }
  ];

  const hudMessage = activeEvent
    ? activeEvent.hud_message || activeEvent.description
    : "Experiment Initialized. Awaiting researcher entrance into camera zone.";

  const guidanceMessage = activeEvent
    ? activeEvent.guidance || `Step ${activeStepIndex + 1} in progress.`
    : "Step 1 ACTIVE: Researcher entering experiment zone.";

  const confidence = activeEvent ? activeEvent.confidence : 0.97;

  return {
    currentTime,
    duration,
    activeStepIndex,
    activeStepNumber: activeStepIndex + 1,
    fsmStatus,
    completedEvents,
    activeEvent,
    pendingEvents,
    detectedObjects,
    hudMessage,
    guidanceMessage,
    confidence,
    isComplete,
    totalEventsCount: events.length,
    completedEventsCount: completedEvents.length,
  };
}
