import { BACKEND_BASE } from '../api/camera';

/**
 * Centralized Strict Voice & TTS Controller for ORBITA
 * ===================================================
 * Enforces explicit session authorization and gating:
 *
 * GATE 1: User MUST be Authenticated (isAuthenticated === true)
 * GATE 2: Experiment MUST be Active/Running (isLiveActive === true)
 * GATE 3: Experiment ID MUST match Active Experiment (experimentId === activeExperimentId)
 * GATE 4: Session ID MUST match Active Session Token (sessionId === currentSessionId)
 * GATE 5: Voice Output MUST be Enabled (voiceEnabled === true)
 */
class VoiceController {
  private isAuthenticated: boolean = false;
  private activeExperimentId: string | null = null;
  private isLiveActive: boolean = false;
  private currentSessionId: string | null = null;

  private lastSpokenText: string = '';
  private lastSpokenStep: number = -1;
  private lastSpokenTime: number = 0;
  private isSpeaking: boolean = false;

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => this.disableAndStopAllVoice());
      window.addEventListener('pagehide', () => this.disableAndStopAllVoice());
    }
  }

  /**
   * Set user authentication state.
   * Disabling auth immediately kills all active and pending voice sessions.
   */
  public setAuthenticated(authenticated: boolean): void {
    this.isAuthenticated = authenticated;
    if (!authenticated) {
      this.disableAndStopAllVoice();
    }
  }

  public getIsAuthenticated(): boolean {
    return this.isAuthenticated;
  }

  /**
   * Start a live experiment voice session.
   * MUST be called only when user starts a live experiment (/experiments/:id/live).
   */
  public startExperimentSession(experimentId: string): string {
    if (!this.isAuthenticated) {
      // Gate Check: Do not permit session creation for unauthenticated users
      this.stopAllSpeech();
      return '';
    }

    // 1. Invalidate previous session first
    this.stopAllSpeech();

    // 2. Generate new session token
    const newSessionId = `${experimentId}_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    this.activeExperimentId = experimentId;
    this.isLiveActive = true;
    this.currentSessionId = newSessionId;

    this.lastSpokenText = '';
    this.lastSpokenStep = -1;
    this.lastSpokenTime = 0;

    // 3. Notify backend API of active experiment voice session
    try {
      fetch(`${BACKEND_BASE}/api/voice/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          active: true,
          experiment_id: experimentId,
          session_id: newSessionId,
        }),
      }).catch(() => {});
    } catch {
      // ignore network errors
    }

    return newSessionId;
  }

  /**
   * Stop an experiment voice session completely.
   * Invalidates session tokens BEFORE canceling speech to prevent async callback races.
   */
  public stopExperimentSession(sessionId?: string): void {
    if (sessionId && this.currentSessionId && sessionId !== this.currentSessionId) {
      // Stale session stop request — ignore if active session is newer
      return;
    }

    // 1. Invalidate session state FIRST
    this.isLiveActive = false;
    this.activeExperimentId = null;
    this.currentSessionId = null;

    // 2. Stop browser speech & backend queue
    this.stopAllSpeech();

    // 3. Notify backend API of deactivated session
    try {
      fetch(`${BACKEND_BASE}/api/voice/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: false }),
      }).catch(() => {});
    } catch {
      // ignore
    }
  }

  /**
   * Complete shutdown of all voice capability (logout / page unload).
   */
  public disableAndStopAllVoice(): void {
    this.isAuthenticated = false;
    this.isLiveActive = false;
    this.activeExperimentId = null;
    this.currentSessionId = null;
    this.stopAllSpeech();
  }

  /**
   * Stop all active and queued speech instantly across browser and backend.
   */
  public stopAllSpeech(): void {
    this.isSpeaking = false;

    // 1. Cancel browser Web Speech API
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (e) {
        console.warn('[VoiceController] Error canceling browser TTS:', e);
      }
    }

    // 2. Clear backend TTS queue
    try {
      fetch(`${BACKEND_BASE}/api/voice/stop`, { method: 'POST' }).catch(() => {});
    } catch {
      // Ignore network errors on teardown
    }
  }

  /**
   * Central Gated Speech Execution.
   * Every speech request MUST pass all 5 security and lifecycle gates.
   */
  public speak(
    text: string,
    options: {
      experimentId: string;
      sessionId: string;
      stepIdx?: number;
      force?: boolean;
      voiceEnabled?: boolean;
      onStart?: () => void;
      onEnd?: () => void;
    }
  ): boolean {
    const { experimentId, sessionId, stepIdx, force = false, voiceEnabled = true, onStart, onEnd } = options;

    // =========================================================================
    // THE 5 MANDATORY LIFECYCLE GATES
    // =========================================================================

    // GATE 1: User MUST be Authenticated
    if (!this.isAuthenticated) {
      return false;
    }

    // GATE 2: Experiment MUST be Active/Running
    if (!this.isLiveActive || !this.activeExperimentId) {
      return false;
    }

    // GATE 3: Experiment ID MUST match Active Experiment
    if (!experimentId || experimentId !== this.activeExperimentId) {
      return false;
    }

    // GATE 4: Session ID MUST match Active Session Token
    if (!sessionId || sessionId !== this.currentSessionId) {
      return false;
    }

    // GATE 5: Voice Output MUST be Enabled
    if (!voiceEnabled) {
      return false;
    }

    const cleanText = (text || '').trim();
    if (!cleanText) {
      return false;
    }

    const now = Date.now();

    // Deduplication & Cooldown Guard
    if (!force) {
      if (stepIdx !== undefined && stepIdx === this.lastSpokenStep && cleanText === this.lastSpokenText) {
        return false;
      }
      if (cleanText === this.lastSpokenText && now - this.lastSpokenTime < 3000) {
        return false;
      }
    }

    // Record last spoken state
    this.lastSpokenText = cleanText;
    this.lastSpokenStep = stepIdx !== undefined ? stepIdx : -1;
    this.lastSpokenTime = now;

    // Prevent Overlap: Cancel previous speech before starting new speech
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }

    // Dispatch to backend TTS with session token check
    fetch(`${BACKEND_BASE}/api/voice/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: cleanText,
        clear_previous: true,
        experiment_id: experimentId,
        session_id: sessionId,
      }),
    }).catch(() => {});

    // Dispatch to Browser Speech Synthesis
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 1.0;
        utterance.pitch = 1.05;
        utterance.volume = 1.0;

        const voices = window.speechSynthesis.getVoices();
        const femaleVoice = voices.find((v) => {
          const name = v.name.toLowerCase();
          return (
            name.includes('zira') ||
            name.includes('hazel') ||
            name.includes('female') ||
            name.includes('samantha') ||
            name.includes('victoria') ||
            name.includes('karen') ||
            name.includes('aria') ||
            name.includes('jenny')
          );
        });
        if (femaleVoice) {
          utterance.voice = femaleVoice;
        }

        utterance.onstart = () => {
          // Re-verify ALL GATES inside async start callback
          if (
            !this.isAuthenticated ||
            !this.isLiveActive ||
            this.activeExperimentId !== experimentId ||
            this.currentSessionId !== sessionId
          ) {
            window.speechSynthesis.cancel();
            return;
          }
          this.isSpeaking = true;
          if (onStart) onStart();
        };

        utterance.onend = () => {
          this.isSpeaking = false;
          if (onEnd) onEnd();
        };

        utterance.onerror = () => {
          this.isSpeaking = false;
          if (onEnd) onEnd();
        };

        // Final Re-check before calling speak()
        if (
          !this.isAuthenticated ||
          !this.isLiveActive ||
          this.activeExperimentId !== experimentId ||
          this.currentSessionId !== sessionId
        ) {
          return false;
        }

        window.speechSynthesis.speak(utterance);
        return true;
      } catch (e) {
        console.warn('[VoiceController] SpeechSynthesis execution failed:', e);
        return false;
      }
    }

    return true;
  }

  public getIsSpeaking(): boolean {
    return this.isSpeaking;
  }

  public getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  public getActiveExperimentId(): string | null {
    return this.activeExperimentId;
  }

  public getIsLiveActive(): boolean {
    return this.isLiveActive;
  }
}

export const voiceController = new VoiceController();
export default voiceController;
