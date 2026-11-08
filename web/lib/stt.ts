// Caregiver speech-to-text with the browser's Web Speech API (Chrome / Edge / Safari).
// startListening(onFinal) → stop(). Auto-restarts. Call muteListening(ms) while BlinkTalk is speaking
// so the app doesn't transcribe its own voice as the caregiver.

type SRResult = { isFinal: boolean; 0: { transcript: string } };
type SREvent = { resultIndex: number; results: ArrayLike<SRResult> };
type SRErrorEvent = { error: string };
interface SR {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((e: SREvent) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: SRErrorEvent) => void) | null;
  start(): void;
  abort(): void;
}
type SRCtor = new () => SR;

function getCtor(): SRCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

let mutedUntil = 0;

/** Ignore anything heard for the next `ms` milliseconds (e.g. while our own TTS plays). */
export function muteListening(ms: number) {
  mutedUntil = Math.max(mutedUntil, performance.now() + ms);
}

export function sttSupported(): boolean {
  return getCtor() !== null;
}

export function startListening(onFinal: (text: string) => void): () => void {
  const Ctor = getCtor();
  if (!Ctor) {
    console.warn("[stt] Web Speech API not supported in this browser (use Chrome)");
    return () => {};
  }
  let stopped = false;
  let rec: SR | null = null;
  let restartDelay = 300;

  const start = () => {
    if (stopped) return;
    rec = new Ctor();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = false;
    rec.onresult = (e) => {
      restartDelay = 300;
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (!r.isFinal) continue;
        const text = r[0].transcript.trim();
        if (text && performance.now() >= mutedUntil) onFinal(text);
      }
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        console.warn("[stt] microphone permission denied");
        stopped = true;
      } else if (e.error !== "no-speech" && e.error !== "aborted") {
        console.warn("[stt] error:", e.error);
        restartDelay = Math.min(5000, restartDelay * 2);
      }
    };
    rec.onend = () => {
      if (!stopped) setTimeout(start, restartDelay);
    };
    try {
      rec.start();
    } catch (err) {
      console.warn("[stt] start failed", err);
      setTimeout(start, 1000);
    }
  };

  start();
  return () => {
    stopped = true;
    try {
      rec?.abort();
    } catch {
      /* ignore */
    }
  };
}
