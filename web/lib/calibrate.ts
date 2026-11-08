// Guided calibration → per-user rest values and gesture peaks, saved in localStorage.
// signals.ts reads them. Steps: rest 3 s → look left ×3 → look right ×3 → long blink ×3 → eyebrows ×2.
import { startFace } from "./face";
import { DEFAULT_CALIBRATION, type Calibration, type Frame, type Rest } from "./signalCore";

const KEY = "blinktalk.calibration.v1";

export function loadCalibration(): Calibration | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Calibration) : null;
  } catch {
    return null;
  }
}

export function saveCalibration(c: Calibration) {
  try {
    localStorage.setItem(KEY, JSON.stringify(c));
  } catch {
    /* private mode etc. */
  }
}

export function clearCalibration() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export type CalStepKey = "rest" | "left" | "right" | "blink" | "brow" | "done";
export type CalStep = {
  key: CalStepKey;
  title: string;
  instruction: string;
  dot: "center" | "left" | "right";
  phase: "ready" | "go" | "relax" | "done";
  rep: number; // 1-based
  reps: number;
  stepIndex: number; // 0-based
  steps: number;
  faceFound: boolean;
};

const median = (xs: number[]) => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

let audio: AudioContext | null = null;
function beep(freq: number) {
  try {
    audio ??= new AudioContext();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq;
    g.gain.value = 0.15;
    o.connect(g);
    g.connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + 0.12);
  } catch {
    /* no audio */
  }
}

/**
 * Runs the guided calibration on `video` (shows the camera if it is visible), reports each step via onStep,
 * saves and returns the calibration. Abort with opts.signal.
 */
export async function runCalibration(
  video: HTMLVideoElement,
  onStep: (s: CalStep) => void,
  opts: { signal?: AbortSignal } = {},
): Promise<Calibration> {
  const frames: Frame[] = [];
  let faceFound = false;
  const stop = await startFace(video, (f) => {
    faceFound = f.faceFound;
    if (f.faceFound) frames.push(f);
  });
  const now = () => performance.now() / 1000;
  const sleep = (s: number) =>
    new Promise<void>((resolve, reject) => {
      const id = setTimeout(resolve, s * 1000);
      opts.signal?.addEventListener("abort", () => {
        clearTimeout(id);
        reject(new DOMException("aborted", "AbortError"));
      });
    });
  const between = (a: number, b: number) => frames.filter((f) => f.t >= a && f.t <= b);

  const STEPS: { key: Exclude<CalStepKey, "done">; title: string; reps: number }[] = [
    { key: "rest", title: "Relax", reps: 1 },
    { key: "left", title: "Look left", reps: 3 },
    { key: "right", title: "Look right", reps: 3 },
    { key: "blink", title: "Long blink", reps: 3 },
    { key: "brow", title: "Raise eyebrows", reps: 2 },
  ];
  const report = (i: number, rep: number, phase: CalStep["phase"], instruction: string, dot: CalStep["dot"] = "center") =>
    onStep({ key: STEPS[i].key, title: STEPS[i].title, instruction, dot, phase, rep, reps: STEPS[i].reps, stepIndex: i, steps: STEPS.length, faceFound });

  try {
    // wait for a face (max ~10 s)
    for (let k = 0; k < 50 && !faceFound; k++) {
      report(0, 1, "ready", "Center your face in the camera");
      await sleep(0.2);
    }

    // 1. rest
    report(0, 1, "ready", "Look at the dot and relax");
    await sleep(1.0);
    report(0, 1, "go", "Keep looking at the dot. Blink normally.");
    let a = now();
    await sleep(3.0);
    const restFrames = between(a, now());
    const rest: Rest = {
      blink: median(restFrames.map((f) => f.blink)),
      h: median(restFrames.map((f) => f.lookA - f.lookB)),
      lookA: median(restFrames.map((f) => f.lookA)),
      lookB: median(restFrames.map((f) => f.lookB)),
      brow: median(restFrames.map((f) => f.brow)),
      yaw: median(restFrames.map((f) => f.yaw)),
      pitch: median(restFrames.map((f) => f.pitch)),
    };

    // 2/3. left & right: mean h over the last 0.8 s of each look
    const lookTrials = async (i: number, dot: "left" | "right") => {
      const vals: number[] = [];
      for (let rep = 1; rep <= STEPS[i].reps; rep++) {
        report(i, rep, "relax", "Look at the center dot");
        await sleep(1.2);
        report(i, rep, "go", "Move only your eyes to the dot", dot);
        await sleep(0.7);
        a = now();
        await sleep(0.8);
        const fs = between(a, now());
        if (fs.length) vals.push(fs.reduce((s, f) => s + (f.lookA - f.lookB), 0) / fs.length);
      }
      return median(vals);
    };
    const leftH = await lookTrials(1, "left");
    const rightH = await lookTrials(2, "right");
    const hsign: 1 | -1 = !(leftH < rightH) ? 1 : -1;

    // 4. long blinks: peak blink while closed
    const blinkPeaks: number[] = [];
    for (let rep = 1; rep <= STEPS[3].reps; rep++) {
      report(3, rep, "relax", "Get ready: close at the first beep, open at the second");
      await sleep(1.5);
      report(3, rep, "go", "Close your eyes…");
      beep(440);
      a = now();
      await sleep(1.2);
      beep(880);
      const fs = between(a, now());
      if (fs.length) blinkPeaks.push(Math.max(...fs.map((f) => f.blink)) - rest.blink);
    }

    // 5. eyebrows
    const browPeaks: number[] = [];
    for (let rep = 1; rep <= STEPS[4].reps; rep++) {
      report(4, rep, "relax", "Relax your face");
      await sleep(1.5);
      report(4, rep, "go", "Raise your eyebrows");
      a = now();
      await sleep(1.2);
      const fs = between(a, now());
      if (fs.length) browPeaks.push(Math.max(...fs.map((f) => f.brow)) - rest.brow);
    }

    const cal: Calibration = {
      rest,
      hsign,
      peaks: {
        left: (leftH - rest.h) * hsign,
        right: (rightH - rest.h) * hsign,
        blink: median(blinkPeaks),
        brow: median(browPeaks),
      },
      ts: Date.now(),
      notes: [],
    };
    // sanity checks: fall back to defaults for anything unusable
    const n = cal.notes!;
    for (const k of Object.keys(cal.rest) as (keyof Rest)[]) {
      if (!Number.isFinite(cal.rest[k])) { cal.rest[k] = DEFAULT_CALIBRATION.rest[k]; n.push(`rest.${k} missing`); }
    }
    if (!(cal.peaks.left > 0.15)) { n.push("left look was weak"); cal.peaks.left = Math.max(cal.peaks.left || 0, 0.3); }
    if (!(cal.peaks.right < -0.15)) { n.push("right look was weak"); cal.peaks.right = Math.min(cal.peaks.right || 0, -0.3); }
    if (!(cal.peaks.blink > 0.3)) n.push("eyes did not close fully: blink detection may be weak");
    if (!(cal.peaks.brow > 0.1)) { n.push("eyebrow raise was weak"); cal.peaks.brow = Math.max(cal.peaks.brow || 0, 0.3); }

    saveCalibration(cal);
    onStep({ key: "done", title: "Done", instruction: n.length ? n.join(" · ") : "Calibration saved", dot: "center", phase: "done", rep: 1, reps: 1, stepIndex: STEPS.length, steps: STEPS.length, faceFound });
    return cal;
  } finally {
    stop();
  }
}
