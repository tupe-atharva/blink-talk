// Eye signals → SignalEvents for the UI (contract in CLAUDE.md).
//   default          LightGBM classifier (public/models/signal_trees.json), every 100 ms on 1 s windows
//   ?signals=rules   threshold rules from calibration (fallback)
//   ?mode=scan       highlight auto-advances every 1.5 s; long blink selects (no gaze needed)
import type { SignalEvent } from "./mockSignals";
import { startFace } from "./face";
import { loadCalibration } from "./calibrate";
import {
  DEFAULT_CALIBRATION, DEFAULT_DECODE, Decoder, FeatureBuffer, predictTrees,
  type Meta, type Row, type Trees,
} from "./signalCore";

export type { SignalEvent };

export type TrackingDebug = { row: Row; probs: Record<string, number> | null; mode: string };

let modelCache: Promise<{ meta: Meta; trees: Trees }> | null = null;
function loadModel() {
  modelCache ??= (async () => {
    const meta = (await (await fetch("/models/signal_meta.json")).json()) as Meta;
    const trees = (await (await fetch("/models/signal_trees.json")).json()) as Trees;
    if (trees.features.join() !== meta.features.join()) throw new Error("model files out of sync");
    return { meta, trees };
  })();
  modelCache.catch(() => (modelCache = null));
  return modelCache;
}

export async function startTracking(
  video: HTMLVideoElement,
  onEvent: (e: SignalEvent) => void,
  opts: { onDebug?: (d: TrackingDebug) => void; swapLeftRight?: boolean } = {},
): Promise<() => void> {
  const q = new URLSearchParams(window.location.search);
  const scan = q.get("mode") === "scan";
  const cal = loadCalibration() ?? DEFAULT_CALIBRATION;
  if (!loadCalibration()) console.warn("[signals] no calibration found, using defaults; open /calibrate");

  let model: { meta: Meta; trees: Trees } | null = null;
  if (q.get("signals") !== "rules") {
    try {
      model = await loadModel();
    } catch (e) {
      console.warn("[signals] classifier unavailable, using rules", e);
    }
  }
  const mode = model ? "classifier" : "rules";
  const buf = new FeatureBuffer(cal, model?.meta.closed_delta ?? 0.3);
  // Speed tuning (live, no rebuild): /talk?dwell=3&blink=0.45
  //   dwell = how many 100 ms checks a look must last before the highlight moves (default 3 → ~0.3 s)
  //   blink = seconds the eyes must stay closed for a select (default 0.45 s; natural blinks are ~0.1-0.2 s)
  const num = (k: string, d: number) => {
    const v = Number(q.get(k));
    return q.has(k) && Number.isFinite(v) && v > 0 ? v : d;
  };
  const decode = {
    ...DEFAULT_DECODE,
    ...(model?.meta.decode ?? {}),
    consec: { ...DEFAULT_DECODE.consec, move: Math.round(num("dwell", DEFAULT_DECODE.consec.move)) },
    select_min_s: num("blink", DEFAULT_DECODE.select_min_s ?? 0.45),
  };
  const dec = new Decoder(cal, decode, mode, !scan);
  const step = model?.meta.step_s ?? 0.1;
  const win = model?.meta.window_s ?? 1.0;
  console.info(`[signals] mode=${mode}${scan ? " +scan" : ""}, calibrated=${cal.ts > 0}, dwell=${decode.consec.move}, blink=${decode.select_min_s}s`);

  const fi = (n: string) => model?.meta.features.indexOf(n) ?? -1;
  const [yMax, yMin, pMax, pMin] = ["yaw_rel_max", "yaw_rel_min", "pitch_rel_max", "pitch_rel_min"].map(fi);
  let lastInfer = 0, lastFaceT = performance.now() / 1000, faceShown = true, lastProbs: Record<string, number> | null = null;

  const stopFace = await startFace(
    video,
    (f) => {
      if (!f.faceFound) {
        if (faceShown && f.t - lastFaceT > 0.5) {
          faceShown = false;
          onEvent({ type: "hold", progress: 0 });
          onEvent({ type: "face", found: false });
        }
        return;
      }
      lastFaceT = f.t;
      if (!faceShown) {
        faceShown = true;
        onEvent({ type: "face", found: true });
      }
      const row = buf.push(f);
      if (!row) return;
      dec.onFrame(row, onEvent);
      if (model && row.t - lastInfer >= step) {
        lastInfer = row.t;
        const x = buf.features(row.t, model.meta.features, win);
        if (x) {
          lastProbs = predictTrees(model.trees, x);
          const headRange = yMax >= 0 ? Math.max(x[yMax] - x[yMin], x[pMax] - x[pMin]) : 0;
          dec.onProbs(lastProbs, row, onEvent, headRange);
        }
      }
      opts.onDebug?.({ row, probs: lastProbs, mode });
    },
    { swapLeftRight: opts.swapLeftRight },
  );

  const scanTimer = scan
    ? window.setInterval(() => {
        if (faceShown && !dec.resting && !dec.eyesClosed()) onEvent({ type: "move", dir: 1 });
      }, 1500)
    : 0;

  return () => {
    stopFace();
    if (scanTimer) window.clearInterval(scanTimer);
  };
}
