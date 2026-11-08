// Pure eye-signal logic (no DOM): per-frame features, 1 s window features, tree model, event decoding.
// Shared by signals.ts (browser) and the parity test. Must match ml/notebooks/02_train_classifier.ipynb.
import type { SignalEvent } from "./mockSignals";

export type Frame = {
  t: number; // seconds
  faceFound: boolean;
  blink: number; lookA: number; lookB: number; brow: number;
  yaw: number; pitch: number; jaw: number; smile: number;
};

export type Rest = { blink: number; h: number; lookA: number; lookB: number; brow: number; yaw: number; pitch: number };

export type Calibration = {
  rest: Rest;
  hsign: 1 | -1; // +1 if looking at screen-LEFT raises h = lookA - lookB
  peaks: { left: number; right: number; brow: number; blink: number }; // relative to rest, after hsign (left > 0, right < 0)
  ts: number;
  notes?: string[];
};

export const DEFAULT_CALIBRATION: Calibration = {
  rest: { blink: 0.15, h: 0.05, lookA: 0.1, lookB: 0.05, brow: 0.01, yaw: 0, pitch: 0 },
  hsign: 1,
  peaks: { left: 0.6, right: -0.6, brow: 0.5, blink: 0.6 },
  ts: 0,
};

export type Decode = {
  p_min: number;
  consec: { select: number; move: number; undo: number };
  rearm_p_idle: number;
  select_min_s?: number; // eyes must also stay closed this long for a select
  head_gate_deg?: number; // ignore moves while the head turns more than this within the window
};
export const DEFAULT_DECODE: Decode = {
  p_min: 0.5, consec: { select: 1, move: 3, undo: 2 }, rearm_p_idle: 0.5, select_min_s: 0.45, head_gate_deg: 8,
};

export type Meta = { features: string[]; window_s: number; step_s: number; closed_delta: number; decode: Decode };
export type Tree = { f: number[]; t: number[]; l: number[]; r: number[]; v: number[] };
export type Trees = { classes: string[]; num_class: number; features: string[]; trees: Tree[] };

export type Row = {
  t: number;
  blink_rel: number; h_rel: number; lookA_rel: number; lookB_rel: number;
  brow_rel: number; yaw_rel: number; pitch_rel: number; jaw: number; smile: number;
  closed: boolean;
  cd: number; // seconds closed so far (0 if open)
};

const SERIES = ["blink_rel", "h_rel", "lookA_rel", "lookB_rel", "brow_rel", "yaw_rel", "pitch_rel", "jaw", "smile"] as const;
type SeriesName = (typeof SERIES)[number];

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Keeps ~1.5 s of frames relative to the user's calibration and computes window features. */
export class FeatureBuffer {
  rows: Row[] = [];
  private closedStart: number | null = null;
  dtEst = 1 / 30;

  constructor(public cal: Calibration, public closedDelta = 0.3, private keepS = 1.5) {}

  push(f: Frame): Row | null {
    if (!f.faceFound) return null;
    const r = this.cal.rest;
    let hRel = f.lookA - f.lookB - r.h;
    let aRel = f.lookA - r.lookA;
    let bRel = f.lookB - r.lookB;
    if (this.cal.hsign < 0) {
      hRel = -hRel;
      [aRel, bRel] = [bRel, aRel];
    }
    const blinkRel = f.blink - r.blink;
    const closed = blinkRel > this.closedDelta;
    const prev = this.rows[this.rows.length - 1];
    if (prev) {
      const d = f.t - prev.t;
      if (d > 0 && d < 0.2) this.dtEst = 0.9 * this.dtEst + 0.1 * d;
    }
    let cd = 0;
    if (closed) {
      if (this.closedStart === null) this.closedStart = f.t;
      cd = f.t - this.closedStart + this.dtEst;
    } else {
      this.closedStart = null;
    }
    const row: Row = {
      t: f.t, blink_rel: blinkRel, h_rel: hRel, lookA_rel: aRel, lookB_rel: bRel,
      brow_rel: f.brow - r.brow, yaw_rel: f.yaw - r.yaw, pitch_rel: f.pitch - r.pitch,
      jaw: f.jaw, smile: f.smile, closed, cd,
    };
    this.rows.push(row);
    while (this.rows.length && this.rows[0].t < f.t - this.keepS) this.rows.shift();
    return row;
  }

  /** Features of the window (te - win, te], in `names` order. null if too few frames. */
  features(te: number, names: string[], win = 1.0): number[] | null {
    const w = this.rows.filter((r) => r.t >= te - win && r.t <= te);
    return windowFeatures(w, names);
  }
}

export function windowFeatures(w: Row[], names: string[]): number[] | null {
  if (w.length < 11) return null;
  const tw = w.map((r) => r.t);
  const tm = tw.reduce((a, b) => a + b, 0) / tw.length;
  const x = tw.map((t) => t - tm);
  const den = x.reduce((a, b) => a + b * b, 0) || 1e-9;
  const feats: Record<string, number> = {};
  for (const s of SERIES) {
    const v = w.map((r) => r[s as SeriesName]);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    let min = Infinity, max = -Infinity, num = 0;
    for (let i = 0; i < v.length; i++) {
      if (v[i] < min) min = v[i];
      if (v[i] > max) max = v[i];
      num += x[i] * (v[i] - mean);
    }
    feats[`${s}_mean`] = mean;
    feats[`${s}_min`] = min;
    feats[`${s}_max`] = max;
    feats[`${s}_last`] = v[v.length - 1];
    feats[`${s}_slope`] = num / den;
  }
  // closures inside the window
  const c = w.map((r) => r.closed);
  const diffs: number[] = [];
  for (let i = 1; i < tw.length; i++) diffs.push(tw[i] - tw[i - 1]);
  const dt = diffs.length ? median(diffs) : 1 / 30;
  const runs: [number, number][] = [];
  for (let i = 0; i < c.length; i++) {
    if (!c[i]) continue;
    let j = i;
    while (j + 1 < c.length && c[j + 1]) j++;
    runs.push([i, j]);
    i = j;
  }
  const lengths = runs.map(([a, b]) => tw[b] - tw[a] + dt);
  let since = 1.0;
  if (runs.length && !c[c.length - 1]) since = Math.min(1.0, tw[tw.length - 1] - tw[runs[runs.length - 1][1]]);
  else if (c[c.length - 1]) since = 0.0;
  feats.closed_dur = w[w.length - 1].cd;
  feats.longest_closure = lengths.length ? Math.max(...lengths) : 0;
  feats.n_closures = runs.length;
  feats.frac_closed = c.filter(Boolean).length / c.length;
  feats.since_closure = since;
  return names.map((n) => {
    const v = feats[n];
    if (v === undefined) throw new Error(`unknown feature ${n}`);
    return v;
  });
}

/** LightGBM multiclass trees → class probabilities (softmax of summed leaf values). */
export function predictTrees(m: Trees, x: number[]): Record<string, number> {
  const raw = new Array<number>(m.num_class).fill(0);
  for (let k = 0; k < m.trees.length; k++) {
    const tr = m.trees[k];
    let leaf: number;
    if (!tr.f.length) leaf = tr.v[0];
    else {
      let n = 0;
      while (n >= 0) n = x[tr.f[n]] <= tr.t[n] ? tr.l[n] : tr.r[n];
      leaf = tr.v[-n - 1];
    }
    raw[k % m.num_class] += leaf;
  }
  const mx = Math.max(...raw);
  const e = raw.map((r) => Math.exp(r - mx));
  const sum = e.reduce((a, b) => a + b, 0);
  const out: Record<string, number> = {};
  m.classes.forEach((c, i) => (out[c] = e[i] / sum));
  return out;
}

// ---------------------------------------------------------------------------------------------
// Event decoding

const DEFAULT_SELECT_S = 0.45; // eyes closed this long (and the model agrees) = long blink
const REST_S = 5.0;
const SHORT_BLINK_S = 0.4;
const EMERGENCY_WINDOW_S = 1.5;

type Emit = (e: SignalEvent) => void;

export class Decoder {
  resting = false;
  private armed = false; // this closure counts as a long blink
  private closureStart: number | null = null;
  private lastHoldEmit = 0;
  private shortBlinks: number[] = [];
  private runs = { select: 0, move: 0, undo: 0 };
  private latched = false; // waiting for idle after a move/undo
  private ruleMove: { side: -1 | 0 | 1; since: number; latched: boolean } = { side: 0, since: 0, latched: false };
  private ruleBrow = { since: -1, latched: false };
  private restEmitted = false;

  constructor(private cal: Calibration, private dec: Decode, public mode: "classifier" | "rules", private gazeMoves = true) {}

  eyesClosed(): boolean {
    return this.closureStart !== null;
  }

  /** Every frame: closures → hold / select / rest / emergency; rules mode also does moves and undo. */
  onFrame(row: Row, emit: Emit) {
    if (row.closed) {
      if (this.closureStart === null) {
        this.closureStart = row.t - (row.cd || 0);
        this.armed = false;
      }
      const dur = row.cd;
      if (this.mode === "rules" && dur >= this.selectMin()) this.armed = true;
      if (!this.resting && row.t - this.lastHoldEmit >= 0.05) {
        this.lastHoldEmit = row.t;
        emit({ type: "hold", progress: Math.min(1, dur / this.selectMin()) });
      }
      if (dur > REST_S && !this.resting && !this.restEmitted) {
        this.resting = true;
        this.restEmitted = true;
        emit({ type: "hold", progress: 0 });
        emit({ type: "rest", on: true });
      }
    } else if (this.closureStart !== null) {
      // eyes just reopened
      const dur = row.t - this.closureStart;
      const wasArmed = this.armed;
      this.closureStart = null;
      this.armed = false;
      emit({ type: "hold", progress: 0 });
      if (this.restEmitted) {
        this.restEmitted = false; // the closure that started rest ends here; stay resting
      } else if (this.resting) {
        if (wasArmed && dur >= this.selectMin()) {
          this.resting = false;
          emit({ type: "rest", on: false }); // long blink wakes up
        }
      } else if (wasArmed && dur >= this.selectMin() && dur <= REST_S) {
        emit({ type: "select" });
      } else if (!wasArmed && dur < SHORT_BLINK_S) {
        this.shortBlinks = this.shortBlinks.filter((t) => row.t - t <= EMERGENCY_WINDOW_S);
        this.shortBlinks.push(row.t);
        if (this.shortBlinks.length >= 3) {
          this.shortBlinks = [];
          emit({ type: "emergency" });
        }
      }
    }
    if (this.mode === "rules" && !row.closed && !this.resting) this.rulesGazeBrow(row, emit);
  }

  private selectMin() {
    return this.dec.select_min_s ?? DEFAULT_SELECT_S;
  }

  /** Every inference (≈100 ms) in classifier mode. headRange = max(yaw range, pitch range) in the window, degrees. */
  onProbs(p: Record<string, number>, row: Row, emit: Emit, headRange = 0) {
    const pl = p.left ?? 0, pr = p.right ?? 0;
    const cand = { select: p.long_blink ?? 0, move: pl + pr, undo: p.brow_up ?? 0 };
    (Object.keys(cand) as (keyof typeof cand)[]).forEach((k) => {
      this.runs[k] = cand[k] >= this.dec.p_min ? this.runs[k] + 1 : 0;
    });
    if (row.closed && this.runs.select >= this.dec.consec.select) this.armed = true;
    if (row.closed || this.resting) return;
    if (this.latched) {
      if ((p.idle ?? 0) + (p.natural_blink ?? 0) >= this.dec.rearm_p_idle) this.latched = false;
      return;
    }
    const headStill = headRange < (this.dec.head_gate_deg ?? Infinity);
    if (this.gazeMoves && this.runs.move >= this.dec.consec.move && headStill) {
      emit({ type: "move", dir: pl >= pr ? -1 : 1 });
      this.latched = true;
    } else if (this.runs.undo >= this.dec.consec.undo) {
      emit({ type: "undo" });
      this.latched = true;
    }
  }

  private rulesGazeBrow(row: Row, emit: Emit) {
    const { left, right, brow } = this.cal.peaks;
    if (this.gazeMoves) {
      const side: -1 | 0 | 1 = row.h_rel > 0.5 * left ? -1 : row.h_rel < 0.5 * right ? 1 : 0;
      const m = this.ruleMove;
      if (m.latched) {
        if (Math.abs(row.h_rel) < 0.25 * Math.min(Math.abs(left), Math.abs(right))) m.latched = false; // back to center
      } else if (side !== m.side) {
        m.side = side;
        m.since = row.t;
      } else if (side !== 0 && row.t - m.since >= 0.3) {
        emit({ type: "move", dir: side });
        m.latched = true;
        m.side = 0;
      }
    }
    const b = this.ruleBrow;
    if (b.latched) {
      if (row.brow_rel < 0.25 * brow) b.latched = false;
    } else if (row.brow_rel > 0.5 * brow) {
      if (b.since < 0) b.since = row.t;
      else if (row.t - b.since >= 0.2) {
        emit({ type: "undo" });
        b.latched = true;
        b.since = -1;
      }
    } else b.since = -1;
  }
}
