"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { runCalibration, loadCalibration, type CalStep } from "@/lib/calibrate";

export default function Calibrate() {
  const video = useRef<HTMLVideoElement>(null);
  const abort = useRef<AbortController | null>(null);
  const [step, setStep] = useState<CalStep | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSaved, setHasSaved] = useState(false);

  useEffect(() => {
    setHasSaved(!!loadCalibration());
    return () => abort.current?.abort();
  }, []);

  const start = async () => {
    if (!video.current || running) return;
    setError(null);
    setRunning(true);
    abort.current = new AbortController();
    try {
      await runCalibration(video.current, setStep, { signal: abort.current.signal });
      setHasSaved(true);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message || "Camera not available");
    } finally {
      setRunning(false);
    }
  };

  const dotLeft = step?.dot === "left" ? "4%" : step?.dot === "right" ? "96%" : "50%";
  const done = step?.key === "done";

  return (
    <main className="relative mx-auto flex min-h-screen max-w-5xl flex-col items-center justify-center gap-8 p-6 text-center">
      {/* target dot (fixed to the screen edges so eye movements are large) */}
      {running && step && step.key !== "done" && (
        <div
          className="pointer-events-none fixed top-1/2 z-30 h-12 w-12 -translate-x-1/2 -translate-y-1/2 rounded-full bg-highlight shadow-glow transition-[left] duration-150"
          style={{ left: dotLeft }}
        />
      )}

      <div className="glass flex w-full flex-col items-center gap-5 rounded-4xl px-8 py-10">
        <p className="caption">
          {step && !done ? `Step ${step.stepIndex + 1} of ${step.steps}${step.reps > 1 ? ` · ${step.rep}/${step.reps}` : ""}` : "Calibration"}
        </p>
        <h1 className="font-display text-6xl font-extrabold italic leading-none tracking-tight md:text-7xl">
          {step ? step.title : "Calibration"}
        </h1>
        <p className="max-w-xl text-2xl leading-relaxed text-muted">
          {error ?? step?.instruction ?? "Sit like you will during use, face centered. About 40 seconds: relax, look left, look right, long blinks, raise eyebrows."}
        </p>
        <video
          ref={video}
          muted
          playsInline
          className="h-40 w-56 rounded-2xl object-cover"
          style={{ transform: "scaleX(-1)", opacity: running ? 1 : 0.4 }}
        />
        {step && !step.faceFound && running && (
          <p className="rounded-full bg-danger/20 px-5 py-2 text-lg text-danger">Looking for your face…</p>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        {!running && (
          <button onClick={start} className="glow rounded-full bg-highlight px-10 py-5 text-2xl font-semibold text-onhighlight shadow-glow">
            {done || hasSaved ? "Calibrate again" : "Start"}
          </button>
        )}
        {running && (
          <button onClick={() => abort.current?.abort()} className="glass glow rounded-full px-8 py-5 text-xl font-medium">
            Cancel
          </button>
        )}
        {!running && (hasSaved || done) && (
          <Link href="/talk" className="glow rounded-full bg-highlight px-10 py-5 text-2xl font-semibold text-onhighlight shadow-glow">
            Done
          </Link>
        )}
        <Link href="/" className="glass glow rounded-full px-8 py-5 text-xl font-medium">Back home</Link>
      </div>
    </main>
  );
}
