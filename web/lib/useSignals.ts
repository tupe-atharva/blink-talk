"use client";
import { useEffect, useRef } from "react";
import { SignalEvent, startKeyboardSignals } from "./mockSignals";
import { startTracking } from "./signals";

// Eye tracking (signals.ts) + keyboard backup, always both.
// ?keys=1 → keyboard only (no camera).
export function useSignals(handler: (e: SignalEvent) => void) {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    const emit = (e: SignalEvent) => ref.current(e);
    const stopKeys = startKeyboardSignals(emit);
    if (new URLSearchParams(window.location.search).has("keys")) return stopKeys;

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("aria-hidden", "true");
    Object.assign(video.style, { position: "fixed", width: "2px", height: "2px", opacity: "0", pointerEvents: "none", bottom: "0", right: "0" });
    document.body.appendChild(video);

    let stopEyes: (() => void) | null = null;
    let cancelled = false;
    startTracking(video, emit)
      .then((stop) => (cancelled ? stop() : (stopEyes = stop)))
      .catch((err) => console.error("[BlinkTalk] eye tracking failed, keyboard still works:", err));

    return () => {
      cancelled = true;
      stopKeys();
      stopEyes?.();
      video.remove();
    };
  }, []);
}
