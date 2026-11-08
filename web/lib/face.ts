// MediaPipe Face Landmarker → per-frame eye features. Video never leaves the browser.
// Assets are served locally (works offline at the demo):
//   public/mediapipe/wasm/*              (copied from node_modules/@mediapipe/tasks-vision/wasm)
//   public/models/face_landmarker.task   (downloaded once, see README steps)
import type { FaceLandmarker, FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import type { Frame } from "./signalCore";

export type { Frame as FaceFrame };

const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/models/face_landmarker.task";

let landmarker: Promise<FaceLandmarker> | null = null;

function getLandmarker(): Promise<FaceLandmarker> {
  if (!landmarker) {
    landmarker = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      const opts = (delegate: "GPU" | "CPU") => ({
        baseOptions: { modelAssetPath: MODEL_PATH, delegate },
        runningMode: "VIDEO" as const,
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      try {
        return await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
      } catch (e) {
        console.warn("[face] GPU delegate failed, using CPU", e);
        return await FaceLandmarker.createFromOptions(fileset, opts("CPU"));
      }
    })();
    landmarker.catch(() => (landmarker = null)); // allow retry after a failure
  }
  return landmarker;
}

export async function openCamera(video: HTMLVideoElement): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  return stream;
}

function toFrame(res: FaceLandmarkerResult, t: number, swap: boolean): Frame {
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (!cats) return { t, faceFound: false, blink: 0, lookA: 0, lookB: 0, brow: 0, yaw: 0, pitch: 0, jaw: 0, smile: 0 };
  const s: Record<string, number> = {};
  for (const c of cats) s[c.categoryName] = c.score;
  let yaw = 0, pitch = 0;
  const m = res.facialTransformationMatrixes?.[0]?.data; // 4x4 column-major
  if (m) {
    yaw = (Math.atan2(m[8], m[10]) * 180) / Math.PI;
    pitch = (Math.asin(Math.max(-1, Math.min(1, -m[9]))) * 180) / Math.PI;
  }
  let lookA = (s.eyeLookOutLeft + s.eyeLookInRight) / 2;
  let lookB = (s.eyeLookInLeft + s.eyeLookOutRight) / 2;
  if (swap) [lookA, lookB] = [lookB, lookA];
  return {
    t,
    faceFound: true,
    blink: (s.eyeBlinkLeft + s.eyeBlinkRight) / 2,
    lookA,
    lookB,
    brow: s.browInnerUp,
    yaw,
    pitch,
    jaw: s.jawOpen ?? 0,
    smile: ((s.mouthSmileLeft ?? 0) + (s.mouthSmileRight ?? 0)) / 2,
  };
}

/**
 * Starts the camera (unless `video` already has a stream) and calls onFrame for every new video frame.
 * Returns stop(), which also stops the camera if this function opened it.
 */
export async function startFace(
  video: HTMLVideoElement,
  onFrame: (f: Frame) => void,
  opts: { swapLeftRight?: boolean } = {},
): Promise<() => void> {
  const ownStream = video.srcObject ? null : await openCamera(video);
  const lm = await getLandmarker();
  let stopped = false, raf = 0, lastVideoTime = -1, lastTs = 0;
  const loop = () => {
    if (stopped) return;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = Math.max(performance.now(), lastTs + 1); // MediaPipe needs increasing timestamps
      lastTs = now;
      try {
        onFrame(toFrame(lm.detectForVideo(video, now), now / 1000, !!opts.swapLeftRight));
      } catch (e) {
        console.warn("[face] detect failed", e);
      }
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  return () => {
    stopped = true;
    cancelAnimationFrame(raf);
    if (ownStream) {
      ownStream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
  };
}
