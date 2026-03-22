import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

type ActionName = "raise_arm";

type FramePayload = {
  pose: number[][];
  left_hand: number[][];
  right_hand: number[][];
};

type EvalResponse = any;

type ProfileState = {
  ok?: boolean;
  exists?: boolean;
  user_name?: string;
  video_path?: string | null;
  photo_path?: string | null;
  avatar_url?: string | null;
  photo_url?: string | null;
  profile?: any;
};

const ACTIONS: ActionName[] = ["raise_arm"];

const TARGET_DURATION_MS = 3000;
const SAMPLE_INTERVAL_MS = 33;
const EVAL_TIMEOUT_MS = 12_000;
const LLM_TIMEOUT_MS = 20_000;

const MP_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const POSE_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const API_BASE = "http://127.0.0.1:8000";
const EVAL_API = "/api/evaluate";
const LLM_PING_API = "/api/llm_ping";
const LLM_CONFIRM_API = "/api/confirm";
const COACH_VIDEO_API = `${API_BASE}/api/coach_video_v2`;
const PROFILE_CHECK_API = `${API_BASE}/api/profile/check`;
const PROFILE_REGISTER_V2_API = `${API_BASE}/api/profile/register_v2`;
const STANDARD_INFO_API = `${API_BASE}/api/standard_video/info`;
const STANDARD_BUILD_API = `${API_BASE}/api/standard_video/build`;

const DEMO_VIDEO_BY_ACTION: Record<ActionName, string> = {
  raise_arm: "/demos/raise_arm.mp4",
};

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

function joinApiUrl(path?: string | null) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...init, signal: controller.signal });
    return resp;
  } finally {
    window.clearTimeout(timer);
  }
}

function RehabMain({ userName }: { userName: string }) {
  const [action, setAction] = useState<ActionName>("raise_arm");

  const [mpStatus, setMpStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle"
  );
  const [mpError, setMpError] = useState<string>("");

  const [cameraOn, setCameraOn] = useState(false);
  const [poseDetected, setPoseDetected] = useState(false);
  const [handsDetected, setHandsDetected] = useState(false);

  const [framesBuffered, setFramesBuffered] = useState(0);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const poseLmStdRef = useRef<PoseLandmarker | null>(null);

  const [isPinging, setIsPinging] = useState(false);
  const [pingResult, setPingResult] = useState<any | null>(null);
  const [pingError, setPingError] = useState<string>("");

  const [isConfirming, setIsConfirming] = useState(false);
  const [confirmResult, setConfirmResult] = useState<any | null>(null);
  const [confirmError, setConfirmError] = useState<string>("");

  const [result, setResult] = useState<EvalResponse | null>(null);
  const [evalError, setEvalError] = useState<string>("");

  const [useLLM, setUseLLM] = useState<boolean>(true);

  const demoVideoSrc = useMemo(() => DEMO_VIDEO_BY_ACTION[action], [action]);

  const [stdOverrideSrc, setStdOverrideSrc] = useState<string | null>(null);
  const [isCoachPlaying, setIsCoachPlaying] = useState(false);
  const [buildingStandard, setBuildingStandard] = useState(false);
  const [statusText, setStatusText] = useState("");

  const userVideoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const standardVideoRef = useRef<HTMLVideoElement | null>(null);
  const stdPlayerRef = useRef<HTMLVideoElement | null>(null);

  const poseLmRef = useRef<PoseLandmarker | null>(null);
  const handLmRef = useRef<HandLandmarker | null>(null);
  const drawingRef = useRef<DrawingUtils | null>(null);

  const rafRef = useRef<number | null>(null);
  const lastSampleTsRef = useRef<number>(0);

  const framesRef = useRef<FramePayload[]>([]);
  const userKeyframesRef = useRef<string[]>([]);
  const stdKeyframesRef = useRef<string[]>([]);

  const captureStartTsRef = useRef<number | null>(null);
  const captureDoneRef = useRef<boolean>(false);

  const standardSeqRef = useRef<FramePayload[]>([]);
  const standardReadyRef = useRef<boolean>(false);

  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setMpStatus("loading");
        setMpError("");

        const vision = await FilesetResolver.forVisionTasks(MP_WASM_BASE);

        const poseLmUser = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: POSE_TASK_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        const poseLmStd = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: POSE_TASK_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        const handLm = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: HAND_TASK_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (cancelled) return;

        poseLmRef.current = poseLmUser;
        poseLmStdRef.current = poseLmStd;
        handLmRef.current = handLm;

        const canvas = overlayRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) drawingRef.current = new DrawingUtils(ctx);
        }

        setMpStatus("ready");
      } catch (e: any) {
        if (cancelled) return;
        setMpStatus("error");
        setMpError(
          String(e?.message || e) +
            "\n（如果提示 Unable to open zip archive，大概率是 .task 文件 URL 404 或本地 models 不存在）"
        );
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  async function ensureUserStandardVideo(currentUserName: string, currentAction: string) {
    try {
      setBuildingStandard(true);
      setStatusText("正在检查个性化标准动作视频...");

      const infoResp = await fetch(
        `${STANDARD_INFO_API}?user_name=${encodeURIComponent(currentUserName)}&action=${encodeURIComponent(currentAction)}`
      );
      const info = await infoResp.json();

      if (info.generated_exists && info.display_video_url) {
        setStdOverrideSrc(joinApiUrl(info.display_video_url));
        setStatusText("已加载个性化标准动作视频");
        return;
      }

      setStatusText("正在生成个性化标准动作视频（MusePose）...");
      const buildResp = await fetch(STANDARD_BUILD_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_name: currentUserName,
          action: currentAction,
          force: false,
        }),
      });

      if (!buildResp.ok) {
        const txt = await buildResp.text();
        throw new Error(txt);
      }

      const buildData = await buildResp.json();
      setStdOverrideSrc(joinApiUrl(buildData.video_url));
      setStatusText("个性化标准动作视频生成完成");
    } catch (e: any) {
      console.error(e);
      setStdOverrideSrc(null);
      setStatusText(`生成失败，回退到原始标准视频：${e?.message || e}`);
    } finally {
      setBuildingStandard(false);
    }
  }

  useEffect(() => {
    if (userName) {
      ensureUserStandardVideo(userName, action);
    }
  }, [userName, action]);

  function captureVideoFrameToDataURL(
    video: HTMLVideoElement,
    maxW = 480,
    quality = 0.7
  ): string | null {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;

    const scale = Math.min(1, maxW / w);
    const cw = Math.max(1, Math.round(w * scale));
    const ch = Math.max(1, Math.round(h * scale));

    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, cw, ch);
    return canvas.toDataURL("image/jpeg", quality);
  }

  function getKeyframeTargets(totalMs: number) {
    return [0, 0.25, 0.5, 0.75, 0.95].map((r) => r * totalMs);
  }

  async function buildStandardSequence() {
    standardReadyRef.current = false;
    standardSeqRef.current = [];
    stdKeyframesRef.current = [];

    const video = standardVideoRef.current;
    const poseLm = poseLmStdRef.current;
    if (!video || !poseLm) return;

    if (video.readyState < 2) {
      await new Promise<void>((resolve) => {
        const onLoaded = () => {
          video.removeEventListener("loadeddata", onLoaded);
          resolve();
        };
        video.addEventListener("loadeddata", onLoaded, { once: true });
        video.load();
      });
    }

    if (!video.videoWidth || !video.videoHeight || !video.duration) {
      await new Promise<void>((resolve) => {
        const onMeta = () => {
          video.removeEventListener("loadedmetadata", onMeta);
          resolve();
        };
        video.addEventListener("loadedmetadata", onMeta, { once: true });
      }).catch(() => {});
    }

    const videoDurationMs = (video.duration || 3) * 1000;
    const durMs = Math.min(TARGET_DURATION_MS, videoDurationMs);

    const seq: FramePayload[] = [];
    const baseTs = performance.now();
    const keyTargets = getKeyframeTargets(durMs);
    let nextKeyIdx = 0;

    for (let t = 0; t < durMs; t += SAMPLE_INTERVAL_MS) {
      const ct = Math.min(video.duration || 3, t / 1000);
      video.currentTime = ct;

      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked, { once: true });
      });

      if (nextKeyIdx < keyTargets.length && t >= keyTargets[nextKeyIdx]) {
        const img = captureVideoFrameToDataURL(video, 480, 0.7);
        if (img) stdKeyframesRef.current.push(img);
        nextKeyIdx += 1;
      }

      const ts = baseTs + t;
      const poseRes = poseLm.detectForVideo(video, ts);
      const posePts = poseRes.landmarks?.[0];
      if (!posePts) continue;

      const pose33x3 = posePts.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0]);
      seq.push({
        pose: pose33x3,
        left_hand: [],
        right_hand: [],
      });
    }

    while (stdKeyframesRef.current.length < keyTargets.length) {
      const img = captureVideoFrameToDataURL(video, 480, 0.7);
      if (!img) break;
      stdKeyframesRef.current.push(img);
    }

    standardSeqRef.current = seq;
    standardReadyRef.current = seq.length >= 3;
  }

  useEffect(() => {
    if (mpStatus !== "ready") return;
    buildStandardSequence().catch(() => {
      standardReadyRef.current = false;
      standardSeqRef.current = [];
    });
  }, [mpStatus, action, demoVideoSrc]);

  async function startCamera() {
    setEvalError("");
    setResult(null);
    setConfirmResult(null);
    setConfirmError("");

    const video = userVideoRef.current;
    if (!video) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540 },
        audio: false,
      });
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();

      setCameraOn(true);
      framesRef.current = [];
      setFramesBuffered(0);
      lastSampleTsRef.current = 0;
      captureStartTsRef.current = null;
      captureDoneRef.current = false;
      userKeyframesRef.current = [];

      startLoop();
    } catch (e: any) {
      setEvalError("无法打开摄像头：" + String(e?.message || e));
    }
  }

  function stopCamera() {
    setCameraOn(false);
    setPoseDetected(false);
    setHandsDetected(false);

    framesRef.current = [];
    setFramesBuffered(0);
    lastSampleTsRef.current = 0;
    captureStartTsRef.current = null;
    captureDoneRef.current = false;

    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }

  function startLoop() {
    if (rafRef.current != null) return;

    const step = (ts: number) => {
      rafRef.current = requestAnimationFrame(step);

      const video = userVideoRef.current;
      const canvas = overlayRef.current;
      const poseLm = poseLmRef.current;
      const handLm = handLmRef.current;
      const drawer = drawingRef.current;

      if (!video || !canvas || !poseLm || !handLm || !drawer) return;
      if (video.readyState < 2) return;

      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (vw === 0 || vh === 0) return;
      if (canvas.width !== vw) canvas.width = vw;
      if (canvas.height !== vh) canvas.height = vh;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const poseRes = poseLm.detectForVideo(video, ts);
      const posePts = poseRes.landmarks?.[0] ?? null;
      const hasPose = !!posePts;
      setPoseDetected(hasPose);

      const handRes = handLm.detectForVideo(video, ts);
      const hasHands = !!(handRes.landmarks && handRes.landmarks.length > 0);
      setHandsDetected(hasHands);

      let leftHand: any[] | null = null;
      let rightHand: any[] | null = null;
      if (hasHands) {
        for (let i = 0; i < handRes.landmarks!.length; i++) {
          const lm = handRes.landmarks![i];
          const label =
            handRes.handedness?.[i]?.[0]?.categoryName ||
            handRes.handedness?.[i]?.[0]?.displayName ||
            "";
          const low = label.toLowerCase();
          if (low.includes("left")) leftHand = lm;
          else if (low.includes("right")) rightHand = lm;
          else {
            if (i === 0) leftHand = lm;
            else if (i === 1) rightHand = lm;
          }
        }
      }

      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);

      if (standardReadyRef.current && captureStartTsRef.current != null) {
        const elapsed = ts - captureStartTsRef.current;
        const idx = Math.min(
          standardSeqRef.current.length - 1,
          Math.max(0, Math.floor(elapsed / SAMPLE_INTERVAL_MS))
        );
        const stdPose = standardSeqRef.current[idx]?.pose;

        if (stdPose && stdPose.length === 33) {
          const stdLandmarks = stdPose.map(([x, y]) => ({ x: 1 - x, y, z: 0 }));
          drawer.drawLandmarks(stdLandmarks as any, { radius: 3, color: "#00ff66" } as any);
          drawer.drawConnectors(
            stdLandmarks as any,
            PoseLandmarker.POSE_CONNECTIONS,
            { color: "#00ff66", lineWidth: 2 } as any
          );
        }
      }

      if (posePts) {
        drawer.drawLandmarks(posePts, { radius: 3 } as any);
        drawer.drawConnectors(posePts, PoseLandmarker.POSE_CONNECTIONS);
      }
      if (leftHand) {
        drawer.drawLandmarks(leftHand, { radius: 3 } as any);
        drawer.drawConnectors(leftHand, HandLandmarker.HAND_CONNECTIONS);
      }
      if (rightHand) {
        drawer.drawLandmarks(rightHand, { radius: 3 } as any);
        drawer.drawConnectors(rightHand, HandLandmarker.HAND_CONNECTIONS);
      }

      ctx.restore();

      if (!posePts) return;

      if (captureStartTsRef.current == null) {
        captureStartTsRef.current = ts;
        userKeyframesRef.current = [];
      }

      const elapsed = ts - captureStartTsRef.current;
      if (elapsed >= TARGET_DURATION_MS) {
        captureDoneRef.current = true;
        return;
      }

      if (ts - lastSampleTsRef.current < SAMPLE_INTERVAL_MS) return;
      lastSampleTsRef.current = ts;

      const keyTargets = getKeyframeTargets(TARGET_DURATION_MS);
      if (userKeyframesRef.current.length < keyTargets.length) {
        const nextTarget = keyTargets[userKeyframesRef.current.length];
        if (elapsed >= nextTarget) {
          const img = captureVideoFrameToDataURL(video, 480, 0.7);
          if (img) userKeyframesRef.current.push(img);
        }
      }

      const pose33x3 = posePts.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0]);
      const left21x3 = leftHand ? leftHand.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0]) : [];
      const right21x3 = rightHand ? rightHand.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0]) : [];

      framesRef.current.push({
        pose: pose33x3,
        left_hand: left21x3,
        right_hand: right21x3,
      });

      setFramesBuffered(framesRef.current.length);
    };

    rafRef.current = requestAnimationFrame(step);
  }

  async function evaluate() {
    setEvalError("");
    setResult(null);
    setConfirmResult(null);
    setConfirmError("");

    if (isEvaluating) return;

    if (!captureDoneRef.current) {
      setEvalError(`还没采集满 ${TARGET_DURATION_MS / 1000}s，请继续保持动作...`);
      return;
    }

    const frames = framesRef.current;
    if (frames.length < 3) {
      setEvalError(`有效帧过少：${frames.length}（至少需要 3 帧）`);
      return;
    }

    setIsEvaluating(true);

    try {
      const payload: any = {
        action,
        frames,
        user_seq: frames,
        standard_seq: standardReadyRef.current ? standardSeqRef.current : null,
        use_llm: useLLM,
      };

      const resp = await fetchWithTimeout(
        EVAL_API,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        EVAL_TIMEOUT_MS
      );

      const text = await resp.text();
      if (!resp.ok) {
        setEvalError(`HTTP ${resp.status}:\n${text}`);
        return;
      }

      try {
        setResult(JSON.parse(text));
      } catch {
        setResult({ raw: text });
      }
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setEvalError(
          `请求超时（${EVAL_TIMEOUT_MS}ms）\n请确认后端 http://127.0.0.1:8000 正在运行，并且 /api/evaluate 没卡住。`
        );
      } else {
        setEvalError("请求失败：" + String(e?.message || e));
      }
    } finally {
      setIsEvaluating(false);
    }
  }

  async function pingLLM() {
    setPingError("");
    setPingResult(null);
    if (isPinging) return;
    setIsPinging(true);

    try {
      const resp = await fetchWithTimeout(LLM_PING_API, { method: "GET" }, 8_000);
      const text = await resp.text();
      if (!resp.ok) {
        setPingError(`HTTP ${resp.status}:\n${text}`);
        return;
      }
      try {
        setPingResult(JSON.parse(text));
      } catch {
        setPingResult({ raw: text });
      }
    } catch (e: any) {
      setPingError("Ping 失败：" + String(e?.message || e));
    } finally {
      setIsPinging(false);
    }
  }

  async function playCoachVideoFromLLM() {
    const feedbackText =
      (result as any)?.llm_feedback ||
      (confirmResult as any)?.llm_confirm?.overall ||
      "";

    if (!feedbackText.trim()) {
      alert("没有可用的教练反馈文本（请先勾选用大模型生成教练反馈，并 Evaluate 一次）");
      return;
    }

    setIsCoachPlaying(true);

    try {
      const resp = await fetch(COACH_VIDEO_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_name: userName,
          text: feedbackText,
          version: "v1.5",
          mode: "normal",
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.detail || JSON.stringify(data));

      const url = `${API_BASE}${data.url}?t=${Date.now()}`;
      setStdOverrideSrc(url);

      setTimeout(() => {
        const el = stdPlayerRef.current;
        if (el) {
          el.currentTime = 0;
          el.play().catch(() => {});
        }
      }, 50);
    } catch (e: any) {
      alert("生成口播失败：" + (e?.message || String(e)));
      setStdOverrideSrc(null);
    } finally {
      setIsCoachPlaying(false);
    }
  }

  async function confirmLLM() {
    setConfirmError("");
    setConfirmResult(null);

    if (!captureDoneRef.current) {
      setConfirmError(`还没采集满 ${TARGET_DURATION_MS / 1000}s，请继续保持动作...`);
      return;
    }

    const frames = framesRef.current;
    if (frames.length < 3) {
      setConfirmError(`有效帧过少：${frames.length}（至少需要 3 帧）`);
      return;
    }

    if (!result) {
      setConfirmError("请先点击 Evaluate 得到评估结果，再进行 Confirm。");
      return;
    }

    if (isConfirming) return;

    const stdImgs = stdKeyframesRef.current || [];
    const usrImgs = userKeyframesRef.current || [];

    if (stdImgs.length < 2) {
      setConfirmError(`标准关键帧不足：${stdImgs.length}（至少需要 2 张，建议 5 张）`);
      return;
    }
    if (usrImgs.length < 2) {
      setConfirmError(`用户关键帧不足：${usrImgs.length}（至少需要 2 张，建议 5 张）`);
      return;
    }

    const ok = window.confirm("将调用大模型进行关键帧对比确认（可能产生费用），是否继续？");
    if (!ok) return;

    setIsConfirming(true);

    try {
      const payload = {
        action,
        frames,
        standard_seq: standardReadyRef.current ? standardSeqRef.current : null,
        eval_result: result,
        standard_images: stdImgs,
        user_images: usrImgs,
      };

      const resp = await fetchWithTimeout(
        LLM_CONFIRM_API,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        LLM_TIMEOUT_MS
      );

      const text = await resp.text();
      if (!resp.ok) {
        setConfirmError(`HTTP ${resp.status}:\n${text}`);
        return;
      }

      try {
        const parsed = JSON.parse(text);
        setConfirmResult(parsed);
      } catch {
        setConfirmResult({ raw: text });
      }
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setConfirmError(`Confirm 超时（${LLM_TIMEOUT_MS}ms）。`);
      } else {
        setConfirmError("Confirm 失败：" + String(e?.message || e));
      }
    } finally {
      setIsConfirming(false);
    }
  }

  return (
    <div
      style={{
        padding: 24,
        background: "#0b0b0c",
        color: "#fff",
        minHeight: "100vh",
        fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      <h1 style={{ fontSize: 44, margin: 0, lineHeight: 1.1 }}>
        Rehab Web MVP (Standard ↔ User)
      </h1>
      <div style={{ opacity: 0.8, marginTop: 8 }}>
        Camera → 3s window → {EVAL_API} → {LLM_CONFIRM_API}
      </div>
      <div style={{ marginTop: 10, opacity: 0.95 }}>
        当前用户：<b>{userName}</b>
      </div>

      <div style={{ marginTop: 12, opacity: 0.9 }}>
        {mpStatus === "ready" && "MediaPipe ready."}
        {mpStatus === "loading" && "Loading MediaPipe models..."}
        {mpStatus === "error" && (
          <pre
            style={{
              background: "#3a0b0b",
              border: "1px solid #ff5a5a",
              padding: 12,
              borderRadius: 12,
              whiteSpace: "pre-wrap",
            }}
          >
            初始化失败：{mpError}
          </pre>
        )}
      </div>

      <div style={{ marginTop: 12, opacity: 0.9 }}>
        {buildingStandard ? "MusePose 处理中..." : statusText}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
          marginTop: 18,
        }}
      >
        <div
          style={{
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 16,
            padding: 16,
            background: "rgba(255,255,255,0.04)",
          }}
        >
          <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 10 }}>
            Standard Demo ({action})
          </div>

          <video
            ref={stdPlayerRef}
            src={stdOverrideSrc || demoVideoSrc}
            controls
            autoPlay
            loop
            muted
            playsInline
            style={{ width: "100%", borderRadius: 12, background: "#000" }}
          />

          <video
            ref={standardVideoRef}
            src={demoVideoSrc}
            muted
            playsInline
            crossOrigin="anonymous"
            preload="auto"
            style={{ display: "none" }}
          />

          <div style={{ marginTop: 10, opacity: 0.7 }}>
            展示视频：{stdOverrideSrc || demoVideoSrc}
            <div style={{ marginTop: 6 }}>标准骨架：{standardReadyRef.current ? "✅ ready" : "⌛ building..."}</div>
          </div>
        </div>

        <div
          style={{
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 16,
            padding: 16,
            background: "rgba(255,255,255,0.04)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <div style={{ fontSize: 18, opacity: 0.9 }}>Action:</div>
            <select
              value={action}
              onChange={(e) => setAction(e.target.value as ActionName)}
              style={{
                background: "#1a1a1b",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: 10,
                padding: "6px 10px",
              }}
            >
              {ACTIONS.map((a) => (
                <option value={a} key={a}>
                  {a}
                </option>
              ))}
            </select>

            <button
              onClick={startCamera}
              disabled={mpStatus !== "ready" || cameraOn}
              style={{
                marginLeft: 8,
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.2)",
                background: cameraOn ? "#333" : "#111",
                color: "#fff",
                cursor: cameraOn ? "not-allowed" : "pointer",
              }}
            >
              Start Camera
            </button>

            <button
              onClick={stopCamera}
              disabled={!cameraOn}
              style={{
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.2)",
                background: !cameraOn ? "#333" : "#111",
                color: "#fff",
                cursor: !cameraOn ? "not-allowed" : "pointer",
              }}
            >
              Stop
            </button>

            <button
              onClick={evaluate}
              disabled={!cameraOn || !captureDoneRef.current || isEvaluating || isConfirming}
              style={{
                marginLeft: "auto",
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.25)",
                background: !cameraOn || !captureDoneRef.current || isEvaluating || isConfirming ? "#333" : "#111",
                color: "#fff",
                cursor: !cameraOn || !captureDoneRef.current || isEvaluating || isConfirming ? "not-allowed" : "pointer",
              }}
            >
              {isEvaluating ? "Evaluating..." : "Evaluate (3s window)"}
            </button>

            <button
              onClick={pingLLM}
              disabled={isPinging}
              style={{
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(120,170,255,0.55)",
                background: isPinging ? "#333" : "rgba(120,170,255,0.15)",
                color: "#fff",
                cursor: isPinging ? "not-allowed" : "pointer",
              }}
            >
              {isPinging ? "Pinging..." : "Ping LLM"}
            </button>

            <button
              onClick={confirmLLM}
              disabled={!cameraOn || !captureDoneRef.current || isEvaluating || isConfirming || !result}
              style={{
                padding: "10px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,180,80,0.55)",
                background:
                  !cameraOn || !captureDoneRef.current || isEvaluating || isConfirming || !result
                    ? "#333"
                    : "rgba(255,180,80,0.15)",
                color: "#fff",
                cursor:
                  !cameraOn || !captureDoneRef.current || isEvaluating || isConfirming || !result
                    ? "not-allowed"
                    : "pointer",
              }}
            >
              {isConfirming ? "Confirming..." : "Confirm (LLM)"}
            </button>

            <button onClick={playCoachVideoFromLLM} disabled={isCoachPlaying}>
              {isCoachPlaying ? "生成口播中..." : "播放教练口播讲解"}
            </button>

            <label style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 10, opacity: 0.9 }}>
              <input type="checkbox" checked={useLLM} onChange={(e) => setUseLLM(e.target.checked)} />
              用大模型生成教练反馈
            </label>
          </div>

          <div style={{ marginTop: 12, opacity: 0.75 }}>
            Frames: {framesBuffered} ｜ Captured: {captureDoneRef.current ? "✅ 3s" : "⌛ collecting"} ｜ Pose: {poseDetected ? "✅" : "❌"} ｜ Hands: {handsDetected ? "✅" : "❌"} ｜ Std: {standardReadyRef.current ? "✅" : "⌛"}
          </div>

          <div>
            StdKeyframes: {stdKeyframesRef.current.length} | UserKeyframes: {userKeyframesRef.current.length}
          </div>

          <div style={{ position: "relative", marginTop: 12 }}>
            <video
              ref={userVideoRef}
              playsInline
              muted
              style={{
                width: "100%",
                borderRadius: 14,
                background: "#111",
                transform: "scaleX(-1)",
              }}
            />
            <canvas
              ref={overlayRef}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
              }}
            />
          </div>

          <div style={{ marginTop: 14, fontSize: 22, fontWeight: 700 }}>Result</div>

          <div
            style={{
              marginTop: 8,
              border: "1px solid rgba(255,255,255,0.15)",
              borderRadius: 14,
              padding: 14,
              minHeight: 210,
              background: "rgba(0,0,0,0.25)",
              whiteSpace: "pre-wrap",
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas",
              lineHeight: 1.4,
            }}
          >
            {evalError ? evalError : confirmError ? confirmError : pingError ? pingError : JSON.stringify({ ping: pingResult ?? undefined, evaluate: result ?? undefined, confirm: confirmResult ?? undefined }, null, 2)}
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [inputName, setInputName] = useState<string>(() => localStorage.getItem("rehab_user_name") || "");
  const [resolvedName, setResolvedName] = useState<string>("");
  const [profile, setProfile] = useState<ProfileState | null>(null);
  const [step, setStep] = useState<"entry" | "record" | "main">("entry");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [previewPhotoUrl, setPreviewPhotoUrl] = useState<string>("");

  const recorderVideoRef = useRef<HTMLVideoElement | null>(null);
  const recorderStreamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      recorderStreamRef.current?.getTracks().forEach((t) => t.stop());
      if (previewPhotoUrl) URL.revokeObjectURL(previewPhotoUrl);
    };
  }, [previewPhotoUrl]);

  async function checkProfile() {
    const name = inputName.trim();
    if (!name) {
      setError("请先输入姓名");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const resp = await fetch(`${PROFILE_CHECK_API}?name=${encodeURIComponent(name)}`);
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.detail || JSON.stringify(data));
      setResolvedName(name);
      setProfile(data);
      localStorage.setItem("rehab_user_name", name);
      if (data.exists) {
        setStep("main");
      } else {
        setStep("record");
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function startRecorderPreview() {
    if (recorderStreamRef.current) {
      recorderStreamRef.current.getTracks().forEach((t) => t.stop());
      recorderStreamRef.current = null;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    recorderStreamRef.current = stream;
    if (recorderVideoRef.current) {
      recorderVideoRef.current.srcObject = stream;
      await recorderVideoRef.current.play().catch(() => {});
    }
  }

  useEffect(() => {
    if (step === "record") {
      startRecorderPreview().catch((e) => {
        setError(String(e?.message || e));
      });
    }
  }, [step]);

  async function recordAndUploadV2() {
    const name = resolvedName || inputName.trim();
    if (!name) {
      setError("姓名为空");
      return;
    }
    if (!photoFile) {
      setError("请先上传一张个人照片");
      return;
    }
    if (!recorderStreamRef.current) {
      await startRecorderPreview();
    }

    setBusy(true);
    setError("");
    try {
      const stream = recorderStreamRef.current!;
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
        ? "video/webm;codecs=vp9,opus"
        : "video/webm";
      const chunks: BlobPart[] = [];
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      const stopped = new Promise<void>((resolve) => {
        recorder.onstop = () => resolve();
      });

      recorder.start();
      await new Promise((resolve) => window.setTimeout(resolve, 3000));
      recorder.stop();
      await stopped;

      const blob = new Blob(chunks, { type: mimeType });
      const file = new File([blob], `${name}.webm`, { type: mimeType });
      const fd = new FormData();
      fd.append("name", name);
      fd.append("video", file);
      fd.append("photo", photoFile);

      const resp = await fetch(PROFILE_REGISTER_V2_API, { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data?.detail || JSON.stringify(data));

      setProfile(data?.profile || { exists: true, user_name: name });
      setResolvedName(data?.profile?.user_name || name);
      localStorage.setItem("rehab_user_name", data?.profile?.user_name || name);
      recorderStreamRef.current?.getTracks().forEach((t) => t.stop());
      recorderStreamRef.current = null;
      setStep("main");
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  if (step === "main") {
    return <RehabMain userName={resolvedName || inputName.trim()} />;
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0b0b0c",
        color: "#fff",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
      }}
    >
      <div
        style={{
          width: "min(720px, 100%)",
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.14)",
          borderRadius: 20,
          padding: 24,
        }}
      >
        <h1 style={{ marginTop: 0, fontSize: 36 }}>Rehab Web MVP</h1>
        <div style={{ opacity: 0.8, marginBottom: 18 }}>
          进入系统前先输入姓名。若该姓名没有个人模板视频，则先录制一段 3 秒视频并上传一张个人照片，之后标准动作展示会自动使用该用户对应的 MusePose 生成视频。
        </div>

        {step === "entry" && (
          <>
            <div style={{ fontSize: 18, marginBottom: 8 }}>姓名</div>
            <input
              value={inputName}
              onChange={(e) => setInputName(e.target.value)}
              placeholder="请输入姓名，例如 Mike"
              style={{
                width: "100%",
                padding: "12px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.18)",
                background: "#151517",
                color: "#fff",
                fontSize: 16,
                boxSizing: "border-box",
              }}
            />
            <button
              onClick={checkProfile}
              disabled={busy}
              style={{
                marginTop: 16,
                padding: "12px 18px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.2)",
                background: busy ? "#333" : "#111",
                color: "#fff",
                cursor: busy ? "not-allowed" : "pointer",
              }}
            >
              {busy ? "检查中..." : "进入系统"}
            </button>
          </>
        )}

        {step === "record" && (
          <>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>
              首次使用：请为 “{resolvedName}” 录制 3 秒个人模板视频并上传一张照片
            </div>
            <div style={{ opacity: 0.78, marginBottom: 14 }}>
              建议正脸、光线稳定、嘴部清晰。视频仅首次录制一次，照片用于生成个性化标准动作展示视频。
            </div>

            <div style={{ marginBottom: 16 }}>
              <div style={{ marginBottom: 8, fontSize: 16 }}>上传个人照片</div>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setPhotoFile(f);
                  if (previewPhotoUrl) URL.revokeObjectURL(previewPhotoUrl);
                  setPreviewPhotoUrl(f ? URL.createObjectURL(f) : "");
                }}
              />
              {photoFile && (
                <div style={{ marginTop: 8, opacity: 0.85 }}>
                  已选择：{photoFile.name}
                </div>
              )}
              {previewPhotoUrl && (
                <div style={{ marginTop: 10 }}>
                  <img
                    src={previewPhotoUrl}
                    alt="preview"
                    style={{ width: 220, borderRadius: 12, display: "block" }}
                  />
                </div>
              )}
            </div>

            <div style={{ position: "relative", width: "100%" }}>
              <video
                ref={recorderVideoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: "100%",
                  borderRadius: 14,
                  background: "#000",
                  transform: "scaleX(-1)",
                  display: "block",
                  minHeight: 320,
                  objectFit: "cover",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 12,
                  left: 12,
                  padding: "6px 10px",
                  borderRadius: 999,
                  background: "rgba(0,0,0,0.55)",
                  border: "1px solid rgba(255,255,255,0.18)",
                  fontSize: 13,
                }}
              >
                摄像头实时预览
              </div>
            </div>

            <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
              <button
                onClick={recordAndUploadV2}
                disabled={busy}
                style={{
                  padding: "12px 18px",
                  borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: busy ? "#333" : "#111",
                  color: "#fff",
                  cursor: busy ? "not-allowed" : "pointer",
                }}
              >
                {busy ? "录制并上传中..." : "开始录制 3 秒并保存"}
              </button>
              <button
                onClick={() => {
                  recorderStreamRef.current?.getTracks().forEach((t) => t.stop());
                  recorderStreamRef.current = null;
                  setStep("entry");
                }}
                disabled={busy}
                style={{
                  padding: "12px 18px",
                  borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.2)",
                  background: "#111",
                  color: "#fff",
                }}
              >
                返回
              </button>
            </div>
          </>
        )}

        {profile && (
          <div style={{ marginTop: 18, opacity: 0.75, whiteSpace: "pre-wrap" }}>
            {JSON.stringify(profile, null, 2)}
          </div>
        )}

        {error && (
          <pre
            style={{
              marginTop: 18,
              background: "#3a0b0b",
              border: "1px solid #ff5a5a",
              padding: 12,
              borderRadius: 12,
              whiteSpace: "pre-wrap",
            }}
          >
            {error}
          </pre>
        )}
      </div>
    </div>
  );
}

export default App;
