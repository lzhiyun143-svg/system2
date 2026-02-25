import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

type ActionName = "raise_arm";

type FramePayload = {
  // 统一用 33x3 / 21x3（z 强制 0，避免 z 不一致导致误差爆炸）
  pose: number[][];
  left_hand: number[][];
  right_hand: number[][];
};

type EvalResponse = any;

const ACTIONS: ActionName[] = ["raise_arm"];

// ====== 采集策略：采满 3 秒 ======
const TARGET_DURATION_MS = 3000; // 采集 3 秒
const SAMPLE_INTERVAL_MS = 33; // ~30fps 采样（可改成 40/50 更省）
const EVAL_TIMEOUT_MS = 12_000;
const LLM_TIMEOUT_MS = 20_000;

// MediaPipe wasm & model（CDN，避免 public/models 缺文件导致 “Unable to open zip archive”）
const MP_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const POSE_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// 前端请求后端（走 vite proxy：/api -> http://127.0.0.1:8000）
const EVAL_API = "/api/evaluate";
const LLM_PING_API = "/api/llm_ping";
const LLM_CONFIRM_API = "/api/confirm";

// 标准视频（public/demos/raise_arm.mp4）
const DEMO_VIDEO_BY_ACTION: Record<ActionName, string> = {
  raise_arm: "/demos/raise_arm.mp4",
};

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// fetch + timeout 小工具
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

export default function App() {
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
  // ✅ LLM 状态
  const [isPinging, setIsPinging] = useState(false);
  const [pingResult, setPingResult] = useState<any | null>(null);
  const [pingError, setPingError] = useState<string>("");

  const [isConfirming, setIsConfirming] = useState(false);
  const [confirmResult, setConfirmResult] = useState<any | null>(null);
  const [confirmError, setConfirmError] = useState<string>("");

  const [result, setResult] = useState<EvalResponse | null>(null);
  const [evalError, setEvalError] = useState<string>("");

  const [useLLM, setUseLLM] = useState<boolean>(true); // 默认打开

  const demoVideoSrc = useMemo(() => DEMO_VIDEO_BY_ACTION[action], [action]);

  // DOM refs
  const userVideoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // ✅ 标准视频隐藏 video（用于抽帧得到标准骨架序列）
  const standardVideoRef = useRef<HTMLVideoElement | null>(null);

  // MediaPipe refs
  const poseLmRef = useRef<PoseLandmarker | null>(null);
  const handLmRef = useRef<HandLandmarker | null>(null);
  const drawingRef = useRef<DrawingUtils | null>(null);

  // loop refs
  const rafRef = useRef<number | null>(null);
  const lastSampleTsRef = useRef<number>(0);

  // frames buffer
  const framesRef = useRef<FramePayload[]>([]);

  // 采集窗口：3 秒
  const captureStartTsRef = useRef<number | null>(null);
  const captureDoneRef = useRef<boolean>(false);

  // ✅ 标准序列（3秒）缓存
  const standardSeqRef = useRef<FramePayload[]>([]);
  const standardReadyRef = useRef<boolean>(false);

  // stream
  const streamRef = useRef<MediaStream | null>(null);

  // ====== 1) 初始化 MediaPipe ======
  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setMpStatus("loading");
        setMpError("");

        const vision = await FilesetResolver.forVisionTasks(MP_WASM_BASE);

        // ✅ 摄像头用
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

        // ✅ 标准视频抽帧用（单独实例，避免 timestamp 互相打架）
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

        // 绑定到 ref
        poseLmRef.current = poseLmUser;      // 你 step() 里摄像头 detectForVideo 用这个
        poseLmStdRef.current = poseLmStd;    // 你 buildStandardSequence() 用这个

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

        // ✅ 摄像头用
        poseLmRef.current = poseLmUser;

        // ✅ 标准视频抽帧用
        poseLmStdRef.current = poseLmStd;
        handLmRef.current = handLm;

        // drawing utils
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

  // ====== 1.5) 构建标准骨架序列（从标准视频抽帧，取3秒窗口） ======
  async function buildStandardSequence() {
    standardReadyRef.current = false;
    standardSeqRef.current = [];

    const video = standardVideoRef.current;
    const poseLm = poseLmStdRef.current;
    if (!video || !poseLm) return;

    // 确保加载完成
    if (video.readyState < 2) {
      await new Promise<void>((resolve) => {
        const onLoaded = () => {
          video.removeEventListener("loadeddata", onLoaded);
          resolve();
        };
        video.addEventListener("loadeddata", onLoaded);
        video.load();
      });
    }

    const videoDurationMs = (video.duration || 3) * 1000;
    const durMs = Math.min(TARGET_DURATION_MS, videoDurationMs);

    const seq: FramePayload[] = [];
    const baseTs = performance.now();

    // 通过 seek 抽帧
    for (let t = 0; t < durMs; t += SAMPLE_INTERVAL_MS) {
      const ct = Math.min(video.duration || 3, t / 1000);
      video.currentTime = ct;

      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
      });

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

    standardSeqRef.current = seq;
    standardReadyRef.current = seq.length >= 3;
  }

  // MediaPipe ready + action变化后，自动重建标准序列
  useEffect(() => {
    if (mpStatus !== "ready") return;
    buildStandardSequence().catch(() => {
      standardReadyRef.current = false;
      standardSeqRef.current = [];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mpStatus, action]);

  // ====== 2) 启动/停止摄像头 ======
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

      // reset buffer + 3秒窗口
      framesRef.current = [];
      setFramesBuffered(0);
      lastSampleTsRef.current = 0;
      captureStartTsRef.current = null;
      captureDoneRef.current = false;

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

  // ====== 3) 检测循环（draw + buffer frames） ======
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

      // overlay 跟 video 像素尺寸一致
      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (vw === 0 || vh === 0) return;
      if (canvas.width !== vw) canvas.width = vw;
      if (canvas.height !== vh) canvas.height = vh;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // ====== 检测 ======
      const poseRes = poseLm.detectForVideo(video, ts);
      const posePts = poseRes.landmarks?.[0] ?? null;
      const hasPose = !!posePts;
      setPoseDetected(hasPose);

      const handRes = handLm.detectForVideo(video, ts);
      const hasHands = !!(handRes.landmarks && handRes.landmarks.length > 0);
      setHandsDetected(hasHands);

      // ====== 左/右手分离（按 handedness） ======
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

      // ====== 画骨架（镜像画法：canvas 内部翻转，和 video 的镜像一致） ======
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);

      // ✅ 先画标准骨架（绿色），再画用户骨架（白色），便于对齐
      if (standardReadyRef.current && captureStartTsRef.current != null) {
        const elapsed = ts - captureStartTsRef.current;
        const idx = Math.min(
          standardSeqRef.current.length - 1,
          Math.max(0, Math.floor(elapsed / SAMPLE_INTERVAL_MS))
        );
        const stdPose = standardSeqRef.current[idx]?.pose;

        if (stdPose && stdPose.length === 33) {
          // ✅ 标准骨架也镜像：x' = 1 - x（因为我们在 canvas 里做了 translate+scale(-1,1)）
          const stdLandmarks = stdPose.map(([x, y]) => ({ x: 1 - x, y, z: 0 }));
          // DrawingUtils 在不同版本里参数类型略有差异，这里用 any 兼容
          drawer.drawLandmarks(stdLandmarks as any, { radius: 3, color: "#00ff66" } as any);
          drawer.drawConnectors(
            stdLandmarks as any,
            PoseLandmarker.POSE_CONNECTIONS,
            { color: "#00ff66", lineWidth: 2 } as any
          );
        }
      }

      // 用户骨架（默认白色）
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

      // ====== 采样入 buffer：采满 3 秒全帧 ======
      if (!posePts) return;

      // 第一次开始采样
      if (captureStartTsRef.current == null) {
        captureStartTsRef.current = ts;
      }

      const elapsed = ts - captureStartTsRef.current;

      // 采满 3 秒：停止采样（继续画即可）
      if (elapsed >= TARGET_DURATION_MS) {
        captureDoneRef.current = true;
        return;
      }

      // 采样间隔控制
      if (ts - lastSampleTsRef.current < SAMPLE_INTERVAL_MS) return;
      lastSampleTsRef.current = ts;

      const pose33x3 = posePts.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0]);
      const left21x3 = leftHand
        ? leftHand.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0])
        : [];
      const right21x3 = rightHand
        ? rightHand.map((p: any) => [clamp01(p.x), clamp01(p.y), 0.0])
        : [];

      framesRef.current.push({
        pose: pose33x3,
        left_hand: left21x3,
        right_hand: right21x3,
      });

      setFramesBuffered(framesRef.current.length);
    };

    rafRef.current = requestAnimationFrame(step);
  }

  // ====== 4) Evaluate：3秒采满后才允许 ======
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
        const json = JSON.parse(text);
        setResult(json);
      } catch {
        setResult({ raw: text });
      }
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setEvalError(
          `请求超时（${EVAL_TIMEOUT_MS}ms）\n` +
            `请确认后端 http://127.0.0.1:8000 正在运行，并且 /api/evaluate 没卡住。`
        );
      } else {
        setEvalError("请求失败：" + String(e?.message || e));
      }
    } finally {
      setIsEvaluating(false);
    }
  }

  // ====== 5) Ping LLM ======
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

  // ====== 6) Confirm (LLM)：可选（需要你后端实现 POST /api/confirm） ======
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

    const ok = window.confirm("将调用大模型确认（可能产生费用），是否继续？");
    if (!ok) return;

    setIsConfirming(true);

    try {
      const payload = {
        action,
        frames,
        standard_seq: standardReadyRef.current ? standardSeqRef.current : null,
        eval_result: result,
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
        setConfirmResult(JSON.parse(text));
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

  // ====== UI ======
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

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
          marginTop: 18,
        }}
      >
        {/* Left: Standard */}
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
            src={demoVideoSrc}
            controls
            style={{
              width: "100%",
              borderRadius: 14,
              background: "#111",
            }}
          />

          {/* ✅ 隐藏 video：用于抽帧构建标准骨架 */}
          <video
            ref={standardVideoRef}
            src={demoVideoSrc}
            muted
            playsInline
            style={{ display: "none" }}
          />

          <div style={{ marginTop: 10, opacity: 0.7 }}>
            标准视频路径：{demoVideoSrc}
            <div style={{ marginTop: 6 }}>
              标准骨架：{standardReadyRef.current ? "✅ ready" : "⌛ building..."}
            </div>
          </div>
        </div>

        {/* Right: User */}
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
              title="可选：需要后端实现 POST /api/confirm"
            >
              {isConfirming ? "Confirming..." : "Confirm (LLM)"}
            </button>

            <label style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 10, opacity: 0.9 }}>
              <input
                type="checkbox"
                checked={useLLM}
                onChange={(e) => setUseLLM(e.target.checked)}
              />
              用大模型生成教练反馈
            </label>
          </div>

          <div style={{ marginTop: 12, opacity: 0.75 }}>
            Frames: {framesBuffered} ｜ Captured: {captureDoneRef.current ? "✅ 3s" : "⌛ collecting"} ｜ Pose:{" "}
            {poseDetected ? "✅" : "❌"} ｜ Hands: {handsDetected ? "✅" : "❌"} ｜ Std:{" "}
            {standardReadyRef.current ? "✅" : "⌛"}
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

          <div style={{ marginTop: 14, fontSize: 22, fontWeight: 700 }}>
            Result
          </div>

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
            {evalError ? (
              evalError
            ) : confirmError ? (
              confirmError
            ) : pingError ? (
              pingError
            ) : (
              JSON.stringify(
                {
                  ping: pingResult ?? undefined,
                  evaluate: result ?? undefined,
                  confirm: confirmResult ?? undefined,
                },
                null,
                2
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
