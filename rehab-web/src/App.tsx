import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import "./App.css";

type ActionName = "raise_arm";

type FramePayload = {
  pose: number[][];
  left_hand: number[][];
  right_hand: number[][];
};

type EvalAutoResponse = any;

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

type StandardVideoItem = {
  id?: string;
  file_name?: string;
  video_url?: string;
  demo_video_url?: string;
  source_demo_video_url?: string;
  local_path?: string;
  cached?: boolean;
};

const FIXED_ACTION: ActionName = "raise_arm";

const DEFAULT_TARGET_DURATION_MS = 3000;
const SAMPLE_INTERVAL_MS = 33;
const EVAL_TIMEOUT_MS = 120_000;

const MP_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const POSE_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_TASK_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const API_BASE = "http://127.0.0.1:8000";
const EVAL_AUTO_API = `${API_BASE}/api/evaluate_auto`;
const PROFILE_CHECK_API = `${API_BASE}/api/profile/check`;
const PROFILE_REGISTER_V2_API = `${API_BASE}/api/profile/register_v2`;
const STANDARD_INFO_API = `${API_BASE}/api/standard_video/info`;
const STANDARD_BUILD_API = `${API_BASE}/api/standard_video/build`;

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

function normalizeStandardItemsFromInfo(info: any): StandardVideoItem[] {
  if (
    info?.generated_exists === true &&
    Array.isArray(info?.generated_videos) &&
    info.generated_videos.length > 0
  ) {
    return info.generated_videos.map((x: any, idx: number) => ({
      id: x?.id || `${idx}`,
      file_name: x?.file_name || `video_${idx + 1}.mp4`,
      video_url: x?.video_url,
      local_path: x?.local_path,
      cached: x?.cached,
    }));
  }
  return [];
}

function normalizeStandardItemsFromBuild(data: any): StandardVideoItem[] {
  if (Array.isArray(data?.results) && data.results.length > 0) {
    return data.results.map((x: any, idx: number) => ({
      id: x?.id || `${idx}`,
      file_name: x?.file_name || `video_${idx + 1}.mp4`,
      video_url: x?.video_url,
      source_demo_video_url: x?.source_demo_video_url,
      local_path: x?.local_path,
      cached: x?.cached,
    }));
  }
  return [];
}

function getKeyframeTargets(totalMs: number) {
  const count = Math.max(5, Math.min(16, Math.ceil(totalMs / 800)));
  if (count <= 1) return [0];

  const targets: number[] = [];
  for (let i = 0; i < count; i++) {
    const ratio = (0.95 * i) / (count - 1);
    targets.push(ratio * totalMs);
  }
  return targets;
}

function getScoreInfo(result: any) {
  const candidates = [
    result?.score,
    result?.final_score,
    result?.total_score,
    result?.overall_score,
    result?.result?.score,
  ];

  const found = candidates.find((x) => typeof x === "number" && Number.isFinite(x));
  const score = typeof found === "number" ? Math.max(0, Math.min(100, found)) : null;

  if (score == null) {
    return { score: "--", percent: 0, level: "待评估" };
  }
  if (score >= 85) return { score: `${score.toFixed(0)}`, percent: score, level: "优秀" };
  if (score >= 70) return { score: `${score.toFixed(0)}`, percent: score, level: "良好" };
  if (score >= 60) return { score: `${score.toFixed(0)}`, percent: score, level: "合格" };
  return { score: `${score.toFixed(0)}`, percent: score, level: "待提升" };
}

function formatMpStatus(status: "idle" | "loading" | "ready" | "error") {
  if (status === "ready") return "识别引擎已就绪";
  if (status === "loading") return "识别引擎加载中";
  if (status === "error") return "识别引擎异常";
  return "识别引擎待初始化";
}

function StatusBadge({
  text,
  tone = "default",
}: {
  text: string;
  tone?: "default" | "success" | "warn" | "danger" | "info";
}) {
  return <span className={`status-badge tone-${tone}`}>{text}</span>;
}

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

function SectionCard({
  title,
  desc,
  extra,
  children,
}: {
  title: string;
  desc?: string;
  extra?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel-card">
      <div className="panel-head">
        <div>
          <h3 className="panel-title">{title}</h3>
          {desc ? <p className="panel-desc">{desc}</p> : null}
        </div>
        {extra ? <div className="panel-extra">{extra}</div> : null}
      </div>
      {children}
    </section>
  );
}

function RehabMain({ userName }: { userName: string }) {
  const action: ActionName = FIXED_ACTION;

  const [mpStatus, setMpStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle"
  );
  const [mpError, setMpError] = useState<string>("");

  const [cameraOn, setCameraOn] = useState(false);
  const [poseDetected, setPoseDetected] = useState(false);
  const [handsDetected, setHandsDetected] = useState(false);

  const [framesBuffered, setFramesBuffered] = useState(0);
  const [isEvaluating, setIsEvaluating] = useState(false);

  const [result, setResult] = useState<EvalAutoResponse | null>(null);
  const [evalError, setEvalError] = useState<string>("");

  const [feedbackText, setFeedbackText] = useState<string>("");
  const [coachVideoUrl, setCoachVideoUrl] = useState<string>("");

  const [stdVideoList, setStdVideoList] = useState<StandardVideoItem[]>([]);
  const [selectedStdVideoUrl, setSelectedStdVideoUrl] = useState<string | null>(null);

  const [buildingStandard, setBuildingStandard] = useState(false);
  const [statusText, setStatusText] = useState("");

  const [currentTrainDurationMs, setCurrentTrainDurationMs] = useState<number>(
    DEFAULT_TARGET_DURATION_MS
  );
  const [trainFinished, setTrainFinished] = useState<boolean>(false);
  const [trainHint, setTrainHint] = useState<string>("");

  const [stdVideoAspect, setStdVideoAspect] = useState<number>(16 / 9);
  const [userVideoAspect, setUserVideoAspect] = useState<number>(16 / 9);

  const userVideoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const standardVideoRef = useRef<HTMLVideoElement | null>(null);
  const stdPlayerRef = useRef<HTMLVideoElement | null>(null);
  const coachPlayerRef = useRef<HTMLVideoElement | null>(null);

  const poseLmRef = useRef<PoseLandmarker | null>(null);
  const poseLmStdRef = useRef<PoseLandmarker | null>(null);
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

  const activeStandardVideoSrc = useMemo(() => {
    return selectedStdVideoUrl || "";
  }, [selectedStdVideoUrl]);

  const scoreInfo = useMemo(() => getScoreInfo(result), [result]);

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
        `${STANDARD_INFO_API}?user_name=${encodeURIComponent(
          currentUserName
        )}&action=${encodeURIComponent(currentAction)}`
      );
      const info = await infoResp.json();

      if (!infoResp.ok) {
        throw new Error(info?.detail || JSON.stringify(info));
      }

      const infoItems = normalizeStandardItemsFromInfo(info);
      if (infoItems.length > 0) {
        const urls = infoItems
          .map((x) => ({
            ...x,
            fullUrl: joinApiUrl(x.video_url || ""),
          }))
          .filter((x) => !!x.fullUrl);

        if (urls.length > 0) {
          setStdVideoList(
            urls.map((x) => ({
              ...x,
              video_url: x.fullUrl,
            }))
          );
          setSelectedStdVideoUrl(urls[0].fullUrl);
          setStatusText(`已加载个性化标准动作视频（${urls.length}个）`);
          return;
        }
      }

      setStatusText("未找到已生成视频，正在调用服务生成...");
      const buildResp = await fetch(STANDARD_BUILD_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_name: currentUserName,
          action: currentAction,
          force: false,
        }),
      });

      const buildData = await buildResp.json();
      if (!buildResp.ok) {
        throw new Error(buildData?.detail || JSON.stringify(buildData));
      }

      const builtItems = normalizeStandardItemsFromBuild(buildData);
      const urls = builtItems
        .map((x) => ({
          ...x,
          fullUrl: joinApiUrl(x.video_url || ""),
        }))
        .filter((x) => !!x.fullUrl);

      if (urls.length > 0) {
        setStdVideoList(
          urls.map((x) => ({
            ...x,
            video_url: x.fullUrl,
          }))
        );
        setSelectedStdVideoUrl(urls[0].fullUrl);
        setStatusText(`个性化标准动作视频生成完成（${urls.length}个）`);
      } else {
        setStdVideoList([]);
        setSelectedStdVideoUrl(null);
        setStatusText("生成完成，但未返回可播放视频");
      }
    } catch (e: any) {
      console.error(e);
      setStdVideoList([]);
      setSelectedStdVideoUrl(null);
      setStatusText(`标准视频生成失败：${e?.message || e}`);
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

  async function buildStandardSequence() {
    standardReadyRef.current = false;
    standardSeqRef.current = [];
    stdKeyframesRef.current = [];

    const video = standardVideoRef.current;
    const poseLm = poseLmStdRef.current;
    if (!video || !poseLm) return;
    if (!activeStandardVideoSrc) return;

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

    if (video.videoWidth > 0 && video.videoHeight > 0) {
      setStdVideoAspect(video.videoWidth / video.videoHeight);
    }

    const videoDurationMs = (video.duration || 3) * 1000;
    const durMs = Math.max(1000, Math.round(videoDurationMs));
    setCurrentTrainDurationMs(durMs);

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
    if (!activeStandardVideoSrc) {
      standardReadyRef.current = false;
      standardSeqRef.current = [];
      stdKeyframesRef.current = [];
      return;
    }

    buildStandardSequence().catch(() => {
      standardReadyRef.current = false;
      standardSeqRef.current = [];
    });
  }, [mpStatus, action, activeStandardVideoSrc]);

  async function startCamera() {
    setEvalError("");
    setResult(null);
    setFeedbackText("");
    setCoachVideoUrl("");
    setTrainFinished(false);
    setTrainHint("");

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

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        setUserVideoAspect(video.videoWidth / video.videoHeight);
      }

      setCameraOn(true);
      framesRef.current = [];
      setFramesBuffered(0);
      lastSampleTsRef.current = 0;
      captureStartTsRef.current = null;
      captureDoneRef.current = false;
      userKeyframesRef.current = [];

      const stdPlayer = stdPlayerRef.current;
      if (stdPlayer && activeStandardVideoSrc) {
        try {
          stdPlayer.pause();
          stdPlayer.currentTime = 0;
          await stdPlayer.play();
        } catch (e) {
          console.warn("标准视频播放失败：", e);
        }
      }

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
    userKeyframesRef.current = [];

    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    const stdPlayer = stdPlayerRef.current;
    if (stdPlayer) {
      try {
        stdPlayer.pause();
      } catch {}
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
          drawer.drawLandmarks(stdLandmarks as any, {
            radius: 3,
            color: "#22c55e",
          } as any);
          drawer.drawConnectors(
            stdLandmarks as any,
            PoseLandmarker.POSE_CONNECTIONS,
            { color: "#22c55e", lineWidth: 2 } as any
          );
        }
      }

      ctx.restore();

      if (!posePts) return;

      if (captureStartTsRef.current == null) {
        captureStartTsRef.current = ts;
        userKeyframesRef.current = [];
      }

      const elapsed = ts - captureStartTsRef.current;
      if (elapsed >= currentTrainDurationMs) {
        captureDoneRef.current = true;
        return;
      }

      if (ts - lastSampleTsRef.current < SAMPLE_INTERVAL_MS) return;
      lastSampleTsRef.current = ts;

      const keyTargets = getKeyframeTargets(currentTrainDurationMs);
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

  async function evaluateAuto() {
    setEvalError("");
    setResult(null);
    setFeedbackText("");
    setCoachVideoUrl("");

    if (isEvaluating) return;

    if (!captureDoneRef.current) {
      setEvalError(
        `当前训练尚未完成，请先跟随左侧视频训练完毕（约 ${(currentTrainDurationMs / 1000).toFixed(1)} 秒）`
      );
      return;
    }

    const frames = framesRef.current;
    if (frames.length < 3) {
      setEvalError(`有效帧过少：${frames.length}（至少需要 3 帧）`);
      return;
    }

    if (!standardReadyRef.current || !standardSeqRef.current.length) {
      setEvalError("当前还没有可用的标准视频骨架序列，请先等待生成完成。");
      return;
    }

    const stdImgs = stdKeyframesRef.current || [];
    const usrImgs = userKeyframesRef.current || [];

    if (stdImgs.length < 2) {
      setEvalError(`标准关键帧不足：${stdImgs.length}（至少需要 2 张）`);
      return;
    }
    if (usrImgs.length < 2) {
      setEvalError(`用户关键帧不足：${usrImgs.length}（至少需要 2 张）`);
      return;
    }

    setIsEvaluating(true);
    setStatusText("正在评估动作并生成反馈...");

    try {
      const payload: any = {
        action,
        user_name: userName,
        frames,
        user_seq: frames,
        standard_seq: standardSeqRef.current,
        standard_images: stdImgs,
        user_images: usrImgs,
        use_llm: true,
      };

      const resp = await fetchWithTimeout(
        EVAL_AUTO_API,
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

      let parsed: any = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = { raw: text };
      }

      setResult(parsed);
      setFeedbackText(parsed?.feedback_text || "");
      setCoachVideoUrl(joinApiUrl(parsed?.coach_video_url || ""));

      setStatusText("评估完成");
      setTimeout(() => {
        const el = coachPlayerRef.current;
        if (el && parsed?.coach_video_url) {
          el.currentTime = 0;
          el.play().catch(() => {});
        }
      }, 80);
    } catch (e: any) {
      if (e?.name === "AbortError") {
        setEvalError(
          `请求超时（${EVAL_TIMEOUT_MS}ms）\n请确认后端 http://127.0.0.1:8000 正在运行，并且 /api/evaluate_auto 没有阻塞。`
        );
      } else {
        setEvalError("请求失败：" + String(e?.message || e));
      }
    } finally {
      setIsEvaluating(false);
    }
  }

  const canEvaluate =
    cameraOn && captureDoneRef.current && !isEvaluating && standardReadyRef.current;

  return (
    <div className="rehab-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-title">智能康复训练平台</div>
            <div className="brand-subtitle">AI Rehabilitation Training Platform</div>
          </div>
        </div>
        <div className="topbar-right">
          <StatusBadge
            text={formatMpStatus(mpStatus)}
            tone={
              mpStatus === "ready"
                ? "success"
                : mpStatus === "error"
                ? "danger"
                : "warn"
            }
          />
          <StatusBadge
            text={buildingStandard ? "标准视频处理中" : "系统运行中"}
            tone={buildingStandard ? "warn" : "info"}
          />
          <div className="user-chip">当前用户：{userName}</div>
        </div>
      </header>

      <section className="hero-banner">
        <div className="hero-text">
          <div className="hero-kicker">个性化 · 智能化 · 可追踪</div>
          <h1 className="hero-title">面向上肢康复训练的智能辅助系统</h1>
          <p className="hero-desc">
            基于姿态识别、动作比对、自动评估与数字人反馈，为用户提供更直观、更专业的康复训练体验。
          </p>
          <div className="hero-tags">
            <StatusBadge text="实时姿态识别" tone="info" />
            <StatusBadge text="自动动作评估" tone="success" />
            <StatusBadge text="数字人反馈" tone="default" />
          </div>
        </div>

        <div className="hero-stats">
          <StatCard label="当前动作" value={action} sub="固定训练动作" />
          <StatCard
            label="训练时长"
            value={`${(currentTrainDurationMs / 1000).toFixed(1)} s`}
            sub="跟练视频时长"
          />
          <StatCard label="评估得分" value={scoreInfo.score} sub={scoreInfo.level} />
          <StatCard label="采集帧数" value={`${framesBuffered}`} sub="动作关键帧累计" />
        </div>
      </section>

      {mpStatus === "error" && (
        <div className="alert-box alert-danger">
          <div className="alert-title">识别模型初始化失败</div>
          <pre className="alert-pre">{mpError}</pre>
        </div>
      )}

      <div className="training-grid">
        <SectionCard
          title="标准示范"
          desc="系统将自动加载当前用户对应的个性化标准动作视频。"
          extra={
            <StatusBadge
              text={standardReadyRef.current ? "标准骨架已就绪" : "标准骨架构建中"}
              tone={standardReadyRef.current ? "success" : "warn"}
            />
          }
        >
          <div className="video-card-wrap">
            <div className="video-header-inline">
              <div className="inline-meta">
                <span>视频数量：{stdVideoList.length}</span>
                <span>当前时长：{(currentTrainDurationMs / 1000).toFixed(1)} 秒</span>
              </div>
            </div>

            <div
              className="video-frame standard-frame adaptive-frame"
              style={{ aspectRatio: `${stdVideoAspect}` }}
            >
              <video
                ref={stdPlayerRef}
                src={activeStandardVideoSrc || undefined}
                controls
                muted
                playsInline
                onLoadedMetadata={(e) => {
                  const el = e.currentTarget;
                  const durSec =
                    el.duration && Number.isFinite(el.duration) ? el.duration : 3;
                  const durMs = Math.max(1000, Math.round(durSec * 1000));
                  setCurrentTrainDurationMs(durMs);

                  if (el.videoWidth > 0 && el.videoHeight > 0) {
                    setStdVideoAspect(el.videoWidth / el.videoHeight);
                  }
                }}
                onEnded={() => {
                  setTrainFinished(true);
                  setTrainHint("本轮示范播放完成，请点击“开始评估”获取结果。");
                  captureDoneRef.current = true;
                }}
                className="video-element fit-contain"
              />
            </div>

            <video
              ref={standardVideoRef}
              src={activeStandardVideoSrc || undefined}
              muted
              playsInline
              crossOrigin="anonymous"
              preload="auto"
              onLoadedMetadata={(e) => {
                const el = e.currentTarget;
                const durSec =
                  el.duration && Number.isFinite(el.duration) ? el.duration : 3;
                const durMs = Math.max(1000, Math.round(durSec * 1000));
                setCurrentTrainDurationMs(durMs);

                if (el.videoWidth > 0 && el.videoHeight > 0) {
                  setStdVideoAspect(el.videoWidth / el.videoHeight);
                }
              }}
              style={{ display: "none" }}
            />

            {!activeStandardVideoSrc && (
              <div className="empty-state-mini">当前暂无可播放的个性化标准视频</div>
            )}

            <div className="train-note">
              {trainFinished
                ? trainHint || "本轮训练结束，请点击评估。"
                : "点击“开启摄像头”后，系统将同步进行跟练采集。"}
            </div>

            {stdVideoList.length > 0 && (
              <div className="video-switcher">
                <div className="switcher-title">可切换的标准视频</div>
                <div className="switcher-grid">
                  {stdVideoList.map((item, idx) => {
                    const url = item.video_url || "";
                    const active = url === activeStandardVideoSrc;
                    return (
                      <button
                        key={`${item.id || idx}_${item.file_name || idx}`}
                        onClick={() => {
                          setSelectedStdVideoUrl(url);
                          setStatusText(`已切换到第 ${idx + 1} 个标准视频`);
                          setTrainFinished(false);
                          setTrainHint("");
                          captureDoneRef.current = false;
                        }}
                        className={`switcher-btn ${active ? "active" : ""}`}
                      >
                        <div className="switcher-file">
                          {item.file_name || `video_${idx + 1}.mp4`}
                        </div>
                        <div className="switcher-state">
                          {item.cached ? "cached" : "generated"}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="mini-status-text">
              {buildingStandard ? "标准视频处理中..." : statusText}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="实时训练"
          desc="跟随左侧标准动作完成训练，系统将自动采集骨架并进行动作评估。"
          extra={
            <div className="panel-action-row">
              <button
                onClick={startCamera}
                disabled={mpStatus !== "ready" || cameraOn}
                className="btn btn-primary"
              >
                开启摄像头
              </button>
              <button
                onClick={stopCamera}
                disabled={!cameraOn}
                className="btn btn-secondary"
              >
                停止采集
              </button>
              <button
                onClick={evaluateAuto}
                disabled={!canEvaluate}
                className="btn btn-accent"
              >
                {isEvaluating ? "评估中..." : "开始评估"}
              </button>
            </div>
          }
        >
          <div className="live-stats-grid">
            <div className="live-pill">
              <span>摄像头</span>
              <strong>{cameraOn ? "已开启" : "未开启"}</strong>
            </div>
            <div className="live-pill">
              <span>人体姿态</span>
              <strong>{poseDetected ? "已检测" : "未检测"}</strong>
            </div>
            <div className="live-pill">
              <span>手部关键点</span>
              <strong>{handsDetected ? "已检测" : "未检测"}</strong>
            </div>
            <div className="live-pill">
              <span>训练状态</span>
              <strong>{captureDoneRef.current ? "已完成" : "进行中"}</strong>
            </div>
          </div>

          <div
            className="video-frame user-frame adaptive-frame"
            style={{ aspectRatio: `${userVideoAspect}` }}
          >
            <video
              ref={userVideoRef}
              playsInline
              muted
              onLoadedMetadata={(e) => {
                const el = e.currentTarget;
                if (el.videoWidth > 0 && el.videoHeight > 0) {
                  setUserVideoAspect(el.videoWidth / el.videoHeight);
                }
              }}
              className="video-element mirrored fit-contain"
            />
            <canvas ref={overlayRef} className="video-overlay" />
            <div className="frame-corner-badge">实时训练画面</div>
          </div>

          <div className="progress-row">
            <div className="progress-meta">
              <span>训练完成度</span>
              <strong>
                {captureDoneRef.current
                  ? "100%"
                  : `${Math.min(
                      99,
                      Math.round(
                        (framesBuffered / Math.max(1, currentTrainDurationMs / SAMPLE_INTERVAL_MS)) *
                          100
                      )
                    )}%`}
              </strong>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${
                    captureDoneRef.current
                      ? 100
                      : Math.min(
                          99,
                          (framesBuffered /
                            Math.max(1, currentTrainDurationMs / SAMPLE_INTERVAL_MS)) *
                            100
                        )
                  }%`,
                }}
              />
            </div>
          </div>

          <div className="helper-text">
            {trainFinished
              ? "标准视频已播放结束，请点击“开始评估”。"
              : `当前训练视频时长：${(currentTrainDurationMs / 1000).toFixed(1)} 秒`}
          </div>
        </SectionCard>
      </div>

      <div className="result-grid">
        <SectionCard
          title="训练结果报告"
          desc="系统将基于标准动作和用户动作关键帧进行自动分析。"
          extra={<StatusBadge text={scoreInfo.level} tone="success" />}
        >
          <div className="score-panel">
            <div className="score-ring">
              <div className="score-ring-inner">
                <div className="score-num">{scoreInfo.score}</div>
                <div className="score-unit">分</div>
              </div>
            </div>

            <div className="score-details">
              <div className="score-bar-head">
                <span>综合评分</span>
                <strong>{scoreInfo.percent}%</strong>
              </div>
              <div className="score-bar">
                <div
                  className="score-bar-fill"
                  style={{ width: `${scoreInfo.percent}%` }}
                />
              </div>

              <div className="report-kv-grid">
                <div className="report-kv">
                  <span>动作名称</span>
                  <strong>{action}</strong>
                </div>
                <div className="report-kv">
                  <span>采集帧数</span>
                  <strong>{framesBuffered}</strong>
                </div>
                <div className="report-kv">
                  <span>标准关键帧</span>
                  <strong>{stdKeyframesRef.current.length}</strong>
                </div>
                <div className="report-kv">
                  <span>用户关键帧</span>
                  <strong>{userKeyframesRef.current.length}</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="feedback-box">
            <div className="feedback-title">文字反馈</div>
            <div className="feedback-content">
              {evalError ? (
                <div className="error-text">{evalError}</div>
              ) : feedbackText ? (
                feedbackText
              ) : (
                "完成训练后，系统将在这里生成动作反馈与改进建议。"
              )}
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="数字人讲解反馈"
          desc="结合评估结果生成视频化口播反馈，辅助用户理解动作问题。"
        >
          <div className="coach-box">
            {coachVideoUrl ? (
              <video
                ref={coachPlayerRef}
                src={coachVideoUrl}
                controls
                autoPlay
                playsInline
                className="coach-video"
              />
            ) : (
              <div className="empty-video-state">暂无数字人反馈视频</div>
            )}
          </div>

          <div className="privacy-note">
            系统仅围绕本轮训练所需的关键帧与结构化评估数据进行处理，用于生成训练反馈。
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="系统调试信息"
        desc="用于开发阶段查看后端返回的结构化结果。正式展示时可隐藏该模块。"
      >
        <pre className="debug-box">{JSON.stringify(result ?? {}, null, 2)}</pre>
      </SectionCard>
    </div>
  );
}

function App() {
  const [inputName, setInputName] = useState<string>(
    () => localStorage.getItem("rehab_user_name") || ""
  );
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
    <div className="entry-shell">
      <div className="entry-bg-orb orb-a" />
      <div className="entry-bg-orb orb-b" />

      <div className="entry-card">
        <div className="entry-left">
          <div className="entry-brand">
            <div className="brand-mark big">R</div>
            <div>
              <div className="brand-title dark">智能康复训练平台</div>
              <div className="brand-subtitle dark">
                Personalized AI-Assisted Rehabilitation
              </div>
            </div>
          </div>

          <h1 className="entry-title">欢迎进入康复训练系统</h1>
          <p className="entry-desc">
            系统将基于用户身份自动加载个性化标准动作视频，并结合摄像头姿态采集、动作评估和数字人讲解完成本轮训练。
          </p>

          <div className="entry-feature-list">
            <div className="entry-feature">实时姿态识别与关键点采集</div>
            <div className="entry-feature">动作跟练与自动评估</div>
            <div className="entry-feature">数字人视频化训练反馈</div>
          </div>
        </div>

        <div className="entry-right">
          {step === "entry" && (
            <div className="form-card">
              <div className="form-card-title">用户登录</div>
              <div className="form-card-desc">
                请输入姓名。若系统中不存在个人模板，将进入首次采集流程。
              </div>

              <label className="field-label">姓名</label>
              <input
                value={inputName}
                onChange={(e) => setInputName(e.target.value)}
                placeholder="请输入姓名，例如 Liu"
                className="modern-input"
              />

              <button
                onClick={checkProfile}
                disabled={busy}
                className="btn btn-primary btn-block"
              >
                {busy ? "检查中..." : "进入系统"}
              </button>
            </div>
          )}

          {step === "record" && (
            <div className="form-card">
              <div className="form-card-title">首次使用信息采集</div>
              <div className="form-card-desc">
                请为 <strong>{resolvedName}</strong> 上传个人照片，并录制 3 秒模板视频。
              </div>

              <label className="field-label">上传个人照片</label>
              <input
                type="file"
                accept="image/*"
                className="modern-file"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setPhotoFile(f);
                  if (previewPhotoUrl) URL.revokeObjectURL(previewPhotoUrl);
                  setPreviewPhotoUrl(f ? URL.createObjectURL(f) : "");
                }}
              />

              {photoFile && <div className="selected-file">已选择：{photoFile.name}</div>}

              {previewPhotoUrl && (
                <img src={previewPhotoUrl} alt="preview" className="preview-photo" />
              )}

              <div className="record-preview-wrap">
                <video
                  ref={recorderVideoRef}
                  autoPlay
                  playsInline
                  muted
                  className="record-preview-video"
                />
                <div className="frame-corner-badge">摄像头预览</div>
              </div>

              <div className="entry-btn-row">
                <button
                  onClick={recordAndUploadV2}
                  disabled={busy}
                  className="btn btn-accent"
                >
                  {busy ? "录制并上传中..." : "开始录制并保存"}
                </button>
                <button
                  onClick={() => {
                    recorderStreamRef.current?.getTracks().forEach((t) => t.stop());
                    recorderStreamRef.current = null;
                    setStep("entry");
                  }}
                  disabled={busy}
                  className="btn btn-secondary"
                >
                  返回
                </button>
              </div>
            </div>
          )}

          {profile && (
            <details className="details-box">
              <summary>查看用户信息</summary>
              <pre className="debug-box compact">{JSON.stringify(profile, null, 2)}</pre>
            </details>
          )}

          {error && (
            <div className="alert-box alert-danger">
              <div className="alert-title">操作失败</div>
              <pre className="alert-pre">{error}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;