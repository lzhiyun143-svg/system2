import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const commonOptions = {
    runningMode: "VIDEO" as const,
    numPoses: 1,
    // 把阈值调低，更容易检测到
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  };

  // 先尝试 GPU，失败则自动降级 CPU（Windows 上很常见）
  try {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      ...commonOptions,
    });
  } catch (e) {
    console.warn("[pose] GPU delegate failed, fallback to CPU.", e);
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "CPU",
      },
      ...commonOptions,
    });
  }
}
