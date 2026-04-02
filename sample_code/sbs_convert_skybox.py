import cv2
import numpy as np
import torch
import subprocess

# ======================
# 入出力
# ======================
INPUT_PATH = "input.mp4"
TMP_PATH   = "tmp_sbs.mp4"
OUTPUT_PATH = "output_360_sbs.mp4"

# ======================
# MiDaS（軽量）
# ======================
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# ======================
# 疑似 equirectangular
# ======================
def to_equirectangular_np(frame):
    h, w = frame.shape[:2]

    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    # 歪み調整（重要）
    xv2 = np.tan(xv * np.pi / 3)
    yv2 = np.tan(yv * np.pi / 4)

    src_x = ((xv2 + 1) * 0.5 * (w - 1)).astype(np.int32)
    src_y = ((yv2 + 1) * 0.5 * (h - 1)).astype(np.int32)

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    return frame[src_y, src_x]

# ======================
# 深度推定
# ======================
def estimate_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth

# ======================
# 感性ワープ（単眼立体）
# ======================
def perceptual_warp(frame, depth):
    h, w = depth.shape

    # 中央強調（人物前提）
    x = np.linspace(-1, 1, w)
    center_weight = np.exp(-(x**2) * 3)

    # 深度制限
    depth = np.clip(depth, 0.3, 0.8)

    # Z方向っぽいスケール
    scale = 1 + depth * 0.15 * center_weight[None, :]

    xv, yv = np.meshgrid(np.arange(w), np.arange(h))

    cx = w // 2
    cy = h // 2

    x_new = (xv - cx) / scale + cx
    y_new = (yv - cy) / scale + cy

    x_new = np.clip(x_new, 0, w - 1).astype(np.int32)
    y_new = np.clip(y_new, 0, h - 1).astype(np.int32)

    return frame[y_new, x_new]

# ======================
# SBS生成（左右同一＝安定）
# ======================
def make_sbs(frame):
    return np.hstack((frame, frame))

# ======================
# メイン処理
# ======================
def process_video(input_path, tmp_path):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w * 2, h)
    )

    frame_idx = 0
    depth = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 深度は間引き（高速化）
        if frame_idx % 2 == 0 or depth is None:
            depth = estimate_depth(frame)

        # 感性立体化
        warped = perceptual_warp(frame, depth)

        # 360化
        eq = to_equirectangular_np(warped)

        # SBS（左右同一）
        sbs = make_sbs(eq)

        out.write(sbs)

        frame_idx += 1

    cap.release()
    out.release()

# ======================
# ffmpeg整形（SKYBOX用）
# ======================
def convert_for_skybox(input_path, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,

        # フラット映像 → equirectangular（湾曲前提）
        "-vf", "v360=input=flat:output=equirect:h_fov=90:v_fov=60",

        # H.264で安定化
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",

        # Oculus対応
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        # 360動画として認識
        "-metadata:s:v", "projection=equirectangular",

        output_path
    ]

    subprocess.run(cmd)

# ======================
# 実行
# ======================
process_video(INPUT_PATH, TMP_PATH)
convert_for_skybox(TMP_PATH, OUTPUT_PATH)

print("✅ 完了:", OUTPUT_PATH)
