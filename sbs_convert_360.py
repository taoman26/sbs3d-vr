import os
import cv2
import numpy as np
import torch
import subprocess

# ======================
# 入出力
# ======================
INPUT_PATH      = "input.mp4"
TMP_PATH        = "tmp_sbs.mp4"
TMP_AUDIO_PATH  = "tmp_sbs_audio.mp4"
OUTPUT_PATH     = "output_360_sbs.mp4"

# MiDaS（軽量）
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# ---- 疑似 equirectangular 変換（NumPy版） ----
def to_equirectangular_np(frame):
    h, w = frame.shape[:2]

    # 正規化座標 [-1,1]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)

    # 球面っぽく歪ませる（軽量版）
    xv2 = np.tan(xv * np.pi/2)
    yv2 = np.tan(yv * np.pi/4)

    # 元画像座標に戻す
    src_x = ((xv2 + 1) * 0.5 * (w-1)).astype(np.int32)
    src_y = ((yv2 + 1) * 0.5 * (h-1)).astype(np.int32)

    src_x = np.clip(src_x, 0, w-1)
    src_y = np.clip(src_y, 0, h-1)

    return frame[src_y, src_x]

# ---- 深度推定 ----
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

# ---- NumPy版ステレオ生成（核心） ----
def stereo_from_depth_np(frame, depth):
    h, w = depth.shape

    # 視差（弱めが重要）
    disparity = (depth * 8).astype(np.int32)

    # 中央強調（人物想定）
    x = np.linspace(-1, 1, w)
    center_weight = np.exp(-(x**2) * 4)  # ガウス
    disparity = (disparity * center_weight[None, :]).astype(np.int32)

    # 座標グリッド
    xx = np.arange(w)[None, :].repeat(h, axis=0)

    # 左右座標
    x_left  = np.clip(xx - disparity//2, 0, w-1)
    x_right = np.clip(xx + disparity//2, 0, w-1)

    # 画素取得（高速インデクシング）
    left  = frame[np.arange(h)[:,None], x_left]
    right = frame[np.arange(h)[:,None], x_right]

    return left, right

# ---- メイン処理 ----
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w*2, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ① 深度
        depth = estimate_depth(frame)

        # ② 安定化
        depth = np.clip(depth, 0.2, 0.8)

        # ③ ステレオ生成
        left, right = stereo_from_depth_np(frame, depth)

        # ④ SBS
        sbs = np.hstack((left, right))

        out.write(sbs)

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

        # 音声をそのままコピー
        "-c:a", "copy",

        # Oculus対応
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        # 360動画として認識
        "-metadata:s:v", "projection=equirectangular",

        output_path
    ]

    subprocess.run(cmd)

# ---- 実行 ----
process_video(INPUT_PATH, TMP_PATH)

# 元動画の音声を映像のみの一時ファイルにマージ
subprocess.run([
    "ffmpeg", "-y",
    "-i", TMP_PATH,
    "-i", INPUT_PATH,
    "-map", "0:v:0",
    "-map", "1:a:0?",
    "-c:v", "copy",
    "-c:a", "copy",
    "-shortest",
    TMP_AUDIO_PATH
], check=True)

convert_for_skybox(TMP_AUDIO_PATH, OUTPUT_PATH)

# 一時ファイルを削除
for tmp in [TMP_PATH, TMP_AUDIO_PATH]:
    if os.path.exists(tmp):
        os.remove(tmp)

print("✅ 完了:", OUTPUT_PATH)
