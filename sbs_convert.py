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
OUTPUT_PATH     = "output_sbs.mp4"

# MiDaS（軽量）
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

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

# ---- NumPy版ステレオ生成 ----
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
# 左右眼をそれぞれ独立して equirectangular に変換し
# 再度 SBS に結合することで SKYBOX の
# 「180° 3D SBS」設定のみで視聴できる形式を生成する
# ======================
def convert_for_skybox(input_path, output_path):
    filter_complex = (
        # ① SBSを左右に分割
        "[0:v]crop=iw/2:ih:0:0[L];"
        "[0:v]crop=iw/2:ih:iw/2:0[R];"
        # ② 各眼を h_fov:v_fov = 90:60 = 3:2 にクロップ（水平中央）
        #    16:9横長動画は「高さ維持・横を縮める」方向でクロップ
        #    例: 1920x1080 → 1620x1080（左右各150pxカット）
        #    trunc(ih*3/2/2)*2 で偶数保証（yuv420p要件）
        "[L]crop=trunc(ih*3/2/2)*2:ih:(iw-trunc(ih*3/2/2)*2)/2:0[LC];"
        "[R]crop=trunc(ih*3/2/2)*2:ih:(iw-trunc(ih*3/2/2)*2)/2:0[RC];"
        # ③ 各眼を独立して equirectangular へ変換
        "[LC]v360=input=flat:output=equirect:h_fov=90:v_fov=60[LE];"
        "[RC]v360=input=flat:output=equirect:h_fov=90:v_fov=60[RE];"
        # ④ 上下に黒帯を追加して 1:1 に（VR180規格: 各眼が正方形、全体2:1）
        #    3:2クロップ後は iw > ih なので pad=iw:iw:0:(iw-ih)/2 で上下追加
        #    例: 1620x1080 → 1620x1620
        "[LE]pad=iw:iw:0:(iw-ih)/2[LEP];"
        "[RE]pad=iw:iw:0:(iw-ih)/2[REP];"
        # ⑤ SBS 再結合（出力例: 3840x1920 = 標準VR180形式）
        "[LEP][REP]hstack[OUT]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[OUT]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-metadata:s:v", "projection=equirectangular",
        output_path
    ]
    subprocess.run(cmd, check=True)

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
