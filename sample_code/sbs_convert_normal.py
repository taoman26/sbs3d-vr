import cv2
import torch
import numpy as np

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_sbs.mp4"
DISPARITY_SCALE = 10

print("Loading MiDaS model...")
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to("cpu")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width * 2, height)
)

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 軽量化（任意）
    # frame = cv2.resize(frame, (640, 360))

    h, w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to("cpu")

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    # -----------------------------
    # ★ NumPyで視差処理（高速化ポイント）
    # -----------------------------
    disparity = ((1.0 - depth_norm) * DISPARITY_SCALE).astype(np.int32)

    # 座標グリッド生成
    x_coords = np.tile(np.arange(w), (h, 1))

    # 左右シフト座標
    left_x = np.clip(x_coords + disparity, 0, w - 1)
    right_x = np.clip(x_coords - disparity, 0, w - 1)

    # 行インデックス
    y_coords = np.arange(h)[:, None]

    # 高速ピクセル取得
    left_img = frame[y_coords, left_x]
    right_img = frame[y_coords, right_x]

    # SBS合成
    sbs = np.hstack((left_img, right_img))

    out.write(sbs)

cap.release()
out.release()

print("Done!")
