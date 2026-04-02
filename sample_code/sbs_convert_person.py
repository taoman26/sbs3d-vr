import cv2
import torch
import numpy as np

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_sbs.mp4"

BASE_DISPARITY = 6      # 背景
PERSON_BOOST = 2.0      # 人物の強調倍率

print("Loading MiDaS model...")
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to("cpu")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# -----------------------------
# 人物検出（HOG）
# -----------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # 軽量化（推奨）
    # frame = cv2.resize(frame, (640, 360))

    h, w = frame.shape[:2]

    # -----------------------------
    # Depth推定
    # -----------------------------
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
    # 人物検出
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, _ = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # 人物マスク作成
    person_mask = np.zeros((h, w), dtype=np.float32)

    for (x, y, bw, bh) in boxes:
        person_mask[y:y+bh, x:x+bw] = 1.0

    # マスクをぼかして自然に
    person_mask = cv2.GaussianBlur(person_mask, (31, 31), 0)

    # -----------------------------
    # 視差生成（人物強調）
    # -----------------------------
    disparity = (1.0 - depth_norm) * BASE_DISPARITY

    # 人物部分だけ強調
    disparity = disparity * (1 + person_mask * (PERSON_BOOST - 1))
    disparity = disparity.astype(np.int32)

    # -----------------------------
    # NumPyで高速シフト
    # -----------------------------
    x_coords = np.tile(np.arange(w), (h, 1))

    left_x = np.clip(x_coords + disparity, 0, w - 1)
    right_x = np.clip(x_coords - disparity, 0, w - 1)

    y_coords = np.arange(h)[:, None]

    left_img = frame[y_coords, left_x]
    right_img = frame[y_coords, right_x]

    # SBS
    sbs = np.hstack((left_img, right_img))
    out.write(sbs)

cap.release()
out.release()

print("Done!")
