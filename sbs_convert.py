import os
import cv2
import numpy as np
import torch
import subprocess
import threading

# ======================
# 入出力
# ======================
INPUT_PATH      = "input.mp4"
TMP_PATH        = "tmp_sbs.mp4"
TMP_AUDIO_PATH  = "tmp_sbs_audio.mp4"
OUTPUT_PATH     = "output_sbs.mp4"

# ======================
# デフォルトパラメータ
# ======================
DEFAULT_DISPARITY_SCALE = 8      # 3D強度（視差）
DEFAULT_CENTER_FOCUS    = 4.0    # 中央フォーカス幅（ガウスσ）
DEFAULT_DEPTH_MIN       = 0.2    # 深度クリップ下限
DEFAULT_DEPTH_MAX       = 0.8    # 深度クリップ上限
DEFAULT_H_FOV           = 90     # VR180変換の水平FOV
DEFAULT_V_FOV           = 70     # VR180変換の垂直FOV

class ConversionCancelled(Exception):
    """ユーザー操作による変換中断"""
    pass


class CancelToken:
    """GUIのスレッドから変換処理に中断を伝えるためのトークン"""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def is_cancelled(self):
        return self._event.is_set()


def _run_ffmpeg(cmd, cancel_token=None):
    """ffmpegをサブプロセスとして実行し、cancel_tokenがセットされたら中断する"""
    proc = subprocess.Popen(cmd)
    try:
        while True:
            try:
                returncode = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if cancel_token and cancel_token.is_cancelled:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise ConversionCancelled()
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        raise

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


# MiDaS（軽量）モデルは初回使用時に遅延ロードする
_model = None
_transform = None


def get_model():
    global _model, _transform
    if _model is None:
        _model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        _model.eval()
        _transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return _model, _transform


# ---- 深度推定 ----
def estimate_depth(frame):
    model, transform = get_model()

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
def stereo_from_depth_np(frame, depth, disparity_scale=DEFAULT_DISPARITY_SCALE,
                          center_focus=DEFAULT_CENTER_FOCUS):
    h, w = depth.shape

    # 視差（弱めが重要）
    disparity = (depth * disparity_scale).astype(np.int32)

    # 中央強調（人物想定）
    x = np.linspace(-1, 1, w)
    center_weight = np.exp(-(x**2) * center_focus)  # ガウス
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
def process_video(input_path, output_path,
                   disparity_scale=DEFAULT_DISPARITY_SCALE,
                   center_focus=DEFAULT_CENTER_FOCUS,
                   depth_min=DEFAULT_DEPTH_MIN,
                   depth_max=DEFAULT_DEPTH_MAX,
                   progress_callback=None,
                   cancel_token=None):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w*2, h)
    )

    frame_idx = 0
    while True:
        if cancel_token and cancel_token.is_cancelled:
            cap.release()
            out.release()
            raise ConversionCancelled()

        ret, frame = cap.read()
        if not ret:
            break

        # ① 深度
        depth = estimate_depth(frame)

        # ② 安定化
        depth = np.clip(depth, depth_min, depth_max)

        # ③ ステレオ生成
        left, right = stereo_from_depth_np(frame, depth, disparity_scale, center_focus)

        # ④ SBS
        sbs = np.hstack((left, right))

        out.write(sbs)

        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()


# ======================
# ffmpeg整形（SKYBOX用）
# 左右眼をそれぞれ独立して equirectangular に変換し
# 再度 SBS に結合することで SKYBOX の
# 「180° 3D SBS」設定のみで視聴できる形式を生成する
# ======================
def convert_for_skybox(input_path, output_path,
                        h_fov=DEFAULT_H_FOV, v_fov=DEFAULT_V_FOV,
                        cancel_token=None):
    filter_complex = (
        # ① SBSを左右に分割
        "[0:v]crop=iw/2:ih:0:0[L];"
        "[0:v]crop=iw/2:ih:iw/2:0[R];"
        # ② 各眼を 3:2 にクロップ（水平中央）
        #    16:9横長動画は「高さ維持・横を縮める」方向でクロップ
        #    例: 1920x1080 → 1620x1080（左右各150pxカット）
        #    trunc(ih*3/2/2)*2 で偶数保証（yuv420p要件）
        "[L]crop=trunc(ih*3/2/2)*2:ih:(iw-trunc(ih*3/2/2)*2)/2:0[LC];"
        "[R]crop=trunc(ih*3/2/2)*2:ih:(iw-trunc(ih*3/2/2)*2)/2:0[RC];"
        # ③ 各眼を独立して equirectangular へ変換
        f"[LC]v360=input=flat:output=equirect:h_fov={h_fov}:v_fov={v_fov}[LE];"
        f"[RC]v360=input=flat:output=equirect:h_fov={h_fov}:v_fov={v_fov}[RE];"
        "[LE]scale=iw:trunc(ih*1.5/2)*2[LES];"
        "[RE]scale=iw:trunc(ih*1.5/2)*2[RES];"
        # ④ 上下に黒帯を追加して 1:1 に（VR180規格: 各眼が正方形、全体2:1）
        #    3:2クロップ後は iw > ih なので pad=iw:iw:0:(iw-ih)/2 で上下追加
        #    例: 1620x1080 → 1620x1620
        "[LES]pad=iw:iw:0:(iw-ih)/2[LEP];"
        "[RES]pad=iw:iw:0:(iw-ih)/2[REP];"
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
    _run_ffmpeg(cmd, cancel_token)


def convert_to_vr180(input_path, output_path,
                      disparity_scale=DEFAULT_DISPARITY_SCALE,
                      center_focus=DEFAULT_CENTER_FOCUS,
                      depth_min=DEFAULT_DEPTH_MIN,
                      depth_max=DEFAULT_DEPTH_MAX,
                      h_fov=DEFAULT_H_FOV,
                      v_fov=DEFAULT_V_FOV,
                      progress_callback=None,
                      cancel_token=None):
    """2D動画をVR180 SBSに変換する一連の処理をまとめたエントリポイント。

    GUI・CLIの両方から呼び出せるよう、入出力パスとパラメータを引数化している。
    cancel_token（CancelToken）がキャンセルされると ConversionCancelled を送出する。
    """
    base, _ = os.path.splitext(output_path)
    tmp_path = base + "_tmp_sbs.mp4"
    tmp_audio_path = base + "_tmp_sbs_audio.mp4"

    try:
        process_video(
            input_path, tmp_path,
            disparity_scale=disparity_scale,
            center_focus=center_focus,
            depth_min=depth_min,
            depth_max=depth_max,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )

        # 元動画の音声を映像のみの一時ファイルにマージ
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            tmp_audio_path
        ], cancel_token)

        convert_for_skybox(tmp_audio_path, output_path, h_fov=h_fov, v_fov=v_fov,
                            cancel_token=cancel_token)
    except ConversionCancelled:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        for tmp in [tmp_path, tmp_audio_path]:
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == "__main__":
    convert_to_vr180(INPUT_PATH, OUTPUT_PATH)
    print("✅ 完了:", OUTPUT_PATH)
