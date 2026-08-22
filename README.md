# sbs3d-vr

2D動画をAI深度推定によって疑似3D立体化し、**Oculus GO + SKYBOX** で VR180 SBS として視聴できる形式に変換するPythonスクリプト集です。

> **推奨スクリプト:** `sbs_convert.py`  
> **SKYBOXでの設定:** 180度 / 3D SBS / 標準（魚眼・モノスコープ不要）

---

## 概要

```
2D動画 (MP4)
  ↓ MiDaS で各フレームの深度推定
  ↓ 深度に応じた横方向ピクセルシフト → 左眼・右眼フレーム生成
  ↓ ffmpeg filter_complex:
      左右眼を個別に equirectangular 変換 → 各眼を正方形 pad → SBS 再結合
output_sbs.mp4  ← SKYBOX「180度 / 3D SBS」で視聴
```

---

## 動作環境

- Python 3.10+
- ffmpeg（システムにインストール済みであること）
- CPU動作（GPU不要）

### ffmpeg のインストール

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

---

## セットアップ

```bash
git clone https://github.com/taoman26/sbs3d-vr.git
cd sbs3d-vr

pip install -r requirements.txt
```

---

## 使い方

### 推奨スクリプト: `sbs_convert.py`

```bash
# input.mp4 を同じディレクトリに置いてから実行
python sbs_convert.py
```

出力: `output_sbs.mp4`

**SKYBOX での設定:** `180度` / `3D SBS` / `標準`

---

### GUIアプリ: `sbs_convert_gui.py`

ファイル選択とパラメータ調整をGUIで行える PySide6 製アプリです。`sbs_convert.py` と同じ変換処理をライブラリとして呼び出します。

```bash
python sbs_convert_gui.py
```

- ソース動画を選択すると、出力先が自動的に `元ファイル名_vr180.拡張子` になります（同じディレクトリに出力）
- 基本パラメータ（3D強度・中央フォーカス幅）をその場で調整可能
- 詳細設定（FOV・深度クリップ範囲）は歪みに直結するため折りたたみ式・警告表示付き
- 変換中は「中断」ボタンでいつでも中止でき、一時ファイル・不完全な出力ファイルは自動削除されます

---

### スクリプト一覧

| スクリプト | 説明 | SKYBOX設定 |
|---|---|---|
| `sbs_convert.py` | **推奨。** 左右眼を個別に equirectangular 変換し、正規の VR180 SBS 形式を生成 | 180度 / 3D SBS / 標準 |
| `sbs_convert_gui.py` | `sbs_convert.py` のGUI版（PySide6）。ファイル選択・パラメータ調整・中断に対応 | 180度 / 3D SBS / 標準 |
| `sbs_convert_360.py` | 初期バージョン。SBS全体を1枚としてequirect変換するため、特殊な設定が必要 | 180度 / モノスコープ / 魚眼 |
| `sample_code/sbs_convert_normal.py` | シンプルなSBS変換のみ（ffmpeg処理なし） | — |
| `sample_code/sbs_convert_person.py` | HOGによる人物検出付きで人物部分の視差を強調 | — |
| `sample_code/sbs_convert_skybox.py` | equirectangular変換付きの試作版 | 180度 / モノスコープ / 魚眼 |

---

## 入出力ファイル名の変更

各スクリプト冒頭の定数を変更してください。

```python
INPUT_PATH  = "input.mp4"   # 入力ファイル名
OUTPUT_PATH = "output_sbs.mp4"  # 出力ファイル名
```

---

## パラメータ調整

`sbs_convert.py` の主要なパラメータ（`convert_to_vr180()` の引数、GUI版では対応する項目から調整可能）：

| 引数 | デフォルト | 効果 | GUIでの位置 |
|---|---|---|---|
| `disparity_scale` | `8` | 視差強度（3D強度）。大きいほど立体感が強まる（目が疲れるので8以下推奨） | 基本パラメータ |
| `center_focus` | `4.0` | 中央ガウス重みのσ（中央フォーカス幅）。人物が中央にいる動画向け | 基本パラメータ |
| `depth_min` / `depth_max` | `0.2` / `0.8` | 深度の有効範囲。端の値を捨てて安定化 | 詳細設定 |
| `h_fov` / `v_fov` | `90` / `70` | equirectangular変換のFOV。変更すると歪みに直結するため基本的にデフォルト推奨 | 詳細設定 |

左右眼を3:2でクロップする比率は歪み対策として固定値になっており、`h_fov`/`v_fov`とは独立している（詳細は次項）。

---

## 変換パイプラインの詳細（`sbs_convert.py`）

```
SBS動画 (3840×1080)  ← process_video() が生成
  ↓ 音声マージ
  ↓ ffmpeg filter_complex:
      crop 左眼(1920×1080)        crop 右眼(1920×1080)
        ↓ 3:2 クロップ               ↓ 3:2 クロップ
      (1620×1080)                (1620×1080)
        ↓ v360(flat→equirect)       ↓ v360(flat→equirect)
      (1620×1080)                (1620×1080)
        ↓ pad 上下黒帯               ↓ pad 上下黒帯
      (1620×1620)                (1620×1620)
                ↓ hstack
          SBS (3240×1620)  ← 2:1 = 標準 VR180 SBS 規格
```

**3:2 クロップの理由:**  
元映像が 16:9 のため、ピクセル密度が水平・垂直でずれて横伸びが生じる。v360 の前に 3:2 にクロップすることで解消。この比率は `h_fov`/`v_fov` の値によらず固定しており、両者を連動させると歪みが悪化することを確認済み。

**正方形 pad の理由:**  
SKYBOX の 180° モードは各眼を 180°×180° の半球に均等マッピングする。各眼が正方形（1:1）でないと縦伸びが生じる。

---

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| PyTorch | MiDaS 深度推定モデルの実行 |
| OpenCV | 動画読み書き、画像処理 |
| NumPy | 深度マップに基づくピクセルシフト処理 |
| timm | MiDaS の内部依存 |
| ffmpeg | equirectangular 変換・H.264 エンコード（外部ツール） |
| PySide6 | `sbs_convert_gui.py` のGUI |

---

## ライセンス

MIT License

---

## 参考

- [MiDaS - Intel ISL](https://github.com/isl-org/MiDaS)
- [SKYBOX VR Player](https://skybox.xyz/)
- ffmpeg `v360` filter
