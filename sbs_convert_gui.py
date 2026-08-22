import os
import sys
import traceback

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QDoubleSpinBox, QGroupBox, QProgressBar, QMessageBox, QToolButton,
)

import sbs_convert


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_vr180{ext}"


class ConvertWorker(QObject):
    progress = Signal(int, int)      # frame_idx, total_frames
    status = Signal(str)
    finished = Signal(str, str)      # status ("success"/"cancelled"/"error"), message

    def __init__(self, input_path, output_path, params, cancel_token):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.params = params
        self.cancel_token = cancel_token

    def run(self):
        try:
            self.status.emit("深度推定・立体化処理中...")
            sbs_convert.convert_to_vr180(
                self.input_path,
                self.output_path,
                disparity_scale=self.params["disparity_scale"],
                center_focus=self.params["center_focus"],
                depth_min=self.params["depth_min"],
                depth_max=self.params["depth_max"],
                h_fov=self.params["h_fov"],
                v_fov=self.params["v_fov"],
                progress_callback=lambda i, total: self.progress.emit(i, total),
                cancel_token=self.cancel_token,
            )
            self.finished.emit("success", self.output_path)
        except sbs_convert.ConversionCancelled:
            self.finished.emit("cancelled", "")
        except Exception:
            self.finished.emit("error", traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBS 3D VR180 コンバータ")
        self.setMinimumWidth(520)

        self.thread = None
        self.worker = None
        self.cancel_token = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ---- ソースファイル選択 ----
        source_box = QGroupBox("ソース動画")
        source_layout = QVBoxLayout(source_box)

        src_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("変換する動画ファイルを選択...")
        self.source_edit.textChanged.connect(self._on_source_changed)
        src_browse = QPushButton("参照...")
        src_browse.clicked.connect(self._browse_source)
        src_row.addWidget(self.source_edit)
        src_row.addWidget(src_browse)
        source_layout.addLayout(src_row)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("出力先（ソース名 + _vr180）")
        out_browse = QPushButton("参照...")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(out_browse)
        source_layout.addLayout(out_row)

        layout.addWidget(source_box)

        # ---- 基本パラメータ ----
        basic_box = QGroupBox("基本パラメータ")
        basic_form = QFormLayout(basic_box)

        self.disparity_spin = QDoubleSpinBox()
        self.disparity_spin.setRange(0.0, 20.0)
        self.disparity_spin.setSingleStep(0.5)
        self.disparity_spin.setValue(sbs_convert.DEFAULT_DISPARITY_SCALE)
        basic_form.addRow("3D強度（視差の強さ）", self.disparity_spin)

        self.center_focus_spin = QDoubleSpinBox()
        self.center_focus_spin.setRange(0.5, 10.0)
        self.center_focus_spin.setSingleStep(0.5)
        self.center_focus_spin.setValue(sbs_convert.DEFAULT_CENTER_FOCUS)
        basic_form.addRow("中央フォーカス幅", self.center_focus_spin)

        layout.addWidget(basic_box)

        # ---- 詳細設定（折りたたみ） ----
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("詳細設定 ▸")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)

        self.advanced_box = QGroupBox()
        self.advanced_box.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_box)

        warning = QLabel(
            "⚠ FOVは映像の歪みに直結します。基本的にデフォルト値のまま推奨します。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #b35c00;")
        advanced_layout.addWidget(warning)

        advanced_form = QFormLayout()

        self.h_fov_spin = QDoubleSpinBox()
        self.h_fov_spin.setRange(30.0, 180.0)
        self.h_fov_spin.setValue(sbs_convert.DEFAULT_H_FOV)
        advanced_form.addRow("水平FOV (h_fov)", self.h_fov_spin)

        self.v_fov_spin = QDoubleSpinBox()
        self.v_fov_spin.setRange(20.0, 120.0)
        self.v_fov_spin.setValue(sbs_convert.DEFAULT_V_FOV)
        advanced_form.addRow("垂直FOV (v_fov)", self.v_fov_spin)

        self.depth_min_spin = QDoubleSpinBox()
        self.depth_min_spin.setRange(0.0, 1.0)
        self.depth_min_spin.setSingleStep(0.05)
        self.depth_min_spin.setValue(sbs_convert.DEFAULT_DEPTH_MIN)
        advanced_form.addRow("深度クリップ下限", self.depth_min_spin)

        self.depth_max_spin = QDoubleSpinBox()
        self.depth_max_spin.setRange(0.0, 1.0)
        self.depth_max_spin.setSingleStep(0.05)
        self.depth_max_spin.setValue(sbs_convert.DEFAULT_DEPTH_MAX)
        advanced_form.addRow("深度クリップ上限", self.depth_max_spin)

        advanced_layout.addLayout(advanced_form)
        layout.addWidget(self.advanced_box)

        # ---- 実行 ----
        button_row = QHBoxLayout()
        self.convert_button = QPushButton("変換開始")
        self.convert_button.clicked.connect(self._start_conversion)
        button_row.addWidget(self.convert_button)

        self.cancel_button = QPushButton("中断")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_conversion)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("待機中")
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ---- UI操作 ----
    def _toggle_advanced(self):
        expanded = self.advanced_toggle.isChecked()
        self.advanced_box.setVisible(expanded)
        self.advanced_toggle.setText("詳細設定 ▾" if expanded else "詳細設定 ▸")

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ソース動画を選択", "",
            "動画ファイル (*.mp4 *.mov *.avi *.mkv);;すべてのファイル (*)"
        )
        if path:
            self.source_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "出力先を選択", self.output_edit.text() or "",
            "MP4ファイル (*.mp4);;すべてのファイル (*)"
        )
        if path:
            self.output_edit.setText(path)

    def _on_source_changed(self, text):
        if text:
            self.output_edit.setText(default_output_path(text))

    # ---- 変換処理 ----
    def _start_conversion(self):
        input_path = self.source_edit.text().strip()
        output_path = self.output_edit.text().strip()

        if not input_path or not os.path.isfile(input_path):
            QMessageBox.warning(self, "エラー", "有効なソース動画を選択してください。")
            return
        if not output_path:
            QMessageBox.warning(self, "エラー", "出力先を指定してください。")
            return

        params = {
            "disparity_scale": self.disparity_spin.value(),
            "center_focus": self.center_focus_spin.value(),
            "depth_min": self.depth_min_spin.value(),
            "depth_max": self.depth_max_spin.value(),
            "h_fov": self.h_fov_spin.value(),
            "v_fov": self.v_fov_spin.value(),
        }
        if params["depth_min"] >= params["depth_max"]:
            QMessageBox.warning(self, "エラー", "深度クリップ下限は上限より小さくしてください。")
            return

        self.convert_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("準備中...")

        self.cancel_token = sbs_convert.CancelToken()

        self.thread = QThread()
        self.worker = ConvertWorker(input_path, output_path, params, self.cancel_token)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _cancel_conversion(self):
        if self.cancel_token:
            self.cancel_token.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("中断中...")

    def _on_progress(self, frame_idx, total_frames):
        if total_frames > 0:
            percent = int(frame_idx / total_frames * 90)  # 90%までは深度推定パス
            self.progress_bar.setValue(percent)
            self.status_label.setText(f"深度推定・立体化処理中... ({frame_idx}/{total_frames})")
        else:
            self.status_label.setText(f"深度推定・立体化処理中... ({frame_idx}フレーム)")

    def _on_finished(self, status, message):
        self.convert_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.cancel_token = None
        if status == "success":
            self.progress_bar.setValue(100)
            self.status_label.setText(f"完了: {message}")
            QMessageBox.information(self, "完了", f"変換が完了しました:\n{message}")
        elif status == "cancelled":
            self.progress_bar.setValue(0)
            self.status_label.setText("中断しました")
        else:
            self.status_label.setText("エラーが発生しました")
            QMessageBox.critical(self, "エラー", message)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
