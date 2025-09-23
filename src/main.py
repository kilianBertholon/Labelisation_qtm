"""Multicam Annotator - prototype

Objectif:
- Afficher 6 vidéos synchronisées (.avi recommandé)
- Contrôles play/pause/seek
- Annotation simple (labels timestampés) et export Excel

Notes/assomptions:
- Prototype: on utilise OpenCV pour le décodage (CPU) et une option "GPU" affichée
  si une build d'OpenCV CUDA est disponible. Pour un .exe final, utilisez PyInstaller
  (voir README.md).
- On suppose des vidéos avec framerate similaire; si différent, la synchronisation
  suit le fps minimum commun.
"""

"""Multicam Annotator - stable prototype

Fonctionnalités conservées dans cette version stable:
- Lecture synchronisée de jusqu'à 6 vidéos
- Contrôles: Play/Pause, Prev/Next frame, Seek (slider), vitesse
- Annotations simples et export Excel (pandas + openpyxl)

Cette version évite les mécanismes avancés (préfetch asynchrone, réglage de cache
dans l'UI) pour rester robuste.
"""

import sys
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLineEdit, QMessageBox, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class Annotation:
    cam: int
    frame_idx: int
    time_sec: float
    label: str


class VideoWorker(QThread):
    frames_ready = Signal(list, int)  # list of QImage or None, current_frame_idx

    def __init__(self, paths, cache_size=120, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.caps = [None] * len(paths)
        self.fps = []
        self.frame_counts = []
        self.running = False
        self.playing = False
        self.current_frame = 0
        self.playback_rate = 1.0
        self.cache_size = cache_size
        self.caches = [OrderedDict() for _ in paths]

    def open_all(self):
        for i, p in enumerate(self.paths):
            cap = None
            try:
                cap = cv2.VideoCapture(str(p))
            except Exception:
                cap = None
            if cap is None or not cap.isOpened():
                self.caps[i] = None
                self.fps.append(30.0)
                self.frame_counts.append(0)
                continue
            self.caps[i] = cap
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps.append(fps)
            self.frame_counts.append(frames)

    def close_all(self):
        for cap in self.caps:
            if cap:
                cap.release()
        self.caps = []

    def run(self):
        if cv2 is None:
            return
        try:
            self.open_all()
        except Exception:
            self.frames_ready.emit([None] * len(self.paths), 0)
            return

        master_fps = min(self.fps) if self.fps else 30.0
        self.running = True
        last_time = time.time()
        while self.running:
            if not self.playing:
                time.sleep(0.02)
                continue
            period = 1.0 / (master_fps * max(1e-6, self.playback_rate))
            now = time.time()
            elapsed = now - last_time
            if elapsed < period:
                time.sleep(max(0, period - elapsed))
                now = time.time()
                elapsed = now - last_time
            last_time = now
            imgs = []
            for i, cap in enumerate(self.caps):
                frame = self._get_frame(i, self.current_frame)
                if frame is None:
                    imgs.append(None)
                else:
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    imgs.append(image.copy())
            self.frames_ready.emit(imgs, self.current_frame)
            self.current_frame += 1
            if any(self.current_frame >= cnt for cnt in self.frame_counts if cnt > 0):
                self.playing = False
        self.close_all()

    def _get_frame(self, cam_idx, frame_idx):
        cache = self.caches[cam_idx]
        if frame_idx in cache:
            cache.move_to_end(frame_idx)
            return cache[frame_idx]
        cap = self.caps[cam_idx]
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        cache[frame_idx] = frame
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return frame

    def seek(self, frame_idx):
        self.current_frame = int(frame_idx)

    def step(self, n=1):
        self.current_frame = max(0, self.current_frame + int(n))

    def set_playback_rate(self, rate: float):
        self.playback_rate = float(rate)

    def set_playing(self, v: bool):
        self.playing = bool(v)

    def stop(self):
        self.running = False


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('multicam_annotator')
        self.video_labels = []
        self.paths = []
        self.worker = None
        self.annotations = []
        self.init_ui()

    def init_ui(self):
        main = QVBoxLayout()
        grid = QGridLayout()
        for i in range(6):
            lbl = QLabel(f"Cam {i+1}\n(aucune vidéo)")
            lbl.setFixedSize(480, 360)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #000; color: #fff; border: 1px solid #444;")
            grid.addWidget(lbl, i // 3, i % 3)
            self.video_labels.append(lbl)

        main.addLayout(grid)

        # Controls
        ctrls = QHBoxLayout()
        btn_load = QPushButton('Charger 6 vidéos')
        btn_load.clicked.connect(self.load_videos)
        self.btn_prev = QPushButton('◀')
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self.step_prev)
        self.btn_play = QPushButton('Play')
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton('▶')
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.step_next)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_seek)
        self.speed_box = QComboBox()
        self.speed_box.addItems(['0.25x', '0.5x', '1x', '2x'])
        self.speed_box.setCurrentText('1x')
        self.speed_box.currentTextChanged.connect(self.on_speed_changed)

        ctrls.addWidget(btn_load)
        ctrls.addWidget(self.btn_prev)
        ctrls.addWidget(self.btn_play)
        ctrls.addWidget(self.btn_next)
        ctrls.addWidget(self.slider)
        ctrls.addWidget(self.speed_box)
        main.addLayout(ctrls)

        # Annotation controls
        ann = QHBoxLayout()
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText('Label (ex: walking, error...)')
        btn_add_label = QPushButton('Ajouter label (cam active)')
        btn_add_label.clicked.connect(self.add_label)
        btn_export = QPushButton('Exporter annotations (Excel)')
        btn_export.clicked.connect(self.export_annotations)
        self.gpu_checkbox = QCheckBox('Essayer GPU (si OpenCV CUDA disponible)')
        ann.addWidget(self.label_input)
        ann.addWidget(btn_add_label)
        ann.addWidget(self.gpu_checkbox)
        ann.addWidget(btn_export)
        main.addLayout(ann)

        self.setLayout(main)

    @Slot()
    def load_videos(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Sélectionner vidéos (1..6)', str(Path.cwd()), 'Videos (*.avi *.mp4 *.mov)')
        if not files:
            return
        self.paths = files[:6]
        for i in range(6):
            if i < len(self.paths):
                self.video_labels[i].setText(Path(self.paths[i]).name)
            else:
                self.video_labels[i].setText(f'Cam {i+1}\n(aucune vidéo)')

        if self.worker:
            self.worker.stop()
            self.worker.wait(200)
        self.worker = VideoWorker(self.paths, cache_size=120)
        self.worker.frames_ready.connect(self.on_frames)
        self.worker.start()
        self.btn_play.setEnabled(True)
        self.slider.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

    @Slot()
    def step_prev(self):
        if self.worker:
            self.worker.step(-1)

    @Slot()
    def step_next(self):
        if self.worker:
            self.worker.step(1)

    @Slot(str)
    def on_speed_changed(self, txt: str):
        try:
            rate = float(txt.replace('x', ''))
        except Exception:
            rate = 1.0
        if self.worker:
            self.worker.set_playback_rate(rate)

    @Slot(list, int)
    def on_frames(self, qimages, frame_idx):
        for i, img in enumerate(qimages):
            if img is None:
                continue
            pix = QPixmap.fromImage(img).scaled(self.video_labels[i].size(), Qt.KeepAspectRatio)
            self.video_labels[i].setPixmap(pix)
        try:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_idx)
        finally:
            self.slider.blockSignals(False)

    @Slot()
    def toggle_play(self):
        if not self.worker:
            return
        if self.worker.playing:
            self.worker.set_playing(False)
            self.btn_play.setText('Play')
        else:
            if self.worker.frame_counts:
                vals = [c for c in self.worker.frame_counts if c > 0]
                if vals:
                    self.slider.setMaximum(min(vals))
            self.worker.set_playing(True)
            self.btn_play.setText('Pause')

    @Slot(int)
    def on_seek(self, v):
        if self.worker:
            self.worker.seek(v)

    @Slot()
    def add_label(self):
        txt = self.label_input.text().strip()
        if not txt:
            QMessageBox.information(self, 'Info', 'Saisir un texte pour le label avant d\'ajouter.')
            return
        current = self.slider.value()
        fps = min(self.worker.fps) if self.worker and self.worker.fps else 30.0
        t = current / fps
        for i in range(len(self.paths)):
            ann = Annotation(cam=i+1, frame_idx=current, time_sec=t, label=txt)
            self.annotations.append(ann)
        QMessageBox.information(self, 'Annotation', f'Label "{txt}" ajouté au frame {current} ({t:.3f}s) pour {len(self.paths)} cam(s).')

    @Slot()
    def export_annotations(self):
        if not self.annotations:
            QMessageBox.information(self, 'Aucune annotation', 'Il n\'y a pas d\'annotations à exporter.')
            return
        if pd is None:
            QMessageBox.critical(self, 'Dépendance manquante', 'pandas est requis pour exporter. Installer pandas et openpyxl.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Enregistrer annotations', str(Path.cwd() / 'annotations.xlsx'), 'Excel (*.xlsx)')
        if not path:
            return
        rows = []
        for a in self.annotations:
            rows.append({'camera': a.cam, 'frame': a.frame_idx, 'time_s': a.time_sec, 'label': a.label})
        df = pd.DataFrame(rows)
        try:
            df.to_excel(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, 'Erreur', f'Impossible d\'écrire le fichier: {e}')
            return
        QMessageBox.information(self, 'Exporté', f'Annotations exportées vers: {path}')

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait(200)
        event.accept()


def main():
    app = QApplication(sys.argv)
    if cv2 is None:
        QMessageBox.critical(None, 'Dépendance manquante', 'OpenCV (cv2) n\'est pas installé. Voir requirements.txt')
        return
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
