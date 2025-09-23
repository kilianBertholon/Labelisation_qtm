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
import shutil
import subprocess
import tempfile
import os
import threading

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLineEdit, QMessageBox, QCheckBox, QComboBox
)
from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QListWidget, QTextEdit, QVBoxLayout
try:
    # prefer absolute import so running `python src/main.py` works
    from settings import catalog as settings_catalog
except Exception:
    # fallback to package-relative import when module is used as package
    from .settings import catalog as settings_catalog

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pandas as pd
except Exception:
    pd = None


def is_opencv_cuda_available():
    """Return True if OpenCV was built with CUDA support and a device is available."""
    if cv2 is None:
        return False
    try:
        # prefer runtime query
        if hasattr(cv2, 'cuda'):
            try:
                return int(cv2.cuda.getCudaEnabledDeviceCount() or 0) > 0
            except Exception:
                # continue to build-info fallback
                pass
        info = cv2.getBuildInformation()
        return 'CUDA' in info
    except Exception:
        return False


@dataclass
class Annotation:
    cam: int
    frame_idx: int
    time_sec: float
    label: str


class VideoWorker(QThread):
    frames_ready = Signal(list, int)  # list of QImage or None, current_frame_idx
    transcode_progress = Signal(int, int)  # cam_idx, percent

    def __init__(self, paths, cache_size=120, use_hwaccel=False, parent=None):
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
        # failure tracking to avoid crashing on bad decodes
        self.cap_failed = [False] * len(paths)
        self.cap_fail_count = [0] * len(paths)
        # ffmpeg fallback state
        self.ffmpeg_available = bool(shutil.which('ffmpeg'))
        self.transcoded_paths = [None] * len(paths)
        # whether to request hwaccel flags for ffmpeg transcode (if available)
        self.use_hwaccel = bool(use_hwaccel)
        # locks and state for safe concurrent prefetching/reading
        self.cap_locks = [threading.Lock() for _ in paths]
        self.last_pos = [-1] * len(paths)
        self.prefetch_thread = None
        self.prefetch_stop = threading.Event()

    def open_all(self):
        for i, p in enumerate(self.paths):
            cap = None
            try:
                cap = cv2.VideoCapture(str(p))
            except Exception:
                cap = None
            if cap is None or not cap.isOpened():
                # try ffmpeg transcode fallback if available
                if self.ffmpeg_available:
                    newp = self._try_transcode_once(i)
                    if newp:
                        try:
                            cap = cv2.VideoCapture(str(newp))
                        except Exception:
                            cap = None
                if cap is None or not cap.isOpened():
                    self.caps[i] = None
                    self.fps.append(30.0)
                    self.frame_counts.append(0)
                    continue
            else:
                # cap opened; but avoid reading frames in-process if ffmpeg detects decode errors
                if self.ffmpeg_available:
                    try:
                        ok = self._ffmpeg_probe(p)
                    except Exception:
                        ok = True
                    if not ok:
                        # transcode and reopen
                        newp = self._try_transcode_once(i)
                        if newp:
                            try:
                                # release old cap and open new one
                                try:
                                    cap.release()
                                except Exception:
                                    pass
                                cap = cv2.VideoCapture(str(newp))
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
        # start initial prefetch in background to fill caches and avoid heavy seeks
        try:
            if any(cap is not None for cap in self.caps):
                self.prefetch_stop.clear()
                self.prefetch_thread = threading.Thread(target=self._prefetch_initial, daemon=True)
                self.prefetch_thread.start()
        except Exception:
            pass

    def close_all(self):
        # stop prefetch thread
        try:
            self.prefetch_stop.set()
            if self.prefetch_thread and self.prefetch_thread.is_alive():
                self.prefetch_thread.join(timeout=1.0)
        except Exception:
            pass
        for cap in self.caps:
            if cap:
                cap.release()
        self.caps = []
        # attempt to remove any transcoded temporary files we created
        try:
            for p in list(self.transcoded_paths or []):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass
        # reset tracking
        try:
            self.transcoded_paths = [None] * len(self.paths)
        except Exception:
            self.transcoded_paths = []

    def cleanup_transcoded(self):
        """Explicit cleanup method to remove any transcoded temp files.
        Can be called from the GUI thread on shutdown to ensure files are removed.
        """
        try:
            for p in list(self.transcoded_paths or []):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.transcoded_paths = [None] * len(self.paths)
        except Exception:
            self.transcoded_paths = []

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
                try:
                    frame = self._get_frame(i, self.current_frame)
                except Exception:
                    # unexpected C++/cv2 exception - mark camera failed
                    try:
                        self.cap_fail_count[i] += 1
                        if self.cap_fail_count[i] > 5:
                            self.cap_failed[i] = True
                    except Exception:
                        pass
                    imgs.append(None)
                    continue
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
        # fast path: cached
        with self.cap_locks[cam_idx]:
            if frame_idx in cache:
                cache.move_to_end(frame_idx)
                return cache[frame_idx]
            cap = self.caps[cam_idx]
            if cap is None:
                return None
            try:
                # prefer sequential read when possible to avoid heavy cap.set calls
                if self.last_pos[cam_idx] == frame_idx - 1:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.last_pos[cam_idx] = frame_idx
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.last_pos[cam_idx] = frame_idx
            except Exception:
                # treat as a failure; increment counter and possibly mark failed
                self.cap_fail_count[cam_idx] += 1
                if self.cap_fail_count[cam_idx] > 5:
                    self.cap_failed[cam_idx] = True
                # try to transcode once if available and not already tried
                if self.ffmpeg_available and not self.transcoded_paths[cam_idx]:
                    newp = self._try_transcode_once(cam_idx)
                    if newp:
                        try:
                            try:
                                if self.caps[cam_idx]:
                                    self.caps[cam_idx].release()
                            except Exception:
                                pass
                            self.caps[cam_idx] = cv2.VideoCapture(str(newp))
                            self.cap_fail_count[cam_idx] = 0
                            cap = self.caps[cam_idx]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                self.last_pos[cam_idx] = frame_idx
                        except Exception:
                            ok = False
                            frame = None
                else:
                    ok = False
                    frame = None
            if not ok or frame is None:
                self.cap_fail_count[cam_idx] += 1
                if self.cap_fail_count[cam_idx] > 5:
                    self.cap_failed[cam_idx] = True
                return None
            # store in cache
            cache[frame_idx] = frame
            if len(cache) > self.cache_size:
                cache.popitem(last=False)
            return frame

    def _try_transcode_once(self, cam_idx):
        """Attempt to transcode the original path for cam_idx using ffmpeg.
        Returns the path to the transcoded file on success, or None.
        This is intentionally conservative: only one transcode attempt per camera.
        """
        try:
            src = str(self.paths[cam_idx])
        except Exception:
            return None
        if not os.path.exists(src):
            return None
        if self.transcoded_paths[cam_idx]:
            return self.transcoded_paths[cam_idx]
        # create target temp path
        base = Path(src).stem
        out_dir = tempfile.gettempdir()
        out_path = os.path.join(out_dir, f"{base}_transcoded.mp4")
        if self.use_hwaccel:
            # request hw-accelerated decode (cuda) and encode (nvenc) if available
            cmd = [
                'ffmpeg', '-y', '-v', 'error',
                '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                '-i', src,
                '-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '23',
                '-c:a', 'aac', '-movflags', '+faststart', out_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-v', 'error', '-i', src,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-c:a', 'aac', '-movflags', '+faststart', out_path
            ]
        # determine duration via ffprobe if possible
        duration = self._ffprobe_duration(src)
        try:
            if self.use_hwaccel:
                print(f"ffmpeg: transcoding with hwaccel {src} -> {out_path}")
            else:
                print(f"ffmpeg: transcoding {src} -> {out_path}")
            # run ffmpeg with -progress pipe:1 to get periodic progress updates
            popen = subprocess.Popen(cmd + ['-progress', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            percent = 0
            if popen.stdout:
                for line in popen.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    # parse key=value lines
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip()
                        if k in ('out_time_ms', 'out_time_us'):
                            try:
                                out_ms = int(v)
                                out_s = out_ms / 1000.0
                                if duration and duration > 0:
                                    newp = int(min(100, (out_s / duration) * 100))
                                    if newp != percent:
                                        percent = newp
                                        try:
                                            self.transcode_progress.emit(cam_idx, percent)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        elif k == 'out_time':
                            # format HH:MM:SS.micro
                            try:
                                parts = v.split(':')
                                h = float(parts[0]); m = float(parts[1]); s = float(parts[2])
                                out_s = h*3600 + m*60 + s
                                if duration and duration > 0:
                                    newp = int(min(100, (out_s / duration) * 100))
                                    if newp != percent:
                                        percent = newp
                                        try:
                                            self.transcode_progress.emit(cam_idx, percent)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        elif k == 'progress' and v == 'end':
                            percent = 100
                            try:
                                self.transcode_progress.emit(cam_idx, percent)
                            except Exception:
                                pass
            rc = popen.wait()
            if rc == 0 and os.path.exists(out_path):
                self.transcoded_paths[cam_idx] = out_path
                return out_path
            else:
                stderr = ''
                try:
                    stderr = popen.stderr.read()
                except Exception:
                    pass
                print(f"ffmpeg failed for {src}: rc={rc} stderr={stderr}")
        except Exception as e:
            print(f"ffmpeg exception: {e}")
        # ensure UI gets 100 on failure end to avoid stuck bars
        try:
            self.transcode_progress.emit(cam_idx, 100)
        except Exception:
            pass
        return None

    def _ffprobe_duration(self, path):
        try:
            res = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if res.returncode == 0:
                out = res.stdout.strip()
                try:
                    return float(out)
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def _prefetch_initial(self):
        """Background prefetch: fill caches with the first N frames (round-robin) to
        allow seeking back/forward immediately without heavy cap.set operations.
        """
        try:
            # compute target per camera
            targets = [min(self.cache_size, c) if c > 0 else 0 for c in self.frame_counts]
            max_t = max(targets) if targets else 0
            for f in range(max_t):
                if self.prefetch_stop.is_set():
                    break
                for i, cap in enumerate(self.caps):
                    if self.prefetch_stop.is_set():
                        break
                    if f >= targets[i]:
                        continue
                    if self.cap_failed[i] or cap is None:
                        continue
                    with self.cap_locks[i]:
                        cache = self.caches[i]
                        if f in cache:
                            continue
                        try:
                            # if last_pos indicates we can read sequentially, do so
                            if self.last_pos[i] == f - 1:
                                ok, frame = cap.read()
                                if ok and frame is not None:
                                    self.last_pos[i] = f
                                    cache[f] = frame
                            else:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                                ok, frame = cap.read()
                                if ok and frame is not None:
                                    self.last_pos[i] = f
                                    cache[f] = frame
                        except Exception:
                            # mark failure and stop prefetch for this camera
                            try:
                                self.cap_fail_count[i] += 1
                                if self.cap_fail_count[i] > 5:
                                    self.cap_failed[i] = True
                            except Exception:
                                pass
                        finally:
                            if len(cache) > self.cache_size:
                                # trim oldest
                                try:
                                    cache.popitem(last=False)
                                except Exception:
                                    pass
        except Exception:
            pass

    def _ffmpeg_probe(self, path):
        """Run a lightweight ffmpeg read to check for obvious decode errors.
        Returns True if probe looks ok, False if ffmpeg reports decode errors.
        """
        try:
            cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-']
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode != 0:
                # ffmpeg reported errors; consider this a bad file
                return False
            return True
        except Exception:
            return True

    @Slot(int)
    def step_request(self, n: int):
        """Slot callable from main thread via signal/queued connection to change
        current frame and emit the corresponding images from the worker thread."""
        self.current_frame = max(0, self.current_frame + int(n))
        imgs = []
        for i, cap in enumerate(self.caps):
            if self.cap_failed[i]:
                imgs.append(None)
                continue
            try:
                frame = self._get_frame(i, self.current_frame)
            except Exception:
                # mark failed and continue
                try:
                    self.cap_fail_count[i] += 1
                    if self.cap_fail_count[i] > 5:
                        self.cap_failed[i] = True
                except Exception:
                    pass
                imgs.append(None)
                continue
            if frame is None:
                imgs.append(None)
            else:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                imgs.append(image.copy())
        self.frames_ready.emit(imgs, self.current_frame)

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
    step_signal = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('multicam_annotator')
        self.video_labels = []
        self.paths = []
        self.worker = None
        self._progress_per_cam = {}
        self.annotations = []
        # signal to request a step from worker thread
        self.init_ui()

    def init_ui(self):
        main = QVBoxLayout()
        # settings toolbar
        toolbar = QHBoxLayout()
        btn_settings = QPushButton('Paramètres')
        btn_settings.clicked.connect(self.open_settings)
        toolbar.addWidget(btn_settings)
        # place toolbar at top
        main.addLayout(toolbar)

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
        # indicate GPU availability at startup
        available = is_opencv_cuda_available()
        print(f"OpenCV CUDA available: {available}")
        self.gpu_checkbox.setChecked(True)
        # connect checkbox to print changes
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_toggled)
        ann.addWidget(self.label_input)
        ann.addWidget(btn_add_label)
        ann.addWidget(self.gpu_checkbox)
        ann.addWidget(btn_export)
        main.addLayout(ann)

    # Global transcode progress bar (hidden until used)
        self.global_progress = QProgressBar()
        self.global_progress.setRange(0, 100)
        self.global_progress.setValue(0)
        self.global_progress.setVisible(False)
        main.addWidget(self.global_progress)

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
        use_hw = bool(self.gpu_checkbox.isChecked())
        print(f"GPU checkbox requested: {use_hw}")
        self.worker = VideoWorker(self.paths, cache_size=120, use_hwaccel=use_hw)
        self.worker.frames_ready.connect(self.on_frames)
        try:
            self.worker.transcode_progress.connect(self.on_transcode_progress)
        except Exception:
            pass
        # reset progress tracking
        self._progress_per_cam = {i: 0 for i in range(len(self.paths))}
        # connect the step signal to the worker slot (queued connection)
        try:
            self.step_signal.connect(self.worker.step_request)
        except Exception:
            # fallback: direct method assignment (shouldn't happen)
            pass
        self.worker.start()
        self.btn_play.setEnabled(True)
        self.slider.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

    @Slot()
    def step_prev(self):
        if self.worker:
            # request a single-step backwards
            self.step_signal.emit(-1)

    @Slot()
    def step_next(self):
        if self.worker:
            self.step_signal.emit(1)

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
            # set current frame in worker thread by seeking and emitting a frame
            self.worker.seek(v)
            # emit current frame images immediately
            self.step_signal.emit(0)

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
            try:
                self.worker.cleanup_transcoded()
            except Exception:
                pass
        try:
            self.global_progress.setVisible(False)
        except Exception:
            pass
        event.accept()

    def open_settings(self):
        """Open a simple settings dialog showing categories and defaults."""
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle('Paramètres')
            layout = QVBoxLayout()
            listw = QListWidget()
            for c in settings_catalog.list_categories():
                listw.addItem(c)
            txt = QTextEdit()
            txt.setReadOnly(True)
            def on_sel():
                it = listw.currentItem()
                if not it:
                    txt.setPlainText('')
                    return
                cat = it.text()
                desc = settings_catalog.describe(cat) or []
                lines = [f"{k}: {d}" for k, d in desc]
                vals = settings_catalog.get_defaults(cat) or {}
                lines.append('\nDefaults:')
                for kk, vv in vals.items():
                    lines.append(f"  {kk} = {vv}")
                txt.setPlainText('\n'.join(lines))
            listw.currentItemChanged.connect(lambda _i, _j: on_sel())
            layout.addWidget(listw)
            layout.addWidget(txt)
            dlg.setLayout(layout)
            dlg.resize(600, 400)
            dlg.exec()
        except Exception:
            pass

    @Slot(int)
    def on_gpu_toggled(self, state: int):
        enabled = bool(state)
        if enabled:
            print('GPU requested by user (will be used if OpenCV CUDA is available).')
        else:
            print('GPU not requested; using CPU decoding.')

    @Slot(int, int)
    def on_transcode_progress(self, cam_idx: int, percent: int):
        # update per-cam progress and show averaged global progress
        try:
            n = max(1, len(self.paths))
            # ensure dict keys for all cams
            if not self._progress_per_cam or len(self._progress_per_cam) != n:
                self._progress_per_cam = {i: 0 for i in range(n)}
            self._progress_per_cam[cam_idx] = int(percent)
            total = sum(self._progress_per_cam.get(i, 0) for i in range(n))
            avg = int(total / n)
            # show progress bar
            self.global_progress.setVisible(True)
            self.global_progress.setValue(avg)
            # hide when complete
            if avg >= 100:
                # reset after showing 100%
                self.global_progress.setVisible(False)
                self._progress_per_cam = {i: 0 for i in range(n)}
        except Exception:
            pass


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
