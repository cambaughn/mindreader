"""The Mindreader window: camera preview, Decode / Record / Train tabs, hold Space to capture.

Everything heavy is the engine's: CameraEngine (capture + face tracking), Recognizer (crops +
lip-reading + optional LLM rescoring), the personal dataset and the fine-tuner. This file is glue
and widgets. Run from the macOS Terminal app so the camera permission attaches to Terminal.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import traceback

from .. import engine  # noqa: F401  (before silentdict imports)
from .. import __version__
from ..recorder import (CONDITIONS, DatasetStore, RawVideoWriter, RecordSession, counts,
                        prompts_for_condition)
from . import theme
from .theme import OP_COLORS, P, ActivityMeter, repolish

import numpy as np
from PySide6.QtCore import QObject, QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                               QListWidget, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget, QLineEdit)

PREVIEW_W = 480


class Signals(QObject):
    status = Signal(str)
    models = Signal(bool, str)          # loaded?, description
    decoded = Signal(object)            # pipeline.Result
    recorded = Signal(dict)             # RecordSession snapshot
    log = Signal(str)
    progress = Signal(str, float)
    job_done = Signal(str, bool)        # name, ok


def bgr_to_pixmap(bgr: np.ndarray, obs, width: int = PREVIEW_W) -> QPixmap:
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    if obs is not None:
        p = QPainter(pm)
        p.setPen(QPen(QColor(P["accent"]), 2))
        x0, y0, x1, y1 = obs.bbox
        p.drawRect(x0, y0, x1 - x0, y1 - y0)
        mx, my = obs.points[3]
        p.setPen(QPen(QColor(P["rec"]), 2))
        p.drawEllipse(int(mx) - 4, int(my) - 4, 8, 8)
        p.end()
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


def align_html(ops: list[dict]) -> str:
    parts = []
    for op in ops:
        word = op["hyp"] or op["ref"]
        style = f"color:{OP_COLORS[op['op']]};"
        if op["op"] == "del":
            style += "text-decoration:line-through;"
        if op["op"] == "ins":
            style += "font-style:italic;"
        parts.append(f'<span style="{style}">{word}</span>')
    return " ".join(parts)


def ensure_camera_access(timeout: float = 120.0) -> bool:
    """Ask macOS for the camera on the main thread; returns True when allowed.

    The engine's own helper passes a lambda that returns a value to the Objective-C completion
    handler, which PyObjC rejects ("expecting void return value") and the process aborts. This copy
    uses a callback that returns None."""
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    from Foundation import NSDate, NSRunLoop
    from silentdict.permissions import camera_status

    status = camera_status()
    if status == 3:
        return True
    if status in (1, 2):
        return False
    answer: dict = {}

    def done(granted):
        answer["granted"] = bool(granted)

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, done)
    end = time.time() + timeout
    while "granted" not in answer and time.time() < end:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return answer.get("granted", False)


class Main(QMainWindow):
    def __init__(self, camera: str = "0"):
        super().__init__()
        self.setWindowTitle("Mindreader")
        self._capture_state = None
        self.sig = Signals()
        self.camera = None
        self.camera_name = ""
        self.recognizer = None
        self.rescorer = None
        self.cleaner = None
        self.model_lock = threading.Lock()
        self.segments: queue.Queue = queue.Queue()
        self.session: RecordSession | None = None
        self.space_down = False
        self.busy_job = False
        self.history: list[str] = []
        self._build()
        self.sig.status.connect(self.status_bar.setText)
        self.sig.models.connect(self._models_ready)
        self.sig.decoded.connect(self._show_decoded)
        self.sig.recorded.connect(self._show_recorded)
        self.sig.log.connect(self._append_log)
        self.sig.progress.connect(self._progress)
        self.sig.job_done.connect(self._job_done)
        self._refresh_checks()
        self._start_camera(camera)
        self._load_models_async()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._tick)
        self.preview_timer.start(66)
        threading.Thread(target=self._worker, daemon=True).start()
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 10)
        outer.setSpacing(10)

        head = QHBoxLayout()
        outer.addLayout(head)
        title = QLabel("Mindreader")
        title.setStyleSheet("font-size:15px; font-weight:600; letter-spacing:0.5px;")
        head.addWidget(title)
        ver = QLabel(__version__)
        ver.setProperty("role", "muted")
        head.addWidget(ver)
        head.addSpacing(12)
        self.checks_row = QHBoxLayout()
        self.checks_row.setSpacing(6)
        head.addLayout(self.checks_row, 1)

        body = QHBoxLayout()
        body.setSpacing(18)
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(10)
        body.addLayout(left)
        self.preview = QLabel("camera…")
        self.preview.setObjectName("preview")
        self.preview.setFixedSize(PREVIEW_W, int(PREVIEW_W * 9 / 16))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.preview)
        meter_row = QHBoxLayout()
        self.bars = ActivityMeter()
        self.bars.setFixedWidth(7 * 14 + 6 * 5)
        meter_row.addWidget(self.bars)
        self.cam_info = QLabel("")
        self.cam_info.setProperty("role", "muted")
        meter_row.addWidget(self.cam_info, 1)
        left.addLayout(meter_row)
        self.space_label = QLabel("Hold  SPACE  to capture")
        self.space_label.setObjectName("capture")
        self.space_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.space_label)
        left.addStretch(1)

        self.tabs = QTabWidget()
        body.addWidget(self.tabs, 1)
        self.tabs.addTab(self._decode_tab(), "Decode")
        self.tabs.addTab(self._record_tab(), "Record")
        self.tabs.addTab(self._train_tab(), "Train / Eval")

        bottom = QHBoxLayout()
        outer.addLayout(bottom)
        self.status_bar = QLabel("starting…")
        self.status_bar.setProperty("role", "muted")
        bottom.addWidget(self.status_bar, 1)
        b = QPushButton("Open data folder")
        b.clicked.connect(lambda: subprocess.Popen(["open", str(engine.DATA)]))
        bottom.addWidget(b)
        b = QPushButton("Dictate anywhere…")
        b.setToolTip("Launches the engine's hotkey app with Mindreader's data (hold fn in any app).")
        b.clicked.connect(self._dictate)
        bottom.addWidget(b)
        self.resize(1100, 640)

    def _decode_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.personal_box = QCheckBox("Use personal model")
        self.personal_box.setChecked(True)
        self.personal_box.toggled.connect(lambda _: self._load_models_async())
        row.addWidget(self.personal_box)
        self.llm_box = QCheckBox("LLM re-rank (choose among guesses)")
        self.llm_box.toggled.connect(self._toggle_llm)
        row.addWidget(self.llm_box)
        self.rewrite_box = QCheckBox("LLM rewrite (clean-up)")
        self.rewrite_box.setToolTip("Rewrite the guesses into a clean sentence. Needs the re-rank LLM loaded. "
                                    "Measured neutral-to-slightly-positive on words; adds caps and punctuation.")
        self.rewrite_box.toggled.connect(lambda on: on and not self.llm_box.isChecked() and self.llm_box.setChecked(True))
        row.addWidget(self.rewrite_box)
        row.addStretch(1)
        v.addLayout(row)
        card = QFrame()
        card.setProperty("role", "card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(18, 16, 18, 14)
        cv.setSpacing(8)
        self.decoded_text = QLabel("Hold Space, mouth a sentence, release.")
        self.decoded_text.setProperty("role", "hero")
        self.decoded_text.setWordWrap(True)
        self.decoded_text.setMinimumHeight(84)
        cv.addWidget(self.decoded_text)
        self.raw_label = QLabel("")
        self.raw_label.setProperty("role", "muted")
        self.raw_label.setWordWrap(True)
        cv.addWidget(self.raw_label)
        self.timing_label = QLabel("")
        self.timing_label.setProperty("role", "mono")
        cv.addWidget(self.timing_label)
        v.addWidget(card)
        v.addWidget(self._eyebrow("CANDIDATES"))
        self.nbest = QListWidget()
        self.nbest.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        v.addWidget(self.nbest, 1)
        v.addWidget(self._eyebrow("RECENT"))
        self.recent = QListWidget()
        self.recent.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.recent.setMaximumHeight(120)
        v.addWidget(self.recent)
        b = QPushButton("Copy last result")
        b.clicked.connect(lambda: QApplication.clipboard().setText(self.history[-1] if self.history else ""))
        v.addWidget(b)
        return w

    def _record_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Condition:"))
        self.condition = QComboBox()
        self.condition.addItems(list(CONDITIONS))
        row.addWidget(self.condition)
        row.addWidget(QLabel("Sentences:"))
        self.count = QSpinBox()
        self.count.setRange(1, 200)
        self.count.setValue(20)
        row.addWidget(self.count)
        self.raw_box = QCheckBox("Keep raw video")
        self.raw_box.setChecked(True)
        row.addWidget(self.raw_box)
        self.start_btn = QPushButton("Start session")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start_session)
        row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_session)
        self.stop_btn.setEnabled(False)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.progress_label = QLabel("")
        self.progress_label.setProperty("role", "muted")
        v.addWidget(self.progress_label)
        card = QFrame()
        card.setProperty("role", "card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(18, 14, 18, 14)
        cv.setSpacing(8)
        self.now_label = QLabel("")
        self.now_label.setProperty("role", "eyebrow")
        cv.addWidget(self.now_label)
        self.prompt_label = QLabel("Pick a condition and press Start session.")
        self.prompt_label.setProperty("role", "prompt")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setMinimumHeight(70)
        cv.addWidget(self.prompt_label)
        self.next_label = QLabel("")
        self.next_label.setProperty("role", "muted")
        cv.addWidget(self.next_label)
        v.addWidget(card)
        self.rec_status = QLabel("")
        v.addWidget(self.rec_status)
        box = QFrame()
        box.setProperty("role", "card")
        bv = QVBoxLayout(box)
        bv.setContentsMargins(18, 14, 18, 14)
        bv.setSpacing(8)
        self.readback_title = self._eyebrow("READ-BACK OF THE LAST TAKE")
        bv.addWidget(self.readback_title)
        self.readback = QLabel("")
        self.readback.setProperty("role", "readback")
        self.readback.setWordWrap(True)
        bv.addWidget(self.readback)
        self.rec_acc = QLabel("")
        self.rec_acc.setProperty("role", "muted")
        bv.addWidget(self.rec_acc)
        v.addWidget(box)
        row = QHBoxLayout()
        self.redo_btn = QPushButton("Redo last")
        self.redo_btn.clicked.connect(lambda: self._session_call("redo"))
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(lambda: self._session_call("skip"))
        row.addWidget(self.redo_btn)
        row.addWidget(self.skip_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.counts_label = QLabel("")
        self.counts_label.setProperty("role", "muted")
        v.addWidget(self.counts_label)
        v.addStretch(1)
        self._refresh_counts()
        return w

    def _train_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Takes:"))
        self.train_cond = QComboBox()
        self.train_cond.addItems(["all conditions", *CONDITIONS])
        row.addWidget(self.train_cond)
        row.addWidget(QLabel("Strategy:"))
        self.strategy = QComboBox()
        self.strategy.addItems(["light", "both", "full"])
        row.addWidget(self.strategy)
        self.train_btn = QPushButton("Train personal model")
        self.train_btn.setObjectName("primary")
        self.train_btn.clicked.connect(self._train)
        row.addWidget(self.train_btn)
        self.eval_btn = QPushButton("Evaluate")
        self.eval_btn.clicked.connect(self._eval)
        row.addWidget(self.eval_btn)
        self.eval_llm = QCheckBox("with LLM re-rank")
        row.addWidget(self.eval_llm)
        self.eval_rewrite = QCheckBox("+ rewrite")
        row.addWidget(self.eval_rewrite)
        row.addStretch(1)
        v.addLayout(row)
        self.job_progress = QProgressBar()
        self.job_progress.setRange(0, 1000)
        v.addWidget(self.job_progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        v.addWidget(self.log, 1)
        hint = QLabel("Train needs 20+ takes. Every 5th take is held out; the personal model is kept only if it "
                      "beats the base model on those. Results also land in docs/results/.")
        hint.setWordWrap(True)
        hint.setProperty("role", "muted")
        v.addWidget(hint)
        return w

    @staticmethod
    def _eyebrow(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("role", "eyebrow")
        return lab

    # ------------------------------------------------------------------ checks / camera / models
    def _refresh_checks(self):
        while self.checks_row.count():
            item = self.checks_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for name, ok, detail in engine.checks():
            lab = QLabel(("● " if ok else "○ ") + name)
            lab.setToolTip(detail)
            lab.setProperty("role", "chip")
            lab.setProperty("ok", "true" if ok else "false")
            self.checks_row.addWidget(lab)
        self.checks_row.addStretch(1)

    def _start_camera(self, camera: str):
        from silentdict.capture import CameraEngine
        index: int | str = int(camera) if camera.isdigit() else camera
        if isinstance(index, int) and not ensure_camera_access():
            self.sig.status.emit("Camera access denied: System Settings > Privacy & Security > Camera (Terminal).")
            return
        try:
            from silentdict.devices import cameras
            names = cameras()
            self.camera_name = names[index] if isinstance(index, int) and index < len(names) else str(camera)
        except Exception:
            self.camera_name = str(camera)
        self.camera = CameraEngine(on_segment=self.segments.put, camera_index=index, fps=30)
        self.camera.full_rate = True
        self.camera.start()
        self.camera.ready.wait(5)
        if self.camera.error:
            self.sig.status.emit(self.camera.error)

    def _load_models_async(self):
        self.sig.status.emit("Loading lip-reading model…")

        def go():
            try:
                from silentdict.pipeline import Recognizer, resolve_model
                with self.model_lock:
                    self.recognizer = None
                    rec = Recognizer(model="auto", personal=self.personal_box.isChecked())
                    rec.rescorer = self.rescorer
                    rec.warm_up()
                    self.recognizer = rec
                desc = f"{resolve_model('auto')} ({'personal' if getattr(rec.reader, 'personalized', False) else 'base'})"
                self.sig.models.emit(True, desc)
            except Exception as exc:
                self.sig.models.emit(False, f"{exc}\n{traceback.format_exc()}")
        threading.Thread(target=go, daemon=True).start()

    def _models_ready(self, ok: bool, desc: str):
        if ok:
            self.sig.status.emit(f"Ready: {desc}. Hold Space to capture.")
        else:
            self.sig.status.emit("Model failed to load (see Train / Eval log).")
            self._append_log(desc)
        self._refresh_checks()

    def _toggle_llm(self, on: bool):
        if not on:
            self.rescorer = self.cleaner = None
            self.rewrite_box.setChecked(False)
            if self.recognizer:
                self.recognizer.rescorer = None
            return
        from silentdict import paths
        if not paths.LLM_DIR.exists():
            self.sig.status.emit("No local LLM installed (bootstrap with --with-llm).")
            self.llm_box.setChecked(False)
            return
        self.sig.status.emit("Loading language model…")

        def go():
            try:
                from silentdict.rewrite import Rescorer
                from ..cleanup import Cleaner
                r = Rescorer()
                r.logprob("HELLO THERE")
                self.rescorer, self.cleaner = r, Cleaner(rescorer=r)
                if self.recognizer:
                    self.recognizer.rescorer = r
                self.sig.status.emit("Language model ready.")
            except Exception as exc:
                self.sig.status.emit(f"LLM failed: {exc}")
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------------------ preview + space
    def _tick(self):
        if not self.camera:
            return
        f = self.camera.latest
        if f is not None:
            self.preview.setPixmap(bgr_to_pixmap(f.bgr, f.obs))
            from silentdict.activity import bar_levels
            levels = bar_levels(self.camera.recent_obs(7))
            self.bars.set_levels(levels, recording=self.camera.recording)
            self.cam_info.setText(f"{self.camera_name}  ·  {self.camera.measured_fps:.0f} fps  ·  "
                                  f"{'face' if f.obs else 'no face'}")
        rec = self.camera.recording
        state = (rec, self.recognizer is not None)
        if state != self._capture_state:
            self._capture_state = state
            self.space_label.setProperty("recording", "true" if rec else "false")
            self.space_label.setProperty("armed", "true" if state[1] else "false")
            self.space_label.setText("●  RECORDING   release Space to stop" if rec
                                     else ("Hold  SPACE  to capture" if state[1] else "Loading model…"))
            repolish(self.space_label)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and ev.key() == Qt.Key.Key_Space \
                and not ev.isAutoRepeat() and self.isActiveWindow():
            if isinstance(QApplication.focusWidget(), QLineEdit):
                return False
            if ev.type() == QEvent.Type.KeyPress:
                self._space(True)
            else:
                self._space(False)
            return True
        return super().eventFilter(obj, ev)

    def _space(self, down: bool):
        if not self.camera or self.camera.error:
            return
        if down and not self.space_down:
            if self.recognizer is None:
                self.sig.status.emit("Model still loading…")
                return
            if self.session and not self.session.accepting():
                return
            self.space_down = True
            self.camera.start_segment()
        elif not down and self.space_down:
            self.space_down = False
            self.camera.stop_segment()

    # ------------------------------------------------------------------ worker
    def _worker(self):
        while True:
            frames = self.segments.get()
            try:
                with self.model_lock:
                    rec = self.recognizer
                    if rec is None:
                        continue
                    if self.session is not None and self.tabs.currentIndex() == 1:
                        self.sig.status.emit("Saving take…")
                        self.sig.recorded.emit(self.session.handle(frames))
                    else:
                        self.sig.status.emit("Decoding…")
                        t = time.perf_counter()
                        res = rec.recognize_frames(frames, rescore=self.llm_box.isChecked())
                        res.cleaned = None
                        if self.rewrite_box.isChecked() and self.cleaner and res.hyps:
                            from silentdict.text import sentence_case
                            cands = [h.text for h in res.hyps]
                            chosen = next((c for c in cands if sentence_case(c) == res.text), cands[0])
                            t1 = time.perf_counter()
                            res.cleaned = self.cleaner.rewrite([chosen] + [c for c in cands if c != chosen])
                            res.timings["rewrite"] = time.perf_counter() - t1
                        res.timings["total"] = time.perf_counter() - t
                        res.timings["frames"] = len(frames)
                        self.sig.decoded.emit(res)
            except Exception as exc:
                self.sig.status.emit(f"Error: {exc}")
                self.sig.log.emit(traceback.format_exc())

    def _show_decoded(self, res):
        if res.error:
            self.decoded_text.setText(f"({res.error})")
            self.sig.status.emit(res.error)
            return
        cleaned = getattr(res, "cleaned", None)
        shown = cleaned or res.text
        self.decoded_text.setText(shown)
        self.raw_label.setText(f"lip reader: {res.text}" if cleaned else "")
        self.history.append(shown)
        self.recent.insertItem(0, shown)
        self.nbest.clear()
        for h in res.hyps:
            self.nbest.addItem(f"{h.score:8.2f}   {h.text.lower()}")
        t = res.timings
        self.timing_label.setText("  ".join(f"{k} {v:.2f}s" if isinstance(v, float) else f"{k} {v}"
                                            for k, v in t.items()))
        self.sig.status.emit("Ready. Hold Space to capture.")

    # ------------------------------------------------------------------ record
    def _start_session(self):
        cond = self.condition.currentText()
        prompts = prompts_for_condition(self.count.value(), cond)
        if not prompts:
            self.progress_label.setText("No prompts left for this condition.")
            return
        try:
            store = DatasetStore()
        except Exception as exc:
            self.progress_label.setText(f"Cannot start: {exc}")
            return
        self.session = RecordSession(prompts, cond, store=store, recognizer=self.recognizer,
                                     raw=RawVideoWriter(enabled=self.raw_box.isChecked()), camera=self.camera_name)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.condition.setEnabled(False)
        self.tabs.setCurrentIndex(1)
        self._show_recorded(self.session.snapshot())
        self.sig.status.emit(f"Recording session: {cond}. Hold Space and mouth the sentence.")

    def _stop_session(self):
        self.session = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.condition.setEnabled(True)
        self.progress_label.setText("")
        self.prompt_label.setText("Session stopped. Train when you have 20+ takes.")
        self.now_label.setText("")
        self.next_label.setText("")
        self._refresh_counts()

    def _session_call(self, name: str):
        if self.session:
            getattr(self.session, name)()
            self._show_recorded(self.session.snapshot())
            self._refresh_counts()

    def _show_recorded(self, snap: dict):
        self.progress_label.setText(f"{snap['condition']}  ·  {snap['idx']} / {snap['total']}  ·  saved {snap['saved']}")
        self.prompt_label.setText(snap["prompt"] or "Session complete. Train when you have 20+ takes.")
        self.now_label.setText("Now hold SPACE and mouth:" if snap["prompt"] else "")
        self.next_label.setText(f"next: {snap['next']}" if snap["next"] else "")
        color = {"saved": P["good"], "retry": P["bad"], "saving": P["warn"]}.get(snap["status"], P["muted"])
        self.rec_status.setText(f'<span style="color:{color}">{snap["status"]}</span>  {snap["message"]}')
        last = snap.get("last")
        if last:
            self.readback_title.setText(f'READ-BACK OF THE LAST TAKE  ·  "{last["prompt"]}"')
            self.readback.setText(align_html(last["align"]) if last["align"] else last["read"])
        else:
            self.readback_title.setText("READ-BACK OF THE LAST TAKE")
            self.readback.setText("")
        acc = snap.get("accuracy")
        self.rec_acc.setText(f"session word accuracy with current model: {acc:.0%}" if acc is not None else "")
        if snap["status"] == "saved":
            self._refresh_counts()
            self.sig.status.emit("Saved. Hold Space for the next sentence.")
        if snap["done"]:
            self.sig.status.emit("Session complete.")

    def _refresh_counts(self):
        try:
            c = counts()
        except Exception:
            c = {}
        self.counts_label.setText("takes so far:  " + ("  ·  ".join(
            f"{k}: {v.get('train', 0)} train / {v.get('val', 0)} held-out" for k, v in sorted(c.items())) or "none"))

    # ------------------------------------------------------------------ train / eval
    def _append_log(self, text: str):
        self.log.appendPlainText(text.rstrip())

    def _progress(self, stage: str, frac: float):
        self.job_progress.setFormat(f"{stage} %p%")
        self.job_progress.setValue(int(frac * 1000))

    def _run_job(self, name: str, fn):
        if self.busy_job:
            return
        self.busy_job = True
        self.train_btn.setEnabled(False)
        self.eval_btn.setEnabled(False)
        self.tabs.setCurrentIndex(2)
        self._append_log(f"--- {name} {time.strftime('%H:%M:%S')} ---")

        def go():
            ok = True
            try:
                with self.model_lock:
                    fn()
            except Exception as exc:
                ok = False
                self.sig.log.emit(f"{name} failed: {exc}")
                self.sig.log.emit(traceback.format_exc())
            self.sig.job_done.emit(name, ok)
        threading.Thread(target=go, daemon=True).start()

    def _job_done(self, name: str, ok: bool):
        self.busy_job = False
        self.train_btn.setEnabled(True)
        self.eval_btn.setEnabled(True)
        self._append_log(f"--- {name} {'done' if ok else 'FAILED'} ---")
        self.sig.progress.emit("idle", 1.0 if ok else 0.0)
        if name == "train":
            self._load_models_async()   # pick up the new personal weights
        self._refresh_checks()
        self._refresh_counts()

    def _train(self):
        from ..evaluate import train
        cond = self.train_cond.currentText()
        cond = None if cond == "all conditions" else cond
        strat = self.strategy.currentText()
        strategies = ("light", "full") if strat == "both" else (strat,)
        self._run_job("train", lambda: train(condition=cond, strategies=strategies, log=self.sig.log.emit,
                                             progress=self.sig.progress.emit))

    def _eval(self):
        from ..evaluate import evaluate
        self._run_job("eval", lambda: evaluate(use_llm=self.eval_llm.isChecked(), use_rewrite=self.eval_rewrite.isChecked(),
                                               log=self.sig.log.emit,
                                               on_progress=lambda f: self.sig.progress.emit("eval", f)))

    # ------------------------------------------------------------------ dictate / close
    def _dictate(self):
        from ..__main__ import dictate_command, dictate_env
        subprocess.Popen(dictate_command(), env=dictate_env(), cwd=str(engine.ENGINE))
        self.sig.status.emit("Dictation app launched: hold fn in any app. Needs Input Monitoring + Accessibility.")

    def closeEvent(self, ev):
        if self.camera:
            self.camera.shutdown()
        super().closeEvent(ev)


def main(camera: str = "0") -> int:
    app = QApplication([])
    app.setApplicationName("Mindreader")
    theme.apply(app)
    win = Main(camera=camera)
    win.show()
    shot = os.environ.get("MINDREADER_SHOT")   # debug: save a window render per tab after 6 s, then quit
    if shot:
        def grab():
            for i in range(win.tabs.count()):
                win.tabs.setCurrentIndex(i)
                app.processEvents()
                win.grab().save(f"{shot}-{i}.png")
            app.quit()
        QTimer.singleShot(6000, grab)
    return app.exec()
