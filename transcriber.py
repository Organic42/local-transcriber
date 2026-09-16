"""
Local Transcriber — a small tool for transcribing audio and video on this machine.

Nothing is uploaded. Models run locally through two open-source engines:
  * faster-whisper (OpenAI Whisper and fine-tunes, CTranslate2)
  * onnx-asr (NVIDIA Parakeet TDT, ONNX Runtime)

Pick a file, a model, a language, GPU or CPU, and the output formats you want.
The window shows how much of the media has been transcribed and what the job
costs in CPU, RAM and GPU.

Run:  python transcriber.py
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import wave
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ── CUDA runtime DLLs ────────────────────────────────────────────────────────
# On Windows the GPU path needs cuBLAS/cuDNN. The pip wheels (nvidia-cublas-cu12,
# nvidia-cudnn-cu12) put them in site-packages, which is not on the DLL search
# path, so register those folders before anything loads CUDA.
def _register_cuda_dlls():
    try:
        import nvidia  # namespace package from the nvidia-* wheels
    except ImportError:
        return
    for root in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            d = os.path.join(root, sub, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()

import psutil  # noqa: E402

try:
    import pynvml  # from nvidia-ml-py  # noqa: E402

    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_NAME = pynvml.nvmlDeviceGetName(_GPU_HANDLE)
    if isinstance(_GPU_NAME, bytes):
        _GPU_NAME = _GPU_NAME.decode()
except Exception:
    _GPU_HANDLE, _GPU_NAME = None, None


# ── Models ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    label: str
    engine: str                   # "whisper" or "parakeet"
    model_id: str | None          # None = user supplies a custom repo id / folder
    language: str | None = None   # forced language for single-language models
    translate: bool = True        # supports "translate to English"
    note: str = ""


MODELS = [
    ModelSpec("Whisper large-v3 — best accuracy, 99 languages", "whisper", "large-v3"),
    ModelSpec("Whisper large-v3-turbo — much faster, near large-v3", "whisper", "large-v3-turbo"),
    ModelSpec("Distil-Whisper large-v3.5 — English only, fast", "whisper",
              "distil-whisper/distil-large-v3.5-ct2", language="en", translate=False),
    ModelSpec("Whisper large-v2 Hindi (Collabora) — Hindi fine-tune", "whisper",
              "collabora/faster-whisper-large-v2-hindi", language="hi", translate=False),
    ModelSpec("NVIDIA Parakeet TDT 0.6B v3 — English + 24 European, very fast", "parakeet",
              "nemo-parakeet-tdt-0.6b-v3", translate=False,
              note="Detects language itself · always splits on silence · no Indian languages"),
    ModelSpec("Whisper medium — lighter", "whisper", "medium"),
    ModelSpec("Whisper small — good on CPU", "whisper", "small"),
    ModelSpec("Whisper base", "whisper", "base"),
    ModelSpec("Whisper tiny — quick test", "whisper", "tiny"),
    ModelSpec("Custom Whisper model (Hugging Face repo or folder)…", "whisper", None,
              note="Any CTranslate2 Whisper model: a Hugging Face repo id or a local folder"),
]
MODEL_BY_LABEL = {m.label: m for m in MODELS}

LANGUAGES = [
    ("Auto-detect", None), ("English", "en"), ("Hindi", "hi"), ("Marathi", "mr"),
    ("Gujarati", "gu"), ("Tamil", "ta"), ("Telugu", "te"), ("Kannada", "kn"),
    ("Malayalam", "ml"), ("Bengali", "bn"), ("Punjabi", "pa"), ("Urdu", "ur"),
    ("Arabic", "ar"), ("Chinese", "zh"), ("French", "fr"), ("German", "de"),
    ("Japanese", "ja"), ("Spanish", "es"),
]
LANG_NAME = {code: name for name, code in LANGUAGES}
FORMATS = ["txt", "srt", "vtt", "json", "md"]
MEDIA_TYPES = [
    ("Audio / video", "*.mp4 *.mkv *.mov *.webm *.avi *.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
    ("All files", "*.*"),
]
CONFIG_PATH = Path.home() / ".local-transcriber.json"


# ── Themes ───────────────────────────────────────────────────────────────────
THEMES = {
    "Light": dict(bg="#f3f4f6", fg="#1f2328", muted="#6b7280", field="#ffffff",
                  button="#e5e7eb", accent="#2563eb", on_accent="#ffffff", select="#cfe0ff",
                  border="#d1d5db", font=("Segoe UI", 10), mono=("Consolas", 10)),
    "Dark": dict(bg="#1e1f22", fg="#e3e5e8", muted="#9aa0a6", field="#2b2d31",
                 button="#383a40", accent="#5b9bff", on_accent="#0b1220", select="#35507a",
                 border="#3f4147", font=("Segoe UI", 10), mono=("Consolas", 10)),
    "Terminal": dict(bg="#050805", fg="#39ff6a", muted="#23a045", field="#0b140c",
                     button="#0f1f11", accent="#39ff6a", on_accent="#050805", select="#15522a",
                     border="#23a045", font=("Consolas", 10), mono=("Consolas", 10)),
}


# ── Formatting helpers ───────────────────────────────────────────────────────
def clock(seconds, sep=","):
    """HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def short(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def write_outputs(segments, meta, out_dir, stem, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "txt":
            body = "\n".join(f"[{short(s['start'])}] {s['text']}" for s in segments)
        elif fmt == "srt":
            body = "\n".join(
                f"{i}\n{clock(s['start'])} --> {clock(s['end'])}\n{s['text']}\n"
                for i, s in enumerate(segments, 1)
            )
        elif fmt == "vtt":
            body = "WEBVTT\n\n" + "\n".join(
                f"{clock(s['start'], '.')} --> {clock(s['end'], '.')}\n{s['text']}\n"
                for s in segments
            )
        elif fmt == "json":
            body = json.dumps({**meta, "segments": segments}, ensure_ascii=False, indent=2)
        elif fmt == "md":
            head = (f"# Transcript — {meta['file']}\n\n"
                    f"*Language: {meta['language']} · Model: {meta['model']} · "
                    f"Device: {meta['device']} · Duration: {short(meta['duration'])}"
                    f"{' · PARTIAL' if meta['partial'] else ''}*\n\n")
            body = head + "\n\n".join(f"**[{short(s['start'])}]** {s['text']}" for s in segments)
        path.write_text(body + "\n", encoding="utf-8")
        written.append(path)
    return written


def cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(data):
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


class Cancelled(Exception):
    pass


# ── Worker ───────────────────────────────────────────────────────────────────
class Job(threading.Thread):
    """Runs one transcription and reports back through a queue."""

    def __init__(self, opts, events):
        super().__init__(daemon=True)
        self.opts, self.events = opts, events
        self.cancelled = threading.Event()

    def emit(self, kind, **data):
        self.events.put((kind, data))

    def prepare_audio(self, src, boost, workdir):
        """Decode to 16 kHz mono WAV with ffmpeg, optionally levelling quiet speech."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH. Install it, or turn off "
                               "'Boost quiet audio' and use a Whisper model.")
        out = Path(workdir) / "audio.wav"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000"]
        if boost:
            cmd += ["-af", "dynaudnorm=f=150:g=15:p=0.95:m=30"]
        cmd.append(str(out))
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, creationflags=flags)
        while proc.poll() is None:
            if self.cancelled.is_set():
                proc.kill()
                raise Cancelled()
            time.sleep(0.2)
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg could not read this file:\n"
                               + proc.stderr.read().decode(errors="ignore"))
        with wave.open(str(out)) as w:
            duration = w.getnframes() / w.getframerate()
        return str(out), duration

    def run_whisper(self, audio, model_id, device, language):
        from faster_whisper import WhisperModel

        o = self.opts
        compute = "float16" if device == "cuda" else "int8"
        self.emit("status", text=f"Loading {model_id} on {device.upper()} (first use downloads it)…")
        try:
            model = WhisperModel(model_id, device=device, compute_type=compute)
        except Exception as e:
            if device != "cuda":
                raise
            self.emit("log", text=f"GPU unavailable ({e}); falling back to CPU.")
            device = "cpu"
            model = WhisperModel(model_id, device=device, compute_type="int8")

        self.emit("status", text="Transcribing…")
        segs, info = model.transcribe(
            audio, language=language, task=o["task"], beam_size=o["beam"],
            vad_filter=o["vad"], condition_on_previous_text=False,
            hotwords=o["prompt"] or None,
        )
        self.emit("started", duration=info.duration or 0, device=device,
                  language=f"{LANG_NAME.get(info.language, info.language)} "
                           f"({info.language_probability:.0%})")
        return device, info.language, info.duration or 0, ((s.start, s.end, s.text) for s in segs)

    def run_parakeet(self, audio, model_id, device, duration):
        import onnx_asr
        import onnxruntime as ort

        if device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            self.emit("log", text="ONNX Runtime has no CUDA provider; using CPU.")
            device = "cpu"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        self.emit("status", text=f"Loading Parakeet on {device.upper()} (first use downloads it)…")
        vad = onnx_asr.load_vad("silero", providers=providers)
        model = onnx_asr.load_model(model_id, providers=providers)
        self.emit("status", text="Transcribing…")
        self.emit("started", duration=duration, device=device, language="auto")
        stream = ((s.start, s.end, s.text) for s in model.with_vad(vad).recognize(audio))
        return device, "auto", duration, stream

    def run(self):
        o = self.opts
        spec = o["spec"]
        segments, device, lang, duration = [], o["device"], None, 0
        try:
            if device == "auto":
                device = "cuda" if cuda_available() else "cpu"
            language = spec.language or o["language"]

            with tempfile.TemporaryDirectory(prefix="transcriber-") as work:
                audio = o["file"]
                if spec.engine == "parakeet" or o["boost"]:
                    self.emit("status", text="Extracting audio"
                                             f"{' and levelling quiet speech' if o['boost'] else ''}…")
                    audio, duration = self.prepare_audio(o["file"], o["boost"], work)

                if spec.engine == "parakeet":
                    device, lang, duration, stream = self.run_parakeet(audio, o["model_id"], device, duration)
                else:
                    device, lang, duration, stream = self.run_whisper(audio, o["model_id"], device, language)

                for start, end, text in stream:
                    if self.cancelled.is_set():
                        break
                    text = text.strip()
                    if not text:
                        continue
                    segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
                    self.emit("progress", position=end, text=f"[{short(start)}] {text}")
        except Cancelled:
            pass
        except Exception as e:
            self.emit("error", text=f"{type(e).__name__}: {e}")
            return

        partial = self.cancelled.is_set()
        if partial and not segments:
            self.emit("done", files=[], partial=True, count=0)
            return
        src = Path(o["file"])
        meta = {
            "file": src.name, "language": lang, "model": o["model_id"], "engine": spec.engine,
            "device": device, "task": o["task"], "duration": duration, "partial": partial,
        }
        stem = src.stem + ("_partial" if partial else "")
        written = write_outputs(segments, meta, Path(o["out_dir"]), stem, o["formats"])
        self.emit("done", files=[str(p) for p in written], partial=partial, count=len(segments))


# ── UI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Transcriber")
        self.geometry("840x860")
        self.minsize(760, 780)

        self.config_data = load_config()
        self.events = queue.Queue()
        self.job = None
        self.duration = 0
        self.started_at = None
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.combos = []

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        saved_theme = self.config_data.get("theme")
        self.theme_var = tk.StringVar(value=saved_theme if saved_theme in THEMES else "Dark")

        self._build()
        self._apply_theme()
        self._on_model_change()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(150, self._drain_events)
        self.after(1000, self._sample_resources)

    # layout
    def _combo(self, parent, var, values, width):
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", width=width, values=values)
        self.combos.append(cb)
        return cb

    def _build(self):
        pad = {"padx": 10, "pady": 4}
        cfg = self.config_data
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=10, pady=(2, 0))
        ttk.Label(top, text="Local Transcriber", style="Title.TLabel").pack(side="left")
        cb = self._combo(top, self.theme_var, list(THEMES), 10)
        cb.pack(side="right")
        cb.bind("<<ComboboxSelected>>", lambda _: self._apply_theme())
        ttk.Label(top, text="Theme", style="Muted.TLabel").pack(side="right", padx=6)

        files = ttk.LabelFrame(root, text="Files")
        files.pack(fill="x", **pad)
        self.file_var = tk.StringVar()
        self.out_var = tk.StringVar()
        for row, (label, var, cmd) in enumerate([
            ("Input", self.file_var, self._pick_file),
            ("Output folder", self.out_var, self._pick_out),
        ]):
            ttk.Label(files, text=label, width=13).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(files, textvariable=var).grid(row=row, column=1, sticky="ew", pady=3)
            ttk.Button(files, text="Browse…", command=cmd).grid(row=row, column=2, padx=6, pady=3)
        files.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(root, text="Model")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Model", width=13).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        saved_model = cfg.get("model")
        self.model_var = tk.StringVar(value=saved_model if saved_model in MODEL_BY_LABEL else MODELS[0].label)
        cb = self._combo(opts, self.model_var, [m.label for m in MODELS], 60)
        cb.grid(row=0, column=1, columnspan=5, sticky="w", pady=3)
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_model_change())

        ttk.Label(opts, text="Custom model").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.custom_var = tk.StringVar(value=cfg.get("custom_model", ""))
        self.custom_entry = ttk.Entry(opts, textvariable=self.custom_var)
        self.custom_entry.grid(row=1, column=1, columnspan=5, sticky="ew", pady=3)

        ttk.Label(opts, text="Language").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        saved_lang = cfg.get("language")
        self.lang_var = tk.StringVar(value=saved_lang if saved_lang in dict(LANGUAGES) else "Auto-detect")
        self.lang_combo = self._combo(opts, self.lang_var, [l for l, _ in LANGUAGES], 14)
        self.lang_combo.grid(row=2, column=1, sticky="w")

        ttk.Label(opts, text="Task").grid(row=2, column=2, sticky="w", padx=(18, 6))
        self.task_var = tk.StringVar(value="transcribe")
        self.task_combo = self._combo(opts, self.task_var, ["transcribe", "translate"], 11)
        self.task_combo.grid(row=2, column=3, sticky="w")

        ttk.Label(opts, text="Beam size").grid(row=2, column=4, sticky="w", padx=(18, 6))
        self.beam_var = tk.IntVar(value=5)
        self.beam_spin = ttk.Spinbox(opts, from_=1, to=10, width=4, textvariable=self.beam_var)
        self.beam_spin.grid(row=2, column=5, sticky="w")

        self.model_note = tk.StringVar()
        ttk.Label(opts, textvariable=self.model_note, style="Muted.TLabel").grid(
            row=3, column=1, columnspan=5, sticky="w", pady=(0, 4))
        opts.columnconfigure(5, weight=1)

        run = ttk.LabelFrame(root, text="Run")
        run.pack(fill="x", **pad)

        ttk.Label(run, text="Device", width=13).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        dev = ttk.Frame(run)
        dev.grid(row=0, column=1, sticky="w")
        self.device_var = tk.StringVar(value="auto")
        ttk.Radiobutton(dev, text="Auto", value="auto", variable=self.device_var).pack(side="left")
        gpu_btn = ttk.Radiobutton(dev, text=f"GPU{f' ({_GPU_NAME})' if _GPU_NAME else ''}",
                                  value="cuda", variable=self.device_var)
        gpu_btn.pack(side="left", padx=10)
        if not cuda_available():
            gpu_btn.state(["disabled"])
        ttk.Radiobutton(dev, text="CPU", value="cpu", variable=self.device_var).pack(side="left")

        ttk.Label(run, text="Audio").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        aud = ttk.Frame(run)
        aud.grid(row=1, column=1, sticky="w")
        self.vad_var = tk.BooleanVar(value=True)
        self.vad_check = ttk.Checkbutton(aud, text="Skip silence (VAD)", variable=self.vad_var)
        self.vad_check.pack(side="left")
        self.boost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aud, text="Boost quiet audio", variable=self.boost_var).pack(side="left", padx=18)

        ttk.Label(run, text="Formats").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        fmt = ttk.Frame(run)
        fmt.grid(row=2, column=1, sticky="w")
        saved_formats = cfg.get("formats", ["txt", "srt"])
        self.fmt_vars = {f: tk.BooleanVar(value=f in saved_formats) for f in FORMATS}
        for f, v in self.fmt_vars.items():
            ttk.Checkbutton(fmt, text=f.upper(), variable=v).pack(side="left", padx=(0, 12))

        ttk.Label(run, text="Vocabulary").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        self.prompt_var = tk.StringVar(value=cfg.get("vocabulary", ""))
        self.prompt_entry = ttk.Entry(run, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(run, text="Names and terms to spell correctly, comma-separated (Whisper models)",
                  style="Muted.TLabel").grid(row=4, column=1, sticky="w")
        run.columnconfigure(1, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", **pad)
        self.start_btn = ttk.Button(actions, text="Start", command=self._start, style="Accent.TButton")
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(actions, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(actions, text="Open output folder", command=self._open_out).pack(side="right")

        prog = ttk.LabelFrame(root, text="Progress")
        prog.pack(fill="x", **pad)
        self.bar = ttk.Progressbar(prog, maximum=1000)
        self.bar.pack(fill="x", padx=6, pady=(6, 2))
        self.prog_var = tk.StringVar(value="Idle")
        ttk.Label(prog, textvariable=self.prog_var).pack(anchor="w", padx=6, pady=(0, 6))

        res = ttk.LabelFrame(root, text="Resources")
        res.pack(fill="x", **pad)
        self.res_vars = {k: tk.StringVar(value="—") for k in ("cpu", "ram", "gpu", "vram")}
        for col, (key, label) in enumerate([("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"), ("vram", "VRAM")]):
            ttk.Label(res, text=label, style="Muted.TLabel").grid(row=0, column=col, sticky="w", padx=8, pady=(4, 0))
            ttk.Label(res, textvariable=self.res_vars[key]).grid(row=1, column=col, sticky="w", padx=8, pady=(0, 6))
            res.columnconfigure(col, weight=1)

        logf = ttk.LabelFrame(root, text="Live transcript")
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=8, wrap="word", state="disabled", relief="flat",
                           borderwidth=0, highlightthickness=0, padx=6, pady=4)
        scroll = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True, padx=(6, 0), pady=6)

    # theme
    def _apply_theme(self):
        name = self.theme_var.get()
        t = THEMES.get(name, THEMES["Dark"])
        s = self.style
        bold = (t["font"][0], t["font"][1], "bold")
        s.configure(".", background=t["bg"], foreground=t["fg"], fieldbackground=t["field"],
                    selectbackground=t["select"], selectforeground=t["fg"], bordercolor=t["border"],
                    lightcolor=t["field"], darkcolor=t["border"], troughcolor=t["field"],
                    focuscolor=t["accent"], insertcolor=t["fg"], font=t["font"])
        s.configure("TLabelframe", background=t["bg"], bordercolor=t["border"])
        s.configure("TLabelframe.Label", background=t["bg"], foreground=t["accent"], font=bold)
        s.configure("Muted.TLabel", foreground=t["muted"])
        s.configure("Title.TLabel", foreground=t["accent"], font=(t["font"][0], 14, "bold"))
        s.configure("TButton", background=t["button"], foreground=t["fg"], bordercolor=t["border"])
        s.map("TButton", background=[("disabled", t["bg"]), ("active", t["select"])],
              foreground=[("disabled", t["muted"])])
        s.configure("Accent.TButton", background=t["accent"], foreground=t["on_accent"], font=bold)
        s.map("Accent.TButton", background=[("disabled", t["button"]), ("active", t["select"])],
              foreground=[("disabled", t["muted"]), ("active", t["fg"])])
        for widget in ("TEntry", "TSpinbox", "TCombobox"):
            s.configure(widget, fieldbackground=t["field"], foreground=t["fg"],
                        background=t["field"], arrowcolor=t["fg"])
            s.map(widget, fieldbackground=[("disabled", t["bg"]), ("readonly", t["field"])],
                  foreground=[("disabled", t["muted"]), ("readonly", t["fg"])],
                  selectbackground=[("readonly", t["field"])],
                  selectforeground=[("readonly", t["fg"])])
        for widget in ("TCheckbutton", "TRadiobutton"):
            s.configure(widget, indicatorbackground=t["field"], indicatorforeground=t["accent"])
            s.map(widget, foreground=[("disabled", t["muted"])], background=[("disabled", t["bg"])])
        s.configure("TProgressbar", background=t["accent"], troughcolor=t["field"])
        s.configure("TScrollbar", background=t["button"], troughcolor=t["bg"])

        self.configure(background=t["bg"])
        self.log.configure(background=t["field"], foreground=t["fg"], insertbackground=t["fg"],
                           selectbackground=t["select"], font=t["mono"])
        for opt, key in (("background", "field"), ("foreground", "fg"),
                         ("selectBackground", "select"), ("selectForeground", "fg")):
            self.option_add(f"*TCombobox*Listbox.{opt}", t[key])
        self.option_add("*TCombobox*Listbox.font", t["font"])
        for cb in self.combos:  # popdowns that already exist ignore option_add
            try:
                popdown = self.tk.eval(f"ttk::combobox::PopdownWindow {cb}")
                self.tk.call(f"{popdown}.f.l", "configure", "-background", t["field"],
                             "-foreground", t["fg"], "-selectbackground", t["select"],
                             "-selectforeground", t["fg"], "-font", t["font"])
            except tk.TclError:
                pass
        self._save_prefs()

    # model-dependent controls
    def _on_model_change(self):
        spec = MODEL_BY_LABEL[self.model_var.get()]
        whisper = spec.engine == "whisper"
        self.custom_entry.state(["!disabled"] if spec.model_id is None else ["disabled"])
        if spec.language:
            self.lang_var.set(LANG_NAME[spec.language])
        elif not whisper:
            self.lang_var.set("Auto-detect")
        self.lang_combo.state(["!disabled", "readonly"] if whisper and not spec.language else ["disabled"])
        if not spec.translate:
            self.task_var.set("transcribe")
        self.task_combo.state(["!disabled", "readonly"] if spec.translate else ["disabled"])
        for widget in (self.beam_spin, self.vad_check, self.prompt_entry):
            widget.state(["!disabled"] if whisper else ["disabled"])
        self.model_note.set(spec.note)
        self._save_prefs()

    # actions
    def _pick_file(self):
        path = filedialog.askopenfilename(filetypes=MEDIA_TYPES)
        if path:
            self.file_var.set(path)
            if not self.out_var.get():
                self.out_var.set(str(Path(path).parent))

    def _pick_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _open_out(self):
        folder = self.out_var.get()
        if folder and Path(folder).is_dir():
            os.startfile(folder) if sys.platform == "win32" else subprocess.Popen(["xdg-open", folder])

    def _start(self):
        file = self.file_var.get().strip()
        formats = [f for f, v in self.fmt_vars.items() if v.get()]
        spec = MODEL_BY_LABEL[self.model_var.get()]
        model_id = spec.model_id or self.custom_var.get().strip()
        if not file or not Path(file).is_file():
            return messagebox.showwarning("Transcriber", "Choose an audio or video file first.")
        if not formats:
            return messagebox.showwarning("Transcriber", "Pick at least one output format.")
        if not model_id:
            return messagebox.showwarning("Transcriber", "Enter a custom model repo id or folder.")
        out_dir = self.out_var.get().strip() or str(Path(file).parent)
        self.out_var.set(out_dir)
        try:
            beam = max(1, min(10, int(self.beam_var.get())))
        except (tk.TclError, ValueError):
            beam = 5

        opts = {
            "file": file, "out_dir": out_dir, "formats": formats, "spec": spec, "model_id": model_id,
            "language": dict(LANGUAGES)[self.lang_var.get()], "task": self.task_var.get(),
            "device": self.device_var.get(), "vad": self.vad_var.get(), "boost": self.boost_var.get(),
            "beam": beam, "prompt": self.prompt_var.get().strip(),
        }
        self._save_prefs()
        self._clear_log()
        self.bar["value"] = 0
        self.duration, self.started_at = 0, time.time()
        self.start_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.job = Job(opts, self.events)
        self.job.start()

    def _cancel(self):
        if self.job:
            self.job.cancelled.set()
            self.prog_var.set("Cancelling — saving what has been transcribed so far…")
            self.cancel_btn.state(["disabled"])

    def _save_prefs(self):
        if not hasattr(self, "prompt_var"):
            return
        save_config({
            "theme": self.theme_var.get(), "model": self.model_var.get(),
            "language": self.lang_var.get(), "custom_model": self.custom_var.get(),
            "formats": [f for f, v in self.fmt_vars.items() if v.get()],
            "vocabulary": self.prompt_var.get(),
        })

    def _close(self):
        self._save_prefs()
        if self.job:
            self.job.cancelled.set()
        self.destroy()

    # event loop
    def _drain_events(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(**data)
        except queue.Empty:
            pass
        self.after(150, self._drain_events)

    def _on_status(self, text):
        self.prog_var.set(text)

    def _on_log(self, text):
        self._append(text)

    def _on_started(self, duration, device, language):
        self.duration = duration
        self.started_at = time.time()
        self._append(f"— {short(duration)} of media · {device.upper()} · language {language} —")
        self.prog_var.set(f"00:00 / {short(duration)} (0%)")

    def _on_progress(self, position, text):
        self._append(text)
        if not self.duration:
            return
        frac = min(position / self.duration, 1.0)
        elapsed = time.time() - self.started_at
        speed = position / elapsed if elapsed > 0 else 0
        eta = (self.duration - position) / speed if speed > 0 else 0
        self.bar["value"] = int(frac * 1000)
        self.prog_var.set(
            f"{short(position)} / {short(self.duration)} ({frac:.0%}) · elapsed {short(elapsed)}"
            f" · ETA {short(eta)} · {speed:.1f}× realtime"
        )

    def _on_done(self, files, partial, count):
        if not partial:
            self.bar["value"] = 1000
        elapsed = short(time.time() - (self.started_at or time.time()))
        if partial:
            state = "Cancelled — partial transcript saved" if files else "Cancelled"
        else:
            state = "Done"
        self.prog_var.set(f"{state} · {count} segments · {elapsed}")
        if files:
            self._append("Saved:\n  " + "\n  ".join(files))
        self._reset_buttons()

    def _on_error(self, text):
        self.prog_var.set("Failed")
        self._append(f"ERROR: {text}")
        self._reset_buttons()
        messagebox.showerror("Transcriber", text)

    def _reset_buttons(self):
        self.job = None
        self.start_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

    # resources
    def _sample_resources(self):
        try:
            cores = psutil.cpu_count() or 1
            proc_cpu = min(self.proc.cpu_percent(None) / cores, 100)
            self.res_vars["cpu"].set(f"{proc_cpu:.0f}% app · {psutil.cpu_percent(None):.0f}% system")
            vm = psutil.virtual_memory()
            rss = self.proc.memory_info().rss / 2**30
            self.res_vars["ram"].set(f"{rss:.2f} GB app · {vm.percent:.0f}% of {vm.total / 2**30:.0f} GB")
            if _GPU_HANDLE is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                self.res_vars["gpu"].set(f"{util}%")
                self.res_vars["vram"].set(f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.0f} GB")
            else:
                self.res_vars["gpu"].set("no NVIDIA GPU")
                self.res_vars["vram"].set("—")
        except Exception:
            pass
        self.after(1000, self._sample_resources)

    # log
    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
