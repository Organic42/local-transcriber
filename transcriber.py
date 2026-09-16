"""
Local Transcriber — a small tool for transcribing audio/video.

Runs Whisper (via faster-whisper) entirely on this machine: nothing is uploaded.
Pick a file, a language, a model, GPU or CPU, and the output formats you want.
The window shows how much of the media has been transcribed and what the job
is costing in CPU, RAM and GPU.

Run:  python transcriber.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ── CUDA runtime DLLs ────────────────────────────────────────────────────────
# On Windows, faster-whisper's GPU path needs cuBLAS/cuDNN. The pip wheels
# (nvidia-cublas-cu12, nvidia-cudnn-cu12) ship them in site-packages, which is
# not on the DLL search path, so register those folders before importing.
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


LANGUAGES = [
    ("Auto-detect", None), ("English", "en"), ("Hindi", "hi"), ("Marathi", "mr"),
    ("Gujarati", "gu"), ("Tamil", "ta"), ("Telugu", "te"), ("Kannada", "kn"),
    ("Malayalam", "ml"), ("Bengali", "bn"), ("Punjabi", "pa"), ("Urdu", "ur"),
    ("Arabic", "ar"), ("Chinese", "zh"), ("French", "fr"), ("German", "de"),
    ("Japanese", "ja"), ("Spanish", "es"),
]
MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
FORMATS = ["txt", "srt", "vtt", "json", "md"]
MEDIA_TYPES = [
    ("Audio / video", "*.mp4 *.mkv *.mov *.webm *.avi *.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
    ("All files", "*.*"),
]


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
            head = f"# Transcript — {meta['file']}\n\n" \
                   f"*Language: {meta['language']} · Model: {meta['model']} · " \
                   f"Device: {meta['device']} · Duration: {short(meta['duration'])}" \
                   f"{' · PARTIAL' if meta['partial'] else ''}*\n\n"
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


# ── Worker ───────────────────────────────────────────────────────────────────
class Job(threading.Thread):
    """Runs one transcription and reports back through a queue."""

    def __init__(self, opts, events):
        super().__init__(daemon=True)
        self.opts, self.events = opts, events
        self.cancelled = threading.Event()

    def emit(self, kind, **data):
        self.events.put((kind, data))

    def run(self):
        o = self.opts
        try:
            from faster_whisper import WhisperModel

            device = o["device"]
            if device == "auto":
                device = "cuda" if cuda_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"

            self.emit("status", text=f"Loading model {o['model']} on {device.upper()} "
                                     f"(first use downloads it)…")
            try:
                model = WhisperModel(o["model"], device=device, compute_type=compute)
            except Exception as e:
                if device != "cuda":
                    raise
                self.emit("log", text=f"GPU unavailable ({e}); falling back to CPU.")
                device, compute = "cpu", "int8"
                model = WhisperModel(o["model"], device=device, compute_type=compute)

            self.emit("status", text="Reading media…")
            segs_iter, info = model.transcribe(
                o["file"],
                language=o["language"],
                task=o["task"],
                beam_size=5,
                vad_filter=o["vad"],
                condition_on_previous_text=False,
                initial_prompt=o["prompt"] or None,
            )
            duration = info.duration or 0
            self.emit("started", duration=duration, device=device,
                      language=f"{info.language} ({info.language_probability:.0%})")

            segments = []
            for seg in segs_iter:
                if self.cancelled.is_set():
                    break
                text = seg.text.strip()
                segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
                self.emit("progress", position=seg.end, text=f"[{short(seg.start)}] {text}")

            partial = self.cancelled.is_set()
            src = Path(o["file"])
            meta = {
                "file": src.name, "language": info.language, "model": o["model"],
                "device": device, "task": o["task"], "duration": duration, "partial": partial,
            }
            stem = src.stem + ("_partial" if partial else "")
            written = write_outputs(segments, meta, Path(o["out_dir"]), stem, o["formats"])
            self.emit("done", files=[str(p) for p in written], partial=partial, count=len(segments))
        except Exception as e:
            self.emit("error", text=f"{type(e).__name__}: {e}")


# ── UI ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Transcriber")
        self.geometry("760x640")
        self.minsize(680, 560)

        self.events = queue.Queue()
        self.job = None
        self.duration = 0
        self.started_at = None
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)

        self._build()
        self.after(150, self._drain_events)
        self.after(1000, self._sample_resources)

    # layout
    def _build(self):
        pad = {"padx": 10, "pady": 4}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=8, pady=8)

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

        opts = ttk.LabelFrame(root, text="Options")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Language").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.lang_var = tk.StringVar(value="Auto-detect")
        ttk.Combobox(opts, textvariable=self.lang_var, state="readonly", width=16,
                     values=[l for l, _ in LANGUAGES]).grid(row=0, column=1, sticky="w")

        ttk.Label(opts, text="Model").grid(row=0, column=2, sticky="w", padx=(18, 6))
        self.model_var = tk.StringVar(value="large-v3")
        ttk.Combobox(opts, textvariable=self.model_var, state="readonly", width=16,
                     values=MODELS).grid(row=0, column=3, sticky="w")

        ttk.Label(opts, text="Task").grid(row=0, column=4, sticky="w", padx=(18, 6))
        self.task_var = tk.StringVar(value="transcribe")
        ttk.Combobox(opts, textvariable=self.task_var, state="readonly", width=18,
                     values=["transcribe", "translate"]).grid(row=0, column=5, sticky="w")

        has_gpu = cuda_available()
        ttk.Label(opts, text="Device").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        dev = ttk.Frame(opts)
        dev.grid(row=1, column=1, columnspan=5, sticky="w")
        self.device_var = tk.StringVar(value="auto")
        ttk.Radiobutton(dev, text="Auto", value="auto", variable=self.device_var).pack(side="left")
        gpu_btn = ttk.Radiobutton(dev, text=f"GPU{f' ({_GPU_NAME})' if _GPU_NAME else ''}",
                                  value="cuda", variable=self.device_var)
        gpu_btn.pack(side="left", padx=10)
        if not has_gpu:
            gpu_btn.state(["disabled"])
        ttk.Radiobutton(dev, text="CPU", value="cpu", variable=self.device_var).pack(side="left")
        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dev, text="Skip silence (VAD)", variable=self.vad_var).pack(side="left", padx=18)

        ttk.Label(opts, text="Formats").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        fmt = ttk.Frame(opts)
        fmt.grid(row=2, column=1, columnspan=5, sticky="w")
        self.fmt_vars = {f: tk.BooleanVar(value=f in ("txt", "srt")) for f in FORMATS}
        for f, v in self.fmt_vars.items():
            ttk.Checkbutton(fmt, text=f.upper(), variable=v).pack(side="left", padx=(0, 12))

        ttk.Label(opts, text="Vocabulary").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        self.prompt_var = tk.StringVar()
        ttk.Entry(opts, textvariable=self.prompt_var).grid(row=3, column=1, columnspan=5, sticky="ew", pady=3)
        ttk.Label(opts, text="Optional names and terms to spell correctly, e.g. ArthaFlow, RoDTEP, Nashik",
                  foreground="gray").grid(row=4, column=1, columnspan=5, sticky="w")
        opts.columnconfigure(5, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", **pad)
        self.start_btn = ttk.Button(actions, text="Start", command=self._start)
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
            ttk.Label(res, text=label, foreground="gray").grid(row=0, column=col, sticky="w", padx=8, pady=(4, 0))
            ttk.Label(res, textvariable=self.res_vars[key]).grid(row=1, column=col, sticky="w", padx=8, pady=(0, 6))
            res.columnconfigure(col, weight=1)

        logf = ttk.LabelFrame(root, text="Live transcript")
        logf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(logf, height=8, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True, padx=(6, 0), pady=6)

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
        if not file or not Path(file).is_file():
            return messagebox.showwarning("Transcriber", "Choose an audio or video file first.")
        if not formats:
            return messagebox.showwarning("Transcriber", "Pick at least one output format.")
        out_dir = self.out_var.get().strip() or str(Path(file).parent)
        self.out_var.set(out_dir)

        opts = {
            "file": file, "out_dir": out_dir, "formats": formats,
            "language": dict(LANGUAGES)[self.lang_var.get()],
            "model": self.model_var.get(), "task": self.task_var.get(),
            "device": self.device_var.get(), "vad": self.vad_var.get(),
            "prompt": self.prompt_var.get().strip(),
        }
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
        self.bar["value"] = self.bar["value"] if partial else 1000
        elapsed = short(time.time() - (self.started_at or time.time()))
        state = "Cancelled — partial transcript saved" if partial else "Done"
        self.prog_var.set(f"{state} · {count} segments · {elapsed}")
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
            proc_cpu = self.proc.cpu_percent(None) / cores
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
