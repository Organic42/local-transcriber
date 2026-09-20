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
import re
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
from tkinter import filedialog, font as tkfont, messagebox, ttk


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

# Extract audio: format -> (ffmpeg codec args, takes a bitrate)
AUDIO_FORMATS = {
    "mp3": (["-c:a", "libmp3lame"], True),
    "m4a": (["-c:a", "aac"], True),
    "opus": (["-c:a", "libopus"], True),
    "ogg": (["-c:a", "libvorbis"], True),
    "flac": (["-c:a", "flac"], False),
    "wav": (["-c:a", "pcm_s16le"], False),
}
BITRATES = ["96k", "128k", "192k", "256k", "320k"]
LEVELLER = "dynaudnorm=f=150:g=15:p=0.95:m=30"  # evens out quiet and loud speakers
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── Look: Windows 95 desktop, Japanese-PC-manual trim ───────────────────────
VERSION = "1.0"
TAGLINE = "SPEECH TO TEXT. ON YOUR MACHINE. ALWAYS."
UI_FONT = ("MS Sans Serif", 8)  # a scheme can swap it with a "font" entry
LOG_FONT = ("Fixedsys", 9)

# Sidebar, header ribbon and the decorative panel are the app's fixed identity — they
# keep this palette no matter which content colour scheme is picked below.
CHROME = dict(bg="#12141c", bg2="#191c28", active="#232840", text="#e8eaf2", muted="#7a8096",
              accent="#3be08a", accent2="#00d4ff")
PANEL = dict(bg="#1c1a36", bg2="#252048", text="#eef0fb", muted="#9c97c4", accent="#6df0c2",
             accent2="#ffd166")

# Colour schemes after the Win95 Appearance tab. face/light/shadow/dark are the four
# shades of a 3D bevel; well/tile/speech/working colour the transcript map.
SCHEMES = {
    "Neon Dusk": dict(
        face="#c9cdd6", light="#f2f3f6", shadow="#8a8fa0", dark="#3a3d4a", text="#1b1d24",
        field="#ffffff", field_text="#1b1d24", select="#00b8d9", select_text="#ffffff",
        disabled="#9aa0ab", trough="#dfe2e8", muted="#5b6070",
        well="#11141c", tile="#3a3f4c", speech="#33e08a", working="#ff3fae",
        # the well is dark here (unlike every other scheme, where well == field), so text
        # painted on it needs its own light colour rather than field_text
        well_text="#eef0f8", well_muted="#9096ac"),
    "Windows Standard": dict(
        face="#c0c0c0", light="#ffffff", shadow="#808080", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#000080", select_text="#ffffff",
        disabled="#808080", trough="#e0e0e0", muted="#4b4b4b",
        well="#ffffff", tile="#c0c0c0", speech="#0000ff", working="#ff0000"),
    "High Contrast Black": dict(
        face="#000000", light="#ffffff", shadow="#808080", dark="#ffffff", text="#ffffff",
        field="#000000", field_text="#ffffff", select="#800080", select_text="#ffffff",
        disabled="#808080", trough="#404040", emboss=False, muted="#b0b0b0",
        well="#000000", tile="#808080", speech="#00ffff", working="#ffff00"),
    "Eggplant": dict(
        face="#90b0a8", light="#d8e4e0", shadow="#587870", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#584078", select_text="#ffffff",
        disabled="#587870", trough="#b8ccc8", muted="#2f4440",
        well="#ffffff", tile="#90b0a8", speech="#800080", working="#ff0000"),
    "Brick": dict(
        face="#c2bfa5", light="#e8e6d8", shadow="#817e6a", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#800000", select_text="#ffffff",
        disabled="#817e6a", trough="#dcdac8", muted="#4a4738",
        well="#ffffff", tile="#c2bfa5", speech="#800000", working="#008000"),
    # the app's earlier Light, Dark and Terminal colours, given Win95 bevels
    "Light": dict(
        face="#f3f4f6", light="#ffffff", shadow="#9ca3af", dark="#4b5563", text="#1f2328",
        field="#ffffff", field_text="#1f2328", select="#2563eb", select_text="#ffffff",
        disabled="#9ca3af", trough="#e5e7eb", muted="#6b7280",
        well="#ffffff", tile="#e5e7eb", speech="#2563eb", working="#dc2626"),
    "Dark": dict(
        face="#2b2d31", light="#4e5058", shadow="#1a1b1e", dark="#0e0f11", text="#e3e5e8",
        field="#1e1f22", field_text="#e3e5e8", select="#35507a", select_text="#ffffff",
        disabled="#72767d", trough="#232428", emboss=False, muted="#9aa0a6",
        well="#1e1f22", tile="#383a40", speech="#5b9bff", working="#ff6b6b"),
    "Terminal": dict(
        face="#050805", light="#39ff6a", shadow="#15522a", dark="#23a045", text="#39ff6a",
        field="#0b140c", field_text="#39ff6a", select="#15522a", select_text="#39ff6a",
        disabled="#1b6b31", trough="#0b140c", emboss=False, muted="#23a045",
        well="#0b140c", tile="#0f1f11", speech="#39ff6a", working="#ffffff",
        font=("Fixedsys", 9), mono_icons=True),  # green phosphor: one typeface, one colour
}
for _scheme in SCHEMES.values():  # every other scheme has well == field, so this is a no-op there
    _scheme.setdefault("well_text", _scheme["field_text"])
    _scheme.setdefault("well_muted", _scheme["muted"])
del _scheme
SCHEME_GROUPS = [["Neon Dusk"], ["Windows Standard", "High Contrast Black", "Eggplant", "Brick"],
                 ["Light", "Dark", "Terminal"]]
DEFAULT_SCHEME = "Neon Dusk"

# 8-bit icons. "k" is the outline and takes the scheme's text colour; "." is transparent.
PIXEL_COLORS = {"w": "#ffffff", "s": "#c0c0c0", "g": "#808080", "r": "#ff0000",
                "m": "#800000", "G": "#00c000", "y": "#ffff00", "o": "#c08000",
                "b": "#0000ff", "n": "#000080", "c": "#00d4ff"}
SPRITES = {
    "play": ["........",
             "kk......",
             "kGkk....",
             "kGGGkk..",
             "kGGGGGk.",
             "kGGGkk..",
             "kGkk....",
             "kk......"],
    "stop": ["........",
             ".kkkkkk.",
             ".kwrrrk.",
             ".krrrrk.",
             ".krrrrk.",
             ".krrrmk.",
             ".kkkkkk.",
             "........"],
    "folder": ["........",
               ".kkk....",
               "kyyykkkk",
               "kywwwwwk",
               "kyyyyyyk",
               "kyyyyyyk",
               "koooooyk",
               "kkkkkkkk"],
    "note": ["...kk...",
             "...kkk..",
             "...k.kk.",
             "...k..k.",
             "...k....",
             ".kkk....",
             "kkkk....",
             ".kk....."],
    "cassette": ["................",
                 "................",
                 "kkkkkkkkkkkkkkkk",
                 "ksssssssssssssgk",
                 "ksrrrrrrrrrrrrgk",
                 "kswwkkwwwwkkwwgk",
                 "kswkwwknnkwwkwgk",
                 "kswkwwknnkwwkwgk",
                 "kswwkkwwwwkkwwgk",
                 "kswwwwwwwwwwwwgk",
                 "ksssskkkkkkssssk",
                 "ksssksggggskssgk",
                 "kgggkgggggggkggk",
                 "kkkkkkkkkkkkkkkk",
                 "................",
                 "................"],
    # section-header icons (8x8, monochrome silhouettes drawn with mono=)
    "doc": [".kkkkk..",
            ".k...k..",
            ".k....k.",
            ".k....k.",
            ".k....k.",
            ".k....k.",
            ".k....k.",
            ".kkkkkk."],
    "chip": ["..k..k..",
             "..k..k..",
             "kkkkkkkk",
             "k......k",
             "k......k",
             "k......k",
             "kkkkkkkk",
             "..k..k.."],
    "wrench": ["kk......",
               "kkk.....",
               ".kkk....",
               "..kkk...",
               "...kkk..",
               "....kkk.",
               ".....kkk",
               "......kk"],
    "chart": ["........",
              "......k.",
              "......k.",
              "....k.k.",
              "....k.k.",
              "..k.k.k.",
              "..k.k.k.",
              "kkkkkkk."],
    "trash": [".kkkkk..",
              "..kkk...",
              "kkkkkkk.",
              ".k...k..",
              ".k.k.k..",
              ".k.k.k..",
              ".k.k.k..",
              ".kkkkk.."],
    # sidebar nav icons (9x9, monochrome silhouettes)
    "home": ["....k....",
             "...kkk...",
             "..kkkkk..",
             ".kkkkkkk.",
             "kkkkkkkkk",
             "kk.....kk",
             "kk.kkk.kk",
             "kk.kkk.kk",
             "kkkkkkkkk"],
    "wave": [".........",
             "....k....",
             "....k....",
             "..k.k.k..",
             "..k.k.k..",
             "k.k.k.k.k",
             "k.k.k.k.k",
             "k.k.k.k.k",
             "k.k.k.k.k"],
    "clock": ["..kkkkk..",
              ".k.....k.",
              "k...k...k",
              "k...k...k",
              "k...kk..k",
              "k.......k",
              "k.......k",
              ".k.....k.",
              "..kkkkk.."],
    "gear": ["..k...k..",
             ".kkk.kkk.",
             "kk..k..kk",
             "k..kkk..k",
             "kk.k.k.kk",
             "k..kkk..k",
             "kk..k..kk",
             ".kkk.kkk.",
             "..k...k.."],
    "question": [".kkkkk...",
                 "kk...kk..",
                 "k.....k..",
                 "....kk...",
                 "...kk....",
                 "..kk.....",
                 "..kk.....",
                 ".........",
                 "..kk....."],
    # decorative panel art
    "computer_face": [".kkkkkkkk.",
                       "kcccccccck",
                       "kc.k..k.ck",
                       "kc.kkkk.ck",
                       "kccccccck.",
                       "kkkkkkkkkk",
                       ".kssssssk.",
                       "..kssssk..",
                       ".kkkkkkkk."],
    "plant": ["..kk..",
              ".kGGk.",
              "kGGGGk",
              ".kGGk.",
              "..kk..",
              ".koook",
              "..kkk."],
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


def hhmmss(seconds):
    """H:MM:SS, always with an hour, for the file-info strip."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


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


def media_duration(ffmpeg, src, timeout=20):
    """Length in seconds from ffmpeg's header dump, or 0 if it can't be read."""
    try:
        proc = subprocess.run([ffmpeg, "-hide_banner", "-i", src], capture_output=True,
                              creationflags=NO_WINDOW, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 0
    m = re.search(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        return 0
    h, mins, s = m.groups()
    return int(h) * 3600 + int(mins) * 60 + float(s)


def cuda_available():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def play_sound(name):
    """Play one of the classic Windows sounds (tada, chord, ding) if Windows still has it."""
    if sys.platform != "win32":
        return
    import winsound
    wav = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Media" / f"{name}.wav"
    try:
        if wav.is_file():
            winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.MessageBeep()
    except RuntimeError:
        pass


def animations_enabled():
    """False when Windows is set to show fewer animations (Settings > Accessibility)."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        on = ctypes.c_int(1)
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(on), 0)  # SPI_GETCLIENTAREAANIMATION
        return bool(on.value)
    except Exception:
        return True


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


HISTORY_PATH = Path.home() / ".local-transcriber-history.json"
HISTORY_LIMIT = 50


def load_history():
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(entries):
    try:
        HISTORY_PATH.write_text(json.dumps(entries[:HISTORY_LIMIT], indent=2), encoding="utf-8")
    except Exception:
        pass


class Cancelled(Exception):
    pass


# ── Workers ──────────────────────────────────────────────────────────────────
class Worker(threading.Thread):
    """Background task that reports back to the UI through a queue."""

    def __init__(self, opts, events):
        super().__init__(daemon=True)
        self.opts, self.events = opts, events
        self.cancelled = threading.Event()

    def emit(self, kind, **data):
        self.events.put((kind, data))


class Job(Worker):
    """Runs one transcription."""

    def __init__(self, opts, events):
        super().__init__(opts, events)
        self.segments = []  # kept on the job so a failed run can still save them

    def prepare_audio(self, src, boost, workdir):
        """Decode to 16 kHz mono WAV with ffmpeg, optionally levelling quiet speech."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on PATH. Install it, or turn off "
                               "'Boost quiet audio' and use a Whisper model.")
        out = Path(workdir) / "audio.wav"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000"]
        if boost:
            cmd += ["-af", LEVELLER]
        cmd.append(str(out))
        # stderr goes to a file so a chatty ffmpeg can't fill the pipe and stall
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stderr=err, creationflags=NO_WINDOW)
            while proc.poll() is None:
                if self.cancelled.is_set():
                    proc.kill()
                    proc.wait()
                    raise Cancelled()
                time.sleep(0.2)
            if proc.returncode != 0:
                err.seek(0)
                raise RuntimeError("ffmpeg could not read this file:\n"
                                   + err.read().decode(errors="ignore").strip())
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
        self.emit("started", duration=info.duration or 0,
                  summary=f"{device.upper()} · language "
                          f"{LANG_NAME.get(info.language, info.language)} "
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
        self.emit("started", duration=duration, summary=f"{device.upper()} · language auto")
        stream = ((s.start, s.end, s.text) for s in model.with_vad(vad).recognize(audio))
        return device, "auto", duration, stream

    def _attempt(self, device):
        """One full pass on one device. Segments land in self.segments as they arrive."""
        o, spec = self.opts, self.opts["spec"]
        language = spec.language or o["language"]
        lang, duration = None, 0
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
                self.segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
                self.emit("progress", position=end, start=start, text=f"[{short(start)}] {text}")
        return device, lang, duration

    def _save(self, lang, duration, device, partial):
        o = self.opts
        src = Path(o["file"])
        meta = {
            "file": src.name, "language": lang, "model": o["model_id"], "engine": o["spec"].engine,
            "device": device, "task": o["task"], "duration": duration, "partial": partial,
        }
        stem = src.stem + ("_partial" if partial else "")
        written = write_outputs(self.segments, meta, Path(o["out_dir"]), stem, o["formats"])
        return [str(p) for p in written]

    def run(self):
        device = self.opts["device"]
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        lang, duration = None, 0
        try:
            try:
                device, lang, duration = self._attempt(device)
            except Cancelled:
                raise
            except Exception as e:
                # cuDNN and friends usually fail on the first inference, not at load time,
                # so the fallback inside run_whisper can't catch them
                if device != "cuda" or self.segments:
                    raise
                self.emit("log", text=f"GPU unavailable ({e}); retrying on CPU.")
                device, lang, duration = self._attempt("cpu")
        except Cancelled:
            pass
        except Exception as e:
            # never throw away what was already transcribed
            files = self._save(lang, duration, device, partial=True) if self.segments else []
            self.emit("error", text=f"{type(e).__name__}: {e}", files=files)
            return

        partial = self.cancelled.is_set()
        if partial and not self.segments:
            self.emit("done", files=[], partial=True, count=0)
            return
        self.emit("done", files=self._save(lang, duration, device, partial),
                  partial=partial, count=len(self.segments))


class Convert(Worker):
    """Saves the audio track of one file (e.g. a screen recording) with ffmpeg."""

    def run(self):
        o = self.opts
        src = Path(o["file"])
        out = Path(o["out_dir"]) / f"{src.stem}.{o['format']}"
        if out.resolve() == src.resolve():  # never overwrite the input
            out = out.with_name(f"{src.stem}_audio.{o['format']}")
        try:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg was not found on PATH. Install it to extract audio.")
            codec, uses_bitrate = AUDIO_FORMATS[o["format"]]
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                   "-progress", "pipe:1", "-i", str(src), "-vn", *codec]
            if uses_bitrate:
                cmd += ["-b:a", o["bitrate"]]
            if o["boost"]:
                cmd += ["-af", LEVELLER]
            cmd.append(str(out))
            out.parent.mkdir(parents=True, exist_ok=True)

            self.emit("status", text=f"Extracting audio to {o['format'].upper()}…")
            self.emit("started", duration=media_duration(ffmpeg, str(src)), summary=f"saving {out.name}")
            # stderr goes to a file so a chatty ffmpeg can't fill the pipe and stall
            with tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err,
                                        creationflags=NO_WINDOW)
                for line in proc.stdout:  # "key=value" progress lines, about twice a second
                    if self.cancelled.is_set():
                        proc.kill()
                        break
                    key, _, value = line.decode(errors="ignore").strip().partition("=")
                    if key == "out_time_us" and value.isdigit():
                        self.emit("progress", position=int(value) / 1e6)
                proc.wait()
                if self.cancelled.is_set():
                    out.unlink(missing_ok=True)
                    self.emit("done", files=[], partial=True)
                    return
                if proc.returncode != 0:
                    out.unlink(missing_ok=True)
                    err.seek(0)
                    raise RuntimeError("ffmpeg could not extract the audio:\n"
                                       + err.read().decode(errors="ignore").strip())
        except Exception as e:
            self.emit("error", text=f"{type(e).__name__}: {e}")
            return
        self.emit("done", files=[str(out)], partial=False)


# ── UI ───────────────────────────────────────────────────────────────────────
LABEL_W = 10                                          # width of inline field labels
MAP_ROWS, MAP_TILE, MAP_GAP, MAP_MARGIN = 3, 11, 2, 4  # transcript map geometry, in pixels
MAP_HEIGHT = 2 * MAP_MARGIN + MAP_ROWS * (MAP_TILE + MAP_GAP) - MAP_GAP
SIDEBAR_W = 152
NAV_ITEMS = [("transcribe", "Transcribe", "home"), ("extract", "Extract Audio", "wave"),
             ("history", "History", "clock"), ("settings", "Settings", "gear"),
             ("about", "About", "question")]
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


def sprite(master, rows, zoom, outline="#000000", emboss=None, mono=None, pad=1):
    """PhotoImage from a pixel-art grid. emboss=(light, shadow) greys it out like a disabled Win95
    icon; mono=fill colour draws every non-transparent pixel in one flat colour (a silhouette)."""
    img = tk.PhotoImage(master=master, width=len(rows[0]) + pad, height=len(rows) + pad)
    pixels = [(x, y, ch) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch != "."]
    if emboss:
        for x, y, _ in pixels:
            img.put(emboss[0], (x + 1, y + 1))
        for x, y, _ in pixels:
            img.put(emboss[1], (x, y))
    else:
        for x, y, ch in pixels:
            img.put(outline if ch == "k" else mono or PIXEL_COLORS[ch], (x, y))
    return img.zoom(zoom)


class Sunken(tk.Frame):
    """Win95 sunken border (1 or 2 pixels deep) made of nested frames. Content goes in .body."""

    # colour of each ring, and the side it shows on: (0, 1) = bottom/right, (1, 0) = top/left
    LAYERS = [("light", (0, 1)), ("shadow", (1, 0)), ("face", (0, 1)), ("dark", (1, 0))]

    def __init__(self, parent, depth=2, body="field"):
        super().__init__(parent, bd=0, highlightthickness=0)
        self.rings, self.body_key = [], body
        frame = self
        for key, pad in self.LAYERS[: depth * 2]:
            self.rings.append((frame, key))
            frame = tk.Frame(frame, bd=0, highlightthickness=0)
            frame.pack(fill="both", expand=True, padx=pad, pady=pad)
        self.body = frame

    def recolor(self, t):
        for frame, key in self.rings:
            frame.configure(background=t[key])
        self.body.configure(background=t[self.body_key])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Local Transcriber v{VERSION}")
        self.geometry("1180x900")
        self.minsize(1080, 820)

        self.config_data = load_config()
        self.events = queue.Queue()
        self.job = None
        self.duration = 0
        self.started_at = None
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.combos, self.sunkens, self.icon_buttons, self.icons = [], [], [], {}
        self.section_icons = []       # (label, sprite_key) awaiting a theme colour
        self.meter_frac = {}          # key -> last fraction drawn, so a resize can redraw it
        self.meter_canvas = {}
        self.t = SCHEMES[DEFAULT_SCHEME]
        # Tk's named fonts: reconfiguring them restyles every widget that uses them,
        # entries and lists included
        self.ui_font = tkfont.nametofont("TkDefaultFont")
        self.text_font = tkfont.nametofont("TkTextFont")
        self.bold_font = tkfont.Font(self, font=UI_FONT, weight="bold")

        # transcript map: how far the job has read, and where it found speech
        self.map_mode, self.map_active, self.map_reached, self.map_speech = "transcribe", False, 0, []
        self.map_tiles, self.map_drawn, self.map_x0 = [], [], 0
        self.blink_on, self.blink_after = True, None
        self.animate = animations_enabled()

        self.history = load_history()
        self.history_rows = {}
        self.active_nav = "transcribe"
        self.status_state = "idle"
        self.probed_path = None
        self._titlebar_done = False
        self._closing = False

        self.style = ttk.Style(self)
        self.style.theme_use("alt")  # Tk's Windows 95 look: bevels, sunken fields, dotted focus
        cfg = self.config_data
        saved_scheme = cfg.get("scheme")
        self.scheme_var = tk.StringVar(value=saved_scheme if saved_scheme in SCHEMES else DEFAULT_SCHEME)
        self.sound_var = tk.BooleanVar(value=cfg.get("sounds", True))
        self.default_out_var = tk.StringVar(value=cfg.get("default_out", ""))
        self.app_icons = [sprite(self, SPRITES["cassette"], zoom, pad=0) for zoom in (1, 2, 4)]
        self.iconphoto(True, *self.app_icons[:2])

        self._build_menu()
        self._build()
        self._apply_scheme()
        self._on_model_change()
        self._on_audio_format_change()
        self._update_file_info()
        self._nav_click("transcribe")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Map>", self._on_map_once)
        self.after(150, self._drain_events)
        self.after(250, self._sample_resources)

    # ── fixed chrome: title bar, header ribbon, sidebar ─────────────────────
    def _on_map_once(self, _event=None):
        if not self._titlebar_done:
            self._titlebar_done = True
            self._style_titlebar()

    def _style_titlebar(self):
        """Best-effort: paint the native title bar navy to match the app (Windows 11 only)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = int(self.wm_frame(), 16)
            dwmapi = ctypes.windll.dwmapi

            def set_attr(attr, value):
                v = ctypes.c_int(value)
                dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

            def bgr(hex_color):
                r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
                return (b << 16) | (g << 8) | r

            set_attr(20, 1)                      # DWMWA_USE_IMMERSIVE_DARK_MODE
            set_attr(35, bgr(CHROME["bg"]))       # DWMWA_CAPTION_COLOR
            set_attr(36, bgr(CHROME["text"]))     # DWMWA_TEXT_COLOR
            set_attr(33, 1)                       # DWMWA_WINDOW_CORNER_PREFERENCE = DO_NOT_ROUND
        except Exception:
            pass  # older Windows, or DWM refused — the app still works, just unstyled

    def _build_menu(self):
        bar = tk.Menu(self)
        file = tk.Menu(bar, tearoff=False)
        file.add_command(label="Open recording...", underline=0, accelerator="Ctrl+O",
                         command=self._pick_file)
        file.add_command(label="Choose output folder...", underline=7, command=self._pick_out)
        file.add_command(label="Open output folder", underline=5, command=self._open_out)
        file.add_separator()
        file.add_command(label="Exit", underline=1, command=self._close)
        bar.add_cascade(label="File", underline=0, menu=file)

        view = tk.Menu(bar, tearoff=False)
        schemes = tk.Menu(view, tearoff=False)
        for i, names in enumerate(SCHEME_GROUPS):
            if i:
                schemes.add_separator()
            for name in names:
                schemes.add_radiobutton(label=name, value=name, variable=self.scheme_var,
                                        command=self._apply_scheme)
        view.add_cascade(label="Color scheme", underline=0, menu=schemes)
        view.add_checkbutton(label="Sounds", underline=0, variable=self.sound_var,
                             command=self._save_prefs)
        bar.add_cascade(label="View", underline=0, menu=view)

        tools = tk.Menu(bar, tearoff=False)
        tools.add_command(label="Extract audio", underline=0, command=lambda: self._nav_click("extract"))
        tools.add_command(label="View history", underline=5, command=lambda: self._nav_click("history"))
        tools.add_separator()
        tools.add_command(label="Clear live transcript", underline=0, command=self._clear_log)
        bar.add_cascade(label="Tools", underline=0, menu=tools)

        helpm = tk.Menu(bar, tearoff=False)
        helpm.add_command(label="About Local Transcriber", underline=0, command=self._about)
        bar.add_cascade(label="Help", underline=0, menu=helpm)
        self.configure(menu=bar)
        self.bind_all("<Control-o>", lambda _: self._pick_file())

    def _build(self):
        # status bar first, then the header ribbon, so the body fills whatever is left
        self._build_status(self)
        self._build_ribbon(self)

        body = tk.Frame(self, bd=0, highlightthickness=0)
        body.pack(side="top", fill="both", expand=True)
        self._build_sidebar(body)

        content = tk.Frame(body, bd=0, highlightthickness=0)
        content.pack(side="left", fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        page = self._build_page_transcribe(content)
        page.grid(row=0, column=0, sticky="nsew")
        self.pages = {"transcribe": page, "extract": page,
                      "history": self._build_page_history(content),
                      "settings": self._build_page_settings(content)}
        for key, pg in self.pages.items():
            if key not in ("transcribe", "extract"):
                pg.grid(row=0, column=0, sticky="nsew")

    def _build_ribbon(self, parent):
        c = CHROME
        ribbon = tk.Frame(parent, background=c["bg"], height=38)
        ribbon.pack(side="top", fill="x")
        ribbon.pack_propagate(False)
        left = tk.Frame(ribbon, background=c["bg"])
        left.pack(side="left", padx=10)
        tk.Label(left, image=self.app_icons[1], background=c["bg"]).pack(side="left", pady=6)
        title_font = tkfont.Font(self, family="Fixedsys", size=12)
        tk.Label(left, text="LOCAL TRANSCRIBER", background=c["bg"], foreground=c["text"],
                 font=title_font).pack(side="left", padx=(8, 6))
        badge = Sunken(left, depth=1, body="face")
        badge.rings[0][0].configure(background=c["accent"])
        badge.rings[1][0].configure(background=c["bg2"])
        badge.body.configure(background=c["bg2"])
        tk.Label(badge.body, text=f"v{VERSION}", background=c["bg2"], foreground=c["accent2"],
                 font=("Fixedsys", 8)).pack(padx=4)
        badge.pack(side="left")
        tk.Label(ribbon, text=TAGLINE, background=c["bg"], foreground=c["muted"],
                 font=("Fixedsys", 8)).pack(side="right", padx=12)

    def _build_status(self, parent):
        status = self._sunken(parent, depth=1, body="face")
        status.pack(side="bottom", fill="x")
        row = status.body
        left = ttk.Frame(row)
        left.pack(side="left", fill="x", expand=True, padx=6, pady=2)
        self.status_dot = tk.Label(left, text="●", font=UI_FONT)
        self.status_dot.pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, padding=(4, 0)).pack(side="left")
        ttk.Label(left, text="  |  ", style="Hint.TLabel").pack(side="left")
        self.status_summary_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_summary_var, style="Hint.TLabel").pack(side="left")

        right = ttk.Frame(row)
        right.pack(side="right", padx=6, pady=2)
        self.eta_var = tk.StringVar(value="ETA —")
        self.speed_var = tk.StringVar(value="Speed —")
        self.elapsed_var = tk.StringVar(value="Elapsed —")
        for i, var in enumerate((self.eta_var, self.speed_var, self.elapsed_var)):
            if i:
                ttk.Label(right, text="  |  ", style="Hint.TLabel").pack(side="left")
            ttk.Label(right, textvariable=var, style="Hint.TLabel").pack(side="left")

    def _build_sidebar(self, parent):
        c = CHROME
        sidebar = tk.Frame(parent, background=c["bg"], width=SIDEBAR_W)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self.nav_icon_imgs = {key: (sprite(self, SPRITES[icon], 3, mono=c["muted"]),
                                    sprite(self, SPRITES[icon], 3, mono=c["accent"]))
                              for key, _, icon in NAV_ITEMS}
        self.nav_frames = {}
        tk.Frame(sidebar, background=c["bg"], height=8).pack(side="top")
        for key, label, _ in NAV_ITEMS:
            row = tk.Frame(sidebar, background=c["bg"])
            row.pack(side="top", fill="x", pady=1)
            border = tk.Frame(row, background=c["bg"], width=3)
            border.pack(side="left", fill="y")
            inner = tk.Frame(row, background=c["bg"])
            inner.pack(side="left", fill="x", expand=True, pady=7)
            icon_lbl = tk.Label(inner, image=self.nav_icon_imgs[key][0], background=c["bg"])
            icon_lbl.pack()
            text_lbl = tk.Label(inner, text=label, background=c["bg"], foreground=c["muted"],
                                font=("Fixedsys", 8))
            text_lbl.pack(pady=(3, 0))
            self.nav_frames[key] = (row, border, inner, icon_lbl, text_lbl)
            for w in (row, border, inner, icon_lbl, text_lbl):
                w.bind("<Button-1>", lambda _e, k=key: self._nav_click(k))
                w.configure(cursor="hand2")

        tk.Frame(sidebar, background=c["bg"]).pack(side="top", fill="both", expand=True)
        skyline = tk.Canvas(sidebar, width=SIDEBAR_W, height=76, background=c["bg"],
                            bd=0, highlightthickness=0)
        skyline.pack(side="top")
        self._draw_skyline(skyline)
        tags = tk.Frame(sidebar, background=c["bg"])
        tags.pack(side="top", fill="x", pady=(4, 10))
        for tag in ("LOCAL", "PRIVATE", "OPEN SOURCE", "FOREVER ♥"):
            tk.Label(tags, text=f"❘ {tag}", background=c["bg"], foreground=c["muted"],
                     font=("Fixedsys", 7)).pack(anchor="w", padx=14)

    def _draw_skyline(self, c):
        bg = CHROME
        c.create_rectangle(0, 0, SIDEBAR_W, 76, fill=bg["bg"], width=0)
        for x, y, r in ((18, 14, 1), (34, 22, 1), (120, 12, 1), (100, 26, 1), (140, 18, 1)):
            c.create_oval(x - r, y - r, x + r, y + r, fill=bg["muted"], width=0)
        c.create_oval(70, 8, 92, 30, fill="#e7e6f0", width=0)
        c.create_oval(64, 8, 84, 28, fill=bg["bg"], width=0)  # crescent bite
        c.create_polygon(0, 76, 0, 46, 30, 20, 60, 50, 90, 30, 90, 76,
                         fill="#2a2750", width=0)
        c.create_polygon(50, 76, 50, 40, 85, 15, 115, 42, 140, 20, 152, 38, 152, 76,
                         fill="#201d40", width=0)
        for bx, bw, bh in ((14, 14, 26), (36, 10, 34), (110, 16, 30), (132, 12, 22)):
            c.create_rectangle(bx, 76 - bh, bx + bw, 76, fill="#151330", width=0)
            for wy in range(76 - bh + 5, 72, 8):
                c.create_rectangle(bx + 3, wy, bx + 6, wy + 3, fill=bg["accent2"], width=0)

    # ── page: Transcribe / Extract Audio ────────────────────────────────────
    def _section_header(self, frame, icon_key, text):
        lbl = ttk.Label(frame, text=f" {text}", compound="left", style="SectionTitle.TLabel")
        self.section_icons.append((lbl, icon_key))
        return lbl

    def _labelframe(self, parent, icon_key, title, **grid_kw):
        lf = ttk.LabelFrame(parent, padding=(8, 4, 8, 6))
        lf.configure(labelwidget=self._section_header(lf, icon_key, title))
        if grid_kw:
            lf.grid(**grid_kw)
        return lf

    def _build_page_transcribe(self, parent):
        cell = {"sticky": "w", "pady": 2}
        page = ttk.Frame(parent, padding=(10, 8, 10, 6))

        top = ttk.Frame(page)
        top.pack(fill="x")
        top.columnconfigure(0, weight=1, uniform="col")
        top.columnconfigure(1, weight=1, uniform="col")

        # 1. Select File
        files_lf = self._labelframe(top, "doc", "1. Select File", row=0, column=0,
                                    sticky="nsew", padx=(0, 5), pady=(0, 6))
        self.file_var = tk.StringVar()
        self.out_var = tk.StringVar()
        ent = ttk.Entry(files_lf, textvariable=self.file_var)
        ent.grid(row=0, column=0, sticky="ew")
        ent.bind("<Return>", lambda _: self._update_file_info())
        ent.bind("<FocusOut>", lambda _: self._update_file_info())
        ttk.Button(files_lf, text="Browse...", command=self._pick_file).grid(row=0, column=1, padx=(6, 0))
        files_lf.columnconfigure(0, weight=1)
        info = self._sunken(files_lf, depth=1, body="well")
        info.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        info_row = ttk.Frame(info.body, style="Well.TFrame")
        info_row.pack(fill="x", padx=6, pady=5)
        self.finfo_icon = tk.Label(info_row)
        self.finfo_icon.pack(side="left", padx=(0, 8))
        finfo_text = ttk.Frame(info_row, style="Well.TFrame")
        finfo_text.pack(side="left", fill="x", expand=True)
        self.fname_var = tk.StringVar()
        self.fmeta_var = tk.StringVar()
        ttk.Label(finfo_text, textvariable=self.fname_var, style="WellBold.TLabel").pack(anchor="w")
        ttk.Label(finfo_text, textvariable=self.fmeta_var, style="WellHint.TLabel").pack(anchor="w")

        # 2. Model & Language
        model_lf = self._labelframe(top, "chip", "2. Model & Language", row=0, column=1,
                                    sticky="nsew", padx=(5, 0), pady=(0, 6))
        cfg = self.config_data
        ttk.Label(model_lf, text="Model:", width=LABEL_W).grid(row=0, column=0, **cell)
        saved_model = cfg.get("model")
        self.model_var = tk.StringVar(value=saved_model if saved_model in MODEL_BY_LABEL else MODELS[0].label)
        cb = self._combo(model_lf, self.model_var, [m.label for m in MODELS], 30)
        cb.grid(row=0, column=1, columnspan=3, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_model_change())
        model_lf.columnconfigure(1, weight=1)

        ttk.Label(model_lf, text="Custom:", width=LABEL_W).grid(row=1, column=0, **cell)
        self.custom_var = tk.StringVar(value=cfg.get("custom_model", ""))
        self.custom_entry = ttk.Entry(model_lf, textvariable=self.custom_var)
        self.custom_entry.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(model_lf, text="Language:", width=LABEL_W).grid(row=2, column=0, **cell)
        saved_lang = cfg.get("language")
        self.lang_var = tk.StringVar(value=saved_lang if saved_lang in dict(LANGUAGES) else "Auto-detect")
        self.lang_combo = self._combo(model_lf, self.lang_var, [l for l, _ in LANGUAGES], 10)
        self.lang_combo.grid(row=2, column=1, sticky="w")

        ttk.Label(model_lf, text="Task:").grid(row=3, column=0, sticky="w", pady=2)
        self.task_var = tk.StringVar(value="transcribe")
        self.task_combo = self._combo(model_lf, self.task_var, ["transcribe", "translate"], 10)
        self.task_combo.grid(row=3, column=1, sticky="w")
        ttk.Label(model_lf, text="Beam:").grid(row=3, column=2, sticky="w", padx=(10, 4))
        self.beam_var = tk.IntVar(value=5)
        self.beam_spin = ttk.Spinbox(model_lf, from_=1, to=10, width=3, textvariable=self.beam_var)
        self.beam_spin.grid(row=3, column=3, sticky="w")

        self.model_note = tk.StringVar()
        self.model_note_label = ttk.Label(model_lf, textvariable=self.model_note, style="Hint.TLabel",
                                          wraplength=320, justify="left")
        self.model_note_label.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # 3. Options
        opts_lf = self._labelframe(top, "wrench", "3. Options", row=1, column=0,
                                   sticky="nsew", padx=(0, 5))
        ttk.Label(opts_lf, text="Device:", width=LABEL_W).grid(row=0, column=0, **cell)
        dev = ttk.Frame(opts_lf)
        dev.grid(row=0, column=1, sticky="w")
        self.device_var = tk.StringVar(value="auto")
        ttk.Radiobutton(dev, text="Auto", value="auto", variable=self.device_var).pack(side="left")
        gpu_btn = ttk.Radiobutton(dev, text=f"GPU{f' ({_GPU_NAME})' if _GPU_NAME else ''}",
                                  value="cuda", variable=self.device_var)
        gpu_btn.pack(side="left", padx=10)
        if not cuda_available():
            gpu_btn.state(["disabled"])
        ttk.Radiobutton(dev, text="CPU", value="cpu", variable=self.device_var).pack(side="left")

        ttk.Label(opts_lf, text="Audio:", width=LABEL_W).grid(row=1, column=0, **cell)
        aud = ttk.Frame(opts_lf)
        aud.grid(row=1, column=1, sticky="w")
        self.vad_var = tk.BooleanVar(value=True)
        self.vad_check = ttk.Checkbutton(aud, text="Skip silence (VAD)", variable=self.vad_var)
        self.vad_check.pack(side="left")
        self.boost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aud, text="Boost quiet audio", variable=self.boost_var).pack(side="left", padx=(12, 0))

        ttk.Label(opts_lf, text="Formats:", width=LABEL_W).grid(row=2, column=0, **cell)
        fmt = ttk.Frame(opts_lf)
        fmt.grid(row=2, column=1, sticky="w")
        saved_formats = cfg.get("formats", ["txt", "srt"])
        self.fmt_vars = {f: tk.BooleanVar(value=f in saved_formats) for f in FORMATS}
        for f, v in self.fmt_vars.items():
            ttk.Checkbutton(fmt, text=f.upper(), variable=v, command=self._save_prefs).pack(
                side="left", padx=(0, 5))

        ttk.Label(opts_lf, text="Vocabulary:", width=LABEL_W).grid(row=3, column=0, sticky="nw", pady=2)
        self.prompt_var = tk.StringVar(value=cfg.get("vocabulary", ""))
        self.prompt_entry = ttk.Entry(opts_lf, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(opts_lf, text="Names and terms to spell correctly (Whisper models)",
                 style="Hint.TLabel").grid(row=4, column=1, sticky="w")
        opts_lf.columnconfigure(1, weight=1)

        # 4. Audio Extraction
        extract_lf = self._labelframe(top, "note", "4. Audio Extraction (Optional)", row=1, column=1,
                                      sticky="nsew", padx=(5, 0))
        ttk.Label(extract_lf, text="Format:", width=LABEL_W).grid(row=0, column=0, **cell)
        saved_afmt = cfg.get("audio_format")
        self.afmt_var = tk.StringVar(value=saved_afmt if saved_afmt in AUDIO_FORMATS else "mp3")
        cb = self._combo(extract_lf, self.afmt_var, list(AUDIO_FORMATS), 6)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_audio_format_change())
        ttk.Label(extract_lf, text="Bitrate:").grid(row=0, column=2, sticky="w", padx=(10, 4))
        saved_rate = cfg.get("bitrate")
        self.bitrate_var = tk.StringVar(value=saved_rate if saved_rate in BITRATES else "192k")
        self.bitrate_combo = self._combo(extract_lf, self.bitrate_var, BITRATES, 6)
        self.bitrate_combo.grid(row=0, column=3, sticky="w")
        self.bitrate_combo.bind("<<ComboboxSelected>>", lambda _: self._save_prefs())
        ttk.Checkbutton(extract_lf, text="Boost quiet audio", variable=self.boost_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.convert_btn = self._icon_button(extract_lf, "Extract Audio", "note", self._convert)
        self.convert_btn.grid(row=2, column=0, columnspan=4, sticky="e", pady=(8, 0))
        extract_lf.columnconfigure(3, weight=1)

        # decorative panel, spanning both rows
        self._build_decor_panel(top).grid(row=0, column=2, rowspan=2, sticky="ns", padx=(10, 0))

        # 5. Output Folder
        out_lf = self._labelframe(page, "folder", "5. Output Folder")
        out_lf.pack(fill="x", pady=(0, 6))
        ttk.Entry(out_lf, textvariable=self.out_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_lf, text="Browse...", command=self._pick_out).grid(row=0, column=1, padx=(6, 0))
        out_lf.columnconfigure(0, weight=1)

        # actions
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(0, 6))
        self.start_btn = self._icon_button(actions, "Transcribe", "play", self._start,
                                           default="active", style="Go.TButton")
        self.start_btn.pack(side="left")
        self.cancel_btn = self._icon_button(actions, "Stop", "stop", self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self.clear_btn = self._icon_button(actions, "Clear", "trash", self._clear_all)
        self.clear_btn.pack(side="right")
        self._icon_button(actions, "Open Output Folder", "folder", self._open_out).pack(
            side="right", padx=(0, 6))

        # transcript map
        map_lf = self._labelframe(page, "chart", "Transcript Map")
        map_lf.pack(fill="x", pady=(0, 6))
        head = ttk.Frame(map_lf)
        head.pack(fill="x")
        ttk.Label(head, text="Visual overview of where speech occurs in the recording.",
                 style="Hint.TLabel").pack(side="left")
        self.legend = tk.Canvas(head, height=MAP_TILE + 4, bd=0, highlightthickness=0)
        self.legend.pack(side="right")
        well = self._sunken(map_lf, depth=2, body="well")
        well.pack(fill="x", pady=(6, 0))
        self.map_canvas = tk.Canvas(well.body, height=MAP_HEIGHT, bd=0, highlightthickness=0)
        self.map_canvas.pack(fill="x")
        self.map_canvas.bind("<Configure>", self._layout_map)

        # bottom split: live transcript | system monitor
        bottom = ttk.Frame(page)
        bottom.pack(fill="both", expand=True)
        log_lf = self._labelframe(bottom, "doc", "Live Transcript")
        log_lf.pack(side="left", fill="both", expand=True, padx=(0, 6))
        well = self._sunken(log_lf, depth=2, body="field")
        well.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(well.body)
        scroll.pack(side="right", fill="y")
        self.log = tk.Text(well.body, height=8, wrap="word", state="disabled", relief="flat", bd=0,
                           highlightthickness=0, padx=4, pady=2, yscrollcommand=scroll.set)
        scroll.configure(command=self.log.yview)
        self.log.tag_configure("path", wrap="char")  # long paths have no spaces to break at
        self.log.pack(fill="both", expand=True)

        mon_wrap = tk.Frame(bottom, width=200)
        mon_wrap.pack(side="right", fill="y")
        mon_wrap.pack_propagate(False)
        mon_lf = self._labelframe(mon_wrap, "chart", "System Monitor")
        mon_lf.pack(fill="both", expand=True)
        self.res_vars = {k: tk.StringVar() for k in ("cpu", "ram", "gpu", "vram")}
        for key, label in (("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"), ("vram", "VRAM")):
            row = ttk.Frame(mon_lf)
            row.pack(fill="x", pady=4)
            top_row = ttk.Frame(row)
            top_row.pack(fill="x")
            ttk.Label(top_row, text=label, width=5).pack(side="left")
            ttk.Label(top_row, textvariable=self.res_vars[key], style="Hint.TLabel").pack(side="right")
            meter = self._sunken(row, depth=1, body="well")
            meter.pack(fill="x", pady=(2, 0))
            canvas = tk.Canvas(meter.body, height=10, bd=0, highlightthickness=0)
            canvas.pack(fill="x")
            canvas.bind("<Configure>", lambda _e, k=key: self._set_meter(k, self.meter_frac.get(k, 0)))
            self.meter_canvas[key] = canvas
            self.meter_frac[key] = 0.0

        return page

    def _build_decor_panel(self, parent):
        p = PANEL
        outer = tk.Frame(parent, background=p["bg"], width=190)
        outer.pack_propagate(False)
        stars = tk.Frame(outer, background=p["bg"])
        stars.pack(fill="x", pady=(8, 0))
        tk.Label(stars, text="✦", background=p["bg"], foreground=p["accent2"],
                 font=("Fixedsys", 10)).pack(side="left", padx=(12, 0))
        tk.Label(stars, text="✦", background=p["bg"], foreground=p["accent"],
                 font=("Fixedsys", 8)).pack(side="right", padx=(0, 14))

        bullets = tk.Frame(outer, background=p["bg"])
        bullets.pack(fill="x", padx=14, pady=(6, 10))
        for line in ("NO CLOUD.", "NO DATA LEAKS.", "JUST YOUR MACHINE."):
            tk.Label(bullets, text=f"■ {line}", background=p["bg"], foreground=p["text"],
                     font=("Fixedsys", 8), anchor="w").pack(fill="x", pady=1)

        art = tk.Frame(outer, background=p["bg"])
        art.pack(pady=4)
        # keep references — an unreferenced PhotoImage is garbage-collected and vanishes
        self._panel_imgs = [sprite(self, SPRITES["computer_face"], 6, mono=p["accent"]),
                            sprite(self, SPRITES["plant"], 5, mono=p["accent2"])]
        tk.Label(art, image=self._panel_imgs[0], background=p["bg"]).pack(side="left")
        tk.Label(art, image=self._panel_imgs[1], background=p["bg"]).pack(
            side="left", padx=(4, 0), anchor="s")

        tk.Frame(outer, background=p["bg"]).pack(fill="both", expand=True)
        foot = tk.Frame(outer, background=p["bg"])
        foot.pack(fill="x", pady=(0, 12))
        tk.Label(foot, text="GOOD TRANSCRIPTS", background=p["bg"], foreground=p["accent"],
                 font=("Fixedsys", 8), justify="center").pack()
        tk.Label(foot, text="BETTER IDEAS ♥", background=p["bg"], foreground=p["accent"],
                 font=("Fixedsys", 8), justify="center").pack()
        return outer

    # ── page: History ────────────────────────────────────────────────────────
    def _build_page_history(self, parent):
        page = ttk.Frame(parent, padding=(10, 8, 10, 6))
        head = ttk.Frame(page)
        head.pack(fill="x", pady=(0, 6))
        self._section_header_standalone(head, "clock", "History").pack(side="left")
        ttk.Label(head, text="   Recent transcription and extraction jobs.",
                 style="Hint.TLabel").pack(side="left")
        ttk.Button(head, text="Clear history", command=self._clear_history).pack(side="right")

        wrap = self._sunken(page, depth=2, body="field")
        wrap.pack(fill="both", expand=True)
        cols = ("time", "action", "file", "status")
        self.history_tree = ttk.Treeview(wrap.body, columns=cols, show="headings", height=18)
        for col, label, w in (("time", "Time", 130), ("action", "Action", 140),
                              ("file", "File", 340), ("status", "Status", 220)):
            self.history_tree.heading(col, text=label)
            self.history_tree.column(col, width=w, anchor="w")
        scroll = ttk.Scrollbar(wrap.body, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.history_tree.pack(fill="both", expand=True)
        self.history_tree.bind("<Double-1>", self._history_open)
        ttk.Label(page, text="Double-click a row to open its output folder.",
                 style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
        self._refresh_history()
        return page

    def _section_header_standalone(self, parent, icon_key, text):
        """A section title with an icon, for pages that have no LabelFrame to hang it on."""
        return self._section_header(parent, icon_key, text)

    def _refresh_history(self):
        if not hasattr(self, "history_tree"):
            return
        self.history_tree.delete(*self.history_tree.get_children())
        self.history_rows = {}
        for entry in self.history:
            iid = self.history_tree.insert("", "end", values=(
                entry.get("time", ""), entry.get("action", ""),
                entry.get("file", ""), entry.get("status", "")))
            self.history_rows[iid] = entry

    def _history_open(self, _event):
        sel = self.history_tree.selection()
        if not sel:
            return
        entry = self.history_rows.get(sel[0])
        outputs = entry.get("outputs") if entry else None
        if not outputs:
            return
        folder = str(Path(outputs[0]).parent)
        if Path(folder).is_dir():
            os.startfile(folder) if sys.platform == "win32" else subprocess.Popen(["xdg-open", folder])

    def _clear_history(self):
        if self.history and messagebox.askyesno("Transcriber", "Clear the job history?"):
            self.history = []
            save_history(self.history)
            self._refresh_history()

    def _record_history(self, action, file, outputs, status):
        entry = {"time": time.strftime("%Y-%m-%d %H:%M"), "action": action,
                 "file": Path(file).name if file else "—", "outputs": outputs, "status": status}
        self.history.insert(0, entry)
        self.history = self.history[:HISTORY_LIMIT]
        save_history(self.history)
        self._refresh_history()

    # ── page: Settings ──────────────────────────────────────────────────────
    def _build_page_settings(self, parent):
        page = ttk.Frame(parent, padding=(10, 8, 10, 6))
        self._section_header_standalone(page, "gear", "Settings").pack(anchor="w", pady=(0, 8))

        appearance = self._labelframe(page, "chart", "Appearance")
        appearance.pack(fill="x", pady=(0, 6))
        group_names = {0: "Signature", 1: "Windows classics", 2: "Modern"}
        col = 0
        for i, names in enumerate(SCHEME_GROUPS):
            ttk.Label(appearance, text=group_names.get(i, ""), style="Hint.TLabel").grid(
                row=0, column=col, sticky="w", padx=(0 if col == 0 else 20, 0))
            for r, name in enumerate(names, start=1):
                ttk.Radiobutton(appearance, text=name, value=name, variable=self.scheme_var,
                               command=self._apply_scheme).grid(
                    row=r, column=col, sticky="w", padx=(0 if col == 0 else 20, 0), pady=1)
            col += 1

        behavior = self._labelframe(page, "wrench", "Behavior")
        behavior.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(behavior, text="Play a sound when a job finishes", variable=self.sound_var,
                       command=self._save_prefs).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(behavior, text="Default output folder:", width=20).grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(behavior, textvariable=self.default_out_var).grid(
            row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(behavior, text="Browse...", command=self._pick_default_out).grid(
            row=1, column=2, padx=(6, 0), pady=(6, 0))
        ttk.Label(behavior, text="Used to fill Output Folder when it's empty.",
                 style="Hint.TLabel").grid(row=2, column=1, sticky="w")
        behavior.columnconfigure(1, weight=1)

        ttk.Label(page, text=f"Preferences: {CONFIG_PATH}\nHistory: {HISTORY_PATH}",
                 style="Hint.TLabel", justify="left").pack(anchor="w", pady=(6, 0))
        return page

    def _pick_default_out(self):
        path = filedialog.askdirectory()
        if path:
            self.default_out_var.set(path)
            self._save_prefs()

    # ── navigation ───────────────────────────────────────────────────────────
    def _nav_click(self, key):
        if key == "about":
            self._about()
            return
        self.active_nav = key
        self._restyle_nav()
        self.pages[key].tkraise()

    def _restyle_nav(self):
        c = CHROME
        for key, (row, border, inner, icon_lbl, text_lbl) in self.nav_frames.items():
            active = key == self.active_nav
            bg = c["active"] if active else c["bg"]
            border.configure(background=c["accent"] if active else c["bg"])
            for w in (row, inner, icon_lbl, text_lbl):
                w.configure(background=bg)
            icon_lbl.configure(image=self.nav_icon_imgs[key][1 if active else 0])
            text_lbl.configure(foreground=c["text"] if active else c["muted"])

    # ── colour scheme (content area only — sidebar/ribbon/panel stay fixed) ─
    def _apply_scheme(self):
        t = self.t = SCHEMES.get(self.scheme_var.get(), SCHEMES[DEFAULT_SCHEME])
        s = self.style
        family, size = t.get("font", UI_FONT)
        for f in (self.ui_font, self.text_font, self.bold_font):
            f.configure(family=family, size=size)
        s.configure(".", background=t["face"], foreground=t["text"], font=self.ui_font,
                    fieldbackground=t["field"], selectbackground=t["select"],
                    selectforeground=t["select_text"], insertcolor=t["field_text"],
                    bordercolor=t["dark"], lightcolor=t["light"], darkcolor=t["shadow"],
                    troughcolor=t["trough"], arrowcolor=t["text"], focuscolor=t["text"],
                    indicatorcolor=t["field"])
        s.configure("TLabelframe.Label", foreground=t["text"])
        s.configure("SectionTitle.TLabel", font=self.bold_font, foreground=t["text"])
        s.configure("Hint.TLabel", foreground=t["muted"])
        s.configure("WellBold.TLabel", font=self.bold_font, foreground=t["well_text"],
                    background=t["well"])
        s.configure("WellHint.TLabel", foreground=t["well_muted"], background=t["well"])
        s.configure("Well.TFrame", background=t["well"])
        # "well" (a scheme's extreme background) reliably contrasts against "speech" — unlike
        # field_text, which Terminal deliberately sets equal to speech (one colour, on purpose)
        s.configure("Go.TButton", background=t["speech"], foreground=t["well"], font=self.bold_font)
        s.map("Go.TButton", background=[("disabled", t["face"]), ("pressed", t["speech"]),
                                        ("active", t["speech"])],
              foreground=[("disabled", t["disabled"])])
        emboss = t.get("emboss", True)
        s.map(".", foreground=[("disabled", t["disabled"])], embossed=[("disabled", int(emboss))],
              background=[("disabled", t["face"]), ("active", t["face"])])
        # Win95 controls don't light up on hover, and disabled ones keep the face colour
        s.map("TButton", background=[("disabled", t["face"]), ("pressed", t["face"]), ("active", t["face"])],
              foreground=[("disabled", t["disabled"])])
        for widget in ("TCheckbutton", "TRadiobutton"):  # alt sets these per style, over "."
            s.configure(widget, indicatorcolor=t["field"], foreground=t["text"])
            s.map(widget, background=[("active", t["face"])], foreground=[("disabled", t["disabled"])],
                  indicatorcolor=[("pressed", t["face"]), ("disabled", t["face"])])
        for widget in ("TEntry", "TCombobox", "TSpinbox"):
            s.configure(widget, foreground=t["field_text"], fieldbackground=t["field"])
            s.map(widget, fieldbackground=[("disabled", t["face"]), ("readonly", t["field"])],
                  foreground=[("disabled", t["disabled"])],
                  background=[("active", t["face"]), ("pressed", t["face"])],
                  arrowcolor=[("disabled", t["disabled"])])
        s.map("TScrollbar", background=[("active", t["face"]), ("pressed", t["face"])],
              arrowcolor=[("disabled", t["disabled"])])
        s.configure("Treeview", background=t["field"], fieldbackground=t["field"],
                    foreground=t["field_text"])
        s.map("Treeview", background=[("selected", t["select"])],
              foreground=[("selected", t["select_text"])])
        s.configure("Treeview.Heading", background=t["face"], foreground=t["text"])
        s.map("Treeview.Heading", background=[("active", t["face"])])

        self.configure(background=t["face"])
        for well in self.sunkens:
            well.recolor(t)
        self.log.configure(background=t["field"], foreground=t["field_text"], insertbackground=t["field_text"],
                           selectbackground=t["select"], selectforeground=t["select_text"], font=LOG_FONT)
        self.map_canvas.configure(background=t["well"])
        for opt, key in (("background", "field"), ("foreground", "field_text"),
                         ("selectBackground", "select"), ("selectForeground", "select_text")):
            self.option_add(f"*TCombobox*Listbox.{opt}", t[key])
        self.option_add("*TCombobox*Listbox.font", self.ui_font)
        for cb in self.combos:  # popdowns that already exist ignore option_add
            try:
                popdown = self.tk.eval(f"ttk::combobox::PopdownWindow {cb}")
                self.tk.call(f"{popdown}.f.l", "configure", "-background", t["field"],
                             "-foreground", t["field_text"], "-selectbackground", t["select"],
                             "-selectforeground", t["select_text"], "-font", self.ui_font)
            except tk.TclError:
                pass
        for name in ("play", "stop", "folder", "note", "trash"):
            self.icons[name] = sprite(self, SPRITES[name], 2, outline=t["text"],
                                      mono=t["shadow"] if t.get("mono_icons") else None)
            self.icons[name + "_off"] = sprite(self, SPRITES[name], 2,
                                               emboss=(t["light"] if emboss else t["face"], t["shadow"]))
        for button, name in self.icon_buttons:
            button.configure(image=(self.icons[name], "disabled", self.icons[name + "_off"]))
        for label, key in self.section_icons:
            img = sprite(self, SPRITES[key], 2, mono=t["text"])
            self.icons.setdefault("_section", {})[key] = img  # keep a reference alive
            label.configure(image=img)
        self._update_file_icon()
        for key, frac in self.meter_frac.items():
            self._set_meter(key, frac)
        self.map_drawn = []
        self._paint_map()
        self._draw_legend()
        self._update_status_dot()
        self._save_prefs()

    # ── pixel meters (System Monitor) ───────────────────────────────────────
    def _set_meter(self, key, frac):
        self.meter_frac[key] = frac
        canvas = self.meter_canvas.get(key)
        if not canvas:
            return
        canvas.delete("fill")
        w = canvas.winfo_width()
        if w <= 1:
            return
        h = canvas.winfo_height() or 10
        canvas.create_rectangle(0, 0, max(2, int(w * frac)), h, fill=self.t["speech"],
                                width=0, tags="fill")

    # ── transcript map: a Disk Defragmenter-style grid covering the whole media ─
    @staticmethod
    def _draw_tile(canvas, x, y):
        """A raised block: fill, a light top-left edge and a shadow bottom-right edge."""
        e = MAP_TILE - 1
        return (canvas.create_rectangle(x, y, x + MAP_TILE, y + MAP_TILE, width=0),
                canvas.create_line(x, y + e, x, y, x + e, y),
                canvas.create_line(x + e, y, x + e, y + e, x - 1, y + e))

    def _paint_tile(self, canvas, ids, state):
        t = self.t
        if state == "silent":  # read, but nobody spoke: an empty slot
            fill = light = shadow = t["well"]
        else:
            fill, light, shadow = t["tile" if state == "waiting" else state], t["light"], t["shadow"]
        for item, color in zip(ids, (fill, light, shadow)):
            canvas.itemconfigure(item, fill=color)

    def _layout_map(self, _event=None):
        c = self.map_canvas
        width = c.winfo_width()
        pitch = MAP_TILE + MAP_GAP
        cols = max(8, (width - 2 * MAP_MARGIN + MAP_GAP) // pitch)
        x0 = (width - (cols * pitch - MAP_GAP)) // 2
        if len(self.map_tiles) == cols * MAP_ROWS:
            c.move("all", x0 - self.map_x0, 0)
        else:
            c.delete("all")
            self.map_tiles = [self._draw_tile(c, x0 + col * pitch, MAP_MARGIN + row * pitch)
                              for row in range(MAP_ROWS) for col in range(cols)]
            self.map_drawn = []
            self._paint_map()
        self.map_x0 = x0

    def _map_states(self):
        n = len(self.map_tiles)
        if not self.duration:
            return ["waiting"] * n
        span = self.duration / n
        speech = [False] * n
        for start, end in self.map_speech:
            for i in range(int(start / span), min(int(end / span), n - 1) + 1):
                speech[i] = True
        read = min(int(self.map_reached / span), n)  # blocks read to the end
        states = []
        for i in range(n):
            if i < read:
                states.append("speech" if speech[i] or self.map_mode == "extract" else "silent")
            elif i == read and self.map_active:
                states.append("working" if self.blink_on else "waiting")
            else:
                states.append("speech" if speech[i] else "waiting")
        return states

    def _paint_map(self):
        states = self._map_states()
        for i, (ids, state) in enumerate(zip(self.map_tiles, states)):
            if i >= len(self.map_drawn) or self.map_drawn[i] != state:
                self._paint_tile(self.map_canvas, ids, state)
        self.map_drawn = states

    def _draw_legend(self):
        c, t = self.legend, self.t
        c.delete("all")
        c.configure(background=t["face"])
        if self.map_mode == "extract":
            items = [("speech", "Extracted"), ("working", "Processing"), ("waiting", "Not reached")]
        else:
            items = [("speech", "Speech"), ("silent", "No speech"), ("working", "Processing"),
                     ("waiting", "Not reached")]
        x, mid = 2, (MAP_TILE + 4) // 2
        for state, label in items:
            c.create_rectangle(x - 2, 0, x + MAP_TILE + 1, MAP_TILE + 3, fill=t["well"], outline=t["shadow"])
            self._paint_tile(c, self._draw_tile(c, x, 2), state)
            c.create_text(x + MAP_TILE + 6, mid, text=label, anchor="w", font=self.ui_font, fill=t["text"])
            x += MAP_TILE + 6 + self.ui_font.measure(label) + 16
        c.configure(width=x)

    def _blink(self):
        self.blink_after = None
        if self.map_active:
            self.blink_on = not self.blink_on
            self._paint_map()
            self.blink_after = self.after(450, self._blink)

    # ── model-dependent controls ─────────────────────────────────────────────
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
        self.model_note_label.grid() if spec.note else self.model_note_label.grid_remove()
        self._save_prefs()

    def _on_audio_format_change(self):
        lossy = AUDIO_FORMATS[self.afmt_var.get()][1]
        self.bitrate_combo.state(["!disabled", "readonly"] if lossy else ["disabled"])
        self._save_prefs()

    # ── file selection ───────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(filetypes=MEDIA_TYPES)
        if path:
            self.file_var.set(path)
            if not self.out_var.get():
                self.out_var.set(self.default_out_var.get() or str(Path(path).parent))
            self._update_file_info()

    def _update_file_icon(self):
        if not hasattr(self, "finfo_icon"):
            return
        ext = Path(self.file_var.get()).suffix.lower()
        key = "doc" if ext in VIDEO_EXT or not ext else "note"
        img = sprite(self, SPRITES[key], 3, mono=self.t["well_muted"])
        self.icons.setdefault("_finfo", {})[key] = img
        self.finfo_icon.configure(image=img, background=self.t["well"])

    def _update_file_info(self):
        path = self.file_var.get().strip()
        if path == self.probed_path:  # focus-out fires often; only probe real changes
            return
        self.probed_path = path
        p = Path(path)
        if not path or not p.is_file():
            self.fname_var.set("No file selected")
            self.fmeta_var.set("Choose a recording to see its details.")
            self._update_file_icon()
            return
        try:
            size = human_size(p.stat().st_size)
        except OSError:
            size = "—"
        ext = p.suffix.upper().lstrip(".")
        self.fname_var.set(p.name)
        self.fmeta_var.set(f"{ext}  ·  reading…  ·  {size}")
        self._update_file_icon()
        # ffmpeg can take a moment (or hang on a dead network path), so ask off the UI thread
        threading.Thread(target=self._probe_duration, args=(path, ext, size), daemon=True).start()

    def _probe_duration(self, path, ext, size):
        seconds = 0
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                seconds = media_duration(ffmpeg, path)
            except Exception:
                pass
        self.events.put(("probe", {"path": path, "text": f"{ext}  ·  "
                                   f"{hhmmss(seconds) if seconds else '—'}  ·  {size}"}))

    def _on_probe(self, path, text):
        if path == self.file_var.get().strip():  # ignore a probe the user has moved on from
            self.fmeta_var.set(text)

    def _pick_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _open_out(self):
        folder = self.out_var.get()
        if folder and Path(folder).is_dir():
            os.startfile(folder) if sys.platform == "win32" else subprocess.Popen(["xdg-open", folder])

    # ── run / convert ────────────────────────────────────────────────────────
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
        out_dir = self.out_var.get().strip() or self.default_out_var.get() or str(Path(file).parent)
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
        self._launch(Job(opts, self.events))

    def _convert(self):
        file = self.file_var.get().strip()
        if not file or not Path(file).is_file():
            return messagebox.showwarning("Transcriber", "Choose an audio or video file first.")
        out_dir = self.out_var.get().strip() or self.default_out_var.get() or str(Path(file).parent)
        self.out_var.set(out_dir)
        opts = {"file": file, "out_dir": out_dir, "format": self.afmt_var.get(),
                "bitrate": self.bitrate_var.get(), "boost": self.boost_var.get()}
        self._launch(Convert(opts, self.events))

    def _launch(self, job):
        self._save_prefs()
        self._clear_log()
        self.duration, self.started_at = 0, time.time()
        self.map_mode = "extract" if isinstance(job, Convert) else "transcribe"
        self.map_active, self.map_reached, self.map_speech, self.blink_on = True, 0, [], True
        self._paint_map()
        self._draw_legend()
        self.eta_var.set("ETA —")
        self.speed_var.set("Speed —")
        self.elapsed_var.set("Elapsed 00:00")
        self.status_state = "running"
        self._update_status_dot()
        if self.animate and not self.blink_after:
            self.blink_after = self.after(450, self._blink)
        self.start_btn.state(["disabled"])
        self.convert_btn.state(["disabled"])
        self.clear_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.job = job
        self.job.start()

    def _cancel(self):
        if self.job:
            self.job.cancelled.set()
            self.status_var.set("Stopping — saving what has been transcribed so far…"
                                if isinstance(self.job, Job) else "Stopping…")
            self.cancel_btn.state(["disabled"])

    def _clear_all(self):
        if self.job:
            return
        self._clear_log()
        self.duration, self.map_reached, self.map_speech = 0, 0, []
        self.map_active = False
        self._paint_map()
        self.status_var.set("Ready")
        self.status_state = "idle"
        self._update_status_dot()
        self.eta_var.set("ETA —")
        self.speed_var.set("Speed —")
        self.elapsed_var.set("Elapsed —")

    def _update_status_dot(self):
        if not hasattr(self, "status_dot"):
            return
        color = {"idle": self.t.get("muted", self.t["disabled"]), "running": self.t["speech"],
                 "error": self.t["working"]}.get(self.status_state, self.t["text"])
        self.status_dot.configure(background=self.t["face"], foreground=color)

    def _update_status_summary(self):
        if not hasattr(self, "fmt_vars"):
            return
        fmts = ",".join(f.upper() for f, v in self.fmt_vars.items() if v.get()) or "none"
        model_short = self.model_var.get().split(" — ")[0]
        self.status_summary_var.set(
            f"Model: {model_short}   Device: {self.device_var.get()}   "
            f"Language: {self.lang_var.get()}   Formats: {fmts}")

    def _save_prefs(self):
        if not hasattr(self, "afmt_var"):  # still building the window
            return
        save_config({
            "scheme": self.scheme_var.get(), "sounds": self.sound_var.get(),
            "model": self.model_var.get(), "language": self.lang_var.get(),
            "custom_model": self.custom_var.get(),
            "formats": [f for f, v in self.fmt_vars.items() if v.get()],
            "vocabulary": self.prompt_var.get(),
            "audio_format": self.afmt_var.get(), "bitrate": self.bitrate_var.get(),
            "default_out": self.default_out_var.get(),
        })
        self._update_status_summary()

    def _close(self):
        if self._closing:  # already stopping; a second click on X shouldn't re-ask
            return
        self._save_prefs()
        if not (self.job and self.job.is_alive()):
            self.destroy()
            return
        if not messagebox.askyesno(
                "Transcriber", "A job is still running.\n\n"
                "Stop it and save what has been transcribed so far?"):
            return
        # the worker is a daemon thread: destroying the window now would kill it
        # mid-write, so wait for it to finish saving first
        self._closing = True
        self.job.cancelled.set()
        self.status_var.set("Stopping — saving what has been transcribed so far…")
        self.cancel_btn.state(["disabled"])
        self._wait_then_close()

    def _wait_then_close(self, waited=0):
        self._drain_once()  # let the worker's final "done" land, so history records it
        if self.job is None or not self.job.is_alive() or waited > 20000:
            self._save_prefs()
            self.destroy()
            return
        self.after(100, lambda: self._wait_then_close(waited + 100))

    def _about(self):
        prev = self.active_nav
        self.active_nav = "about"
        self._restyle_nav()

        win = tk.Toplevel(self, background=self.t["face"])
        win.title("About Local Transcriber")
        win.resizable(False, False)
        win.transient(self)
        body = ttk.Frame(win, padding=(14, 14, 14, 10))
        body.pack(fill="both", expand=True)
        ttk.Label(body, image=self.app_icons[2]).grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 14))
        ttk.Label(body, text=f"Local Transcriber v{VERSION}", font=self.bold_font).grid(
            row=0, column=1, sticky="w")
        ttk.Label(body, text=TAGLINE, style="Hint.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(body, text="Transcribes audio and video on this computer.\nNothing is uploaded.",
                  justify="left").grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Label(body, text="Speech: faster-whisper and onnx-asr (Parakeet)\nAudio: ffmpeg",
                  justify="left").grid(row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Separator(body).grid(row=4, column=1, sticky="ew", pady=10)
        memory = psutil.virtual_memory().total / 2**30
        ttk.Label(body, text=f"Physical memory: {memory:.0f} GB\n"
                             f"Graphics: {_GPU_NAME or 'no NVIDIA GPU found'}",
                  justify="left").grid(row=5, column=1, sticky="w")

        def _close_about():
            win.destroy()
            self.active_nav = prev
            self._restyle_nav()

        ok = ttk.Button(body, text="OK", width=10, default="active", command=_close_about)
        ok.grid(row=6, column=1, sticky="e", pady=(12, 0))
        win.bind("<Return>", lambda _: _close_about())
        win.bind("<Escape>", lambda _: _close_about())
        win.protocol("WM_DELETE_WINDOW", _close_about)
        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{x}+{y}")
        ok.focus_set()
        win.grab_set()

    # ── event loop ───────────────────────────────────────────────────────────
    def _drain_once(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                getattr(self, f"_on_{kind}")(**data)
        except queue.Empty:
            pass

    def _drain_events(self):
        self._drain_once()
        self.after(150, self._drain_events)

    def _on_status(self, text):
        self.status_var.set(text)

    def _on_log(self, text):
        self._append(text)

    def _on_started(self, duration, summary):
        self.duration = duration
        self.started_at = time.time()
        self._append(f"— {short(duration)} of media · {summary} —")
        self._paint_map()

    def _on_progress(self, position, text=None, start=None):
        if text:
            self._append(text)
        if start is not None:
            self.map_speech.append((start, position))
        self.map_reached = max(self.map_reached, position)
        self._paint_map()
        if not self.duration:
            return
        elapsed = time.time() - self.started_at
        speed = position / elapsed if elapsed > 0 else 0
        eta = (self.duration - position) / speed if speed > 0 else 0
        self.eta_var.set(f"ETA {short(eta)}")
        self.speed_var.set(f"Speed {speed:.1f}×")

    def _on_done(self, files, partial, count=None):
        action = "Extracted audio" if isinstance(self.job, Convert) else "Transcribed"
        job_file = self.job.opts.get("file") if self.job else self.file_var.get()
        self.map_active = False
        if not partial and self.duration:
            self.map_reached = self.duration
        self._paint_map()
        elapsed = short(time.time() - (self.started_at or time.time()))
        self.elapsed_var.set(f"Elapsed {elapsed}")
        self.eta_var.set("ETA —")
        self.speed_var.set("Speed —")
        if partial:
            state = "Stopped — partial transcript saved" if files else "Stopped"
        else:
            state = "Done"
        segments = f" · {count} segments" if count is not None else ""
        self.status_var.set(f"{state}{segments} · {elapsed}")
        self.status_state = "idle"
        self._update_status_dot()
        if files:
            self._append("Saved:")
            for f in files:
                self._append(f"  {f}", "path")
        if not partial and self.sound_var.get():
            play_sound("tada")
        self._record_history(action, job_file, files, state)
        self._reset_buttons()

    def _on_error(self, text, files=None):
        files = files or []
        action = "Extracted audio" if isinstance(self.job, Convert) else "Transcribed"
        job_file = self.job.opts.get("file") if self.job else self.file_var.get()
        self.map_active = False
        self._paint_map()
        self.status_var.set("Failed — partial transcript saved" if files else "Failed")
        self.status_state = "error"
        self._update_status_dot()
        self.eta_var.set("ETA —")
        self.speed_var.set("Speed —")
        self._append(f"ERROR: {text}")
        if files:
            self._append("Saved what was transcribed before the error:")
            for f in files:
                self._append(f"  {f}", "path")
        self._record_history(action, job_file, files, "Failed — partial saved" if files else "Failed")
        self._reset_buttons()
        messagebox.showerror("Transcriber", text)

    def _reset_buttons(self):
        self.job = None
        self.start_btn.state(["!disabled"])
        self.convert_btn.state(["!disabled"])
        self.clear_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

    # ── resources ────────────────────────────────────────────────────────────
    def _sample_resources(self):
        try:
            cores = psutil.cpu_count() or 1
            proc_cpu = min(self.proc.cpu_percent(None) / cores, 100)
            sys_cpu = psutil.cpu_percent(None)
            self.res_vars["cpu"].set(f"{sys_cpu:.0f}%")
            self._set_meter("cpu", sys_cpu / 100)
            vm = psutil.virtual_memory()
            rss = self.proc.memory_info().rss / 2**30
            self.res_vars["ram"].set(f"{rss:.1f} / {vm.total / 2**30:.0f} GB")
            self._set_meter("ram", vm.percent / 100)
            if _GPU_HANDLE is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                self.res_vars["gpu"].set(f"{util}%")
                self._set_meter("gpu", util / 100)
                self.res_vars["vram"].set(f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.0f} GB")
                self._set_meter("vram", mem.used / mem.total if mem.total else 0)
            else:
                self.res_vars["gpu"].set("n/a")
                self.res_vars["vram"].set("n/a")
                self._set_meter("gpu", 0)
                self._set_meter("vram", 0)
            if self.job is not None and self.started_at:
                self.elapsed_var.set(f"Elapsed {short(time.time() - self.started_at)}")
        except Exception:
            pass
        self.after(250, self._sample_resources)

    # ── log ──────────────────────────────────────────────────────────────────
    def _append(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── small widget factories used throughout the pages above ──────────────
    def _combo(self, parent, var, values, width):
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", width=width, values=values)
        self.combos.append(cb)
        return cb

    def _sunken(self, parent, depth=2, body="field"):
        well = Sunken(parent, depth, body)
        self.sunkens.append(well)
        return well

    def _icon_button(self, parent, text, icon, command, **kw):
        button = ttk.Button(parent, text=text, command=command, compound="left", **kw)
        self.icon_buttons.append((button, icon))
        return button


if __name__ == "__main__":
    App().mainloop()
