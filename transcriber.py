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


# ── Look: Windows 95 ─────────────────────────────────────────────────────────
UI_FONT = ("MS Sans Serif", 8)  # a scheme can swap it with a "font" entry
LOG_FONT = ("Fixedsys", 9)

# Colour schemes after the Win95 Appearance tab. face/light/shadow/dark are the four
# shades of a 3D bevel; well/tile/speech/working colour the transcript map.
SCHEMES = {
    "Windows Standard": dict(
        face="#c0c0c0", light="#ffffff", shadow="#808080", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#000080", select_text="#ffffff",
        disabled="#808080", trough="#e0e0e0",
        well="#ffffff", tile="#c0c0c0", speech="#0000ff", working="#ff0000"),
    "High Contrast Black": dict(
        face="#000000", light="#ffffff", shadow="#808080", dark="#ffffff", text="#ffffff",
        field="#000000", field_text="#ffffff", select="#800080", select_text="#ffffff",
        disabled="#808080", trough="#404040", emboss=False,  # a white emboss reads as enabled here
        well="#000000", tile="#808080", speech="#00ffff", working="#ffff00"),
    "Eggplant": dict(
        face="#90b0a8", light="#d8e4e0", shadow="#587870", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#584078", select_text="#ffffff",
        disabled="#587870", trough="#b8ccc8",
        well="#ffffff", tile="#90b0a8", speech="#800080", working="#ff0000"),
    "Brick": dict(
        face="#c2bfa5", light="#e8e6d8", shadow="#817e6a", dark="#000000", text="#000000",
        field="#ffffff", field_text="#000000", select="#800000", select_text="#ffffff",
        disabled="#817e6a", trough="#dcdac8",
        well="#ffffff", tile="#c2bfa5", speech="#800000", working="#008000"),
    # the app's earlier Light, Dark and Terminal colours, given Win95 bevels
    "Light": dict(
        face="#f3f4f6", light="#ffffff", shadow="#9ca3af", dark="#4b5563", text="#1f2328",
        field="#ffffff", field_text="#1f2328", select="#2563eb", select_text="#ffffff",
        disabled="#9ca3af", trough="#e5e7eb",
        well="#ffffff", tile="#e5e7eb", speech="#2563eb", working="#dc2626"),
    "Dark": dict(
        face="#2b2d31", light="#4e5058", shadow="#1a1b1e", dark="#0e0f11", text="#e3e5e8",
        field="#1e1f22", field_text="#e3e5e8", select="#35507a", select_text="#ffffff",
        disabled="#72767d", trough="#232428", emboss=False,
        well="#1e1f22", tile="#383a40", speech="#5b9bff", working="#ff6b6b"),
    "Terminal": dict(
        face="#050805", light="#39ff6a", shadow="#15522a", dark="#23a045", text="#39ff6a",
        field="#0b140c", field_text="#39ff6a", select="#15522a", select_text="#39ff6a",
        disabled="#1b6b31", trough="#0b140c", emboss=False,
        well="#0b140c", tile="#0f1f11", speech="#39ff6a", working="#ffffff",
        font=("Fixedsys", 9), mono_icons=True),  # green phosphor: one typeface, one colour
}
SCHEME_GROUPS = [["Windows Standard", "High Contrast Black", "Eggplant", "Brick"],
                 ["Light", "Dark", "Terminal"]]
DEFAULT_SCHEME = "Windows Standard"

# 8-bit icons. "k" is the outline and takes the scheme's text colour; "." is transparent.
PIXEL_COLORS = {"w": "#ffffff", "s": "#c0c0c0", "g": "#808080", "r": "#ff0000",
                "m": "#800000", "G": "#00c000", "y": "#ffff00", "o": "#c08000",
                "b": "#0000ff", "n": "#000080"}
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


def media_duration(ffmpeg, src):
    """Length in seconds from ffmpeg's header dump, or 0 if it can't be read."""
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", src],
                          capture_output=True, creationflags=NO_WINDOW)
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
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
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
                    self.emit("progress", position=end, start=start, text=f"[{short(start)}] {text}")
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
LABEL_W = 14                                          # width of the left-hand field labels
MAP_ROWS, MAP_TILE, MAP_GAP, MAP_MARGIN = 8, 9, 2, 4  # transcript map geometry, in pixels
MAP_HEIGHT = 2 * MAP_MARGIN + MAP_ROWS * (MAP_TILE + MAP_GAP) - MAP_GAP


def sprite(master, rows, zoom, outline="#000000", emboss=None, mono=None, pad=1):
    """PhotoImage from a pixel-art grid. emboss=(light, shadow) greys it out like a disabled Win95
    icon; mono=fill colour draws it in two colours, outline and fill."""
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
        self.title("Local Transcriber")
        self.geometry("840x900")
        self.minsize(760, 800)

        self.config_data = load_config()
        self.events = queue.Queue()
        self.job = None
        self.duration = 0
        self.started_at = None
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.combos, self.sunkens, self.icon_buttons, self.icons = [], [], [], {}
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

        self.style = ttk.Style(self)
        self.style.theme_use("alt")  # Tk's Windows 95 look: bevels, sunken fields, dotted focus
        cfg = self.config_data
        saved_scheme = cfg.get("scheme")
        self.scheme_var = tk.StringVar(value=saved_scheme if saved_scheme in SCHEMES else DEFAULT_SCHEME)
        self.sound_var = tk.BooleanVar(value=cfg.get("sounds", True))
        self.app_icons = [sprite(self, SPRITES["cassette"], zoom, pad=0) for zoom in (1, 2, 4)]
        self.iconphoto(True, *self.app_icons[:2])

        self._build_menu()
        self._build()
        self._apply_scheme()
        self._on_model_change()
        self._on_audio_format_change()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(150, self._drain_events)
        self.after(250, self._sample_resources)

    # layout
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

        helpm = tk.Menu(bar, tearoff=False)
        helpm.add_command(label="About Local Transcriber", underline=0, command=self._about)
        bar.add_cascade(label="Help", underline=0, menu=helpm)
        self.configure(menu=bar)
        self.bind_all("<Control-o>", lambda _: self._pick_file())

    def _build(self):
        cfg = self.config_data
        group = {"fill": "x", "padx": 8, "pady": (0, 6)}
        inner = (8, 2, 8, 6)
        cell = {"sticky": "w", "pady": 2}
        root = ttk.Frame(self, padding=(0, 6, 0, 0))
        root.pack(fill="both", expand=True)

        # status bar first, so it keeps its place when the window gets short
        status = ttk.Frame(root)
        status.pack(side="bottom", fill="x", padx=2, pady=(0, 2))
        ttk.Sizegrip(status).pack(side="right", anchor="se")
        self.status_var = tk.StringVar(value="Ready")
        self.res_vars = {k: tk.StringVar() for k in ("cpu", "ram", "gpu", "vram")}
        for key, width in (("vram", 15), ("gpu", 13), ("ram", 21), ("cpu", 17)):
            pane = self._sunken(status, depth=1, body="face")
            pane.pack(side="right", padx=(2, 0))
            ttk.Label(pane.body, textvariable=self.res_vars[key], width=width, padding=(4, 1)).pack()
        pane = self._sunken(status, depth=1, body="face")
        pane.pack(side="left", fill="x", expand=True)
        ttk.Label(pane.body, textvariable=self.status_var, padding=(4, 1)).pack(fill="x")

        files = ttk.LabelFrame(root, text="Files", padding=inner)
        files.pack(**group)
        self.file_var = tk.StringVar()
        self.out_var = tk.StringVar()
        for row, (label, var, cmd) in enumerate([
            ("Recording:", self.file_var, self._pick_file),
            ("Output folder:", self.out_var, self._pick_out),
        ]):
            ttk.Label(files, text=label, width=LABEL_W).grid(row=row, column=0, **cell)
            ttk.Entry(files, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
            ttk.Button(files, text="Browse...", command=cmd).grid(row=row, column=2, padx=(6, 0), pady=2)
        files.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(root, text="Model", padding=inner)
        opts.pack(**group)

        ttk.Label(opts, text="Model:", width=LABEL_W).grid(row=0, column=0, **cell)
        saved_model = cfg.get("model")
        self.model_var = tk.StringVar(value=saved_model if saved_model in MODEL_BY_LABEL else MODELS[0].label)
        cb = self._combo(opts, self.model_var, [m.label for m in MODELS], 60)
        cb.grid(row=0, column=1, columnspan=5, **cell)
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_model_change())

        ttk.Label(opts, text="Custom model:").grid(row=1, column=0, **cell)
        self.custom_var = tk.StringVar(value=cfg.get("custom_model", ""))
        self.custom_entry = ttk.Entry(opts, textvariable=self.custom_var)
        self.custom_entry.grid(row=1, column=1, columnspan=5, sticky="ew", pady=2)

        ttk.Label(opts, text="Language:").grid(row=2, column=0, **cell)
        saved_lang = cfg.get("language")
        self.lang_var = tk.StringVar(value=saved_lang if saved_lang in dict(LANGUAGES) else "Auto-detect")
        self.lang_combo = self._combo(opts, self.lang_var, [l for l, _ in LANGUAGES], 14)
        self.lang_combo.grid(row=2, column=1, **cell)

        ttk.Label(opts, text="Task:").grid(row=2, column=2, sticky="w", padx=(16, 6))
        self.task_var = tk.StringVar(value="transcribe")
        self.task_combo = self._combo(opts, self.task_var, ["transcribe", "translate"], 11)
        self.task_combo.grid(row=2, column=3, sticky="w")

        ttk.Label(opts, text="Beam size:").grid(row=2, column=4, sticky="w", padx=(16, 6))
        self.beam_var = tk.IntVar(value=5)
        self.beam_spin = ttk.Spinbox(opts, from_=1, to=10, width=4, textvariable=self.beam_var)
        self.beam_spin.grid(row=2, column=5, sticky="w")

        self.model_note = tk.StringVar()
        self.model_note_label = ttk.Label(opts, textvariable=self.model_note)
        self.model_note_label.grid(row=3, column=1, columnspan=5, sticky="w")
        opts.columnconfigure(5, weight=1)

        run = ttk.LabelFrame(root, text="Options", padding=inner)
        run.pack(**group)

        ttk.Label(run, text="Device:", width=LABEL_W).grid(row=0, column=0, **cell)
        dev = ttk.Frame(run)
        dev.grid(row=0, column=1, sticky="w")
        self.device_var = tk.StringVar(value="auto")
        ttk.Radiobutton(dev, text="Auto", value="auto", variable=self.device_var).pack(side="left")
        gpu_btn = ttk.Radiobutton(dev, text=f"GPU{f' ({_GPU_NAME})' if _GPU_NAME else ''}",
                                  value="cuda", variable=self.device_var)
        gpu_btn.pack(side="left", padx=12)
        if not cuda_available():
            gpu_btn.state(["disabled"])
        ttk.Radiobutton(dev, text="CPU", value="cpu", variable=self.device_var).pack(side="left")

        ttk.Label(run, text="Audio:").grid(row=1, column=0, **cell)
        aud = ttk.Frame(run)
        aud.grid(row=1, column=1, sticky="w")
        self.vad_var = tk.BooleanVar(value=True)
        self.vad_check = ttk.Checkbutton(aud, text="Skip silence (VAD)", variable=self.vad_var)
        self.vad_check.pack(side="left")
        self.boost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aud, text="Boost quiet audio", variable=self.boost_var).pack(side="left", padx=16)

        ttk.Label(run, text="Formats:").grid(row=2, column=0, **cell)
        fmt = ttk.Frame(run)
        fmt.grid(row=2, column=1, sticky="w")
        saved_formats = cfg.get("formats", ["txt", "srt"])
        self.fmt_vars = {f: tk.BooleanVar(value=f in saved_formats) for f in FORMATS}
        for f, v in self.fmt_vars.items():
            ttk.Checkbutton(fmt, text=f.upper(), variable=v).pack(side="left", padx=(0, 12))

        ttk.Label(run, text="Vocabulary:").grid(row=3, column=0, **cell)
        self.prompt_var = tk.StringVar(value=cfg.get("vocabulary", ""))
        self.prompt_entry = ttk.Entry(run, textvariable=self.prompt_var)
        self.prompt_entry.grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(run, text="Names and terms to spell correctly, comma-separated (Whisper models)").grid(
            row=4, column=1, sticky="w")
        run.columnconfigure(1, weight=1)

        conv = ttk.LabelFrame(root, text="Extract audio", padding=inner)
        conv.pack(**group)
        ttk.Label(conv, text="Format:", width=LABEL_W).grid(row=0, column=0, **cell)
        saved_afmt = cfg.get("audio_format")
        self.afmt_var = tk.StringVar(value=saved_afmt if saved_afmt in AUDIO_FORMATS else "mp3")
        cb = self._combo(conv, self.afmt_var, list(AUDIO_FORMATS), 7)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda _: self._on_audio_format_change())
        ttk.Label(conv, text="Bitrate:").grid(row=0, column=2, sticky="w", padx=(16, 6))
        saved_rate = cfg.get("bitrate")
        self.bitrate_var = tk.StringVar(value=saved_rate if saved_rate in BITRATES else "192k")
        self.bitrate_combo = self._combo(conv, self.bitrate_var, BITRATES, 6)
        self.bitrate_combo.grid(row=0, column=3, sticky="w")
        self.bitrate_combo.bind("<<ComboboxSelected>>", lambda _: self._save_prefs())
        ttk.Label(conv, text="Boost quiet audio applies.").grid(
            row=0, column=4, sticky="w", padx=(16, 6))
        self.convert_btn = self._icon_button(conv, "Extract", "note", self._convert)
        self.convert_btn.grid(row=0, column=5, sticky="e")
        conv.columnconfigure(4, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", padx=8, pady=(0, 6))
        self.start_btn = self._icon_button(actions, "Transcribe", "play", self._start, default="active")
        self.start_btn.pack(side="left")
        self.cancel_btn = self._icon_button(actions, "Stop", "stop", self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        self._icon_button(actions, "Open output folder", "folder", self._open_out).pack(side="right")

        mapf = ttk.LabelFrame(root, text="Transcript map", padding=inner)
        mapf.pack(**group)
        self.readout_var = tk.StringVar(value="Choose a recording, then click Transcribe.")
        ttk.Label(mapf, textvariable=self.readout_var).pack(anchor="w", pady=(0, 4))
        well = self._sunken(mapf, depth=2, body="well")
        well.pack(fill="x")
        self.map_canvas = tk.Canvas(well.body, height=MAP_HEIGHT, bd=0, highlightthickness=0)
        self.map_canvas.pack(fill="x")
        self.map_canvas.bind("<Configure>", self._layout_map)
        self.legend = tk.Canvas(mapf, height=MAP_TILE + 4, bd=0, highlightthickness=0)
        self.legend.pack(anchor="w", pady=(5, 0))

        logf = ttk.LabelFrame(root, text="Live transcript", padding=inner)
        logf.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        well = self._sunken(logf, depth=2, body="field")
        well.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(well.body)
        scroll.pack(side="right", fill="y")
        self.log = tk.Text(well.body, height=6, wrap="word", state="disabled", relief="flat", bd=0,
                           highlightthickness=0, padx=4, pady=2, yscrollcommand=scroll.set)
        scroll.configure(command=self.log.yview)
        self.log.tag_configure("path", wrap="char")  # long paths have no spaces to break at
        self.log.pack(fill="both", expand=True)

    # colour scheme
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
        for name in ("play", "stop", "folder", "note"):
            self.icons[name] = sprite(self, SPRITES[name], 2, outline=t["text"],
                                      mono=t["shadow"] if t.get("mono_icons") else None)
            self.icons[name + "_off"] = sprite(self, SPRITES[name], 2,
                                               emboss=(t["light"] if emboss else t["face"], t["shadow"]))
        for button, name in self.icon_buttons:
            button.configure(image=(self.icons[name], "disabled", self.icons[name + "_off"]))
        self.map_drawn = []
        self._paint_map()
        self._draw_legend()
        self._save_prefs()

    # transcript map: a Disk Defragmenter-style grid of blocks covering the whole media
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
            items = [("speech", "Extracted"), ("working", "Working"), ("waiting", "Not reached")]
        else:
            items = [("speech", "Speech"), ("silent", "No speech"), ("working", "Working"),
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
        self.model_note_label.grid() if spec.note else self.model_note_label.grid_remove()
        self._save_prefs()

    def _on_audio_format_change(self):
        lossy = AUDIO_FORMATS[self.afmt_var.get()][1]
        self.bitrate_combo.state(["!disabled", "readonly"] if lossy else ["disabled"])
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
        self._launch(Job(opts, self.events))

    def _convert(self):
        file = self.file_var.get().strip()
        if not file or not Path(file).is_file():
            return messagebox.showwarning("Transcriber", "Choose an audio or video file first.")
        out_dir = self.out_var.get().strip() or str(Path(file).parent)
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
        self.readout_var.set("Starting…")
        if self.animate and not self.blink_after:
            self.blink_after = self.after(450, self._blink)
        self.start_btn.state(["disabled"])
        self.convert_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.job = job
        self.job.start()

    def _cancel(self):
        if self.job:
            self.job.cancelled.set()
            self.status_var.set("Stopping — saving what has been transcribed so far…"
                                if isinstance(self.job, Job) else "Stopping…")
            self.cancel_btn.state(["disabled"])

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
        })

    def _close(self):
        self._save_prefs()
        if self.job:
            self.job.cancelled.set()
        self.destroy()

    def _about(self):
        win = tk.Toplevel(self, background=self.t["face"])
        win.title("About Local Transcriber")
        win.resizable(False, False)
        win.transient(self)
        body = ttk.Frame(win, padding=(14, 14, 14, 10))
        body.pack(fill="both", expand=True)
        ttk.Label(body, image=self.app_icons[2]).grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 14))
        ttk.Label(body, text="Local Transcriber", font=self.bold_font).grid(row=0, column=1, sticky="w")
        ttk.Label(body, text="Transcribes audio and video on this computer.\nNothing is uploaded.",
                  justify="left").grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(body, text="Speech: faster-whisper and onnx-asr (Parakeet)\nAudio: ffmpeg",
                  justify="left").grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Separator(body).grid(row=3, column=1, sticky="ew", pady=10)
        memory = psutil.virtual_memory().total / 2**30
        ttk.Label(body, text=f"Physical memory: {memory:.0f} GB\n"
                             f"Graphics: {_GPU_NAME or 'no NVIDIA GPU found'}",
                  justify="left").grid(row=4, column=1, sticky="w")
        ok = ttk.Button(body, text="OK", width=10, default="active", command=win.destroy)
        ok.grid(row=5, column=1, sticky="e", pady=(12, 0))
        win.bind("<Return>", lambda _: win.destroy())
        win.bind("<Escape>", lambda _: win.destroy())
        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{x}+{y}")
        ok.focus_set()
        win.grab_set()

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
        self.status_var.set(text)

    def _on_log(self, text):
        self._append(text)

    def _on_started(self, duration, summary):
        self.duration = duration
        self.started_at = time.time()
        self._append(f"— {short(duration)} of media · {summary} —")
        if duration:
            self.readout_var.set(f"00:00 / {short(duration)}  (0%)")
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
        frac = min(position / self.duration, 1.0)
        elapsed = time.time() - self.started_at
        speed = position / elapsed if elapsed > 0 else 0
        eta = (self.duration - position) / speed if speed > 0 else 0
        self.readout_var.set(f"{short(position)} / {short(self.duration)}  ({frac:.0%})  ·  "
                             f"ETA {short(eta)}  ·  {speed:.1f}× realtime")

    def _on_done(self, files, partial, count=None):
        self.map_active = False
        if not partial and self.duration:
            self.map_reached = self.duration
            self.readout_var.set(f"{short(self.duration)} / {short(self.duration)}  (100%)")
        self._paint_map()
        elapsed = short(time.time() - (self.started_at or time.time()))
        if partial:
            state = "Stopped — partial transcript saved" if files else "Stopped"
        else:
            state = "Done"
        segments = f" · {count} segments" if count is not None else ""
        self.status_var.set(f"{state}{segments} · {elapsed}")
        if files:
            self._append("Saved:")
            for f in files:
                self._append(f"  {f}", "path")
        if not partial and self.sound_var.get():
            play_sound("tada")
        self._reset_buttons()

    def _on_error(self, text):
        self.map_active = False
        self._paint_map()
        self.status_var.set("Failed")
        self._append(f"ERROR: {text}")
        self._reset_buttons()
        messagebox.showerror("Transcriber", text)

    def _reset_buttons(self):
        self.job = None
        self.start_btn.state(["!disabled"])
        self.convert_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

    # resources
    def _sample_resources(self):
        try:
            cores = psutil.cpu_count() or 1
            proc_cpu = min(self.proc.cpu_percent(None) / cores, 100)
            self.res_vars["cpu"].set(f"CPU {proc_cpu:.0f}% · all {psutil.cpu_percent(None):.0f}%")
            vm = psutil.virtual_memory()
            rss = self.proc.memory_info().rss / 2**30
            self.res_vars["ram"].set(f"RAM {rss:.1f} GB · {vm.percent:.0f}% used")
            if _GPU_HANDLE is not None:
                util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                self.res_vars["gpu"].set(f"GPU {util}%")
                self.res_vars["vram"].set(f"VRAM {mem.used / 2**30:.1f}/{mem.total / 2**30:.0f} GB")
            else:
                self.res_vars["gpu"].set("No NVIDIA GPU")
                self.res_vars["vram"].set("VRAM —")
        except Exception:
            pass
        self.after(1000, self._sample_resources)

    # log
    def _append(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
