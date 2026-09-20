# Local Transcriber

A Windows desktop application for transcribing audio and video entirely on your own machine. All processing runs locally through open-source speech models — no audio, video, or transcript ever leaves the computer.

## Features

- **Speech-to-text engines:** Whisper large-v3 and large-v3-turbo, Distil-Whisper, a Hindi fine-tune, NVIDIA Parakeet TDT, or any custom CTranslate2 Whisper model
- **Language and hardware control:** auto-detect or force a language, run on GPU (NVIDIA/CUDA) or CPU
- **Export formats:** TXT, SRT, VTT, JSON and Markdown, any combination in one pass
- **Audio extraction:** save the audio track of a screen recording or video as MP3, M4A, Opus, OGG, FLAC or WAV
- **Live job monitoring:** a transcript map showing which parts of the recording contain speech, plus ETA, processing speed, and CPU/RAM/GPU/VRAM usage
- **Interface:** a Windows 95–styled UI with seven selectable colour schemes, including Light, Dark and Terminal variants

## Requirements

- Python 3.10 or later
- [ffmpeg](https://ffmpeg.org/) available on `PATH`

```bash
pip install -r requirements.txt
```

Models are downloaded on first use and cached locally afterwards (in `~/.cache/huggingface`).

## Running

```bash
python transcriber.py
```

or double-click `run.bat` on Windows.

## Supported models

| Model | Best for | Languages | Size |
|---|---|---|---|
| **Whisper large-v3** | Highest accuracy; mixed or accented speech | 99 | ~3 GB |
| **Whisper large-v3-turbo** | Near large-v3 quality, several times faster | 99 | ~1.6 GB |
| **Distil-Whisper large-v3.5** | Fast English transcription | English | ~1.5 GB |
| **Whisper large-v2 Hindi** ([Collabora](https://huggingface.co/collabora/faster-whisper-large-v2-hindi)) | Hindi speech | Hindi | ~3 GB |
| **NVIDIA Parakeet TDT 0.6B v3** ([onnx-asr](https://github.com/istupakov/onnx-asr)) | Very fast, punctuated transcription | English + 24 European (no Indian languages) | ~2.4 GB |
| Whisper medium / small / base / tiny | Lighter hardware, quick tests | 99 | 75 MB – 1.5 GB |
| **Custom** | Any CTranslate2 Whisper model — a Hugging Face repo ID or local folder | — | — |

## Transcription options

| Option | Description |
|---|---|
| Language | Auto-detect, or force a language for more reliable results on short or mixed-language audio. Locked for single-language models |
| Task | `transcribe` preserves the spoken language; `translate` outputs English (multilingual Whisper models only) |
| Beam size | Higher values are marginally more accurate and slower; 5 is a reasonable default |
| Device | Auto, GPU (CUDA) or CPU. GPU falls back to CPU automatically if CUDA is unavailable |
| Skip silence (VAD) | Speeds up processing and reduces invented text in silent stretches. Always enabled for Parakeet |
| Boost quiet audio | Normalizes volume via ffmpeg before transcribing; helps when one speaker is much quieter than others |
| Vocabulary | Comma-separated names and terms to spell correctly, e.g. `ArthaFlow, RoDTEP, Nashik` (Whisper models) |
| Formats | Output is written as `<file name>.<extension>` in the chosen output folder |

Clicking **Stop** during a run saves everything transcribed so far as `<file name>_partial.<extension>`. Colour scheme, sound preference, model, language, formats, vocabulary, and audio export settings persist between sessions in `~/.local-transcriber.json`.

## Transcript map

A grid of blocks represents the full length of the recording, filling in left to right as the job progresses:

| Block | Meaning |
|---|---|
| Filled | Speech detected in this segment |
| Empty | Processed, no speech detected |
| Blinking | Currently being transcribed |
| Grey | Not yet reached |

Once a job completes, the map gives an at-a-glance view of where the talking happens in the recording, making pauses and silent stretches easy to spot. Blinking respects the system's animation settings. During audio extraction, the map fills as ffmpeg processes the file.

## Interface

The application uses a Windows 95–inspired interface built entirely with Tk's standard widgets — no additional UI dependencies.

- **View → Color scheme:** four classic Windows schemes (Windows Standard, High Contrast Black, Eggplant, Brick) plus Light, Dark and Terminal variants. Terminal renders in green phosphor with a monospace bitmap font throughout. The transcript map recolors to match.
- **View → Sounds:** plays a completion chime when a job finishes; can be disabled.
- **Help → About:** displays the active speech engines, system memory, and GPU information.
- **Keyboard shortcut:** Ctrl+O opens a recording file.

## Audio extraction

Extracts the audio track from the selected recording using ffmpeg and saves it as `<file name>.<format>` in the output folder. Progress is shown on the transcript map, and **Stop** removes the partially written file.

| Format | Codec | Notes |
|---|---|---|
| MP3 | LAME | Universally compatible |
| M4A | AAC | Smaller than MP3 at equivalent quality |
| Opus | Opus | Most efficient for speech; 96 kbps is typically sufficient |
| OGG | Vorbis | Open format |
| FLAC | FLAC | Lossless; bitrate setting does not apply |
| WAV | 16-bit PCM | Uncompressed; bitrate setting does not apply |

Enabling **Boost quiet audio** also normalizes the extracted track. If the source file already occupies the target filename, the output is suffixed `_audio` to avoid overwriting it.

## Recording tip

Screen recorders often capture only system audio (other participants on a call) and not the local microphone. Enable microphone capture before recording, or your side of the conversation will be missing from the transcript.
