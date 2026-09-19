# Local Transcriber

A small desktop tool for transcribing audio and video on your own machine. Nothing is uploaded.

- **Open-source models:** Whisper large-v3 and turbo, Distil-Whisper, a Hindi fine-tune, NVIDIA Parakeet, or any CTranslate2 Whisper model
- **Choice of language and device:** GPU or CPU
- **Output formats:** TXT, SRT, VTT, JSON and Markdown
- **Extract audio:** save the audio track of a screen recording or video as MP3, M4A, Opus, OGG, FLAC or WAV
- **Live status:** how much of the media is done, ETA and speed, plus CPU, RAM, GPU and VRAM usage
- **Themes:** Dark, Light and Terminal

## Setup

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on `PATH`.

```bash
pip install -r requirements.txt
```

Models download the first time you use them and are cached afterwards (in `~/.cache/huggingface`).

## Run

Double-click `run.bat`, or:

```bash
python transcriber.py
```

## Models

| Model | Best for | Languages | Size |
|---|---|---|---|
| **Whisper large-v3** | Highest accuracy; mixed or accented speech | 99 | ~3 GB |
| **Whisper large-v3-turbo** | Nearly large-v3 quality, several times faster | 99 | ~1.6 GB |
| **Distil-Whisper large-v3.5** | Fast English | English | ~1.5 GB |
| **Whisper large-v2 Hindi** ([Collabora](https://huggingface.co/collabora/faster-whisper-large-v2-hindi)) | Hindi speech | Hindi | ~3 GB |
| **NVIDIA Parakeet TDT 0.6B v3** ([onnx-asr](https://github.com/istupakov/onnx-asr)) | Very fast, punctuated English and European languages | English + 24 European (no Indian languages) | ~2.4 GB |
| Whisper medium / small / base / tiny | Lighter machines and quick tests | 99 | 75 MB – 1.5 GB |
| **Custom** | Any CTranslate2 Whisper model: a Hugging Face repo id or a local folder | — | — |

## Options

| Option | Notes |
|---|---|
| Language | Auto-detect, or force one. Forcing is more reliable for short or mixed-language audio. Locked for single-language models |
| Task | `transcribe` keeps the spoken language; `translate` outputs English (multilingual Whisper models only) |
| Beam size | Higher is slightly more accurate and slower. 5 is a good default |
| Device | Auto, GPU (NVIDIA, CUDA) or CPU. GPU falls back to CPU if CUDA can't load |
| Skip silence (VAD) | Faster, and fewer invented words in silent stretches. Parakeet always uses it |
| Boost quiet audio | Evens out volume with ffmpeg before transcribing. Helps when one speaker is much quieter |
| Vocabulary | Names and terms to spell correctly, e.g. `ArthaFlow, RoDTEP, Nashik` (Whisper models) |
| Formats | Saved as `<file name>.<ext>` in the output folder |

Cancel saves whatever has been transcribed so far as `<file name>_partial.<ext>`. Theme, model, language, formats, vocabulary and the audio format and bitrate are remembered in `~/.local-transcriber.json`.

## Extract audio

Converts the **Input** file (for example, a screen recording) to an audio file using ffmpeg. The file is saved as `<file name>.<format>` in the output folder. It uses the same progress bar and Cancel button as transcription. Cancelling deletes the unfinished file.

| Format | Codec | Notes |
|---|---|---|
| MP3 | LAME | Plays everywhere |
| M4A | AAC | Smaller than MP3 at the same quality |
| Opus | Opus | Smallest for speech; 96k is plenty for voice |
| OGG | Vorbis | Open format |
| FLAC | FLAC | Lossless; bitrate doesn't apply |
| WAV | 16-bit PCM | Uncompressed; bitrate doesn't apply |

Ticking **Boost quiet audio** also levels the extracted audio. If the input is already an audio file of the same format in the same folder, the output is named `<file name>_audio.<format>` so the original is never overwritten.

## Recording tip

Screen recorders often capture only system audio (the other people on a call), not your own microphone. Turn on microphone capture before the meeting, or your side of the conversation will be missing.
