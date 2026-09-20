# Local Transcriber

A Windows desktop application for transcribing audio and video entirely on your own machine. All processing runs locally through open-source speech models — no audio, video, or transcript ever leaves the computer.

## Features

- **Speech-to-text engines:** Whisper large-v3 and large-v3-turbo, Distil-Whisper, a Hindi fine-tune, NVIDIA Parakeet TDT, or any custom CTranslate2 Whisper model
- **Language and hardware control:** auto-detect or force a language, run on GPU (NVIDIA/CUDA) or CPU
- **Export formats:** TXT, SRT, VTT, JSON and Markdown, any combination in one pass
- **Audio extraction:** save the audio track of a screen recording or video as MP3, M4A, Opus, OGG, FLAC or WAV
- **Live job monitoring:** a transcript map showing which parts of the recording contain speech, plus ETA, processing speed, and a System Monitor panel for CPU/RAM/GPU/VRAM usage
- **Searchable history:** every job is logged with its full transcript in a local SQLite database, so you can find which recording mentioned a term months later
- **Model manager and offline mode:** see which models are on disk, download or delete them, or block network access entirely
- **Interface:** a retro desktop UI — sidebar navigation, numbered workflow sections and a decorative panel — with eight selectable colour schemes, including Light, Dark and Terminal variants

## Performance

The app keeps the loaded model in memory between runs, so only the first job of a session pays the load cost. On a GPU it can also transcribe several chunks at once, which is roughly **3× faster** on large-v3 — measured at 10.8 s versus 3.1 s for 11 minutes of speech on an RTX 5080. Batching produces longer segments, so it turns itself off automatically when you export SRT or VTT, where precise timings matter. Both behaviours are adjustable under Settings.

On a machine without a GPU, the app defaults to Whisper small rather than large-v3 on first run.

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
| Mixed languages | Detects the language per segment instead of once for the file — for calls that switch between, say, Hindi and English. Multilingual Whisper models on Auto-detect only |
| Vocabulary | Comma-separated names and terms to spell correctly, e.g. `ArthaFlow, RoDTEP, Nashik` (Whisper models) |
| Formats | Output is written as `<file name>.<extension>` in the chosen output folder |

Clicking **Stop** during a run saves everything transcribed so far as `<file name>_partial.<extension>`. Colour scheme, sound preference, model, language, formats, vocabulary, default output folder and audio export settings persist between sessions in `~/.local-transcriber.json`.

## Transcript map

A grid of blocks represents the full length of the recording, filling in left to right as the job progresses:

| Block | Meaning |
|---|---|
| Filled (green) | Speech detected in this segment |
| Empty | Processed, no speech detected |
| Processing (blinking) | Currently being transcribed |
| Grey | Not yet reached |

Once a job completes, the map gives an at-a-glance view of where the talking happens in the recording, making pauses and silent stretches easy to spot. Blinking respects the system's animation settings. During audio extraction, the map simply fills as ffmpeg processes the file.

## Interface

The application uses a retro desktop interface built entirely with Tk's standard widgets — no additional UI dependencies.

- **Sidebar:** Transcribe, Extract Audio, History, Settings and About. Transcribe and Extract Audio open the same workspace; History and Settings are their own pages.
- **Transcribe workspace:** five numbered sections — Select File (with a details strip showing type, length and size), Model & Language, Options, Audio Extraction and Output Folder — followed by the action buttons, the transcript map and a live transcript alongside the System Monitor panel.
- **Models:** every model with its size on disk, plus Download and Delete. **Offline mode** sets the Hugging Face libraries to local-only, so a missing model fails instead of quietly downloading gigabytes.
- **History:** every completed job, with its transcript stored and indexed for full-text search. Type a word to find the recordings that mention it; double-click a row to read the transcript with matches highlighted, or use Open output folder.
- **Settings:** pick a colour scheme, toggle completion sounds, and set a default output folder used whenever the Output Folder field is left empty.
- **View → Color scheme:** the same eight schemes as the Settings page — four classic Windows looks (Windows Standard, High Contrast Black, Eggplant, Brick), Light, Dark, Terminal, and the default Neon Dusk. Terminal renders in green phosphor with a monospace bitmap font throughout. The transcript map and System Monitor recolor to match; the sidebar and header stay fixed.
- **View → Sounds:** plays a completion chime when a job finishes; can be disabled here or from Settings.
- **Help → About** (or the sidebar's About): displays the active speech engines, system memory, and GPU information.
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
