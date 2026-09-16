# Local Transcriber

Small local tool to transcribe audio or video with Whisper. Runs on this machine only — nothing is uploaded.

## Setup (once)

```bash
pip install -r requirements.txt
```

The first time you use a model, it downloads (large-v3 is ~3 GB) and is cached afterwards.

## Run

Double-click `run.bat`, or:

```bash
python transcriber.py
```

## Options

| Option | Notes |
|---|---|
| Language | Auto-detect, or force one (English, Hindi, Marathi, Gujarati, …). Forcing is more reliable for short or mixed-language audio |
| Model | `large-v3` most accurate · `large-v3-turbo` much faster, nearly as good · `small`/`base` for CPU |
| Task | `transcribe` keeps the spoken language · `translate` outputs English |
| Device | Auto / GPU (NVIDIA, CUDA) / CPU. GPU falls back to CPU if CUDA can't load |
| Skip silence (VAD) | Faster, fewer hallucinations in silent stretches. Turn off if quiet speech is being dropped |
| Formats | TXT, SRT, VTT, JSON, Markdown — saved as `<file name>.<ext>` in the output folder |
| Vocabulary | Names and terms to spell correctly (e.g. `ArthaFlow, RoDTEP, Nashik`) |

Cancel saves whatever has been transcribed so far as `<file name>_partial.<ext>`.

## Recording tip

Screen recorders often capture only system audio (the other people on the call), not your own microphone. Enable microphone capture in the recorder before the meeting, or your side of the conversation will be missing.
