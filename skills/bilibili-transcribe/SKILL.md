---
name: bilibili-transcribe
description: Fetch a Bilibili video's title, description, and audio URL, and transcribe its speech to text locally with FunASR (Fun-ASR-Nano), returning both a full transcript and timestamped segments. Use when the user gives a Bilibili video link or BV/av id and wants the transcript, subtitles, timestamps, summary, or metadata of that video.
---

# Bilibili Transcribe

Given a Bilibili video, fetch its metadata and a local speech transcript. Everything runs
offline via FunASR — no API key, login, or cookie is required.

## Quick start

The bundled script needs [uv](https://docs.astral.sh/uv/). It declares its own
dependencies inline (PEP 723), so `uv run` installs them on first use — nothing else
to set up.

```bash
uv run --no-project "${CLAUDE_PLUGIN_ROOT}/skills/bilibili-transcribe/scripts/bili_info.py" BV1onNWz9EHB
```

This prints JSON to stdout:

```json
{
  "bvid": "BV1onNWz9EHB",
  "title": "Video title",
  "desc": "Video description",
  "audio_url": "https://....m4s?...",
  "transcript": "The full transcribed speech ...",
  "segments": [
    {"start_ms": 140, "end_ms": 1700, "text": "第一句字幕，"},
    {"start_ms": 1700, "end_ms": 2960, "text": "第二句字幕。"}
  ]
}
```

`transcript` is the whole text; `segments` is a subtitle-sized list with start/end times in
milliseconds (handy for subtitles or jumping to a moment in the video).

## Workflow

1. **Extract the id.** Accept a BV id (`BV1onNWz9EHB`), an av id (`av114707...`), or a full
   `bilibili.com/video/...` URL — pull the `BV...`/`av...` token out of the URL.
2. **Run the script** with that id (see Quick start). On the first run uv resolves the deps
   and FunASR downloads the Fun-ASR-Nano model (~800 MB) plus the VAD model — slow once, then
   cached.
3. **Parse the JSON** from stdout. Progress bars and model logs go to stderr, so stdout is
   always clean JSON.
4. **Use `transcript` / `segments`** for whatever the user asked (full text, summary,
   subtitles, timestamps, etc.).

## Options

- `--dry-run` — only fetch metadata; skip the audio download and transcription. Use this when
  the user just wants the title/description or to confirm the video resolves.
- `--language 中文|英文|日文` — optional language hint (default: auto-detect). Fun-ASR-Nano
  also handles 7 Chinese dialects and many accents.
- `--device cpu|cuda:0` — inference device (default `cpu`). Use a CUDA device if available
  to speed up long videos.
- `--model FunAudioLLM/Fun-ASR-Nano-2512` — alternate FunASR model. The timestamped segments
  work with models that emit token timestamps (Fun-ASR-Nano) or `sentence_info` (Paraformer).

```bash
# metadata only, no transcription
uv run --no-project "${CLAUDE_PLUGIN_ROOT}/skills/bilibili-transcribe/scripts/bili_info.py" BV1onNWz9EHB --dry-run

# force Chinese, run on GPU
uv run --no-project "${CLAUDE_PLUGIN_ROOT}/skills/bilibili-transcribe/scripts/bili_info.py" BV1onNWz9EHB --language 中文 --device cuda:0
```

## Notes

- Only the **first part** of a multi-part video is transcribed.
- The audio URL in the output is hotlink-protected and time-limited; it cannot be opened
  directly in a browser. The script already downloads and transcribes it in memory, so you
  rarely need the raw URL.
- Long videos take a while on CPU — set expectations and consider `--device cuda:0` when a
  GPU is present. On CPU, expect roughly real-time/4 (a ~4.5-min clip ≈ 75–90 s).
- **Segment timestamps are best-effort.** Fun-ASR-Nano's own timestamp output is still an
  upstream TODO, so the timing comes from FunASR's generic CTC forced-alignment. `start` times
  are reliable; a segment `end` can occasionally be stretched at a VAD-segment boundary. Good
  for subtitles/navigation, not frame-accurate.
