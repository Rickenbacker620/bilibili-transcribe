---
name: bilibili-transcribe
description: Fetch a Bilibili video's title, description, and audio URL, and transcribe its speech to text locally with FunASR (SenseVoiceSmall). Use when the user gives a Bilibili video link or BV/av id and wants the transcript, subtitles, summary, or metadata of that video.
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
  "transcript": "The transcribed speech ..."
}
```

## Workflow

1. **Extract the id.** Accept a BV id (`BV1onNWz9EHB`), an av id (`av114707...`), or a full
   `bilibili.com/video/...` URL — pull the `BV...`/`av...` token out of the URL.
2. **Run the script** with that id (see Quick start). On the first run uv resolves the deps
   and FunASR downloads the SenseVoiceSmall model (~1 GB) — this is slow once, then cached.
3. **Parse the JSON** from stdout. Progress bars and model logs go to stderr, so stdout is
   always clean JSON.
4. **Use `transcript`** for whatever the user asked (full text, summary, subtitles, etc.).

## Options

- `--dry-run` — only fetch metadata; skip the audio download and transcription. Use this when
  the user just wants the title/description or to confirm the video resolves.
- `--language auto|zh|en|yue|ja|ko` — language hint for SenseVoice (default `auto`).
- `--device cpu|cuda:0` — inference device (default `cpu`). Use a CUDA device if available
  to speed up long videos.
- `--model iic/SenseVoiceSmall` — alternate FunASR model.

```bash
# metadata only, no transcription
uv run --no-project "${CLAUDE_PLUGIN_ROOT}/skills/bilibili-transcribe/scripts/bili_info.py" BV1onNWz9EHB --dry-run

# force Chinese, run on GPU
uv run --no-project "${CLAUDE_PLUGIN_ROOT}/skills/bilibili-transcribe/scripts/bili_info.py" BV1onNWz9EHB --language zh --device cuda:0
```

## Notes

- Only the **first part** of a multi-part video is transcribed.
- The audio URL in the output is hotlink-protected and time-limited; it cannot be opened
  directly in a browser. The script already downloads and transcribes it in memory, so you
  rarely need the raw URL.
- Long videos take a while on CPU — set expectations and consider `--device cuda:0` when a
  GPU is present.
