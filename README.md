# Bilibili transcribe

A minimal Bilibili video info extractor, packaged as an installable **agent skill**. Give it a video ID and it returns the **bvid / title / description / audio URL**, and by default downloads the audio into memory and transcribes it locally with [FunASR](https://github.com/modelscope/FunASR) (Tongyi Lab's Fun-ASR-Nano), producing both a full transcript and timestamped segments.

## What it does

Take a BV id (or av id) as input and print JSON:

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

`audio_url` points to the video's **lowest-bitrate audio stream** (64K AAC), which is ideal for speech transcription. The script downloads that stream into a buffer (no file written to disk) and runs FunASR's Fun-ASR-Nano locally to produce `transcript` (full text) and `segments` (subtitle-sized chunks with start/end times in milliseconds). Both are omitted in `--dry-run` mode.

> **Note on timestamps:** Fun-ASR-Nano's own timestamp output is still an upstream TODO, so segment timing comes from FunASR's generic CTC forced-alignment. `start` times are reliable; a segment `end` can occasionally be stretched at a voice-activity boundary. Good for subtitles/navigation, not frame-accurate.

## Install (as an agent skill)

Install the skill into your agent with the [`skills`](https://github.com/vercel-labs/skills) CLI — no clone or global install needed:

```bash
# Install the skill from this GitHub repo
npx skills add Rickenbacker620/bilibili-transcribe
```

This drops the skill into your project's skills directory (e.g. `.claude/skills/bilibili-transcribe/`), where your agent picks it up automatically. Once installed, just give the agent a Bilibili link or `BV`/`av` id and it runs the skill.

Useful variations:

```bash
# See which skills the repo exposes before installing
npx skills add Rickenbacker620/bilibili-transcribe --list

# Install only this skill by name
npx skills add Rickenbacker620/bilibili-transcribe --skill bilibili-transcribe
```

You still need [uv](https://docs.astral.sh/uv/) on your `PATH` — the skill shells out to `uv run` to fetch dependencies and run the transcription script (see [Dependencies](#dependencies)).

## Dependencies

Declared inline in the script via PEP 723 (`requests`, `tqdm`, `funasr`, `torch`, `torchaudio`, `av`, `numpy`, plus `regex`, `safetensors`, `tiktoken` for Fun-ASR-Nano's tokenizer) and resolved automatically by [uv](https://docs.astral.sh/uv/) — there is no `pyproject.toml`/lockfile to manage. FunASR decodes the audio via PyAV, so no separate `ffmpeg` binary is required.

## Usage (standalone)

You can run the script directly without installing the plugin:

```bash
SCRIPT=skills/bilibili-transcribe/scripts/bili_info.py

# Fetch metadata, download the audio, and transcribe it (default)
uv run --no-project "$SCRIPT" BV1onNWz9EHB

# Only print metadata; skip the download and transcription
uv run --no-project "$SCRIPT" BV1onNWz9EHB --dry-run

# Choose a FunASR model (default: FunAudioLLM/Fun-ASR-Nano-2512), language, or device
uv run --no-project "$SCRIPT" BV1onNWz9EHB --language 中文 --device cuda:0

# av ids are also supported and converted to BV automatically
uv run --no-project "$SCRIPT" av114707520295544
```

The audio stream is fetched in the unauthenticated `try_look` mode, so no login/cookie is needed.

## How it works

The script reproduces the core info-fetching steps from [BilibiliDown](https://github.com/nICEnnnnnnnLee/BilibiliDown):

1. **av ↔ BV conversion** — pure local computation, no network needed.
2. **WBI signature** — newer Bilibili endpoints require a `w_rid` signature or the request gets rejected by their anti-abuse system. The script fetches the key from `x/web-interface/nav` and signs the parameters.
3. **Fetch title/description** — `x/web-interface/wbi/view/detail`.
4. **Fetch cid** — `x/player/pagelist`, locating the first part.
5. **Fetch audio URL** — `x/player/wbi/playurl` (DASH format), picking the lowest-bitrate entry from `dash.audio[]`.

## What those "magic constants" in the script are about

There are a few seemingly arbitrary constants at the top of the script. They exist to deal with Bilibili's two obfuscation algorithms — they are not random.

### 1. av ↔ BV conversion: `_TABLE` / `_XOR` / `_MAX_AID`

Early Bilibili videos used purely numeric av ids (`av170001`); these were later replaced with scrambled-letter BV ids (`BV17x411w7KC`). A BV id is essentially the av number **encoded with a shuffled base-58 alphabet plus XOR scrambling**, so the two can be converted locally without any network request.

```python
_TABLE   = "FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf"
_XOR     = 23442827791579     # XOR constant; scrambles the number so BV ids look random
_MAX_AID = 1 << 51            # fixed high bit; keeps every generated BV id the same length
```

- `_TABLE`: a **shuffled** base-58 alphabet (with easily confused characters like `0OIl` removed). It maps numbers ↔ characters; the order is deliberately scrambled by Bilibili.
- `_XOR`: XORing the av number with this constant makes BV ids from adjacent av numbers look completely unrelated (preventing guessing/enumeration).
- `_MAX_AID`: ORing this fixed high bit before conversion guarantees the result always fills the fixed BV id length.
- The lines like `bv[3], bv[9] = bv[9], bv[3]`: Bilibili also **swaps a few character positions** in the result for extra scrambling. The conversion has to swap them back exactly.

This is Bilibili's fixed (reverse-engineered) algorithm; the constants are hard-coded, so you just copy them as-is.

### 2. WBI signature: `_MIXIN_TAB`

Newer Bilibili endpoints require every request to carry a `w_rid` signature parameter, or it gets blocked by their anti-abuse system (returning -403 or similar). The signing key (mixin key) is computed like this:

1. First request `x/web-interface/nav` and pull two strings from the response (the filenames in `img_url` and `sub_url`), concatenated into a **64-character** raw string.
2. Use the **reorder table** `_MIXIN_TAB` to pick the first 32 characters out of those 64 by index, forming the final mixin key.

```python
_MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, ...]   # 64 indices in total
```

In other words, `_MIXIN_TAB` says "take character #46, then #47, then #18, …" — a fixed shuffle order. This is logic hard-coded in Bilibili's frontend JS; once reverse-engineered, you just copy it.

Once you have the mixin key, signing goes: sort the request parameters lexicographically → append the timestamp `wts` → take `MD5(param_string + mixin_key)` over the whole thing, and the result is `w_rid`.

## ⚠️ Caveats

- **The audio URL cannot be opened directly in a browser**: Bilibili's CDN has hotlink protection, so requests must carry `Referer: https://www.bilibili.com/` or you get a 403.
- **The URL expires**: the `deadline` in the URL is an expiry timestamp; once past, the link fails even with the correct Referer, so fetch it right before use.
- `.m4s` is a DASH segment file, not a directly playable complete audio file.

The correct way to download the audio:

```bash
curl -A "Mozilla/5.0" -H "Referer: https://www.bilibili.com/" "<audio_url>" -o audio.m4s
```
