#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av>=17.0.1",
#     "funasr>=1.3.9",
#     "numpy",
#     "regex",
#     "requests>=2.34.2",
#     "safetensors",
#     "tiktoken",
#     "torch>=2.12.0",
#     "torchaudio>=2.11.0",
#     "tqdm>=4.67.3",
# ]
# ///
"""
bili_info.py — fetch a Bilibili video's: bvid / title / description / audio URL

By default it also downloads the audio and transcribes it locally with FunASR's
Fun-ASR-Nano-2512 (Tongyi Lab), producing both a full transcript and a list of
timestamped segments. Pass --dry-run to skip the download/transcription and only
print the metadata.

Usage: python bili_info.py BV1xxxxxxx [--dry-run] [--model MODEL] [--language LANG]
Dependencies: requests, tqdm; funasr + torch + av (only when not in --dry-run)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
from functools import reduce

import requests
from tqdm import tqdm

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# --- av <-> BV conversion --------------------------------------------------
_TABLE = "FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf"
_XOR, _MAX_AID = 23442827791579, 1 << 51


def av2bv(av: int) -> str:
    bv = list("BV1000000000")
    idx, tmp = 11, (_MAX_AID | av) ^ _XOR
    while tmp:
        bv[idx] = _TABLE[tmp % 58]
        tmp //= 58
        idx -= 1
    bv[3], bv[9] = bv[9], bv[3]
    bv[4], bv[7] = bv[7], bv[4]
    return "".join(bv)


# --- WBI signature ---------------------------------------------------------
_MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]


class BiliInfo:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self._mixin_key = None

    def _mixin(self) -> str:
        if not self._mixin_key:
            img = self.s.get("https://api.bilibili.com/x/web-interface/nav"
                             ).json()["data"]["wbi_img"]
            raw = (img["img_url"].rsplit("/", 1)[-1].split(".")[0]
                   + img["sub_url"].rsplit("/", 1)[-1].split(".")[0])
            self._mixin_key = reduce(lambda s, i: s + raw[i], _MIXIN_TAB[:32], "")
        return self._mixin_key

    def _get(self, url, bvid, params, wbi=False):
        if wbi:
            params = dict(params, wts=int(time.time()))
            q = urllib.parse.urlencode(sorted(params.items()))
            params["w_rid"] = hashlib.md5((q + self._mixin()).encode()).hexdigest()
        return self.s.get(url, params=params,
                          headers={"Referer": f"https://www.bilibili.com/video/{bvid}"}
                          ).json()

    def collect(self, bvid: str) -> dict:
        # title / description
        view = self._get("https://api.bilibili.com/x/web-interface/wbi/view/detail",
                         bvid, {"bvid": bvid, "platform": "web"}, wbi=True
                         )["data"]["View"]
        # cid of the first part
        cid = self._get("https://api.bilibili.com/x/player/pagelist",
                        bvid, {"bvid": bvid})["data"][0]["cid"]
        # lowest-bitrate audio URL (try_look unlocks streams without a login)
        params = {"cid": cid, "bvid": bvid, "qn": 16, "fnver": 0,
                  "fnval": 4048, "fourk": 1, "otype": "json", "try_look": 1}
        dash = self._get("https://api.bilibili.com/x/player/wbi/playurl",
                         bvid, params, wbi=True)["data"].get("dash") or {}
        audios = dash.get("audio") or []
        audio_url = (min(audios, key=lambda a: a.get("bandwidth") or 0)["base_url"]
                     if audios else None)
        return {
            "bvid": bvid,
            "title": view["title"],
            "desc": view.get("desc", ""),
            "audio_url": audio_url,
        }

    def download_audio(self, audio_url: str, bvid: str, progress: bool = True) -> bytes:
        with self.s.get(
            audio_url,
            headers={"Referer": f"https://www.bilibili.com/video/{bvid}"},
            stream=True,
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            buf = io.BytesIO()
            bar = tqdm(
                total=total, desc="audio", unit="B", unit_scale=True,
                unit_divisor=1024, file=sys.stderr, disable=not progress,
            )
            with bar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    buf.write(chunk)
                    bar.update(len(chunk))
        return buf.getvalue()


def _decode_pcm(audio: bytes, rate: int = 16000):
    """Decode in-memory audio bytes to a 16 kHz mono float32 PCM array.

    FunASR can't parse the DASH .m4s (AAC) container itself; it wants PCM
    samples. PyAV (bundled ffmpeg) decodes and resamples the buffer in memory,
    so nothing touches disk.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks = []
    with av.open(io.BytesIO(audio)) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):  # flush buffered samples
            chunks.append(resampled.to_ndarray())
    pcm = np.concatenate(chunks, axis=1).reshape(-1)
    return pcm.astype(np.float32) / 32768.0


# punctuation that ends a subtitle line (kept attached to the line)
_BREAK = set("。！？；…!?;")          # hard breaks: sentence enders
_SOFT = set("，、：,:")                # soft breaks: only used if a line gets long
_MIN_CHARS = 6                         # don't break a line shorter than this


def _segment(timestamps: list) -> list:
    """Turn Fun-ASR-Nano's char-level `timestamps` into subtitle segments.

    Those tokens already carry punctuation (with timing), so we split on
    sentence-ending marks, falling back to soft marks (comma, etc.) only when a
    line runs long, so segments stay subtitle-sized without over-fragmenting.
    End times are then clamped to the next line's start, which hides the
    occasional inflated end time the CTC alignment emits at VAD-segment seams.
    """
    segs, cur = [], None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            segs.append(cur)
        cur = None

    for t in timestamps:
        tok = t["token"]
        if cur is None:
            cur = {"start": t["start_time"], "end": t["end_time"], "text": ""}
        cur["end"] = t["end_time"]
        cur["text"] += tok
        n = len(cur["text"].strip())
        if any(c in _BREAK for c in tok) and n >= _MIN_CHARS:
            flush()
        elif any(c in _SOFT for c in tok) and n >= 18:
            flush()
    flush()

    out = [{"start_ms": round(s["start"] * 1000),
            "end_ms": round(s["end"] * 1000),
            "text": s["text"].strip()} for s in segs]
    for a, b in zip(out, out[1:]):  # clamp each end to the next start
        if a["end_ms"] > b["start_ms"]:
            a["end_ms"] = b["start_ms"]
    return out


def transcribe(audio: bytes, model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
               device: str = "cpu", language: str | None = None) -> tuple[str, list]:
    """Transcribe in-memory audio bytes with FunASR's Fun-ASR-Nano-2512.

    The .m4s buffer is decoded to a PCM array (see _decode_pcm) and fed to the
    model. Returns (full_text, segments) where segments is a list of
    {start_ms, end_ms, text}. FunASR's own logging/progress is routed to stderr
    so stdout stays clean JSON.

    Timestamps note: Fun-ASR-Nano's own timestamp output is still upstream TODO;
    the per-token timing here comes from FunASR's generic CTC forced-alignment
    wrapped around the model by AutoModel — reliable enough for navigation/
    subtitles, but treat it as best-effort rather than frame-accurate.
    """
    import contextlib

    from funasr import AutoModel

    pcm = _decode_pcm(audio)
    with contextlib.redirect_stdout(sys.stderr):
        asr = AutoModel(
            model=model,
            hub="hf",
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
        gen = dict(input=pcm, cache={}, batch_size=1, itn=True,
                   sentence_timestamp=True)
        if language:
            gen["language"] = language          # e.g. 中文 / 英文 / 日文
        res = asr.generate(**gen)

    r = res[0]
    timestamps = r.get("timestamps") or []
    if timestamps:
        segments = _segment(timestamps)
    else:  # models with native sentence timing (e.g. paraformer)
        segments = [{"start_ms": s.get("start"), "end_ms": s.get("end"),
                     "text": (s.get("text") or "").strip()}
                    for s in (r.get("sentence_info") or [])]
    return r.get("text", ""), segments


def main():
    ap = argparse.ArgumentParser(description="Fetch a Bilibili video's bvid/title/description/audio URL")
    ap.add_argument("id", help="BV id or av id")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print the metadata; skip the audio download and transcription")
    ap.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512",
                    help="FunASR model to transcribe with (default: FunAudioLLM/Fun-ASR-Nano-2512)")
    ap.add_argument("--language", default=None,
                    help="optional language hint: 中文/英文/日文 (default: auto-detect)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for inference, e.g. cpu / cuda:0 (default: cpu)")
    args = ap.parse_args()

    vid = args.id
    if vid.lower().startswith("av"):
        vid = av2bv(int(vid[2:]))
    elif not vid.startswith("BV"):
        raise SystemExit("Please provide a BV id or an av id")

    bili = BiliInfo()
    data = bili.collect(vid)

    if not args.dry_run:
        if not data["audio_url"]:
            raise SystemExit("No audio stream found for this video")
        audio = bili.download_audio(data["audio_url"], vid)
        data["transcript"], data["segments"] = transcribe(
            audio, args.model, args.device, args.language)

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
