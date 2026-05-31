#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av>=17.0.1",
#     "funasr>=1.3.9",
#     "numpy",
#     "requests>=2.34.2",
#     "torch>=2.12.0",
#     "torchaudio>=2.11.0",
#     "tqdm>=4.67.3",
# ]
# ///
"""
bili_info.py — fetch a Bilibili video's: bvid / title / description / audio URL

By default it also downloads the audio and transcribes it with FunASR
(Alibaba's SenseVoiceSmall). Pass --dry-run to skip the download/transcription
and only print the metadata.

Usage: python bili_info.py BV1xxxxxxx [--dry-run] [--model MODEL]
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


def transcribe(audio: bytes, model: str = "iic/SenseVoiceSmall",
               device: str = "cpu", language: str = "auto") -> str:
    """Transcribe in-memory audio bytes with FunASR's SenseVoiceSmall.

    The .m4s buffer is decoded to a PCM array (see _decode_pcm) and fed to the
    model directly. FunASR's own logging/progress is routed to stderr so stdout
    stays clean JSON.
    """
    import contextlib

    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    pcm = _decode_pcm(audio)
    with contextlib.redirect_stdout(sys.stderr):
        asr = AutoModel(
            model=model,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
        res = asr.generate(
            input=pcm,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
    return rich_transcription_postprocess(res[0]["text"])


def main():
    ap = argparse.ArgumentParser(description="Fetch a Bilibili video's bvid/title/description/audio URL")
    ap.add_argument("id", help="BV id or av id")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print the metadata; skip the audio download and transcription")
    ap.add_argument("--model", default="iic/SenseVoiceSmall",
                    help="FunASR model to transcribe with (default: iic/SenseVoiceSmall)")
    ap.add_argument("--language", default="auto",
                    help="language hint for SenseVoice: auto/zh/en/yue/ja/ko (default: auto)")
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
        data["transcript"] = transcribe(audio, args.model, args.device, args.language)

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
