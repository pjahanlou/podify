"""The audio stage — narrate the script with Piper (local TTS) and encode an MP3.

Piper runs fully offline. It has no inter-sentence pause control, so we insert silence
between paragraphs ourselves, then encode the stitched 16-bit PCM to MP3 with lameenc
(no ffmpeg required).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import DEFAULT_VOICE, REPO_ROOT

_VOICES_DIR = REPO_ROOT / "voices"
_LENGTH_SCALE = 1.05       # slightly slower than default -> relaxed walking pace
_PARAGRAPH_SILENCE = 0.45  # seconds of silence between paragraphs
_BITRATE = 128             # kbps


def synthesize(script: str, voice: str, out_path: Path) -> Path:
    from piper import PiperVoice, SynthesisConfig

    model = _ensure_voice(voice)
    v = PiperVoice.load(str(model))
    cfg = SynthesisConfig(length_scale=_LENGTH_SCALE)

    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]
    pcm = bytearray()
    sample_rate = 22050
    silence = b""
    for i, para in enumerate(paragraphs):
        for chunk in v.synthesize(para, cfg):
            pcm += chunk.audio_int16_bytes
            sample_rate = chunk.sample_rate
        if not silence:
            silence = _silence(sample_rate, _PARAGRAPH_SILENCE)
        if i < len(paragraphs) - 1:
            pcm += silence
        print(f"[audio]   narrated paragraph {i + 1}/{len(paragraphs)}")

    _encode_mp3(bytes(pcm), sample_rate, out_path)
    secs = len(pcm) // 2 / sample_rate
    print(f"[audio] ~{secs / 60:.1f} min of audio -> {out_path}")
    return out_path


def _ensure_voice(voice: str) -> Path:
    model = _VOICES_DIR / f"{voice}.onnx"
    if model.exists():
        return model
    _VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[audio] downloading Piper voice {voice} (~60MB, one time)...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", "--download-dir",
         str(_VOICES_DIR), voice],
        check=True,
    )
    return model


def _silence(sample_rate: int, seconds: float) -> bytes:
    return b"\x00\x00" * int(sample_rate * seconds)  # mono 16-bit zeros


def _encode_mp3(pcm: bytes, sample_rate: int, out_path: Path) -> None:
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(_BITRATE)
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(1)
    enc.set_quality(2)  # 0=best/slowest .. 9=worst/fastest
    data = enc.encode(pcm) + enc.flush()
    out_path.write_bytes(data)
