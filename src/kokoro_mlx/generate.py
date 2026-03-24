# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Text-to-audio generation pipeline for Kokoro TTS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import KokoroConfig
from .model import KokoroModel
from .phonemize import Phonemizer
from .voices import VoiceManager

SAMPLE_RATE = 24000


@dataclass
class TokenTiming:
    """Timing information for a single phoneme token."""

    token: str    # phoneme character
    start: float  # start time in seconds
    end: float    # end time in seconds


def _resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample audio by exactly 2x using FFT zero-padding.

    For a real signal of length N, the rfft has N//2+1 bins.  Padding the
    spectrum to 2x length and taking the irfft produces a perfectly
    bandlimited 2x-upsampled signal.  Numpy-only, no extra dependencies.
    """
    n = len(audio)
    spectrum = np.fft.rfft(audio)
    out_len = n * 2
    padded = np.zeros(out_len // 2 + 1, dtype=spectrum.dtype)
    padded[: len(spectrum)] = spectrum
    return np.fft.irfft(padded, n=out_len).astype(np.float32) * 2.0


def _seconds_per_frame(config: KokoroConfig) -> float:
    """Duration of one acoustic frame in seconds at native 24 kHz."""
    hop_samples = math.prod(config.istftnet.upsample_rates) * config.istftnet.gen_istft_hop_size
    return hop_samples / SAMPLE_RATE


def generate(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[TokenTiming]]:
    """Full text-to-audio pipeline.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier (>1 is faster, <1 is slower).
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).

    Returns:
        Tuple of (audio, tokens) where audio is a float32 numpy array at the
        requested sample rate and tokens is a list of TokenTiming with per-phoneme
        start/end times in seconds.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return np.array([], dtype=np.float32), []

    voice_array = voice_manager.load_voice(voice)
    hop_secs = _seconds_per_frame(config)

    audio_chunks: list[np.ndarray] = []
    all_timings: list[TokenTiming] = []
    time_offset = 0.0

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio, pred_dur_np = model.forward(phonemes, style, speed)
        chunk = np.array(audio.tolist(), dtype=np.float32)

        # Build per-token timestamps (pred_dur_np includes BOS at [0] and EOS at [-1])
        phoneme_chars = [c for c in phonemes if c in config.vocab]
        t = time_offset
        for char, dur in zip(phoneme_chars, pred_dur_np[1:-1]):
            end = t + int(dur) * hop_secs
            all_timings.append(TokenTiming(token=char, start=t, end=end))
            t = end
        time_offset += len(chunk) / SAMPLE_RATE

        audio_chunks.append(chunk)

    result = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

    if sample_rate == 48000 and len(result) > 0:
        result = _resample_2x(result)

    return result, all_timings


def generate_stream(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
):
    """Generate audio in chunks as they are produced.

    Yields (chunk, tokens) tuples — one per phoneme chunk — where chunk is a
    float32 numpy array and tokens is a list of TokenTiming with per-phoneme
    start/end times in seconds.  Suitable for low-latency streaming playback.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier.
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).

    Yields:
        (chunk, tokens) tuples, one per sentence chunk.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return

    voice_array = voice_manager.load_voice(voice)
    hop_secs = _seconds_per_frame(config)
    upsample = sample_rate == 48000
    time_offset = 0.0

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio, pred_dur_np = model.forward(phonemes, style, speed)
        chunk = np.array(audio.tolist(), dtype=np.float32)

        # Build per-token timestamps for this chunk
        phoneme_chars = [c for c in phonemes if c in config.vocab]
        timings: list[TokenTiming] = []
        t = time_offset
        for char, dur in zip(phoneme_chars, pred_dur_np[1:-1]):
            end = t + int(dur) * hop_secs
            timings.append(TokenTiming(token=char, start=t, end=end))
            t = end
        time_offset += len(chunk) / SAMPLE_RATE

        if upsample and len(chunk) > 0:
            chunk = _resample_2x(chunk)
        yield chunk, timings
