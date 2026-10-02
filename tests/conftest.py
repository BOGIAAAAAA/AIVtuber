from __future__ import annotations

import io
import struct
import wave

import pytest


def make_wav_bytes(frames: int = 2048, rate: int = 32000, amplitude: int = 0, channels: int = 1) -> bytes:
    """產生一段 16-bit 的 WAV（與 GPT-SoVITS 預設輸出相同格式）；amplitude 為 0 時是全靜音。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        # 正負交替的方波，只要不是全 0 就好
        samples = [amplitude if index % 2 else -amplitude for index in range(frames * channels)]
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@pytest.fixture
def speech_wav() -> bytes:
    """有聲音（不是全靜音）的 WAV，代表合成成功的回應。"""
    return make_wav_bytes(amplitude=1000)


@pytest.fixture
def make_wav():
    """回傳可自訂長度、取樣率與音量的 WAV 產生函式。"""
    return make_wav_bytes
