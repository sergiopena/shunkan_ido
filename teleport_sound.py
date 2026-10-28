"""Non-blocking teleport sound for macOS (uses the built-in ``afplay``).

If ``assets/teleport.MP3`` (or .mp3/.wav) exists it is played; otherwise a short
rising "shiun" whoosh is synthesized once into ``assets/teleport_synth.wav``.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CANDIDATES = ("teleport.MP3", "teleport.mp3", "teleport.wav")
SYNTH_PATH = ASSETS_DIR / "teleport_synth.wav"


def synthesize(path: Path = SYNTH_PATH, rate: int = 44100) -> Path:
    """Write a 0.55 s whoosh: a rising sweep over band-limited noise."""
    duration = 0.55
    t = np.linspace(0.0, duration, int(rate * duration), endpoint=False)
    rng = np.random.default_rng(17)
    frequency = 260.0 * (2400.0 / 260.0) ** (np.clip(t / 0.32, 0, 1) ** 1.4)
    sweep = np.sin(2 * np.pi * np.cumsum(frequency) / rate)
    shimmer = np.sin(2 * np.pi * np.cumsum(frequency * 2.01) / rate) * 0.35
    noise = rng.standard_normal(t.size)
    noise = np.convolve(noise, np.ones(12) / 12, mode="same")  # soften the hiss
    envelope = np.minimum(1.0, t / 0.03) * np.exp(-np.maximum(0.0, t - 0.20) * 9.0)
    signal = (0.55 * sweep + shimmer * 0.5 + 0.45 * noise) * envelope
    signal = signal / max(1e-6, np.abs(signal).max()) * 0.8
    data = (signal * 32767).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(data.tobytes())
    return path


def find_sound() -> Path:
    for name in CANDIDATES:
        candidate = ASSETS_DIR / name
        if candidate.is_file():
            return candidate
    return SYNTH_PATH if SYNTH_PATH.is_file() else synthesize()


class TeleportSound:
    def __init__(self, path: Path | None = None):
        self.path = path or find_sound()
        self.player = shutil.which("afplay") or shutil.which("ffplay")
        self.process: subprocess.Popen | None = None
        if self.player is None:
            print("[Goku] afplay/ffplay not found: running without sound.")

    def play(self) -> None:
        if self.player is None:
            return
        self.stop()
        command = [self.player, str(self.path)]
        if self.player.endswith("ffplay"):
            command = [self.player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(self.path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            print(f"[Goku] Could not play sound: {error}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None

    close = stop
