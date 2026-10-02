"""Offline test for the GIF encoder (no emulator, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from typesafe_mario.record import write_gif

Image = pytest.importorskip("PIL.Image")


def _distinct_frames(n: int = 8, size: int = 24) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        a = np.zeros((size, size, 3), np.uint8)
        a[:, :, i % 3] = (i * 30 + 40) % 256
        frames.append(a)
    return frames


def test_write_gif_produces_a_multiframe_animation(tmp_path):
    out = tmp_path / "clip.gif"
    frames = _distinct_frames(8)
    # decimate=1 keeps every frame.
    n = write_gif(frames, out, fps=30, decimate=1)
    assert n == 8
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) == 8


def test_write_gif_decimates(tmp_path):
    out = tmp_path / "clip2.gif"
    n = write_gif(_distinct_frames(10), out, decimate=2)
    assert n == 5  # every 2nd frame


def test_write_gif_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        write_gif([], tmp_path / "x.gif")
