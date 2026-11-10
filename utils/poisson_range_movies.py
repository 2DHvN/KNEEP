"""Animate cached held-out fields and window-averaged entropy maps.

The evaluator selects one test trajectory before caching these arrays. Field
frames are endpoints of the EP windows; entropy maps are averages over their
whole windows, not instantaneous endpoint values. Both EP map arrays already
use per-site EP/time units, so this module applies no extra dx or grid factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import subprocess
import warnings
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

from utils.poisson_range_plots import condition_label, condition_strength


_MOVIES = ("profile", "medium", "predicted")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _movie_data(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = ("movie_fields", "movie_medium_rate", "movie_predicted_rate", "movie_time")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("Cached movie arrays are missing: " + ", ".join(missing)
                         + ". Run evaluation with movie caching enabled before requesting movies.")
    fields = _array(record["movie_fields"])
    medium = _array(record["movie_medium_rate"])
    predicted = _array(record["movie_predicted_rate"])
    if predicted.ndim == 4:
        if not predicted.shape[0]:
            raise ValueError("At least one fitted model is required for a predicted movie")
        predicted = predicted.mean(axis=0)  # Average independent fits only.
    time = _array(record["movie_time"])
    if (fields.ndim != 3 or not fields.shape[0] or medium.shape != fields.shape
            or predicted.shape != fields.shape or time.shape != (fields.shape[0],)):
        raise ValueError("Movie arrays must share [frame,y,x]; predictions may additionally have a fit-seed axis")
    if any(not np.isfinite(value).all() for value in (fields, medium, predicted, time)):
        raise ValueError("Movie arrays must be finite")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Cached movie times must be strictly increasing")
    interval = float(record["interval"])
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("The saved observation interval must be positive and finite")
    result = dict(profile=fields, medium=medium, predicted=predicted, time=time)
    for key in ("movie_window_start", "movie_window_end", "movie_field_time"):
        if key in record:
            value = _array(record[key])
            if value.shape != time.shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must have one finite value per cached frame")
            result[key] = value
    if ("movie_window_start" in result) != ("movie_window_end" in result):
        raise ValueError("Both movie window start and end indices are required together")
    if "movie_window_start" in result:
        starts, ends = result["movie_window_start"], result["movie_window_end"]
        if np.any(starts < 0) or np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
            raise ValueError("Cached EP windows must be ordered, nonoverlapping and nonempty")
    return result


def _normalization(values: np.ndarray, *, log_scale: bool) -> Normalize:
    limit = max(float(np.abs(values).max()), 1e-12)
    if log_scale:
        return SymLogNorm(linthresh=limit * 1e-3, vmin=-limit, vmax=limit, base=10)
    return Normalize(vmin=-limit, vmax=limit)


def animate_test_trajectory(
    record: Mapping[str, Any],
    movie: str = "combined",
    *,
    log_scale: bool = False,
    max_frames: int | None = 64,
    interval_ms: float = 100,
) -> FuncAnimation:
    """Animate the evaluator's cached test trajectory, with fixed color limits.

    ``movie`` is profile, medium, predicted, or combined (three synchronized
    panels). Profile times are EP-window endpoints. EP timestamps are window
    centers and are explicitly labelled as window averages. ``max_frames``
    selects evenly spaced existing frames; it does not interpolate dynamics.
    Signed logarithmic colors apply only to EP maps; field colors stay linear.
    """
    if movie not in (*_MOVIES, "combined"):
        raise ValueError("movie must be profile, medium, predicted or combined")
    if not np.isfinite(interval_ms) or interval_ms <= 0:
        raise ValueError("interval_ms must be positive and finite")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None")
    data = _movie_data(record)
    count = len(data["time"])
    selected = (np.arange(count) if max_frames is None or count <= max_frames
                else np.unique(np.linspace(0, count - 1, max_frames, dtype=int)))
    modes = _MOVIES if movie == "combined" else (movie,)
    fig, axes = plt.subplots(1, len(modes), figsize=(4.3 * len(modes), 4.5), squeeze=False)
    dx = float(record.get("dx", 1))
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be positive and finite")
    height, width = data["profile"].shape[-2:]
    extent = (0, width * dx, 0, height * dx)
    titles = {"profile": "Field profile phi", "medium": "Direct midpoint medium EP",
              "predicted": "KNEEP EP (fit-seed mean)"}
    artists = []
    for ax, mode in zip(axes.flat, modes):
        artist = ax.imshow(data[mode][selected[0]], origin="lower", cmap="RdBu_r", extent=extent,
                           norm=_normalization(data[mode], log_scale=log_scale and mode != "profile"))
        ax.set(title=titles[mode], xlabel="x", ylabel="y")
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.8, label="phi" if mode == "profile" else "EP / time / site")
        if log_scale and mode != "profile":
            limit = float(artist.norm.vmax)
            colorbar.set_ticks(np.asarray([-1, -.01, 0, .01, 1]) * limit)
            colorbar.formatter = FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.0e}")
            colorbar.update_ticks()
        artists.append(artist)
    time_label = fig.text(0.5, 0.055, "", ha="center", fontsize=9)
    fig.suptitle(condition_label(record), fontsize=12)
    fig.text(0.5, 0.012, "Fixed signed color limits; EP maps are window averages. Learned and medium entropy gauges differ.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    def update(position: int):
        index = int(selected[position])
        for mode, artist in zip(modes, artists):
            artist.set_data(data[mode][index])
        interval = float(record.get("interval", 1))
        bounds = None
        if "movie_window_start" in data:
            bounds = (data["movie_window_start"][index] * interval,
                      data["movie_window_end"][index] * interval)
        field_time = (data["movie_field_time"][index] if "movie_field_time" in data
                      else bounds[1] if bounds is not None else None)
        parts = []
        if "profile" in modes:
            parts.append(f"field endpoint t={field_time:g}" if field_time is not None
                         else f"field endpoint, cached frame {index}")
        if any(mode != "profile" for mode in modes):
            parts.append((f"EP window [{bounds[0]:g}, {bounds[1]:g}]" if bounds is not None else "EP window average")
                         + f"; center t={data['time'][index]:g}")
        time_label.set_text(" | ".join(parts))
        return (*artists, time_label)

    update(0)
    animation = FuncAnimation(fig, update, frames=len(selected), interval=interval_ms,
                              blit=False, repeat=True, cache_frame_data=False)
    # Keep useful metadata available to callers without exposing cache internals.
    animation.cached_frame_indices = selected
    return animation


def _filename(record: Mapping[str, Any]) -> str:
    # Include BOTH physical parameters even when a historical condition_id
    # contains ell alone, preventing collisions in a multi-coupling export.
    raw = ("no_interaction" if condition_strength(record) == 0 else
           f"g_{condition_strength(record):g}_ell_{float(record['ell']):g}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip(".") or "condition"


def save_test_movies(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    output_dir: str | Path,
    *,
    fps: float = 10,
    log_scale: bool = False,
    max_frames: int | None = 64,
    combined: bool = False,
) -> list[Path]:
    """Export profile, medium-EP and learned-EP videos for each cached condition.

    MP4 uses ffmpeg when Matplotlib can find it; otherwise the same frames are
    saved as Pillow GIFs. A failed ffmpeg encoding also falls back to GIF. Set
    ``combined=True`` to additionally export a synchronized three-panel movie.
    No simulation, inference or temporal resampling is performed here.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    collection = [records] if isinstance(records, Mapping) else list(records)
    if not collection:
        raise ValueError("At least one evaluation record is required")
    for record in collection:
        _movie_data(record)  # Validate all records before creating outputs.
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    modes = (*_MOVIES, "combined") if combined else _MOVIES
    paths = []
    for record in collection:
        for mode in modes:
            animation = animate_test_trajectory(record, mode, log_scale=log_scale,
                max_frames=max_frames, interval_ms=1000 / fps)
            prefix = f"{_filename(record)}_{mode}" + ("_symlog" if log_scale and mode != "profile" else "")
            try:
                use_ffmpeg = writers.is_available("ffmpeg")
                path = folder / f"{prefix}.{'mp4' if use_ffmpeg else 'gif'}"
                if use_ffmpeg:
                    try:
                        animation.save(str(path), writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
                    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
                        path.unlink(missing_ok=True)
                        warnings.warn(f"ffmpeg could not encode {prefix}; saving GIF instead: {error}", RuntimeWarning)
                        path = folder / f"{prefix}.gif"
                        animation.save(str(path), writer=PillowWriter(fps=fps))
                else:
                    animation.save(str(path), writer=PillowWriter(fps=fps))
                paths.append(path)
            finally:
                plt.close(animation._fig)
    return paths
