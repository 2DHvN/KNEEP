"""Memory-bounded access to independent trajectory files.

Each file stores a CPU ``video`` tensor with shape ``[T,C,H,W]`` or
``[1,T,C,H,W]``.  Files are memory-mapped, and only the requested frame pairs
are copied into the returned batch.  No concatenated training movie is built.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch


def _channel_moments(chunks):
    """Merge float64 population moments, then return the sample deviation."""
    count = 0
    mean = None
    m2 = None
    for chunk in chunks:
        # Chunk shape is [frames, channels, height, width].  Conversion is
        # limited to a small number of frames, including for mapped videos.
        values = chunk.to(device="cpu", dtype=torch.float64)
        batch_count = values.shape[0] * values.shape[2] * values.shape[3]
        variance, batch_mean = torch.var_mean(values, dim=(0, 2, 3), correction=0)
        batch_m2 = variance * batch_count
        if mean is None:
            mean, m2, count = batch_mean, batch_m2, batch_count
        else:
            total = count + batch_count
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
            mean = mean + delta * (batch_count / total)
            count = total
    if mean is None or count < 2:
        raise ValueError("at least two scalar observations per channel are required")
    std = torch.sqrt(m2 / (count - 1))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise RuntimeError("training data contain a nonfinite or zero-variance channel")
    return mean.reshape(1, 1, -1, 1, 1), std.reshape(1, 1, -1, 1, 1)


class ShardedVideo:
    """Logical ``[M,T,C,H,W]`` movie backed by one file per trajectory.

    At most ``max_cached_shards`` mapped video tensors are retained.  The OS
    may retain file pages in its reclaimable page cache; the Python process
    never allocates the full dataset.  Eviction drops the tensor and its map
    once the current frame selection has been copied.  Instances are intended
    for one training worker; open a separate instance in each spawned process.
    """

    ndim = 5

    def __init__(
        self,
        shards: Sequence[str | Path],
        *,
        key: str = "video",
        max_cached_shards: int = 2,
        shape: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        if isinstance(max_cached_shards, bool) or max_cached_shards < 1:
            raise ValueError("max_cached_shards must be a positive integer")
        self.shards = tuple(Path(path).resolve() for path in shards)
        if not self.shards:
            raise ValueError("at least one trajectory shard is required")
        self.key = key
        self.max_cached_shards = int(max_cached_shards)
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._normalization: tuple[torch.Tensor, torch.Tensor] | None = None
        self.shape = tuple(shape) if shape is not None else None
        if self.shape is not None and (len(self.shape) != 5 or self.shape[0] != len(self.shards)):
            raise ValueError("shape must be [number of shards,T,C,H,W]")
        first = self._load_shard(0)
        observed_shape = (len(self.shards), *first.shape)
        if self.shape is not None and self.shape != observed_shape:
            raise ValueError("declared movie shape does not match its first shard")
        self.shape = observed_shape
        self.dtype = first.dtype
        self.device = torch.device("cpu")
        self._cache[0] = first

    def _load_shard(self, index: int) -> torch.Tensor:
        payload = torch.load(self.shards[index], map_location="cpu", weights_only=True,
                             mmap=True)
        video = payload[self.key]
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"{self.shards[index]} does not contain a video tensor")
        if video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        if video.ndim != 4 or video.shape[0] < 2 or min(video.shape[1:]) < 1:
            raise ValueError("each shard must contain [time>=2,C,H,W] or [1,time>=2,C,H,W]")
        if not video.is_floating_point():
            raise ValueError("video tensors must use a floating-point dtype")
        if self.shape is not None and tuple(video.shape) != tuple(self.shape[1:]):
            raise ValueError(f"inconsistent trajectory shape in {self.shards[index]}")
        if hasattr(self, "dtype") and video.dtype != self.dtype:
            raise ValueError(f"inconsistent video dtype in {self.shards[index]}")
        # Other tensors (reference EP, spectra, diagnostics) are not retained.
        return video

    def _get_shard(self, index: int) -> torch.Tensor:
        video = self._cache.pop(index, None)
        if video is None:
            video = self._load_shard(index)
        self._cache[index] = video
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return video

    def fetch_pairs(self, ensemble: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Copy pairs in the requested order, reading each selected shard once."""
        ensemble = torch.as_tensor(ensemble, dtype=torch.long, device="cpu")
        time = torch.as_tensor(time, dtype=torch.long, device="cpu")
        if ensemble.ndim != 1 or ensemble.shape != time.shape:
            raise ValueError("ensemble and time must be matching one-dimensional indices")
        if (torch.any(ensemble < 0) or torch.any(ensemble >= self.shape[0])
                or torch.any(time < 0) or torch.any(time >= self.shape[1] - 1)):
            raise IndexError("trajectory or transition index is out of range")
        pairs = torch.empty((len(ensemble), 2, *self.shape[2:]), dtype=self.dtype)
        for index in torch.unique(ensemble, sorted=True).tolist():
            positions = torch.nonzero(ensemble == index, as_tuple=True)[0]
            video = self._get_shard(index)
            selected_time = time[positions]
            pairs[positions, 0] = video[selected_time]
            pairs[positions, 1] = video[selected_time + 1]
        return pairs

    def channel_normalization(self, *, frame_batch_size: int = 64):
        """Per-channel moments over every saved frame, with sample correction."""
        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")
        if self._normalization is None:
            def chunks():
                for index in range(self.shape[0]):
                    video = self._get_shard(index)
                    for start in range(0, self.shape[1], frame_batch_size):
                        yield video[start:start + frame_batch_size]

            self._normalization = _channel_moments(chunks())
        return tuple(value.clone() for value in self._normalization)

    def close(self) -> None:
        """Release retained mapped video tensors; the adapter can be reused."""
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def video_channel_normalization(video: torch.Tensor | ShardedVideo):
    """Return identical float64 streaming moments for dense or sharded video."""
    if isinstance(video, ShardedVideo):
        return video.channel_normalization()
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError("expected video with shape [M,time>=2,C,H,W]")
    return _channel_moments(
        video[index, start:start + 64]
        for index in range(video.shape[0])
        for start in range(0, video.shape[1], 64)
    )
