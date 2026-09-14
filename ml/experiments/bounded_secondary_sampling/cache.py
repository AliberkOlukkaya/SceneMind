"""Bounded deterministic JPEG cache used by the experiment."""

import os
import time
from pathlib import Path
from typing import Callable

from .policy import POLICY_VERSION, cache_key

Extractor = Callable[[Path, float, Path], None]


class SecondaryFrameCache:
    def __init__(
        self, root: Path, max_bytes: int, extractor: Extractor,
        policy_version: str = POLICY_VERSION,
    ):
        if max_bytes <= 0:
            raise ValueError("cache byte limit must be positive")
        self.root = Path(root) / policy_version
        self.max_bytes = max_bytes
        self.extractor = extractor
        self.policy_version = policy_version
        self.hits = 0
        self.misses = 0

    def path_for(self, video_id: str, timestamp: float) -> Path:
        key = cache_key(video_id, timestamp, self.policy_version)
        return self.root / key[:2] / f"{key}.jpg"

    def get(self, video_id: str, timestamp: float, source: Path) -> tuple[Path, bool, float]:
        destination = self.path_for(video_id, timestamp)
        if destination.is_file():
            self.hits += 1
            os.utime(destination, None)
            return destination, True, 0.0
        self.misses += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.jpg")
        started = time.perf_counter()
        try:
            self.extractor(source, timestamp, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("secondary frame extraction produced no image")
            if temporary.stat().st_size > self.max_bytes:
                raise RuntimeError("secondary frame exceeds the cache byte limit")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        self.cleanup(protected=destination)
        return destination, False, elapsed

    def cleanup(self, protected: Path | None = None) -> int:
        files = sorted(
            self.root.glob("*/*.jpg"), key=lambda path: path.stat().st_mtime
        )
        total = sum(path.stat().st_size for path in files)
        removed = 0
        for path in files:
            if total <= self.max_bytes:
                break
            if protected is not None and path == protected:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            removed += 1
        return removed

    def stats(self) -> dict:
        files = list(self.root.glob("*/*.jpg"))
        requests = self.hits + self.misses
        return {
            "requests": requests, "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / requests if requests else None,
            "files": len(files), "bytes": sum(path.stat().st_size for path in files),
            "max_bytes": self.max_bytes, "policy_version": self.policy_version,
        }
