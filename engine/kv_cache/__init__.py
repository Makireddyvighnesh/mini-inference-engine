"""Explicit KV-cache ownership and accounting."""

from .contiguous import ContiguousKvCache, KvCacheCapacityError, KvCacheStateError

__all__ = [
    "ContiguousKvCache",
    "KvCacheCapacityError",
    "KvCacheStateError",
]
