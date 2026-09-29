"""Bounded, exact-token prefix reuse over immutable paged KV blocks."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from .paged import PagedKvCache, PagedKvOutOfMemoryError, PagedKvStateError


def _block_digest(tokens: tuple[int, ...]) -> bytes:
    """Stable fingerprint of one complete token block; never used without verification."""

    digest = hashlib.blake2b(digest_size=16)
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.digest()


class PagedPrefixCache:
    """Retain completed prompt blocks for later requests on the same model.

    Keys are exact token IDs. Only whole blocks are shared. A lookup always
    leaves at least one prompt token to run so fresh logits produce the first
    output token. The instance must be scoped to one model and KV dtype.
    """

    def __init__(self, cache: PagedKvCache, *, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.cache = cache
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[int, ...], str] = OrderedDict()
        self._first_block_index: dict[bytes, dict[tuple[int, ...], None]] = {}
        self._next_id = 0
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0
        self.evictions = 0
        self.eviction_wall_ms = 0.0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _evict(self, key: tuple[int, ...]) -> None:
        started_ns = time.perf_counter_ns()
        owner = self._entries.pop(key)
        digest = _block_digest(key[: self.cache.allocator.block_size])
        indexed = self._first_block_index[digest]
        del indexed[key]
        if not indexed:
            del self._first_block_index[digest]
        self.cache.release(owner)
        self.evictions += 1
        self.eviction_wall_ms += (time.perf_counter_ns() - started_ns) / 1_000_000.0

    def clear(self) -> None:
        for key in tuple(self._entries):
            self._evict(key)

    def _freeable_blocks(self, key: tuple[int, ...]) -> int:
        """Blocks that evicting ``key`` would return to the pool right now."""

        allocator = self.cache.allocator
        table = allocator.get_block_table(self._entries[key])
        return sum(1 for block_id in table.block_ids if allocator.block_refcount(block_id) == 1)

    def _evict_lru_until(
        self, required_blocks: int, *, protect: tuple[int, ...] | None = None
    ) -> bool:
        """Evict least-recently-used entries until ``required_blocks`` are free.

        Entries whose blocks are all still referenced by running requests are
        skipped: evicting them frees nothing and only destroys future reuse.
        Evicting one entry can make another's blocks freeable, so the scan
        restarts after every eviction.
        """

        allocator = self.cache.allocator
        while allocator.free_block_count < required_blocks:
            victim = next(
                (key for key in self._entries if key != protect and self._freeable_blocks(key)),
                None,
            )
            if victim is None:
                break
            self._evict(victim)
        return allocator.free_block_count >= required_blocks

    def evict_until_free(self, required_blocks: int) -> bool:
        """Reclaim idle prefix owners without disturbing in-flight references."""

        if required_blocks < 0:
            raise ValueError("required_blocks must be non-negative")
        return self._evict_lru_until(required_blocks)

    def publish(self, sequence_id: str, token_ids: Sequence[int]) -> int:
        """Retain the full prompt blocks already written for a request."""

        tokens = tuple(int(value) for value in token_ids)
        block_size = self.cache.allocator.block_size
        retained = (len(tokens) // block_size) * block_size
        if retained == 0:
            return 0
        table = self.cache.allocator.get_block_table(sequence_id)
        if table.token_count < len(tokens):
            raise PagedKvStateError("source KV is shorter than the published prompt")
        key = tokens[:retained]
        if key in self._entries:
            self._entries.move_to_end(key)
            return retained

        while len(self._entries) >= self.max_entries:
            self._evict(next(iter(self._entries)))
        while True:
            owner = f"__prefix_cache_{self._next_id}"
            self._next_id += 1
            if owner not in self.cache.allocator.sequence_ids:
                break
        self.cache.allocator.share_prefix(sequence_id, owner, retained)
        self._entries[key] = owner
        digest = _block_digest(key[:block_size])
        self._first_block_index.setdefault(digest, {})[key] = None
        return retained

    def _best_match(self, tokens: tuple[int, ...]) -> tuple[tuple[int, ...] | None, int]:
        block_size = self.cache.allocator.block_size
        limit = ((len(tokens) - 1) // block_size) * block_size
        if limit < block_size:
            return None, 0
        best_key: tuple[int, ...] | None = None
        best_length = 0
        first_digest = _block_digest(tokens[:block_size])
        for key in self._first_block_index.get(first_digest, ()):
            common = 0
            for start in range(0, min(len(key), limit), block_size):
                source = key[start : start + block_size]
                target = tokens[start : start + block_size]
                # Hashes avoid most comparisons. Exact IDs still decide reuse,
                # including when two different blocks collide deliberately.
                if _block_digest(source) != _block_digest(target) or source != target:
                    break
                common += block_size
            if common > best_length:
                best_key, best_length = key, common
        return best_key, best_length

    def reusable_tokens(self, token_ids: Sequence[int]) -> int:
        """Estimate prefill work for admission without changing cache ownership."""

        return self._best_match(tuple(int(value) for value in token_ids))[1]

    def attach(
        self,
        sequence_id: str,
        token_ids: Sequence[int],
        *,
        output_tokens: int = 1,
    ) -> int:
        """Attach the longest cached prefix and leave room for the suffix.

        Cold and warm requests must both fit in the physical pool. LRU entries
        are evicted before changing the new request's block table.
        """

        tokens = tuple(int(value) for value in token_ids)
        if not tokens or int(output_tokens) < 1:
            raise ValueError("prompt and output length must be positive")
        allocator = self.cache.allocator
        if sequence_id in allocator.sequence_ids:
            raise PagedKvStateError(f"sequence {sequence_id!r} is already allocated")
        block_size = allocator.block_size
        key, reused = self._best_match(tokens)

        def needed_blocks(prefix_tokens: int) -> int:
            return math.ceil((len(tokens) + int(output_tokens) - 1 - prefix_tokens) / block_size)

        # Prefer keeping the matched entry; give it up only if the request
        # cannot fit even after every other reclaimable entry is gone.
        if not self._evict_lru_until(needed_blocks(reused), protect=key) and key is not None:
            self._evict(key)
            key, reused = None, 0
            self._evict_lru_until(needed_blocks(reused))
        if allocator.free_block_count < needed_blocks(reused):
            raise PagedKvOutOfMemoryError("prompt and output cannot fit in the KV block pool")

        if key is None or reused == 0:
            allocator.allocate(sequence_id)
            return 0
        allocator.share_prefix(self._entries[key], sequence_id, reused)
        self._entries.move_to_end(key)
        return reused

    def record_admission(self, reused_tokens: int) -> None:
        """Count one admitted request as a hit or miss.

        Kept separate from ``attach`` because a scheduler may release and
        defer an attached request; counting at attach time double-counts it.
        """

        if reused_tokens > 0:
            self.hits += 1
            self.reused_tokens += int(reused_tokens)
        else:
            self.misses += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "entries": self.entry_count,
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "evictions": self.evictions,
            "eviction_wall_ms": self.eviction_wall_ms,
            "allocated_physical_blocks": self.cache.allocator.allocated_block_count,
        }
