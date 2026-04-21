'''Short-term memory with fixed-capacity sliding window.'''

from __future__ import annotations

from collections import deque

from .base import MemoryEntry, MemoryStore


def _tokenize(text: str) -> set[str]:
    '''Simple whitespace tokenizer for keyword matching.'''
    return set(text.lower().split())


class ShortTermMemory(MemoryStore):
    '''Fixed-capacity sliding window memory (FIFO eviction).

    Stores entries in-memory only, no persistence.
    Retrieval is recency-based, filtered by keyword overlap with the query.
    '''

    def __init__(self, capacity: int = 50) -> None:
        self._capacity = capacity
        self._entries: deque[MemoryEntry] = deque(maxlen=capacity)

    async def add(self, entry: MemoryEntry) -> None:
        '''Add an entry. Oldest entry is evicted if at capacity.'''
        self._entries.append(entry)

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        '''Retrieve most recent entries that have keyword overlap with query.

        Returns up to top_k entries, most recent first.
        If query is empty, returns the most recent top_k entries.
        '''
        if not self._entries:
            return []

        if not query.strip():
            # Return most recent entries
            entries = list(self._entries)
            entries.reverse()
            return entries[:top_k]

        query_tokens = _tokenize(query)
        scored: list[tuple[float, int, MemoryEntry]] = []

        for idx, entry in enumerate(self._entries):
            entry_tokens = _tokenize(entry.content)
            overlap = len(query_tokens & entry_tokens)
            if overlap > 0:
                # Score by overlap, use index as recency tiebreaker (higher = more recent)
                scored.append((overlap, idx, entry))

        # Sort by overlap descending, then recency descending
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [entry for _, _, entry in scored[:top_k]]

    async def clear(self) -> None:
        '''Clear all entries.'''
        self._entries.clear()

    async def size(self) -> int:
        '''Return the number of entries.'''
        return len(self._entries)

    def get_all(self) -> list[MemoryEntry]:
        '''Return all entries (most recent last).'''
        return list(self._entries)
