'''Long-term memory with persistence and importance-based eviction.'''

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base import MemoryEntry, MemoryStore


def _tokenize(text: str) -> set[str]:
    '''Simple whitespace tokenizer.'''
    return set(text.lower().split())


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    '''Token-level Jaccard similarity.'''
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LongTermMemory(MemoryStore):
    '''Persistent cross-task memory with importance-based eviction.

    Retrieval uses a weighted combination of:
    - Keyword similarity (Jaccard)
    - Importance score
    - Time decay

    Persistence is via JSONL files at ~/.sema/memory/{agent_id}/long_term.jsonl
    '''

    def __init__(
        self,
        agent_id: str = 'default',
        capacity: int = 1000,
        persistence_dir: str | Path | None = None,
        similarity_weight: float = 0.5,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2,
        decay_rate: float = 0.001,
    ) -> None:
        self._agent_id = agent_id
        self._capacity = capacity
        self._entries: list[MemoryEntry] = []
        self._similarity_weight = similarity_weight
        self._importance_weight = importance_weight
        self._recency_weight = recency_weight
        self._decay_rate = decay_rate

        if persistence_dir is None:
            self._persistence_path = Path.home() / '.sema' / 'memory' / agent_id / 'long_term.jsonl'
        else:
            self._persistence_path = Path(persistence_dir) / 'long_term.jsonl'

        self._load()

    def _load(self) -> None:
        '''Load entries from persistence file.'''
        if not self._persistence_path.exists():
            return
        try:
            with open(self._persistence_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._entries.append(MemoryEntry(**data))
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        '''Save all entries to persistence file.'''
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persistence_path, 'w', encoding='utf-8') as f:
            for entry in self._entries:
                f.write(entry.model_dump_json() + '\n')

    def _score_entry(self, entry: MemoryEntry, query_tokens: set[str], now: float) -> float:
        '''Compute retrieval score for an entry.'''
        entry_tokens = _tokenize(entry.content)
        similarity = _jaccard_similarity(query_tokens, entry_tokens)

        # Time decay: exponential decay based on age in seconds
        age = max(0.0, now - entry.timestamp)
        recency = 1.0 / (1.0 + self._decay_rate * age)

        return (
            self._similarity_weight * similarity
            + self._importance_weight * entry.importance
            + self._recency_weight * recency
        )

    async def add(self, entry: MemoryEntry) -> None:
        '''Add an entry. Evicts least important entry if at capacity.'''
        if len(self._entries) >= self._capacity:
            # Evict the entry with lowest importance
            min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].importance)
            self._entries.pop(min_idx)

        self._entries.append(entry)
        self._save()

    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        '''Retrieve entries by weighted similarity, importance, and recency.'''
        if not self._entries:
            return []

        if not query.strip():
            # Return most important entries
            sorted_entries = sorted(self._entries, key=lambda e: e.importance, reverse=True)
            return sorted_entries[:top_k]

        query_tokens = _tokenize(query)
        now = time.time()

        scored = [(self._score_entry(entry, query_tokens, now), entry) for entry in self._entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def clear(self) -> None:
        '''Clear all entries and remove persistence file.'''
        self._entries.clear()
        if self._persistence_path.exists():
            self._persistence_path.unlink()

    async def size(self) -> int:
        '''Return the number of entries.'''
        return len(self._entries)
