'''Unified memory system combining short-term and long-term memory.'''

from __future__ import annotations

from .base import MemoryEntry
from .short_term import ShortTermMemory
from .long_term import LongTermMemory


class MemorySystem:
    '''Unified interface over short-term and long-term memory stores.

    Provides high-level operations: remember, recall, consolidate, summarize.
    '''

    def __init__(
        self,
        agent_id: str = 'default',
        short_term_capacity: int = 50,
        long_term_capacity: int = 1000,
        consolidation_threshold: float = 0.7,
    ) -> None:
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(agent_id=agent_id, capacity=long_term_capacity)
        self._consolidation_threshold = consolidation_threshold

    async def remember(self, content: str, entry_type: str = 'observation', importance: float = 0.5, **metadata) -> MemoryEntry:
        '''Store a new memory in short-term memory.

        Args:
            content: The memory content.
            entry_type: Type of memory (observation, reasoning, result, tool_output).
            importance: Importance score [0, 1].
            **metadata: Additional metadata.

        Returns:
            The created MemoryEntry.
        '''
        entry = MemoryEntry(
            content=content,
            entry_type=entry_type,
            importance=importance,
            metadata=metadata,
        )
        await self.short_term.add(entry)
        return entry

    async def recall(self, query: str, top_k: int = 5, source: str = 'both') -> list[MemoryEntry]:
        '''Retrieve memories relevant to a query.

        Args:
            query: The retrieval query.
            top_k: Maximum number of results.
            source: Which store to query - 'short_term', 'long_term', or 'both'.

        Returns:
            List of relevant MemoryEntry objects.
        '''
        results: list[MemoryEntry] = []

        if source in ('short_term', 'both'):
            results.extend(await self.short_term.retrieve(query, top_k=top_k))

        if source in ('long_term', 'both'):
            results.extend(await self.long_term.retrieve(query, top_k=top_k))

        if source == 'both' and len(results) > top_k:
            # Deduplicate by entry_id and take top_k by importance
            seen = set()
            unique = []
            for entry in results:
                if entry.entry_id not in seen:
                    seen.add(entry.entry_id)
                    unique.append(entry)
            unique.sort(key=lambda e: e.importance, reverse=True)
            results = unique[:top_k]

        return results

    async def consolidate(self) -> int:
        '''Migrate important short-term entries to long-term memory.

        Entries with importance >= consolidation_threshold are promoted.

        Returns:
            Number of entries consolidated.
        '''
        entries = self.short_term.get_all()
        consolidated = 0

        for entry in entries:
            if entry.importance >= self._consolidation_threshold:
                await self.long_term.add(entry)
                consolidated += 1

        return consolidated

    async def summarize_context(self, query: str, max_tokens: int = 500) -> str:
        '''Build a context string from recalled memories.

        Args:
            query: Query to recall relevant memories.
            max_tokens: Approximate max tokens (estimated as words) for the summary.

        Returns:
            A formatted context string.
        '''
        entries = await self.recall(query, top_k=10, source='both')
        if not entries:
            return ''

        lines: list[str] = []
        token_count = 0

        for entry in entries:
            line = f'[{entry.entry_type}] {entry.content}'
            # Rough token estimate: word count
            words = len(line.split())
            if token_count + words > max_tokens:
                break
            lines.append(line)
            token_count += words

        return '''
'''.join(lines)

    async def reset_short_term(self) -> None:
        '''Clear short-term memory only.'''
        await self.short_term.clear()
