'''Base classes for the SEMA memory system.'''

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class MemoryEntry(BaseModel):
    '''A single memory entry.'''

    model_config = ConfigDict(validate_assignment=True)

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description='Unique entry identifier')
    content: str = Field(description='Memory content text')
    entry_type: str = Field(default='observation', description='Type: observation, reasoning, result, tool_output')
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp(),
        description='Unix timestamp of creation',
    )
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description='Importance score')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata')


class MemoryStore(ABC):
    '''Abstract base class for memory stores.'''

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> None:
        '''Add an entry to the store.'''
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        '''Retrieve entries relevant to the query.'''
        ...

    @abstractmethod
    async def clear(self) -> None:
        '''Clear all entries from the store.'''
        ...

    @abstractmethod
    async def size(self) -> int:
        '''Return the number of entries in the store.'''
        ...
