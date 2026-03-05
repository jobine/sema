'''Built-in lookup tool for querying agent memory.'''

from __future__ import annotations

from typing import Any

from ..base import Tool, ToolSpec, ToolParameter, ToolResult


class LookupTool(Tool):
    '''Lookup information from agent memory.'''

    def __init__(self, memory_system=None) -> None:
        self._memory = memory_system

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name='lookup',
            description='Look up information from agent memory.',
            parameters=[
                ToolParameter(name='query', type='string', description='Query to search memory for'),
                ToolParameter(name='top_k', type='int', description='Number of results', required=False),
            ],
            returns='string',
        )

    def set_memory(self, memory_system) -> None:
        '''Set the memory system to use for lookups.'''
        self._memory = memory_system

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get('query', '')
        top_k = int(kwargs.get('top_k', 3))

        if not query:
            return ToolResult(
                tool_name='lookup',
                success=False,
                error='Query is required',
            )

        if self._memory is None:
            return ToolResult(
                tool_name='lookup',
                success=False,
                error='No memory system configured',
            )

        try:
            entries = await self._memory.recall(query, top_k=top_k)
            if not entries:
                return ToolResult(
                    tool_name='lookup',
                    success=True,
                    output='No relevant memories found.',
                )

            lines = [f'[{e.entry_type}] {e.content}' for e in entries]
            return ToolResult(
                tool_name='lookup',
                success=True,
                output='\n'.join(lines),
            )
        except Exception as e:
            return ToolResult(
                tool_name='lookup',
                success=False,
                error=str(e),
            )
