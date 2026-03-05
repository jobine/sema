'''Built-in search tool for keyword search within provided context.'''

from __future__ import annotations

from typing import Any

from ..base import Tool, ToolSpec, ToolParameter, ToolResult


class SearchTool(Tool):
    '''Keyword search within provided context text.'''

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name='search',
            description='Search for keywords within the provided context text.',
            parameters=[
                ToolParameter(name='query', type='string', description='Search query keywords'),
                ToolParameter(name='context', type='string', description='Text to search within'),
            ],
            returns='string',
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get('query', '')
        context = kwargs.get('context', '')

        if not query or not context:
            return ToolResult(
                tool_name='search',
                success=False,
                error='Both query and context are required',
            )

        query_tokens = set(query.lower().split())
        sentences = context.replace('\n', '. ').split('. ')

        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = set(sentence.lower().split())
            overlap = len(query_tokens & sentence_tokens)
            if overlap > 0:
                scored.append((overlap, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        matches = [s for _, s in scored[:5]]

        if not matches:
            return ToolResult(
                tool_name='search',
                success=True,
                output='No relevant matches found.',
            )

        return ToolResult(
            tool_name='search',
            success=True,
            output='\n'.join(matches),
        )
