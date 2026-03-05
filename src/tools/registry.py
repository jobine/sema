'''Tool registry for managing and executing tools.'''

from __future__ import annotations

import time
from typing import Any, ClassVar

from .base import Tool, ToolResult


class ToolRegistry:
    '''Registry for managing available tools.

    Supports both instance-level and global (class-level) tool registration,
    following the same pattern as AsyncLLM's provider registry.
    '''

    _global_registry: ClassVar[dict[str, Tool]] = {}

    def __init__(self) -> None:
        self._local_registry: dict[str, Tool] = {}

    @classmethod
    def register_global(cls, tool: Tool) -> None:
        '''Register a tool globally (available to all ToolRegistry instances).'''
        cls._global_registry[tool.spec.name] = tool

    def register(self, tool: Tool) -> None:
        '''Register a tool on this instance.'''
        self._local_registry[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        '''Get a tool by name (local first, then global).'''
        return self._local_registry.get(name) or self._global_registry.get(name)

    def list_tools(self) -> list[str]:
        '''List all available tool names.'''
        names = set(self._global_registry.keys()) | set(self._local_registry.keys())
        return sorted(names)

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        '''Execute a tool by name with the given arguments.'''
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f'Tool "{name}" not found',
            )

        start = time.monotonic()
        try:
            result = await tool.execute(**kwargs)
            result.execution_time = time.monotonic() - start
            return result
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                execution_time=time.monotonic() - start,
            )

    def format_for_prompt(self) -> str:
        '''Format all available tools for inclusion in an LLM prompt.'''
        tool_names = self.list_tools()
        if not tool_names:
            return ''

        lines: list[str] = []
        for name in tool_names:
            tool = self.get(name)
            if tool is None:
                continue
            spec = tool.spec
            params_str = ', '.join(
                f'{p.name}: {p.type}' + ('' if p.required else ' (optional)')
                for p in spec.parameters
            )
            lines.append(f'- {spec.name}({params_str}): {spec.description}')

        return '''
'''.join(lines)
