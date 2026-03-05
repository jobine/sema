'''Base classes for the SEMA tool system.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class ToolParameter(BaseModel):
    '''Describes a parameter for a tool.'''

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description='Parameter name')
    type: str = Field(default='string', description='Parameter type (string, int, float, bool)')
    description: str = Field(default='', description='Parameter description')
    required: bool = Field(default=True, description='Whether the parameter is required')


class ToolSpec(BaseModel):
    '''Specification describing a tool's interface.'''

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description='Tool name')
    description: str = Field(default='', description='Tool description')
    parameters: list[ToolParameter] = Field(default_factory=list, description='Tool parameters')
    returns: str = Field(default='string', description='Return type description')


class ToolResult(BaseModel):
    '''Result of a tool execution.'''

    model_config = ConfigDict(validate_assignment=True)

    tool_name: str = Field(description='Name of the tool that was executed')
    success: bool = Field(default=True, description='Whether execution succeeded')
    output: Any = Field(default=None, description='Tool output')
    error: str = Field(default='', description='Error message if failed')
    execution_time: float = Field(default=0.0, ge=0.0, description='Execution time in seconds')


class Tool(ABC):
    '''Abstract base class for all tools.'''

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        '''Return the tool specification.'''
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        '''Execute the tool with the given arguments.'''
        ...
