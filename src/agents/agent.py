'''Base Agent class for all agents.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from ..models.models import AsyncLLM, AsyncBaseLLM


class AgentConfig(BaseModel):
    '''Configuration for an agent.'''
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
    )
    
    model: str = Field(default='gpt-4o-mini', description='LLM model name to use')
    max_steps: int = Field(default=5, ge=1, description='Maximum number of reasoning steps')
    verbose: bool = Field(default=False, description='Whether to print verbose output')
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description='LLM temperature')


class AgentState(BaseModel):
    '''State of an agent during execution.'''
    
    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True,
    )
    
    question: str = Field(default='', description='The question being answered')
    context: str = Field(default='', description='Context information for the question')
    steps: List[Dict[str, Any]] = Field(default_factory=list, description='History of reasoning steps')
    answer: str = Field(default='', description='The current answer')
    finished: bool = Field(default=False, description='Whether the agent has finished')


class Agent(BaseModel, ABC):
    '''Abstract base class for all agents.'''
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    config: AgentConfig = Field(default_factory=AgentConfig)
    _llm: AsyncBaseLLM | None = PrivateAttr(default=None)
    
    @property
    def llm(self) -> AsyncBaseLLM:
        '''Lazy initialization of LLM.'''
        if self._llm is None:
            self._llm = AsyncLLM(self.config.model)
        return self._llm
    
    @abstractmethod
    async def run(self, question: str, **kwargs: Any) -> str:
        '''Run the agent on a given question and return the answer.'''
        ...
    
    @abstractmethod
    async def step(self, state: AgentState) -> AgentState:
        '''Execute one step of the agent's reasoning.'''
        ...
    
    def reset(self) -> None:
        '''Reset the agent's state for a new question.'''
        self._llm = None
