'''Base Agent class for all agents.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from ..models.models import AsyncLLM, AsyncBaseLLM


class AgentConfig(BaseModel):
    '''Configuration for an agent.'''

    model_config = ConfigDict(
        extra='allow',
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

    metadata: Dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata')
    question: str = Field(default='', description='The question being answered')
    context: str = Field(default='', description='Context information for the question')
    steps: List[Dict[str, Any]] = Field(default_factory=list, description='History of reasoning steps')
    memory_retrievals: List[Dict[str, Any]] = Field(default_factory=list, description='Memory retrieval results')
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description='Tool call history')
    reasoning_trace: List[str] = Field(default_factory=list, description='Reasoning trace for analysis')
    reward: float = Field(default=0.0, description='Reward signal from environment')
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
    _memory: Any = PrivateAttr(default=None)
    _tools: Any = PrivateAttr(default=None)
    _state: AgentState | None = PrivateAttr(default=None)

    @property
    def llm(self) -> AsyncBaseLLM:
        '''Lazy initialization of LLM.'''
        if self._llm is None:
            self._llm = AsyncLLM(self.config.model)
        return self._llm

    @property
    def memory(self):
        '''Lazy initialization of memory system.'''
        if self._memory is None:
            from ..memory import MemorySystem
            self._memory = MemorySystem()
        return self._memory

    @property
    def tools(self):
        '''Lazy initialization of tool registry.'''
        if self._tools is None:
            from ..tools import ToolRegistry
            self._tools = ToolRegistry()
        return self._tools

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
        self._state = None

    def build_prompt(self, state: AgentState) -> str:
        '''Assemble prompt from config extras, memory, and tools.'''
        parts: list[str] = []

        # System prompt from config extras
        system_prompt = getattr(self.config, 'system_prompt', None)
        if system_prompt:
            parts.append(system_prompt)

        # Instruction prompt from config extras
        instruction_prompt = getattr(self.config, 'instruction_prompt', None)
        if instruction_prompt:
            parts.append(instruction_prompt)

        # Memory context
        if state.memory_retrievals:
            memory_lines = []
            for mem in state.memory_retrievals:
                content = mem.get('content', '')
                if content:
                    memory_lines.append(f'- {content}')
            if memory_lines:
                parts.append('''Relevant memories:
''' + '''
'''.join(memory_lines))

        # Available tools
        if self._tools is not None:
            tool_desc = self._tools.format_for_prompt()
            if tool_desc:
                parts.append(f'Available tools:\n{tool_desc}')

        # Context and question
        if state.context:
            parts.append(f'Context: {state.context}')
        if state.question:
            parts.append(f'Question: {state.question}')

        return '''

'''.join(parts)

    def get_genome(self) -> dict:
        '''Export evolvable parameters as a genome dict.'''
        genome = self.config.model_dump()
        genome['agent_class'] = type(self).__name__
        return genome

    @classmethod
    def from_genome(cls, genome: dict) -> Agent:
        '''Create an agent from a genome dict.'''
        genome = genome.copy()
        genome.pop('agent_class', None)
        config = AgentConfig(**genome)
        return cls(config=config)

    def get_trajectory(self) -> dict:
        '''Export the execution trajectory for feedback analysis.'''
        state = self._state or AgentState()
        return {
            'config': self.config.model_dump(),
            'question': state.question,
            'context': state.context,
            'steps': state.steps,
            'answer': state.answer,
            'memory_retrievals': state.memory_retrievals,
            'tool_calls': state.tool_calls,
            'reasoning_trace': state.reasoning_trace,
            'reward': state.reward,
        }
