'''Workflow executor: runs a Workflow DAG on a task and collects results.'''

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from ..agents.agent import Agent, AgentConfig, AgentState
from .schema import Workflow, WorkflowNode


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class WorkflowResult(BaseModel):
    '''Result of executing a workflow on a single task.'''

    model_config = ConfigDict(validate_assignment=True)

    workflow_id: str = Field(description='ID of the executed workflow')
    answer: str = Field(default='', description='Final answer from exit nodes')
    node_outputs: dict[str, Any] = Field(
        default_factory=dict, description='Per-node output dicts keyed by node_id'
    )
    execution_trace: list[dict[str, Any]] = Field(
        default_factory=list, description='Ordered execution trace entries'
    )
    total_steps: int = Field(default=0, ge=0, description='Total reasoning steps across all nodes')
    total_llm_calls: int = Field(default=0, ge=0, description='Total LLM API calls made')


# ---------------------------------------------------------------------------
# Lightweight per-node agent
# ---------------------------------------------------------------------------

class _NodeAgent(Agent):
    '''Concrete Agent that executes one WorkflowNode using Role + Action prompts.

    Inherits prompt-building logic from Agent.build_prompt(), which reads
    system_prompt and instruction_prompt from config extras.
    '''

    async def run(self, question: str, **kwargs: Any) -> str:
        context = kwargs.get('context', '')
        self._state = AgentState(question=question, context=context)

        for _ in range(self.config.max_steps):
            self._state = await self.step(self._state)
            if self._state.finished:
                break

        return self._state.answer

    async def step(self, state: AgentState) -> AgentState:
        prompt = self.build_prompt(state)
        try:
            response = await self.llm(prompt)
            state.answer = str(response).strip() if response else ''
        except Exception as exc:
            state.answer = f'[Error: {exc}]'

        state.steps.append({
            'step': len(state.steps) + 1,
            'prompt_length': len(prompt),
            'answer_preview': state.answer[:120],
        })
        state.finished = True
        return state


# ---------------------------------------------------------------------------
# WorkflowExecutor
# ---------------------------------------------------------------------------

class WorkflowExecutor:
    '''Executes a Workflow DAG on a single task.

    Execution model:
    1. Topological sort into parallel layers (ignoring feedback/loop edges).
    2. For each layer: run all nodes concurrently via asyncio.gather.
    3. Pass outputs between nodes using edge data_mapping (or full answer if empty).
    4. Feedback/loop edges: re-execute the affected subgraph for additional iterations.
    5. Collect final answer from exit nodes.

    Blank workflows (no nodes) return an empty WorkflowResult immediately.
    '''

    def __init__(self, default_model: str = 'gpt-4o-mini') -> None:
        self._default_model = default_model
        self._trace: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Layer computation
    # -----------------------------------------------------------------------

    def _compute_layers(self, workflow: Workflow) -> list[list[str]]:
        '''BFS level-order to find parallel execution layers.

        Feedback and loop edges are excluded so cyclic structures don't
        prevent topological ordering of the main DAG.
        '''
        node_ids = {n.node_id for n in workflow.nodes}
        active_edges = [
            e for e in workflow.edges
            if e.edge_type not in ('feedback', 'loop')
            and e.source_id in node_ids
            and e.target_id in node_ids
        ]

        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        for edge in active_edges:
            adj[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

        layers: list[list[str]] = []
        current = sorted(nid for nid in node_ids if in_degree[nid] == 0)

        while current:
            layers.append(current)
            next_level: list[str] = []
            for nid in current:
                for neighbor in adj[nid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)
            current = sorted(next_level)

        return layers

    # -----------------------------------------------------------------------
    # Context assembly
    # -----------------------------------------------------------------------

    def _build_upstream_context(
        self,
        workflow: Workflow,
        node_id: str,
        context_map: dict[str, Any],
        include_feedback: bool = False,
    ) -> str:
        '''Concatenate outputs from upstream nodes into a context string.

        Args:
            workflow: The current workflow.
            node_id: Target node whose upstream context is needed.
            context_map: Map of node_id -> output dict from already-executed nodes.
            include_feedback: Whether to include feedback/loop edge outputs.
        '''
        skip_types = set() if include_feedback else {'feedback', 'loop'}
        parts: list[str] = []

        for edge in workflow.edges:
            if edge.target_id != node_id:
                continue
            if edge.edge_type in skip_types:
                continue

            source_output = context_map.get(edge.source_id)
            if source_output is None:
                continue

            if edge.data_mapping:
                for src_field, tgt_label in edge.data_mapping.items():
                    value = source_output.get(src_field, '')
                    if value:
                        parts.append(f'{tgt_label}: {value}')
            else:
                answer = source_output.get('answer', '')
                if answer:
                    src_node = workflow.get_node(edge.source_id)
                    role_name = src_node.role.name if src_node else edge.source_id
                    parts.append(f'[{role_name}]: {answer}')

        return '\n\n'.join(parts)

    # -----------------------------------------------------------------------
    # Single-node execution
    # -----------------------------------------------------------------------

    async def _execute_node(
        self,
        workflow: Workflow,
        node_id: str,
        task: dict[str, Any],
        context_map: dict[str, Any],
        include_feedback: bool = False,
        iteration: int = 0,
    ) -> dict[str, Any]:
        '''Instantiate a _NodeAgent, run it, and return its output dict.'''
        node = workflow.get_node(node_id)
        if node is None:
            return {'node_id': node_id, 'answer': '', 'steps': [], 'llm_calls': 0}

        t0 = time.monotonic()

        # For exit nodes, append answer_format instruction so the agent
        # produces concise output matching the benchmark's expected format.
        instruction = node.action.instruction_prompt
        answer_format = task.get('answer_format', '')
        if answer_format and node_id in workflow.exit_nodes:
            instruction = f'{instruction}\n\n{answer_format}'

        # Build AgentConfig from node settings + extras for prompt building
        config_kwargs: dict[str, Any] = {
            'model': node.agent_config.get('model', self._default_model),
            'max_steps': node.agent_config.get('max_steps', node.action.max_steps),
            'system_prompt': node.role.system_prompt,
            'instruction_prompt': instruction,
        }
        if node.agent_config.get('temperature') is not None:
            config_kwargs['temperature'] = node.agent_config['temperature']

        agent = _NodeAgent(config=AgentConfig(**config_kwargs))

        # Combine task context + upstream node outputs
        upstream = self._build_upstream_context(
            workflow, node_id, context_map, include_feedback
        )
        task_context = task.get('context', '')
        full_context = '\n\n'.join(c for c in [task_context, upstream] if c)
        question = task.get('question', '')

        answer = await agent.run(question=question, context=full_context)

        state = agent._state
        steps = state.steps if state else []
        duration_s = round(time.monotonic() - t0, 3)

        return {
            'node_id': node_id,
            'role': node.role.name,
            'action': node.action.name,
            'answer': answer,
            'steps': steps,
            'llm_calls': len(steps),
            'duration_s': duration_s,
            'iteration': iteration,
        }

    # -----------------------------------------------------------------------
    # Main execute
    # -----------------------------------------------------------------------

    async def execute(self, workflow: Workflow, task: dict[str, Any]) -> WorkflowResult:
        '''Execute the workflow on a single task dict.

        Args:
            workflow: Workflow to execute. May be blank (no nodes).
            task: Dict with at least 'question' and optionally 'context'.

        Returns:
            WorkflowResult with final answer and full execution trace.
        '''
        self._trace = []

        # Blank workflow: optimizer hasn't built structure yet
        if not workflow.nodes:
            return WorkflowResult(
                workflow_id=workflow.workflow_id,
                answer='',
                node_outputs={},
                execution_trace=[],
                total_steps=0,
                total_llm_calls=0,
            )

        layers = self._compute_layers(workflow)
        context_map: dict[str, Any] = {}

        # --- Main DAG execution (sequential layers, parallel within each layer) ---
        for layer in layers:
            results = await asyncio.gather(
                *[self._execute_node(workflow, nid, task, context_map, iteration=0)
                  for nid in layer]
            )
            for result in results:
                context_map[result['node_id']] = result
                self._trace.append(result)

        # --- Feedback / loop edge handling ---
        feedback_edges = [
            e for e in workflow.edges
            if e.edge_type in ('feedback', 'loop') and e.max_iterations > 1
        ]
        if feedback_edges:
            max_extra = max(e.max_iterations for e in feedback_edges) - 1
            feedback_targets = {e.target_id for e in feedback_edges}

            # Identify direct downstream of feedback targets (via sequential edges)
            downstream_of_targets: set[str] = set()
            for edge in workflow.edges:
                if (edge.edge_type not in ('feedback', 'loop')
                        and edge.source_id in feedback_targets):
                    downstream_of_targets.add(edge.target_id)

            for iteration in range(1, max_extra + 1):
                # Re-execute feedback targets (with feedback edge context)
                target_results = await asyncio.gather(
                    *[self._execute_node(
                        workflow, nid, task, context_map,
                        include_feedback=True, iteration=iteration
                      )
                      for nid in sorted(feedback_targets)
                      if workflow.get_node(nid) is not None]
                )
                for result in target_results:
                    context_map[result['node_id']] = result
                    self._trace.append(result)

                # Re-execute downstream nodes in layer order so context is fresh
                downstream_ordered = [
                    nid for layer in layers for nid in layer
                    if nid in downstream_of_targets
                ]
                if downstream_ordered:
                    ds_results = await asyncio.gather(
                        *[self._execute_node(
                            workflow, nid, task, context_map, iteration=iteration
                          )
                          for nid in downstream_ordered]
                    )
                    for result in ds_results:
                        context_map[result['node_id']] = result
                        self._trace.append(result)

        # --- Collect final answer from exit nodes ---
        exit_answers: list[str] = []
        for nid in workflow.exit_nodes:
            out = context_map.get(nid)
            if out and out.get('answer'):
                exit_answers.append(out['answer'])

        # Fallback: last executed node's answer
        if not exit_answers and self._trace:
            exit_answers = [self._trace[-1].get('answer', '')]

        # Single exit node → direct answer; multiple → join
        answer = exit_answers[0] if len(exit_answers) == 1 else '\n\n'.join(exit_answers)

        total_steps = sum(len(out.get('steps', [])) for out in context_map.values())
        total_llm_calls = sum(out.get('llm_calls', 0) for out in context_map.values())

        return WorkflowResult(
            workflow_id=workflow.workflow_id,
            answer=answer,
            node_outputs={nid: out for nid, out in context_map.items()},
            execution_trace=self._trace,
            total_steps=total_steps,
            total_llm_calls=total_llm_calls,
        )

    def get_execution_trace(self) -> list[dict[str, Any]]:
        '''Return the trace from the most recent execute() call.'''
        return list(self._trace)
