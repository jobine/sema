'''Evolutionary optimizer for SEMA — multi-level mutation + crossover + selection.'''

from __future__ import annotations

import asyncio
import json
import math
import random
import uuid
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow, WorkflowNode, WorkflowEdge, Role, Action
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

class NodeConfigMutator:
    '''Micro-level mutation: perturbs numeric agent_config fields (no LLM).'''

    _MODEL_LIST = [
        'gpt-4o-mini',
        'gpt-4o',
        'gpt-3.5-turbo',
        'claude-haiku-4-5-20251001',
        'claude-sonnet-4-6',
    ]

    def mutate(self, workflow: Workflow) -> Workflow:
        for node in workflow.nodes:
            cfg = dict(node.agent_config)

            # Temperature perturbation
            if 'temperature' in cfg and isinstance(cfg['temperature'], (int, float)):
                delta = random.gauss(0, 0.1)
                cfg['temperature'] = max(0.0, min(2.0, cfg['temperature'] + delta))

            # max_steps perturbation
            if 'max_steps' in cfg and isinstance(cfg['max_steps'], int):
                delta = random.randint(-1, 1)
                cfg['max_steps'] = max(1, min(10, cfg['max_steps'] + delta))

            # model random swap (low probability)
            if 'model' in cfg and random.random() < 0.1:
                cfg['model'] = random.choice(self._MODEL_LIST)

            node.agent_config = cfg
        return workflow


class RoleActionMutator:
    '''Meso-level mutation: rewrites system/instruction prompts via LLM.'''

    async def mutate(self, workflow: Workflow, model: str, failures: list[str]) -> Workflow:
        if not workflow.nodes:
            return workflow

        node = random.choice(workflow.nodes)
        failure_text = '\n'.join(failures[:3]) if failures else 'No specific failures recorded.'
        goal = workflow.goal
        goal_line = f'Workflow goal: {goal}\n' if goal else ''

        role_prompt = (
            f'{goal_line}'
            f'Current role name: {node.role.name}\n'
            f'Current system prompt: {node.role.system_prompt}\n\n'
            f'Failure cases:\n{failure_text}\n\n'
            'Return a JSON object with an improved role: {"name": "...", "system_prompt": "..."}\n'
            'The name should be a short snake_case identifier reflecting the role\'s function.\n'
            'Return ONLY the JSON.'
        )
        action_prompt = (
            f'{goal_line}'
            f'Current action name: {node.action.name}\n'
            f'Current instruction prompt: {node.action.instruction_prompt}\n\n'
            f'Failure cases:\n{failure_text}\n\n'
            'Return a JSON object with an improved action: {"name": "...", "instruction_prompt": "..."}\n'
            'The name should be a short snake_case identifier for the action.\n'
            'Return ONLY the JSON.'
        )

        try:
            from ..models.models import AsyncLLM
            llm = AsyncLLM(model)
            new_role_resp, new_action_resp = await asyncio.gather(
                llm(role_prompt),
                llm(action_prompt),
            )
            if new_role_resp:
                text = str(new_role_resp).strip()
                try:
                    data = json.loads(text)
                    if 'name' in data:
                        node.role.name = str(data['name']).strip()
                    if 'system_prompt' in data:
                        node.role.system_prompt = str(data['system_prompt']).strip()
                except Exception:
                    node.role.system_prompt = text  # fallback: plain text
            if new_action_resp:
                text = str(new_action_resp).strip()
                try:
                    data = json.loads(text)
                    if 'name' in data:
                        node.action.name = str(data['name']).strip()
                    if 'instruction_prompt' in data:
                        node.action.instruction_prompt = str(data['instruction_prompt']).strip()
                except Exception:
                    node.action.instruction_prompt = text  # fallback: plain text
        except Exception:
            pass

        return workflow


class TopologyMutator:
    '''Macro-level mutation: adds/removes nodes or rewires edges (no LLM).'''

    def mutate(self, workflow: Workflow) -> Workflow:
        if not workflow.nodes:
            return self._add_node(workflow)

        op = random.choice(['add', 'remove', 'rewire'])
        backup_json = workflow.to_json()

        try:
            if op == 'add':
                workflow = self._add_node(workflow)
            elif op == 'remove' and len(workflow.nodes) > 1:
                workflow = self._remove_node(workflow)
            else:
                workflow = self._rewire_edge(workflow)

            errors = workflow.validate_graph()
            if errors:
                workflow = Workflow.from_json(backup_json)
        except Exception:
            workflow = Workflow.from_json(backup_json)

        return workflow

    def _add_node(self, workflow: Workflow) -> Workflow:
        goal = workflow.goal
        role_sys = (
            f'Overall goal: {goal}\n\nYou are a specialist agent contributing to the above goal.'
            if goal else 'You are a helpful assistant.'
        )
        action_inst = (
            f'Process the input and contribute toward: {goal}'
            if goal else 'Process the input and provide output.'
        )
        new_node = WorkflowNode(
            role=Role(name='specialist', system_prompt=role_sys),
            action=Action(name='process', instruction_prompt=action_inst),
        )
        workflow.nodes.append(new_node)

        if workflow.nodes and len(workflow.nodes) > 1:
            parent = random.choice(workflow.nodes[:-1])
            workflow.edges.append(WorkflowEdge(source_id=parent.node_id, target_id=new_node.node_id))

        if not workflow.entry_nodes:
            workflow.entry_nodes = [workflow.nodes[0].node_id]
        if not workflow.exit_nodes or new_node.node_id not in workflow.exit_nodes:
            workflow.exit_nodes = [new_node.node_id]

        return workflow

    def _remove_node(self, workflow: Workflow) -> Workflow:
        # Find non-entry, non-exit candidate nodes
        protected = set(workflow.entry_nodes) | set(workflow.exit_nodes)
        candidates = [n for n in workflow.nodes if n.node_id not in protected]
        if not candidates:
            return workflow

        target = random.choice(candidates)
        tid = target.node_id

        # Find incoming and outgoing edges
        incoming = [e for e in workflow.edges if e.target_id == tid]
        outgoing = [e for e in workflow.edges if e.source_id == tid]

        # Reconnect: link each incoming source to each outgoing target
        new_edges = []
        for inc in incoming:
            for out in outgoing:
                new_edges.append(WorkflowEdge(source_id=inc.source_id, target_id=out.target_id))

        # Remove old edges and node
        workflow.edges = [e for e in workflow.edges if e.source_id != tid and e.target_id != tid]
        workflow.edges.extend(new_edges)
        workflow.nodes = [n for n in workflow.nodes if n.node_id != tid]

        return workflow

    def _rewire_edge(self, workflow: Workflow) -> Workflow:
        if not workflow.edges or len(workflow.nodes) < 2:
            return workflow

        edge = random.choice(workflow.edges)
        valid_targets = [n.node_id for n in workflow.nodes if n.node_id != edge.source_id]
        if valid_targets:
            edge.target_id = random.choice(valid_targets)

        return workflow


class GoalDrivenTopologyMutator:
    '''Macro-level mutation: asks the LLM to propose a targeted structural change guided by goal + failures.'''

    _fallback = TopologyMutator()

    async def mutate(
        self,
        workflow: Workflow,
        model: str,
        failures: list[str],
        fitness: float = 0.0,
    ) -> Workflow:
        if not workflow.goal or not workflow.nodes:
            return self._fallback.mutate(workflow)

        backup = workflow.to_json()
        node_summary = '; '.join(
            f'{n.node_id}={n.role.name}' for n in workflow.nodes
        )
        removable = [
            n.node_id for n in workflow.nodes
            if n.node_id not in set(workflow.entry_nodes) | set(workflow.exit_nodes)
        ]
        failure_text = '\n'.join(failures[:3]) if failures else 'No specific failures.'

        prompt = (
            f'Goal: {workflow.goal}\n'
            f'Current nodes: {node_summary}\n'
            f'Non-protected nodes (removable): {removable}\n'
            f'Recent failures (fitness={fitness:.3f}):\n{failure_text}\n\n'
            'Propose ONE structural change as JSON. Choose the most impactful op:\n'
            '- {"op":"add_node","role_name":"...","system_prompt":"...","action_name":"...","instruction_prompt":"...","after_node_id":"<existing node_id>"}\n'
            '- {"op":"remove_node","node_id":"<non-protected node_id>"}\n'
            '- {"op":"change_role","node_id":"...","role_name":"...","system_prompt":"...","action_name":"...","instruction_prompt":"..."}\n'
            'Return ONLY the JSON patch.'
        )

        try:
            from ..models.models import AsyncLLM
            resp = await AsyncLLM(model)(prompt)
            text = resp.strip()
            if '```' in text:
                start = text.find('{', text.find('```'))
                end = text.rfind('}') + 1
                text = text[start:end]
            patch = json.loads(text)
            op = patch.get('op', '')

            if op == 'add_node':
                new_node = WorkflowNode(
                    node_id=f'agent_{uuid.uuid4().hex[:6]}',
                    role=Role(
                        name=str(patch.get('role_name', 'specialist')),
                        system_prompt=str(patch.get('system_prompt', '')),
                    ),
                    action=Action(
                        name=str(patch.get('action_name', 'process')),
                        instruction_prompt=str(patch.get('instruction_prompt', '')),
                    ),
                )
                after_id = patch.get('after_node_id') or (workflow.nodes[-1].node_id if workflow.nodes else None)
                workflow.nodes.append(new_node)
                if after_id:
                    workflow.edges.append(WorkflowEdge(source_id=after_id, target_id=new_node.node_id, edge_type='sequential'))
                workflow.exit_nodes = [new_node.node_id]

            elif op == 'remove_node':
                tid = str(patch.get('node_id', ''))
                protected = set(workflow.entry_nodes) | set(workflow.exit_nodes)
                if tid and tid not in protected and any(n.node_id == tid for n in workflow.nodes):
                    incoming = [e for e in workflow.edges if e.target_id == tid]
                    outgoing = [e for e in workflow.edges if e.source_id == tid]
                    new_edges = [
                        WorkflowEdge(source_id=inc.source_id, target_id=out.target_id, edge_type='sequential')
                        for inc in incoming for out in outgoing
                    ]
                    workflow.edges = [e for e in workflow.edges if e.source_id != tid and e.target_id != tid]
                    workflow.edges.extend(new_edges)
                    workflow.nodes = [n for n in workflow.nodes if n.node_id != tid]

            elif op == 'change_role':
                tid = str(patch.get('node_id', ''))
                for node in workflow.nodes:
                    if node.node_id == tid:
                        if 'role_name' in patch:
                            node.role.name = str(patch['role_name'])
                        if 'system_prompt' in patch:
                            node.role.system_prompt = str(patch['system_prompt'])
                        if 'action_name' in patch:
                            node.action.name = str(patch['action_name'])
                        if 'instruction_prompt' in patch:
                            node.action.instruction_prompt = str(patch['instruction_prompt'])
                        break

            errors = workflow.validate_graph()
            if errors:
                workflow = Workflow.from_json(backup)
        except Exception:
            workflow = Workflow.from_json(backup)

        return workflow


# ---------------------------------------------------------------------------
# Selection operators
# ---------------------------------------------------------------------------

class TournamentSelection:
    '''Select the fittest from a random tournament.'''

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def select(self, workflows: list[Workflow]) -> Workflow:
        tournament = random.sample(workflows, min(self.k, len(workflows)))
        return max(tournament, key=lambda wf: wf.fitness)


class FitnessProportionalSelection:
    '''Roulette-wheel selection using softmax of fitness scores.'''

    def select(self, workflows: list[Workflow]) -> Workflow:
        fitnesses = [wf.fitness for wf in workflows]
        # Softmax
        max_f = max(fitnesses)
        exp_f = [math.exp(f - max_f) for f in fitnesses]
        total = sum(exp_f)
        probs = [e / total for e in exp_f]

        r = random.random()
        cumulative = 0.0
        for wf, p in zip(workflows, probs):
            cumulative += p
            if r <= cumulative:
                return wf
        return workflows[-1]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class WorkflowCrossover:
    '''Single-point crossover on the node list of two parent workflows.'''

    def crossover(self, parent1: Workflow, parent2: Workflow) -> Workflow:
        nodes1 = parent1.nodes
        nodes2 = parent2.nodes

        if not nodes1 or not nodes2:
            return Workflow.from_json(parent1.to_json())

        cut = random.randint(1, max(1, min(len(nodes1), len(nodes2)) - 1))
        child_nodes = [
            WorkflowNode.model_validate(n.model_dump()) for n in nodes1[:cut]
        ] + [
            WorkflowNode.model_validate(n.model_dump()) for n in nodes2[cut:]
        ]

        surviving_ids = {n.node_id for n in child_nodes}

        # Keep only edges where both endpoints survive
        all_edges = parent1.edges + parent2.edges
        child_edges = [
            WorkflowEdge.model_validate(e.model_dump())
            for e in all_edges
            if e.source_id in surviving_ids and e.target_id in surviving_ids
        ]
        # Deduplicate edges
        seen: set[tuple[str, str]] = set()
        unique_edges: list[WorkflowEdge] = []
        for e in child_edges:
            key = (e.source_id, e.target_id)
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        entry_nodes = [nid for nid in parent1.entry_nodes if nid in surviving_ids]
        if not entry_nodes and child_nodes:
            entry_nodes = [child_nodes[0].node_id]

        exit_nodes = [nid for nid in parent1.exit_nodes if nid in surviving_ids]
        if not exit_nodes and child_nodes:
            exit_nodes = [child_nodes[-1].node_id]

        child = Workflow(
            nodes=child_nodes,
            edges=unique_edges,
            entry_nodes=entry_nodes,
            exit_nodes=exit_nodes,
            goal=parent1.goal,
            environment=parent1.environment,
            parent_ids=[parent1.workflow_id, parent2.workflow_id],
        )
        return child


# ---------------------------------------------------------------------------
# Config + Optimizer
# ---------------------------------------------------------------------------

class EvolutionaryConfig(OptimizerConfig):
    '''Configuration for the evolutionary optimizer.'''

    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    selection_method: str = Field(default='tournament')  # "tournament" | "proportional"
    tournament_k: int = Field(default=3, ge=1)
    mutation_levels: list[str] = Field(default_factory=lambda: ['micro', 'meso', 'macro'])
    macro_mutation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    meso_model: str = Field(default='', description='LLM model for meso/macro mutations; falls back to optimizer_model')


class EvolutionaryOptimizer(Optimizer):
    '''Multi-level evolutionary optimizer for Workflow populations.'''

    def __init__(self, config: EvolutionaryConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: EvolutionaryConfig = config
        self._node_mutator = NodeConfigMutator()
        self._role_mutator = RoleActionMutator()
        self._topo_mutator = TopologyMutator()
        self._goal_topo_mutator = GoalDrivenTopologyMutator()
        self._crossover = WorkflowCrossover()
        self._stats: dict[str, int] = {
            'crossovers': 0,
            'micro_mutations': 0,
            'meso_mutations': 0,
            'macro_mutations': 0,
        }

    def _select_parent(self, workflows: list[Workflow]) -> Workflow:
        if self.config.selection_method == 'proportional':
            return FitnessProportionalSelection().select(workflows)
        return TournamentSelection(self.config.tournament_k).select(workflows)

    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        # Update fitness scores
        for wf_id, score in fitness_scores.items():
            population.update_fitness(wf_id, score)

        # Collect failure messages for meso mutations
        failures = [
            f'Q: {t.question} | Pred: {t.prediction} | Truth: {t.ground_truth}'
            for t in trajectories
            if t.env_reward < 0.5
        ]

        elite = self._get_elite(population)
        all_workflows = population.workflows
        n_new = self.config.population_size - len(elite)

        new_workflows: list[Workflow] = [self._deep_copy_workflow(wf) for wf in elite]

        offspring_pre_async: list[Workflow] = []

        for _ in range(n_new):
            p1 = self._select_parent(all_workflows)
            p2 = self._select_parent(all_workflows)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child = self._crossover.crossover(p1, p2)
                self._stats['crossovers'] += 1
            else:
                child = self._deep_copy_workflow(p1)

            # Micro mutation (sync, always)
            child = self._node_mutator.mutate(child)
            self._stats['micro_mutations'] += 1

            offspring_pre_async.append(child)

        # Macro (goal-driven topology) + meso mutations merged into one async gather
        levels = self.config.mutation_levels
        macro_rate = self.config.macro_mutation_rate
        mutation_rate = self.config.mutation_rate
        meso_model = self.config.meso_model or self.config.optimizer_model
        stats = self._stats

        async def maybe_macro_and_meso(child: Workflow) -> Workflow:
            if 'macro' in levels and random.random() < macro_rate:
                child = await self._goal_topo_mutator.mutate(child, meso_model, failures, child.fitness)
                stats['macro_mutations'] += 1
            if 'meso' in levels and random.random() < mutation_rate:
                child = await self._role_mutator.mutate(child, meso_model, failures)
                stats['meso_mutations'] += 1
            return child

        mutated = await asyncio.gather(*[maybe_macro_and_meso(c) for c in offspring_pre_async])
        new_workflows.extend(mutated)

        population.replace_workflows(new_workflows)
        population.advance_generation()
        return population

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
