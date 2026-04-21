'''Monte Carlo Tree Search optimizer for workflow topology optimization.'''

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from ..workflow.schema import Workflow, WorkflowNode, WorkflowEdge, Role, Action
from ..feedback.base import Trajectory
from .base import Optimizer, OptimizerConfig
from .population import Population


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

@dataclass
class _MCTSNode:
    '''Search tree node representing one topology decision.'''

    action: str = ''
    parent: '_MCTSNode | None' = field(default=None, repr=False)
    children: list['_MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.total_reward / self.visits) + c * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

    def best_child(self, c: float) -> '_MCTSNode':
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def tree_size(self) -> int:
        return 1 + sum(ch.tree_size() for ch in self.children)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

_NUM_NODE_OPTIONS = ['num_nodes=1', 'num_nodes=2', 'num_nodes=3', 'num_nodes=4']
_ROLE_OPTIONS = ['role=researcher', 'role=analyst', 'role=planner', 'role=executor', 'role=critic']
_EDGE_OPTIONS = ['topology=sequential', 'topology=parallel', 'topology=feedback']


def _action_space_for_depth(depth: int) -> list[str]:
    if depth == 0:
        return _NUM_NODE_OPTIONS
    elif depth == 1:
        return _ROLE_OPTIONS
    else:
        return _EDGE_OPTIONS


# ---------------------------------------------------------------------------
# Config + Optimizer
# ---------------------------------------------------------------------------

class MCTSConfig(OptimizerConfig):
    '''Configuration for MCTS optimizer.'''

    exploration_constant: float = Field(default=1.41, ge=0.0)
    max_iterations: int = Field(default=20, ge=1)
    rollout_samples: int = Field(default=3, ge=1)
    max_depth: int = Field(default=6, ge=1)


class MCTSOptimizer(Optimizer):
    '''Monte Carlo Tree Search for workflow topology optimization.

    The search tree persists across generations to accumulate knowledge.
    '''

    def __init__(self, config: MCTSConfig, population: Population) -> None:
        super().__init__(config, population)
        self.config: MCTSConfig = config
        self._root: _MCTSNode = _MCTSNode(action='root')
        self._stats: dict[str, Any] = {
            'tree_size': 1,
            'best_path': [],
            'best_path_reward': 0.0,
        }

    # -----------------------------------------------------------------------
    # MCTS phases
    # -----------------------------------------------------------------------

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        '''UCB1 tree policy: descend until a leaf.'''
        while not node.is_leaf():
            node = node.best_child(self.config.exploration_constant)
        return node

    def _expand(self, node: _MCTSNode, depth: int) -> _MCTSNode:
        '''Add one untried child action.'''
        action_space = _action_space_for_depth(depth % 3)
        tried = {ch.action for ch in node.children}
        untried = [a for a in action_space if a not in tried]
        if not untried:
            return node
        action = random.choice(untried)
        child = _MCTSNode(action=action, parent=node)
        node.children.append(child)
        return child

    def _rollout(self, node: _MCTSNode, population: Population) -> float:
        '''Estimate reward using average fitness of population as proxy.'''
        # Build a path from root to node
        path: list[_MCTSNode] = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()

        # Score by how many actions were taken (proxy for workflow complexity)
        wf = self._build_workflow_from_path(path)
        workflows = population.workflows
        if not workflows:
            return 0.0

        # Fitness proxy: average fitness of workflows with similar node count
        n_nodes = len(wf.nodes)
        similar = [w for w in workflows if abs(len(w.nodes) - n_nodes) <= 1]
        if similar:
            return sum(w.fitness for w in similar) / len(similar)
        return sum(w.fitness for w in workflows) / len(workflows)

    def _backpropagate(self, node: _MCTSNode, reward: float) -> None:
        current: _MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _build_workflow_from_path(self, path: list[_MCTSNode], goal: str = '', environment: str = '') -> Workflow:
        '''Construct a Workflow from the decisions encoded in the MCTS path.'''
        num_nodes = 2
        roles = ['researcher', 'analyst']
        topology = 'sequential'

        for node in path:
            if node.action.startswith('num_nodes='):
                try:
                    num_nodes = int(node.action.split('=')[1])
                except ValueError:
                    pass
            elif node.action.startswith('role='):
                roles.append(node.action.split('=')[1])
            elif node.action.startswith('topology='):
                topology = node.action.split('=')[1]

        num_nodes = max(1, min(num_nodes, 5))
        wf_nodes: list[WorkflowNode] = []
        for i in range(num_nodes):
            role_name = roles[i % len(roles)] if roles else 'agent'
            wf_nodes.append(WorkflowNode(
                role=Role(name=role_name, system_prompt=f'You are a {role_name}.'),
                action=Action(name=f'action_{i}', instruction_prompt='Process and respond.'),
            ))

        edges: list[WorkflowEdge] = []
        if topology == 'sequential':
            for i in range(len(wf_nodes) - 1):
                edges.append(WorkflowEdge(
                    source_id=wf_nodes[i].node_id,
                    target_id=wf_nodes[i + 1].node_id,
                ))
        elif topology == 'parallel':
            if len(wf_nodes) > 1:
                for i in range(1, len(wf_nodes)):
                    edges.append(WorkflowEdge(
                        source_id=wf_nodes[0].node_id,
                        target_id=wf_nodes[i].node_id,
                    ))
        elif topology == 'feedback' and len(wf_nodes) >= 2:
            edges.append(WorkflowEdge(
                source_id=wf_nodes[0].node_id,
                target_id=wf_nodes[-1].node_id,
            ))

        entry = [wf_nodes[0].node_id] if wf_nodes else []
        exit_ = [wf_nodes[-1].node_id] if wf_nodes else []

        return Workflow(
            nodes=wf_nodes,
            edges=edges,
            entry_nodes=entry,
            exit_nodes=exit_,
            goal=goal,
            environment=environment,
        )

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    async def step(
        self,
        population: Population,
        fitness_scores: dict[str, float],
        trajectories: list[Trajectory],
    ) -> Population:
        for wf_id, score in fitness_scores.items():
            population.update_fitness(wf_id, score)

        # Run MCTS iterations
        best_reward = 0.0
        best_path: list[str] = []

        for _ in range(self.config.max_iterations):
            leaf = self._select(self._root)
            depth = self._get_depth(leaf)

            if depth < self.config.max_depth:
                leaf = self._expand(leaf, depth)

            reward = self._rollout(leaf, population)

            if reward > best_reward:
                best_reward = reward
                best_path = self._get_path(leaf)

            self._backpropagate(leaf, reward)

        # Extract top-K workflows from best paths
        elite = self._get_elite(population)
        elite_ids = {wf.workflow_id for wf in elite}
        n_new = self.config.population_size - len(elite)

        new_workflows: list[Workflow] = [self._deep_copy_workflow(wf) for wf in elite]

        # Inherit goal/environment from existing population
        ref = next((w for w in population.workflows if w.goal), None)
        goal = ref.goal if ref else ''
        environment = ref.environment if ref else ''

        # Generate diverse workflows from top MCTS paths
        top_paths = self._get_top_paths(self.config.rollout_samples)
        for i in range(n_new):
            path_nodes = top_paths[i % len(top_paths)] if top_paths else [self._root]
            wf = self._build_workflow_from_path(path_nodes, goal=goal, environment=environment)
            wf.generation = population.generation + 1
            new_workflows.append(wf)

        new_workflows = new_workflows[: self.config.population_size]
        population.replace_workflows(new_workflows)
        population.advance_generation()

        self._stats['tree_size'] = self._root.tree_size()
        self._stats['best_path'] = best_path
        self._stats['best_path_reward'] = best_reward

        return population

    def _get_depth(self, node: _MCTSNode) -> int:
        depth = 0
        current: _MCTSNode | None = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def _get_path(self, node: _MCTSNode) -> list[str]:
        path: list[str] = []
        current: _MCTSNode | None = node
        while current is not None:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path

    def _get_top_paths(self, k: int) -> list[list[_MCTSNode]]:
        '''BFS to find top-k leaf paths by average reward.'''
        paths: list[tuple[float, list[_MCTSNode]]] = []

        def dfs(node: _MCTSNode, path: list[_MCTSNode]) -> None:
            current_path = path + [node]
            if node.is_leaf() and node.visits > 0:
                avg = node.total_reward / node.visits
                paths.append((avg, current_path))
            for child in node.children:
                dfs(child, current_path)

        dfs(self._root, [])
        paths.sort(key=lambda x: -x[0])
        return [p for _, p in paths[:k]] or [[self._root]]

    def get_statistics(self) -> dict[str, Any]:
        return dict(self._stats)
