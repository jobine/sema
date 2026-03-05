'''Workflow schema models for SEMA framework.'''

from __future__ import annotations

import json
import uuid
from collections import deque
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class Role(BaseModel):
    '''Defines the role an agent plays in a workflow node.'''

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description='Role name')
    description: str = Field(default='', description='Role description')
    system_prompt: str = Field(default='', description='System prompt for this role')
    available_tools: list[str] = Field(default_factory=list, description='Tool names available to this role')
    memory_strategy: str = Field(default='short_term', description='Memory strategy: short_term, long_term, both')


class Action(BaseModel):
    '''Defines what action a workflow node performs.'''

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(description='Action name')
    instruction_prompt: str = Field(default='', description='Instruction prompt template for this action')
    output_schema: dict[str, Any] = Field(default_factory=dict, description='Expected output schema')
    max_steps: int = Field(default=5, ge=1, description='Max reasoning steps for this action')


class WorkflowNode(BaseModel):
    '''A node in the workflow graph, combining a role and action.'''

    model_config = ConfigDict(validate_assignment=True)

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description='Unique node identifier')
    role: Role = Field(description='The role for this node')
    action: Action = Field(description='The action for this node')
    agent_config: dict[str, Any] = Field(default_factory=dict, description='Agent configuration overrides')


class WorkflowEdge(BaseModel):
    '''An edge connecting two nodes in the workflow graph.'''

    model_config = ConfigDict(validate_assignment=True)

    source_id: str = Field(description='Source node ID')
    target_id: str = Field(description='Target node ID')
    edge_type: str = Field(default='sequential', description='Edge type: sequential, conditional, feedback')
    condition: str = Field(default='', description='Condition expression for conditional edges')
    data_mapping: dict[str, str] = Field(default_factory=dict, description='Maps source outputs to target inputs')
    max_iterations: int = Field(default=1, ge=1, description='Max iterations for feedback loops')


class Workflow(BaseModel):
    '''A complete workflow graph defining multi-agent execution topology.'''

    model_config = ConfigDict(validate_assignment=True)

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description='Unique workflow identifier')
    generation: int = Field(default=0, ge=0, description='Evolution generation number')
    parent_ids: list[str] = Field(default_factory=list, description='Parent workflow IDs (for lineage tracking)')
    goal: str = Field(default='', description='High-level goal description')
    environment: str = Field(default='', description='Environment/benchmark name')
    nodes: list[WorkflowNode] = Field(default_factory=list, description='Workflow nodes')
    edges: list[WorkflowEdge] = Field(default_factory=list, description='Workflow edges')
    entry_nodes: list[str] = Field(default_factory=list, description='Node IDs where execution starts')
    exit_nodes: list[str] = Field(default_factory=list, description='Node IDs where execution ends')
    fitness: float = Field(default=0.0, description='Current fitness score')
    fitness_history: list[float] = Field(default_factory=list, description='Historical fitness scores')
    metadata: dict[str, Any] = Field(default_factory=dict, description='Arbitrary metadata')

    def to_json(self) -> str:
        '''Serialize workflow to JSON string.'''
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Workflow:
        '''Deserialize workflow from JSON string.'''
        return cls.model_validate_json(json_str)

    def get_node(self, node_id: str) -> WorkflowNode | None:
        '''Get a node by its ID.'''
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def validate_graph(self) -> list[str]:
        '''Validate the workflow graph structure. Returns list of error messages (empty if valid).'''
        errors: list[str] = []
        node_ids = {n.node_id for n in self.nodes}

        # Check edges reference valid nodes
        for edge in self.edges:
            if edge.source_id not in node_ids:
                errors.append(f'Edge references unknown source node: {edge.source_id}')
            if edge.target_id not in node_ids:
                errors.append(f'Edge references unknown target node: {edge.target_id}')

        # Check entry/exit nodes exist
        for nid in self.entry_nodes:
            if nid not in node_ids:
                errors.append(f'Entry node not found: {nid}')
        for nid in self.exit_nodes:
            if nid not in node_ids:
                errors.append(f'Exit node not found: {nid}')

        # Check for empty workflow
        if not self.nodes:
            errors.append('Workflow has no nodes')

        # Check entry nodes defined
        if self.nodes and not self.entry_nodes:
            errors.append('No entry nodes defined')

        return errors

    def get_execution_order(self) -> list[str]:
        '''Return topological sort of node IDs. Raises ValueError if cycle detected.'''
        node_ids = {n.node_id for n in self.nodes}

        # Build adjacency list and in-degree count
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in self.edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                adj[edge.source_id].append(edge.target_id)
                in_degree[edge.target_id] += 1

        # Kahn's algorithm
        queue = deque(nid for nid in node_ids if in_degree[nid] == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(node_ids):
            raise ValueError('Workflow graph contains a cycle')

        return order
