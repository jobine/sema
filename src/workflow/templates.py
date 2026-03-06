'''Pre-configured workflow templates for common patterns.'''

from __future__ import annotations

from .schema import Role, Action, WorkflowNode, WorkflowEdge, Workflow


class WorkflowTemplate:
    '''Factory for common workflow topologies.'''

    @staticmethod
    def blank(goal: str = '', environment: str = '') -> Workflow:
        '''Create an empty workflow. Optimizer builds structure from scratch.'''
        return Workflow(goal=goal, environment=environment)

    @staticmethod
    def single_agent(
        role_name: str = 'solver',
        action_name: str = 'solve',
        system_prompt: str = 'You are a helpful assistant.',
        instruction_prompt: str = 'Answer the question.',
        goal: str = '',
        environment: str = '',
        **agent_config,
    ) -> Workflow:
        '''Create a single-agent workflow.'''
        node = WorkflowNode(
            node_id='agent_0',
            role=Role(name=role_name, system_prompt=system_prompt),
            action=Action(name=action_name, instruction_prompt=instruction_prompt),
            agent_config=agent_config,
        )
        return Workflow(
            goal=goal,
            environment=environment,
            nodes=[node],
            entry_nodes=['agent_0'],
            exit_nodes=['agent_0'],
        )

    @staticmethod
    def chain(
        roles: list[dict] | None = None,
        goal: str = '',
        environment: str = '',
    ) -> Workflow:
        '''Create a sequential chain of agents.

        Args:
            roles: List of dicts with keys: name, system_prompt, action_name, instruction_prompt.
                   Defaults to a 3-step chain: decompose -> research -> synthesize.
            goal: High-level goal for the workflow.
        '''
        if roles is None:
            roles = [
                {
                    'name': 'decomposer',
                    'system_prompt': 'You decompose complex questions into sub-questions.',
                    'action_name': 'decompose',
                    'instruction_prompt': 'Break down the question into sub-questions.',
                },
                {
                    'name': 'researcher',
                    'system_prompt': 'You research and gather information.',
                    'action_name': 'research',
                    'instruction_prompt': 'Research each sub-question using the context.',
                },
                {
                    'name': 'synthesizer',
                    'system_prompt': 'You synthesize information into a final answer.',
                    'action_name': 'synthesize',
                    'instruction_prompt': 'Combine the research into a final answer.',
                },
            ]

        nodes = []
        edges = []
        for i, r in enumerate(roles):
            node_id = f'agent_{i}'
            nodes.append(WorkflowNode(
                node_id=node_id,
                role=Role(name=r['name'], system_prompt=r.get('system_prompt', '')),
                action=Action(
                    name=r.get('action_name', r['name']),
                    instruction_prompt=r.get('instruction_prompt', ''),
                ),
            ))
            if i > 0:
                edges.append(WorkflowEdge(
                    source_id=f'agent_{i - 1}',
                    target_id=node_id,
                    edge_type='sequential',
                ))

        return Workflow(
            goal=goal,
            environment=environment,
            nodes=nodes,
            edges=edges,
            entry_nodes=[nodes[0].node_id] if nodes else [],
            exit_nodes=[nodes[-1].node_id] if nodes else [],
        )

    @staticmethod
    def debate(
        num_debaters: int = 2,
        num_rounds: int = 2,
        goal: str = '',
        environment: str = '',
    ) -> Workflow:
        '''Create a debate workflow with multiple debaters and a judge.

        Args:
            num_debaters: Number of debater agents.
            num_rounds: Number of debate rounds (feedback loop iterations).
            goal: High-level goal for the workflow.
        '''
        nodes = []
        edges = []

        # Create debater nodes
        for i in range(num_debaters):
            node_id = f'debater_{i}'
            nodes.append(WorkflowNode(
                node_id=node_id,
                role=Role(
                    name=f'debater_{i}',
                    system_prompt=f'You are debater {i}. Argue your position clearly.',
                ),
                action=Action(
                    name='debate',
                    instruction_prompt='Present your argument on the question.',
                ),
            ))

        # Create judge node
        judge_id = 'judge'
        nodes.append(WorkflowNode(
            node_id=judge_id,
            role=Role(
                name='judge',
                system_prompt='You evaluate arguments and determine the best answer.',
            ),
            action=Action(
                name='judge',
                instruction_prompt='Evaluate all arguments and provide the final answer.',
            ),
        ))

        # Debaters feed into judge
        for i in range(num_debaters):
            edges.append(WorkflowEdge(
                source_id=f'debater_{i}',
                target_id=judge_id,
                edge_type='sequential',
            ))

        # Feedback loop: judge back to debaters for multi-round debate
        if num_rounds > 1:
            for i in range(num_debaters):
                edges.append(WorkflowEdge(
                    source_id=judge_id,
                    target_id=f'debater_{i}',
                    edge_type='feedback',
                    condition='round < max_rounds',
                    max_iterations=num_rounds,
                ))

        entry_nodes = [f'debater_{i}' for i in range(num_debaters)]
        return Workflow(
            goal=goal,
            environment=environment,
            nodes=nodes,
            edges=edges,
            entry_nodes=entry_nodes,
            exit_nodes=[judge_id],
        )

    @staticmethod
    def hierarchical(
        num_workers: int = 3,
        goal: str = '',
        environment: str = '',
    ) -> Workflow:
        '''Create a hierarchical manager-worker workflow.

        Args:
            num_workers: Number of worker agents.
            goal: High-level goal for the workflow.
        '''
        nodes = []
        edges = []

        # Manager node
        manager_id = 'manager'
        nodes.append(WorkflowNode(
            node_id=manager_id,
            role=Role(
                name='manager',
                system_prompt='You delegate tasks and aggregate results from workers.',
            ),
            action=Action(
                name='delegate',
                instruction_prompt='Break the task into sub-tasks and assign to workers.',
            ),
        ))

        # Worker nodes
        for i in range(num_workers):
            worker_id = f'worker_{i}'
            nodes.append(WorkflowNode(
                node_id=worker_id,
                role=Role(
                    name=f'worker_{i}',
                    system_prompt=f'You are worker {i}. Complete your assigned sub-task.',
                ),
                action=Action(
                    name='execute',
                    instruction_prompt='Complete the sub-task assigned to you.',
                ),
            ))
            # Manager -> Worker
            edges.append(WorkflowEdge(
                source_id=manager_id,
                target_id=worker_id,
                edge_type='sequential',
            ))

        # Aggregator node
        aggregator_id = 'aggregator'
        nodes.append(WorkflowNode(
            node_id=aggregator_id,
            role=Role(
                name='aggregator',
                system_prompt='You combine worker outputs into a final result.',
            ),
            action=Action(
                name='aggregate',
                instruction_prompt='Combine all worker outputs into a final answer.',
            ),
        ))

        # Workers -> Aggregator
        for i in range(num_workers):
            edges.append(WorkflowEdge(
                source_id=f'worker_{i}',
                target_id=aggregator_id,
                edge_type='sequential',
            ))

        return Workflow(
            goal=goal,
            environment=environment,
            nodes=nodes,
            edges=edges,
            entry_nodes=[manager_id],
            exit_nodes=[aggregator_id],
        )
