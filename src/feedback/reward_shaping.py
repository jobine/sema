'''Reward shaping utilities for SEMA feedback system.'''

from __future__ import annotations


class RewardShaper:
    '''Shapes raw rewards for use in evolutionary optimization.

    Provides normalization, baseline removal, and ranking-based shaping.
    '''

    def __init__(self, baseline_ema_alpha: float = 0.1) -> None:
        self._baseline: float | None = None
        self._alpha = baseline_ema_alpha

    def shape(self, rewards: list[float]) -> list[float]:
        '''Apply reward shaping: normalize and remove baseline.

        Args:
            rewards: Raw reward values.

        Returns:
            Shaped reward values.
        '''
        if not rewards:
            return []

        # Update baseline with exponential moving average
        batch_mean = sum(rewards) / len(rewards)
        if self._baseline is None:
            self._baseline = batch_mean
        else:
            self._baseline = (1 - self._alpha) * self._baseline + self._alpha * batch_mean

        # Remove baseline
        centered = [r - self._baseline for r in rewards]

        # Normalize by standard deviation
        if len(centered) > 1:
            mean_c = sum(centered) / len(centered)
            variance = sum((x - mean_c) ** 2 for x in centered) / len(centered)
            std = variance ** 0.5
            if std > 1e-8:
                centered = [c / std for c in centered]

        return centered

    def fitness_from_rewards(self, rewards: list[float]) -> list[float]:
        '''Convert rewards to fitness scores using ranking-based shaping.

        Maps rewards to ranks, then normalizes ranks to [0, 1].
        This is more robust to outliers than raw rewards.

        Args:
            rewards: Raw reward values.

        Returns:
            Fitness values in [0, 1], based on rank.
        '''
        if not rewards:
            return []

        n = len(rewards)
        if n == 1:
            return [0.5]

        # Create (index, reward) pairs, sort by reward
        indexed = sorted(enumerate(rewards), key=lambda x: x[1])

        # Assign ranks (0-indexed), handle ties with average rank
        fitness = [0.0] * n
        i = 0
        while i < n:
            # Find all elements with the same reward
            j = i
            while j < n and indexed[j][1] == indexed[i][1]:
                j += 1
            # Average rank for ties
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                orig_idx = indexed[k][0]
                fitness[orig_idx] = avg_rank / (n - 1)
            i = j

        return fitness
