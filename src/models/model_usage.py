'''Thread-safe singleton for tracking LLM token usage and cost.'''

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..utils import get_logger, download_file, load_json

logger = get_logger(__name__)

_DEFAULT_PRICING_PATH = Path.home() / '.sema' / 'pricing' / 'model_prices.json'
_LITELLM_PRICES_URL = (
	'https://raw.githubusercontent.com/BerriAI/litellm/main/'
	'model_prices_and_context_window.json'
)


class ModelUsage:
	'''Global singleton that accumulates prompt/completion tokens and cost.

	Thread-safe: the accumulator counters are guarded by an instance lock,
	and price-table swaps are atomic dict reassignments (safe under CPython).
	'''

	_instance: ModelUsage | None = None
	_init_lock: threading.Lock = threading.Lock()

	def __init__(self, pricing_path: Path | None = None) -> None:
		self._lock = threading.Lock()
		self._pricing_path = pricing_path or _DEFAULT_PRICING_PATH
		self._prices: dict[str, dict[str, Any]] = {}
		self._total_prompt_tokens: int = 0
		self._total_completion_tokens: int = 0
		self._total_cost: float = 0.0
		self._load_prices()

	def _load_prices(self) -> None:
		'''Load pricing data from local cache, downloading on first run.'''
		if not self._pricing_path.exists():
			try:
				download_file(url=_LITELLM_PRICES_URL, destination_path=str(self._pricing_path))
			except Exception as exc:
				logger.warning(f'Failed to fetch pricing from {_LITELLM_PRICES_URL}: {exc}. All costs will be $0.')
				return
		try:
			data = load_json(str(self._pricing_path))
			self._prices = data if isinstance(data, dict) else {}
			logger.info(f'Loaded pricing for {len(self._prices)} models from {self._pricing_path}')
		except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
			logger.warning(f'Failed to load pricing: {exc}. All costs will be $0.')

	def update_prices(self, url: str = _LITELLM_PRICES_URL) -> None:
		'''Fetch latest pricing from litellm GitHub and merge into local cache.

		Merge strategy: {**old_local, **new_remote} — remote overwrites matching
		keys; local-only keys (user-added models) are preserved.
		'''
		logger.info(f'Fetching pricing from {url} ...')
		tmp_path = self._pricing_path.with_suffix('.new.json')
		try:
			download_file(url=url, destination_path=str(tmp_path))
			remote_prices = load_json(str(tmp_path))
		finally:
			if tmp_path.exists():
				tmp_path.unlink()

		if not isinstance(remote_prices, dict):
			raise ValueError(f'Expected JSON object at {url}, got {type(remote_prices).__name__}')

		# Merge: old local values as base, remote overwrites
		merged = {**self._prices, **remote_prices}

		# Persist
		self._pricing_path.parent.mkdir(parents=True, exist_ok=True)
		with open(self._pricing_path, 'w', encoding='utf-8') as f:
			json.dump(merged, f, indent=2)

		with self._lock:
			self._prices = merged
		logger.info(
			f'Pricing updated: {len(remote_prices)} remote models merged, '
			f'{len(merged)} total models cached at {self._pricing_path}'
		)

	@classmethod
	def get_instance(cls, pricing_path: Path | None = None) -> ModelUsage:
		'''Return the global singleton, creating it on first call.'''
		if cls._instance is None:
			with cls._init_lock:
				if cls._instance is None:
					cls._instance = cls(pricing_path=pricing_path)
		return cls._instance

	def _compute_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
		'''Compute cost for a single LLM call based on pricing data.'''
		price_info = self._prices.get(model_id)
		if price_info is None:
			return 0.0
		input_cost = prompt_tokens * price_info.get('input_cost_per_token', 0.0)
		output_cost = completion_tokens * price_info.get('output_cost_per_token', 0.0)
		return input_cost + output_cost

	def record(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
		'''Record one LLM call: compute cost, log, accumulate.'''
		cost = self._compute_cost(model_id, prompt_tokens, completion_tokens)
		with self._lock:
			self._total_prompt_tokens += prompt_tokens
			self._total_completion_tokens += completion_tokens
			self._total_cost += cost
		logger.info(
			f'LLM usage: model={model_id}, prompt_tokens={prompt_tokens}, '
			f'completion_tokens={completion_tokens}, cost=${cost:.6f}'
		)

	@property
	def total_prompt_tokens(self) -> int:
		return self._total_prompt_tokens

	@property
	def total_completion_tokens(self) -> int:
		return self._total_completion_tokens

	@property
	def total_cost(self) -> float:
		return self._total_cost

	def summary(self) -> str:
		'''Formatted summary string for console output.'''
		return (
			f'=== Model Usage Summary ===\n'
			f'Total prompt tokens:     {self._total_prompt_tokens:,}\n'
			f'Total completion tokens:  {self._total_completion_tokens:,}\n'
			f'Total cost:              ${self._total_cost:.4f}'
		)

	def report_section(self) -> str:
		'''Markdown section for report.md.'''
		return (
			f'## Model Usage\n\n'
			f'| Metric | Value |\n'
			f'|--------|-------|\n'
			f'| Total prompt tokens | {self._total_prompt_tokens:,} |\n'
			f'| Total completion tokens | {self._total_completion_tokens:,} |\n'
			f'| Total cost | ${self._total_cost:.4f} |\n'
		)
