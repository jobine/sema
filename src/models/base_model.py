import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict


@dataclass
class LLMConfig:
	'''Configuration for a language model loaded from models.json.'''

	name: str
	provider: str
	description: str
	base_url: str
	api_key: str
	temperature: float | None = None
	_file_cache: ClassVar[Dict[Path, Dict[str, Dict[str, Any]]]] = {}
	_instance_cache: ClassVar[Dict[tuple[Path, str], 'LLMConfig']] = {}

	@classmethod
	def load(cls, model: str, path: str | Path | None = None) -> 'LLMConfig':
		'''Load configuration for a given model key.

		Args:
			model: Model key defined in the JSON file.
			path: Optional override path to the JSON config file.
		'''
		config_path = (Path(path) if path else Path(__file__).with_name('models.json')).resolve()
		cache_key = (config_path, model)

		if cache_key in cls._instance_cache:
			return cls._instance_cache[cache_key]

		if not config_path.is_file():
			raise FileNotFoundError(f'Config file not found: {config_path}')

		if config_path not in cls._file_cache:
			with config_path.open('r', encoding='utf-8') as fp:
				cls._file_cache[config_path] = json.load(fp)

		data = cls._file_cache[config_path]

		if model not in data:
			raise ValueError(f'Model "{model}" not found in {config_path}')

		entry = data[model]
		instance = cls(
			name=model,
			provider=entry.get('provider', entry.get('type', '')),
			description=entry.get('description', ''),
			base_url=entry.get('base_url', ''),
			api_key=entry.get('api_key', ''),
			temperature=entry.get('temperature', 1.0),
		)
		cls._instance_cache[cache_key] = instance
		return instance


class AsyncBaseLLM(ABC):
	'''Abstract base class for async LLM implementations.'''

	def __init__(self, config: LLMConfig) -> None:
		self.config = config

	@abstractmethod
	async def __call__(self, prompt: str, **kwargs: Any) -> str:
		'''Call the LLM asynchronously and return text.'''
		...
