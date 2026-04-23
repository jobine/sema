'''Resolved filesystem paths for the SEMA runtime, loaded from paths.json.

Mirrors the `LLMConfig.load()` caching pattern: a file cache keyed by the
resolved config path, so repeated `Paths.load()` calls are free.
'''

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


_DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / 'paths.json'


@dataclass(frozen=True)
class SEMAPaths:
    '''Filesystem paths for SEMA runtime artefacts.'''

    root: Path
    experiments: Path
    benchmarks: Path
    memory: Path
    trajectories: Path
    pricing: Path
    logs: Path

    _cache: ClassVar[dict[Path, 'SEMAPaths']] = {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'SEMAPaths':
        '''Load resolved paths from ``src/config/paths.json`` (or an override).

        All ``{root}`` placeholders are expanded against ``data["root"]``, then
        ``~`` is expanded via ``os.path.expanduser``. The returned object holds
        absolute ``Path`` values ready for direct use.
        '''
        config_path = (Path(path) if path else _DEFAULT_CONFIG_FILE).resolve()
        if config_path in cls._cache:
            return cls._cache[config_path]

        if not config_path.is_file():
            raise FileNotFoundError(f'Paths config not found: {config_path}')

        with config_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        root_raw = data.get('root', '~/.sema')
        root = _expand(root_raw)
        paths_section = data.get('paths', {}) or {}

        def resolve(key: str, default_template: str) -> Path:
            template = paths_section.get(key, default_template)
            expanded = template.replace('{root}', str(root))
            return _expand(expanded)

        instance = cls(
            root=root,
            experiments=resolve('experiments', '{root}/experiments'),
            benchmarks=resolve('benchmarks', '{root}/benchmarks'),
            memory=resolve('memory', '{root}/memory'),
            trajectories=resolve('trajectories', '{root}/trajectories'),
            pricing=resolve('pricing', '{root}/pricing/model_prices.json'),
            logs=resolve('logs', '{root}/logs/sema.log'),
        )
        cls._cache[config_path] = instance
        return instance


def _expand(p: str) -> Path:
    return Path(os.path.normpath(os.path.expanduser(str(p))))
