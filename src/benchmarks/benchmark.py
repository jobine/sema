from __future__ import annotations

import os
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, List

from ..utils import get_logger

logger = get_logger(__name__)

class Benchmark(ABC):
    PASS = 'PASS'
    FAIL = 'FAIL'

    def __init__(self, name: str, data_folder: str):
        self.name = name
        self.data_folder = data_folder

        self._train_data: List[dict] | None = None
        self._validate_data: List[dict] | None = None
        self._test_data: List[dict] | None = None

        # if data folder does not exist, create it
        os.makedirs(self.data_folder, exist_ok=True)

        # Load the data
        self.load_data()
        
    @abstractmethod
    def load_data(self, force_reload: bool = False) -> None:
        '''
        Abstract method to download datasets if self.data_folder does not exist, then load data from `self.data_folder`, and assigned to _train_data/_validate_data/_test_data datasets.
        '''
        pass

    @abstractmethod
    def load_dataset(self, dataset: dict | None, force_reload: bool = False) -> List[dict] | None:
        '''
        Abstract method to load a specific dataset (train/validate/test).
        '''
        pass

    @abstractmethod
    async def evaluate(self, prediction: Any, label: Any) -> dict:
        '''
        Abstract method to evaluate the given model on the benchmark dataset.
        Should return either Benchmark.PASS or Benchmark.FAIL.
        '''
        pass

    @abstractmethod
    async def run(
        self,
        callback: Any,
        dataset: str = 'validate',
        num_samples: int | None = None,
        verbose: bool = False
    ) -> dict:
        '''
        Abstract method to run the benchmark evaluation using the provided callback function.
        
        Args:
            callback: The callback function to evaluate.
            dataset: The dataset to use ('train', 'validate', 'test').
            num_samples: Number of samples to evaluate (None for all).
            verbose: Whether to print detailed logs.
        
        Returns:
            Dictionary with aggregated metrics and individual results
        '''
        pass

    @property
    def train_data(self) -> List[dict] | None:
        if self._train_data is None:
            logger.error('Train data not loaded. Please call load_data() first.')
            # raise ValueError("Train data not loaded. Please call load_data() first.")
        return self._train_data
    
    @property
    def validate_data(self) -> List[dict] | None:
        if self._validate_data is None:
            logger.error('Validate data not loaded. Please call load_data() first.')
            # raise ValueError("Validate data not loaded. Please call load_data() first.")
        return self._validate_data
    
    @property
    def test_data(self) -> List[dict] | None:
        if self._test_data is None:
            logger.error('Test data not loaded. Please call load_data() first.')
            # raise ValueError("Test data not loaded. Please call load_data() first.")
        return self._test_data


class DatasetType(Enum):
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'
    ALL = 'all'
    
    @classmethod
    def from_value(cls, value: str, default: 'DatasetType' = VALIDATE) -> 'DatasetType':
        '''
        Get enum object by value (case-insensitive).
        
        Args:
            value: Enum value string
            default: Default value to return if not found
            
        Returns:
            Corresponding enum object, or default if not found
        '''
        value_lower = value.lower()
        for item in cls:
            if item.value == value_lower:
                return item
        return default

