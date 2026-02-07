import os
from typing import Any, List, Dict, Callable, Awaitable

from .benchmark import Benchmark, DatasetType
from .measures import exact_match_score, f1_score
from .tools import load_json, download_file
from ..utils import get_logger

logger = get_logger(__name__)


class HotpotQA(Benchmark):
    def __init__(self, data_folder: str = None, dataset_type: DatasetType = DatasetType.ALL):
        self.dataset_type = dataset_type

        location = os.path.normpath(os.path.expanduser(data_folder or '~/.sema/benchmarks'))
        super().__init__(name=type(self).__name__.lower(), data_folder=location)

    def load_data(self, force_reload: bool = False) -> None:
        '''
        Load the HotpotQA dataset into the benchmark.
        Downloads the dataset files if they do not exist or if force_reload is True.
        '''

        name = type(self).__name__.lower()
        # Get the directory where this module is located
        module_dir = os.path.dirname(os.path.abspath(__file__))
        benchmarks = load_json(os.path.join(module_dir, 'benchmarks.json'))

        if name in benchmarks:
            # Implement data loading logic here
            # For example, download dataset files if not present or force_reload is True
            # Then load the data into self._train_data, self._validate_data, self._test_data
            datasets = benchmarks[name]

            # Load training data
            if self.dataset_type in (DatasetType.ALL, DatasetType.TRAIN):
                self._train_data = self.load_dataset(dataset=datasets.get(DatasetType.TRAIN.value), force_reload=force_reload)
            
            # Load validation data
            if self.dataset_type in (DatasetType.ALL, DatasetType.VALIDATE):
                self._validate_data = self.load_dataset(dataset=datasets.get(DatasetType.VALIDATE.value), force_reload=force_reload)
            
            # Load test data
            if self.dataset_type in (DatasetType.ALL, DatasetType.TEST):
                self._test_data = self.load_dataset(dataset=datasets.get(DatasetType.TEST.value), force_reload=force_reload)
        else:
            raise ValueError(f'Benchmark {name} not found in benchmarks.json')
        
    def load_dataset(self, dataset: dict | None, force_reload: bool = False) -> list[dict] | None:
        '''
        Load a specific dataset (train/validate/test) based on the provided dataset info.
        If the dataset file does not exist or force_reload is True, it downloads the dataset.
        '''

        if dataset is None:
            return None
        
        file_path = os.path.join(self.data_folder, dataset['name'])
        if not os.path.exists(file_path) or force_reload:
            download_file(url=dataset['url'], destination_path=file_path)
        
        data = load_json(file_path)
        return data
        
    async def evaluate(self, prediction: Any, label: Any) -> dict:
        '''
        Evaluate a single prediction against the ground truth.
        
        Override of Benchmark.evaluate() for HotpotQA-specific evaluation.
        
        Args:
            prediction: The predicted answer string
            label: The ground truth answer string
            
        Returns:
            Dictionary with 'em' (exact match), 'f1' scores and 'result'
        '''
        # Ensure inputs are strings
        pred_str = str(prediction) if prediction is not None else ""
        label_str = str(label) if label is not None else ""
        
        em = exact_match_score(pred_str, label_str)
        f1 = f1_score(pred_str, label_str)
        
        return {
            'em': em,
            'f1': f1,
            'result': Benchmark.PASS if em == 1.0 else Benchmark.FAIL
        }
    
    async def run(
        self,
        callback: Callable[[str, Any], Awaitable[str]],
        dataset: str = 'validate',
        num_samples: int | None = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        '''
        Run the benchmark on a dataset using the provided agent function.
        
        Args:
            callback: Async function that takes (question, context) and returns answer string
            dataset: Which dataset to use ('train', 'validate', 'test')
            num_samples: Number of samples to evaluate (None for all)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with aggregated metrics and individual results
        '''
        # Get the appropriate dataset
        if dataset == 'train':
            data = self.train_data
        elif dataset == 'validate':
            data = self.validate_data
        elif dataset == 'test':
            data = self.test_data
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        
        if data is None:
            raise ValueError(f'Dataset "{dataset}" not loaded')
        
        # Limit samples if specified
        if num_samples is not None:
            data = data[:num_samples]
        
        results = []
        total_em = 0.0
        total_f1 = 0.0
        
        for i, item in enumerate(data):
            question = item['question']
            context = item.get('context', [])
            ground_truth = item['answer']
            
            if verbose:
                logger.info(f'Processing {i+1}/{len(data)}: {question[:50]}...')
            
            try:
                # Get prediction from agent
                prediction = await callback(question, context)
                
                # Evaluate
                eval_result = await self.evaluate(prediction, ground_truth)
                
                result = {
                    'id': item.get('_id', i),
                    'question': question,
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'em': eval_result['em'],
                    'f1': eval_result['f1'],
                    'result': eval_result['result']
                }
                
                total_em += eval_result['em']
                total_f1 += eval_result['f1']
                
            except Exception as e:
                logger.error(f'Error processing question {i}: {e}')
                result = {
                    'id': item.get('_id', i),
                    'question': question,
                    'prediction': '',
                    'ground_truth': ground_truth,
                    'em': 0.0,
                    'f1': 0.0,
                    'result': Benchmark.FAIL,
                    'error': str(e)
                }
            
            results.append(result)
            
            if verbose:
                logger.info(f'  EM: {result["em"]:.2f}, F1: {result["f1"]:.2f}')
        
        # Calculate aggregate metrics
        n = len(results)
        avg_em = total_em / n if n > 0 else 0.0
        avg_f1 = total_f1 / n if n > 0 else 0.0
        
        return {
            'metrics': {
                'exact_match': avg_em,
                'f1': avg_f1,
                'num_samples': n,
                'num_passed': sum(1 for r in results if r['result'] == Benchmark.PASS)
            },
            'results': results
        }

