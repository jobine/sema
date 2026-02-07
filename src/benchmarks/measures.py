"""Common evaluation utilities for benchmarks."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation.
    Lower text, remove punctuation, articles and extra whitespace.
    
    Args:
        s: The string to normalize
        
    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
        
    Returns:
        F1 score as a float between 0.0 and 1.0
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate exact match score between prediction and ground truth.
    
    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string
        
    Returns:
        1.0 if normalized strings match exactly, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
