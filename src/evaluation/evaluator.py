"""
Main evaluation class
"""
from typing import List, Dict, Union, Optional, Any
import numpy as np

from .bleu import BLEUMetric
from .rouge import ROUGEMetric
from .meteor import METEORMetric


class Evaluator:
    """
    Main evaluation class for machine translation evaluation.

    -> Orchestrates multiple metrics for evaluation.
    
    Attributes:
        metrics (Dict): Dictionary of metric instances
    """
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        **metric_kwargs
    ):
        """
        Initialize evaluator with specified metrics.

        Args:
            metrics (List[str], optional): List of metric names to use
                Options: 'bleu', 'rouge_l', 'meteor'
                If None, uses all available metrics
            **metric_kwargs: Additional keyword arguments for each metric
                - bleu_max_n: Maximum n-gram length for BLEU (default: 4)
                - rouge_beta: Beta parameter for ROUGE (default: 1.0)
                - meteor_alpha: Alpha parameter for METEOR (default: 0.9)
                - meteor_beta: Beta parameter for METEOR (default: 3.0)
                - meteor_gamma: Gamma parameter for METEOR (default: 0.5)
        """
        if metrics is None:
            metrics = ['bleu', 'rouge_l', 'meteor']

        self.metrics = {}
        self._initialize_metrics(metrics, metric_kwargs)

    def _initialize_metrics(self, metrics: List[str], metric_kwargs: Dict[str, Any]):
        """
        Initializa metric instrances.

        Args:
            metrics (List[str]): List of metric names
            metric_kwargs (Dict[str, Any]): Additional keyword arguments for each metric
        """
        for metric_name in metrics:
            if metric_name == 'bleu':
                self.metrics[metric_name] = BLEUMetric(
                    max_n=metric_kwargs.get('bleu_max_n', 4),
                    smoothing=metric_kwargs.get('bleu_smoothing', False)
                )
            
            elif metric_name == 'rouge_l':
                self.metrics[metric_name] = ROUGEMetric(
                    beta=metric_kwargs.get('rouge_beta', 1.0)
                )

            elif metric_name == 'meteor':
                self.metrics['meteor'] = METEORMetric(
                    alpha=metric_kwargs.get('meteor_alpha', 0.9),
                    beta=metric_kwargs.get('meteor_beta', 3.0),
                    gamma=metric_kwargs.get('meteor_gamma', 0.5)
                )

    def evaluate(
        self,
        references: Union[str, List[str]],
        hypotheses: Union[str, List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate translation quality using configured metrics.

        Args:
            references (Union[str, List[str]]): Reference translations
            hypotheses (Union[str, List[str]]): Hypothesis translations

        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        all_scores = {}

        for metric_name, metric in self.metrics.items():
            scores = metric.compute(references, hypotheses)

            if metric_name == 'bleu':
                all_scores.update(scores)
            elif metric_name == 'rouge_l':
                all_scores['rouge_l'] = scores.get('rouge_l_f', 0.0)
            elif metric_name == 'meteor':
                all_scores['meteor'] = scores.get('meteor', 0.0)
        
        return all_scores

    def evaluate_model(
        self,
        references_phoneme: List[str],
        hypotheses_phoneme: List[str],
        references_word: List[str],
        hypotheses_word: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model at both phoneme and word levels.

        Args:
            references_phoneme (List[str]): Phoneme-level reference translations
            hypotheses_phoneme (List[str]): Phoneme-level hypothesis translations
            references_word (List[str]): Word-level reference translations
            hypotheses_word (List[str]): Word-level hypothesis translations
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of metric scores for each level
        """
        results = {}
        
        print("Evaluating Phoneme level...")
        results['phoneme'] = self.evaluate(references_phoneme, hypotheses_phoneme)
        
        print("Evaluating Word level...")
        results['word'] = self.evaluate(references_word, hypotheses_word)
        
        return results

    def print_results(self, results: Dict[str, Dict[str, float]], model_name: str = "Model"):
        """
        Print evaluation results.
        
        Args:
            results: Results from evaluate_model()
            model_name: Name of the model being evaluated
        """
        print("\n" + "="*70)
        print(f"EVALUATION RESULTS: {model_name}")
        print("="*70)
        
        for level in ['phoneme', 'word']:
            if level in results:
                print(f"\n{level.upper()} LEVEL:")
                print("-" * 50)
                scores = results[level]
                
                # Print BLEU scores
                if 'bleu_1' in scores:
                    print(f"  BLEU@1:    {scores['bleu_1']:.4f}")
                if 'bleu_2' in scores:
                    print(f"  BLEU@2:    {scores['bleu_2']:.4f}")
                if 'bleu_3' in scores:
                    print(f"  BLEU@3:    {scores['bleu_3']:.4f}")
                if 'bleu_4' in scores:
                    print(f"  BLEU@4:    {scores['bleu_4']:.4f}")
                
                # Print ROUGE-L
                if 'rouge_l' in scores:
                    print(f"  ROUGE-L:   {scores['rouge_l']:.4f}")
                
                # Print METEOR
                if 'meteor' in scores:
                    print(f"  METEOR:    {scores['meteor']:.4f}")
        
        print("="*70 + "\n")

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """
        Compare multiple models.
        
        Args:
            model_results: Dict[model_name -> Dict[level -> Dict[metric -> score]]]
        """
        print("\n" + "="*110)
        print("MODEL COMPARISON")
        print("="*110)
        
        # Header
        model_names = list(model_results.keys())
        header = f"{'#':<5} {'Metrics':<15} {'Level':<10}"
        for model_name in model_names:
            header += f"{model_name:<25}"
        print(header)
        print("-" * 110)
        
        # Print scores
        metrics_list = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge_l', 'meteor']
        metric_names = ['BLEU@1', 'BLEU@2', 'BLEU@3', 'BLEU@4', 'ROUGE-L', 'Meteor']
        
        row_num = 1
        for metric, metric_name in zip(metrics_list, metric_names):
            for level in ['Phoneme', 'Word']:
                level_key = level.lower()
                row = f"{row_num:<5} {metric_name:<15} {level:<10}"
                
                for model_name in model_names:
                    if level_key in model_results[model_name]:
                        score = model_results[model_name][level_key].get(metric, 0.0)
                        row += f"{score:<25.4f}"
                    else:
                        row += f"{'N/A':<25}"
                
                print(row)
                row_num += 1
        
        print("="*110 + "\n")