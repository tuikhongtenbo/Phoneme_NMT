"""
Score Aggregator Utility

Aggregates and compares evaluation scores from multiple models and tokenization methods.
Supports exporting to CSV/JSON and displaying comparison tables.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import pandas as pd


class ScoreAggregator:
    """
    Aggregates evaluation scores from multiple models and tokenization methods.
    
    Structure:
        {
            "model_name": {
                "tokenization_method": {
                    "level": {
                        "metric": score
                    }
                }
            }
        }
    """
    
    def __init__(self):
        self.scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    
    def add_scores(
        self,
        model_name: str,
        tokenization: str,
        level: str,
        scores: Dict[str, float]
    ):
        """
        Add scores for a specific model, tokenization method, and level.
        
        Args:
            model_name (str): Name of the model (e.g., "LSTM-Bahdanau", "Transformer")
            tokenization (str): Tokenization method (e.g., "word", "phoneme", "bpe")
            level (str): Evaluation level (e.g., "phoneme", "word")
            scores (Dict[str, float]): Dictionary of metric scores
                Example: {"bleu_1": 0.5, "bleu_2": 0.4, "rouge_l": 0.6, "meteor": 0.55}
        
        Example:
            >>> aggregator = ScoreAggregator()
            >>> aggregator.add_scores("LSTM-Bahdanau", "word", "word", {"bleu_1": 0.5, "rouge_l": 0.6})
        """
        if model_name not in self.scores:
            self.scores[model_name] = {}
        
        if tokenization not in self.scores[model_name]:
            self.scores[model_name][tokenization] = {}
        
        self.scores[model_name][tokenization][level] = scores
    
    def add_from_evaluator_result(
        self,
        model_name: str,
        tokenization: str,
        evaluator_result: Dict[str, Dict[str, float]]
    ):
        """
        Add scores from Evaluator.evaluate_model() result.
        
        Args:
            model_name (str): Name of the model
            tokenization (str): Tokenization method used
            evaluator_result (Dict[str, Dict[str, float]]): Result from evaluator
                Format: {"phoneme": {...}, "word": {...}}
        
        Example:
            >>> from src.evaluation import Evaluator
            >>> evaluator = Evaluator()
            >>> results = evaluator.evaluate_model(...)
            >>> aggregator.add_from_evaluator_result("LSTM-Bahdanau", "word", results)
        """
        for level, level_scores in evaluator_result.items():
            self.add_scores(model_name, tokenization, level, level_scores)
    
    def get_all_models(self) -> List[str]:
        return list(self.scores.keys())
    
    def get_all_tokenizations(self) -> List[str]:
        tokenizations = set()
        for model_scores in self.scores.values():
            tokenizations.update(model_scores.keys())
        return sorted(list(tokenizations))
    
    def get_all_levels(self) -> List[str]:
        levels = set()
        for model_scores in self.scores.values():
            for token_scores in model_scores.values():
                levels.update(token_scores.keys())
        return sorted(list(levels))
    
    def get_all_metrics(self) -> List[str]:
        metrics = set()
        for model_scores in self.scores.values():
            for token_scores in model_scores.values():
                for level_scores in token_scores.values():
                    metrics.update(level_scores.keys())
        return sorted(list(metrics))
    
    def get_score(
        self,
        model_name: str,
        tokenization: str,
        level: str,
        metric: str
    ) -> Optional[float]:
        """
        Get a specific score.
        
        Args:
            model_name (str): Model name
            tokenization (str): Tokenization method
            level (str): Evaluation level
            metric (str): Metric name
        
        Returns:
            Optional[float]: Score value or None if not found
        """
        try:
            return self.scores[model_name][tokenization][level][metric]
        except KeyError:
            return None
    
    def to_dataframe(
        self,
        level: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert scores to pandas DataFrame for easy analysis.
        
        Args:
            level (str, optional): Filter by specific level. If None, includes all levels.
            metrics (List[str], optional): Filter by specific metrics. If None, includes all.
        
        Returns:
            pd.DataFrame: DataFrame with columns: model, tokenization, level, metric, score
        
        Example:
            >>> df = aggregator.to_dataframe(level="word", metrics=["bleu_4", "rouge_l"])
            >>> print(df)
        """
        rows = []
        
        for model_name, model_data in self.scores.items():
            for tokenization, token_data in model_data.items():
                for eval_level, level_data in token_data.items():
                    # Filter by level if specified
                    if level is not None and eval_level != level:
                        continue
                    
                    for metric, score in level_data.items():
                        # Filter by metrics if specified
                        if metrics is not None and metric not in metrics:
                            continue
                        
                        rows.append({
                            "model": model_name,
                            "tokenization": tokenization,
                            "level": eval_level,
                            "metric": metric,
                            "score": score
                        })
        
        return pd.DataFrame(rows)
    
    def to_pivot_table(
        self,
        level: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        index: str = "model",
        columns: str = "metric",
        values: str = "score"
    ) -> pd.DataFrame:
        """
        Create a pivot table for easy comparison.
        
        Args:
            level (str, optional): Filter by level
            metrics (List[str], optional): Filter by metrics
            index (str): Column to use as index (default: "model")
            columns (str): Column to use as columns (default: "metric")
            values (str): Column to use as values (default: "score")
        
        Returns:
            pd.DataFrame: Pivot table
        
        Example:
            >>> pivot = aggregator.to_pivot_table(level="word", metrics=["bleu_4", "rouge_l"])
            >>> print(pivot)
        """
        df = self.to_dataframe(level=level, metrics=metrics)
        
        if df.empty:
            return pd.DataFrame()
        
        # Add combined identifier if needed
        if index == "model" and "tokenization" in df.columns:
            df["model_token"] = df["model"] + " (" + df["tokenization"] + ")"
            index = "model_token"
        
        pivot = df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc='first'
        )
        
        return pivot
    
    def print_comparison(
        self,
        level: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        descending: bool = True
    ):
        """
        Print a formatted comparison table.
        
        Args:
            level (str, optional): Filter by level
            metrics (List[str], optional): Filter by metrics. If None, uses common metrics.
            sort_by (str, optional): Metric to sort by
            descending (bool): Sort in descending order
        
        Example:
            >>> aggregator.print_comparison(level="word", metrics=["bleu_4", "rouge_l"], sort_by="bleu_4")
        """
        if metrics is None:
            metrics = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor"]
        
        # Filter metrics that exist in data
        all_metrics = self.get_all_metrics()
        metrics = [m for m in metrics if m in all_metrics]
        
        if not metrics:
            print("No metrics found!")
            return
        
        # Get all combinations
        models = self.get_all_models()
        tokenizations = self.get_all_tokenizations()
        
        if level is None:
            levels = self.get_all_levels()
        else:
            levels = [level]
        
        print("\n" + "=" * 120)
        print("EVALUATION SCORES COMPARISON")
        print("=" * 120)
        
        for eval_level in levels:
            print(f"\n{'LEVEL: ' + eval_level.upper():^120}")
            print("-" * 120)
            
            # Build table
            rows = []
            for model in models:
                for tokenization in tokenizations:
                    row = {
                        "Model": f"{model} ({tokenization})",
                    }
                    row_scores = {}
                    for metric in metrics:
                        score = self.get_score(model, tokenization, eval_level, metric)
                        if score is not None:
                            row[metric] = f"{score:.4f}"
                            row_scores[metric] = score
                        else:
                            row[metric] = "N/A"
                            row_scores[metric] = -1
                    
                    rows.append((row, row_scores))
            
            # Sort if specified
            if sort_by and sort_by in metrics:
                rows.sort(key=lambda x: x[1].get(sort_by, -1), reverse=descending)
            
            # Print header
            header = f"{'Model':<40}"
            metric_labels = {
                "bleu_1": "BLEU@1",
                "bleu_2": "BLEU@2",
                "bleu_3": "BLEU@3",
                "bleu_4": "BLEU@4",
                "rouge_l": "ROUGE-L",
                "meteor": "METEOR"
            }
            
            for metric in metrics:
                label = metric_labels.get(metric, metric)
                header += f"{label:>12}"
            print(header)
            print("-" * 120)
            
            # Print rows
            for row, _ in rows:
                row_str = f"{row['Model']:<40}"
                for metric in metrics:
                    row_str += f"{row[metric]:>12}"
                print(row_str)
            
            print("-" * 120)
        
        print("=" * 120 + "\n")
    
    def save_json(self, filepath: Union[str, Path]):
        """
        Save scores to JSON file.
        
        Args:
            filepath (Union[str, Path]): Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.scores, f, indent=2, ensure_ascii=False)
        
        print(f"Scores saved to {filepath}")
    
    def load_json(self, filepath: Union[str, Path]):
        """
        Load scores from JSON file.
        
        Args:
            filepath (Union[str, Path]): Path to JSON file
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.scores = json.load(f)
        
        print(f"Scores loaded from {filepath}")
    
    def save_csv(
        self,
        filepath: Union[str, Path],
        level: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Save scores to CSV file.
        
        Args:
            filepath (Union[str, Path]): Path to output CSV file
            level (str, optional): Filter by level
            metrics (List[str], optional): Filter by metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe(level=level, metrics=metrics)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Scores saved to {filepath}")
    
    def save_excel(
        self,
        filepath: Union[str, Path],
        level: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Save scores to Excel file with multiple sheets.
        
        Args:
            filepath (Union[str, Path]): Path to output Excel file
            level (str, optional): Filter by level
            metrics (List[str], optional): Filter by metrics
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Save full data
            df_full = self.to_dataframe(level=level, metrics=metrics)
            df_full.to_excel(writer, sheet_name='All Scores', index=False)
            
            # Save pivot tables by level
            levels = [level] if level else self.get_all_levels()
            for eval_level in levels:
                pivot = self.to_pivot_table(level=eval_level, metrics=metrics)
                if not pivot.empty:
                    pivot.to_excel(writer, sheet_name=f'{eval_level.capitalize()} Level')
        
        print(f"Scores saved to {filepath}")
    
    def get_best_model(
        self,
        metric: str,
        level: str,
        tokenization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the best model for a specific metric and level.
        
        Args:
            metric (str): Metric name (e.g., "bleu_4")
            level (str): Evaluation level (e.g., "word")
            tokenization (str, optional): Filter by tokenization method
        
        Returns:
            Dict with keys: model, tokenization, score
        """
        best_score = -1
        best_model = None
        best_tokenization = None
        
        for model_name, model_data in self.scores.items():
            for tokenization_method, token_data in model_data.items():
                if tokenization is not None and tokenization_method != tokenization:
                    continue
                
                if level in token_data:
                    score = token_data[level].get(metric, -1)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_tokenization = tokenization_method
        
        return {
            "model": best_model,
            "tokenization": best_tokenization,
            "score": best_score,
            "metric": metric,
            "level": level
        }


if __name__ == "__main__":

    print("=" * 60)
    print("Testing ScoreAggregator")
    print("=" * 60)
    
    # Create aggregator
    aggregator = ScoreAggregator()
    
    # Add sample scores
    print("\n1. Adding sample scores...")
    
    # LSTM-Bahdanau with word tokenization
    aggregator.add_scores(
        model_name="LSTM-Bahdanau",
        tokenization="word",
        level="word",
        scores={"bleu_1": 0.5234, "bleu_2": 0.4123, "bleu_3": 0.3567, "bleu_4": 0.3123, "rouge_l": 0.5678, "meteor": 0.5234}
    )
    aggregator.add_scores(
        model_name="LSTM-Bahdanau",
        tokenization="word",
        level="phoneme",
        scores={"bleu_1": 0.4567, "bleu_2": 0.3456, "bleu_3": 0.2987, "bleu_4": 0.2654, "rouge_l": 0.5123, "meteor": 0.4789}
    )
    
    # LSTM-Luong with word tokenization
    aggregator.add_scores(
        model_name="LSTM-Luong",
        tokenization="word",
        level="word",
        scores={"bleu_1": 0.5456, "bleu_2": 0.4321, "bleu_3": 0.3765, "bleu_4": 0.3234, "rouge_l": 0.5789, "meteor": 0.5345}
    )
    
    # Transformer with phoneme tokenization
    aggregator.add_scores(
        model_name="Transformer",
        tokenization="phoneme",
        level="phoneme",
        scores={"bleu_1": 0.5123, "bleu_2": 0.4234, "bleu_3": 0.3678, "bleu_4": 0.3234, "rouge_l": 0.5890, "meteor": 0.5456}
    )
    
    print(f"   Models: {aggregator.get_all_models()}")
    print(f"   Tokenizations: {aggregator.get_all_tokenizations()}")
    print(f"   Levels: {aggregator.get_all_levels()}")
    print(f"   Metrics: {aggregator.get_all_metrics()}")
    
    # Print comparison
    print("\n2. Printing comparison table (word level)...")
    aggregator.print_comparison(level="word", sort_by="bleu_4")
    
    # Get best model
    print("\n3. Finding best model for BLEU-4 at word level...")
    best = aggregator.get_best_model("bleu_4", "word")
    print(f"   Best: {best['model']} ({best['tokenization']}) - {best['score']:.4f}")
    
    # Save to files
    print("\n4. Saving to files...")
    aggregator.save_json("logs/test_scores.json")
    aggregator.save_csv("logs/test_scores.csv", level="word")
    
    # Test DataFrame
    print("\n5. Creating DataFrame...")
    df = aggregator.to_dataframe(level="word", metrics=["bleu_4", "rouge_l"])
    print(df)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)