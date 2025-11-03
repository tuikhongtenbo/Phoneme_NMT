"""
BLEU (Bilingual Evaluation Understudy) metric implementation

Reference:
    Papineni et al. (2002) "BLEU: a Method for Automatic Evaluation of Machine Translation"
    https://aclanthology.org/P02-1040.pdf
"""
import math
from collections import Counter
from typing import List, Tuple, Dict, Union

from .base_metric import BaseMetric


class BLEUMetric(BaseMetric):
    """
    BLEU Score metric for machine translation evaluation.
    
    BLEU measures n-gram precision with brevity penalty for short translations.
    """
    
    def __init__(self, max_n: int = 4, smoothing: bool = False):
        """
        Initialize BLEU metric.
        
        Args:
            max_n (int): Maximum n-gram size (default: 4 for BLEU-4)
            smoothing (bool): Apply smoothing for zero counts
        """
        super().__init__(name="BLEU")
        self.max_n = max_n
        self.smoothing = smoothing
    
    def compute(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]], **kwargs) -> Dict[str, float]:
        """
        Calculate BLEU scores (BLEU@1, @2, @3, @4).
        
        Args:
            references: Single reference string or list of reference strings
            hypotheses: Single hypothesis string or list of hypothesis strings
            **kwargs: Additional arguments
        
        Returns:
            Dict[str, float]: Dictionary with BLEU scores
        """
        max_n = kwargs.get('max_n', self.max_n)
        smoothing = kwargs.get('smoothing', self.smoothing)
        
        # Handle single sentence case
        if isinstance(references, str):
            references = [references]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        
        # Tokenize if needed
        if isinstance(references[0], str):
            reference_tokens = [[ref.split()] for ref in references]
            hypothesis_tokens = [hyp.split() for hyp in hypotheses]
        else:
            reference_tokens = [[ref] for ref in references]
            hypothesis_tokens = hypotheses
        
        results = {}
        
        # Calculate BLEU-n (cumulative: geometric mean of P1...Pn)
        for n in range(1, max_n + 1):
            # Uniform weights for P1...Pn only
            weights = tuple([1.0/n if i < n else 0.0 for i in range(max_n)])
            
            if len(hypothesis_tokens) == 1:
                score = self._sentence_bleu(
                    reference_tokens[0], 
                    hypothesis_tokens[0], 
                    weights, 
                    smoothing,
                    max_n=n  
                )
            else:
                score = self._corpus_bleu(
                    reference_tokens, 
                    hypothesis_tokens, 
                    weights, 
                    smoothing,
                    max_n=n 
                )
            
            results[f'bleu_{n}'] = score
        
        # Overall BLEU-4 is the same as bleu_4
        results['bleu'] = results[f'bleu_{max_n}']
        
        return results
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """
        Extract n-grams from a list of tokens.
        Args:
            tokens (List[str]): List of tokens
            n (int): N-gram size
        
        Returns:
            Counter: Counter of n-grams
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    @staticmethod
    def _modified_precision(reference_tokens: List[List[str]], hypothesis_tokens: List[str], n: int) -> float:
        """
        Calculate modified n-gram precision with clipping.
        Args:
            reference_tokens (List[List[str]]): List of reference tokens
            hypothesis_tokens (List[str]): List of hypothesis tokens
            n (int): N-gram size
        
        Returns:
            float: Modified n-gram precision
        """
        hyp_ngrams = BLEUMetric._get_ngrams(hypothesis_tokens, n)
        
        if len(hyp_ngrams) == 0:
            return 0.0
        
        max_ref_counts = Counter()
        for ref_tokens in reference_tokens:
            ref_ngrams = BLEUMetric._get_ngrams(ref_tokens, n)
            for ngram in ref_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        clipped_counts = {
            ngram: min(count, max_ref_counts[ngram])
            for ngram, count in hyp_ngrams.items()
        }
        
        numerator = sum(clipped_counts.values())
        denominator = sum(hyp_ngrams.values())
        
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def _brevity_penalty(reference_length: int, hypothesis_length: int) -> float:
        """
        Calculate brevity penalty to penalize short translations.
        Args:
            reference_length (int): Length of the reference
            hypothesis_length (int): Length of the hypothesis
        
        Returns:
            float: Brevity penalty
        """
        if hypothesis_length > reference_length:
            return 1.0
        elif hypothesis_length == 0:
            return 0.0
        else:
            return math.exp(1 - reference_length / hypothesis_length)
    
    @staticmethod
    def _closest_ref_length(reference_tokens: List[List[str]], hypothesis_length: int) -> int:
        """
        Find the reference length closest to hypothesis length.
        Args:
            reference_tokens (List[List[str]]): List of reference tokens
            hypothesis_length (int): Length of the hypothesis
        
        Returns:
            int: Closest reference length
        """
        ref_lengths = [len(ref) for ref in reference_tokens]
        closest_length = min(ref_lengths, key=lambda x: abs(x - hypothesis_length))
        return closest_length
    
    def _sentence_bleu(
        self,
        reference_tokens: List[List[str]], 
        hypothesis_tokens: List[str],
        weights: Tuple[float, ...],
        smoothing: bool = False,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score for a single sentence.
        Args:
            reference_tokens (List[List[str]]): List of reference tokens
            hypothesis_tokens (List[str]): List of hypothesis tokens
            weights (Tuple[float, ...]): Weights for each n-gram size
            smoothing (bool): Apply smoothing for zero counts
            max_n (int): Maximum n-gram size to consider
        
        Returns:
            float: BLEU score
        """
        ref_len = self._closest_ref_length(reference_tokens, len(hypothesis_tokens))
        bp = self._brevity_penalty(ref_len, len(hypothesis_tokens))
        
        precisions = []
        for n in range(1, max_n + 1):
            precision = self._modified_precision(reference_tokens, hypothesis_tokens, n)
            if smoothing and precision == 0.0:
                precision = 1e-7
            precisions.append(precision)
        
        # Calculate weighted geometric mean
        log_sum = 0.0
        for p, w in zip(precisions, weights[:max_n]):
            if w > 0 and p > 0:
                log_sum += w * math.log(p)
        
        geo_mean = math.exp(log_sum)
        
        bleu = bp * geo_mean
        return bleu
    
    def _corpus_bleu(
        self,
        list_of_references: List[List[List[str]]], 
        hypotheses: List[List[str]],
        weights: Tuple[float, ...],
        smoothing: bool = False,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score for entire corpus.
        Args:
            list_of_references (List[List[List[str]]]): List of reference tokens
            hypotheses (List[List[str]]): List of hypothesis tokens
            weights (Tuple[float, ...]): Weights for each n-gram size
            smoothing (bool): Apply smoothing for zero counts
            max_n (int): Maximum n-gram size to consider
        
        Returns:
            float: BLEU score
        """
        total_ref_length = 0
        total_hyp_length = 0
        clipped_counts = [Counter() for _ in range(max_n)]
        total_counts = [Counter() for _ in range(max_n)]
        
        for references, hypothesis in zip(list_of_references, hypotheses):
            total_hyp_length += len(hypothesis)
            total_ref_length += self._closest_ref_length(references, len(hypothesis))
            
            for n in range(1, max_n + 1):
                hyp_ngrams = self._get_ngrams(hypothesis, n)
                
                max_ref_counts = Counter()
                for ref_tokens in references:
                    ref_ngrams = self._get_ngrams(ref_tokens, n)
                    for ngram in ref_ngrams:
                        max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
                
                for ngram, count in hyp_ngrams.items():
                    clipped_counts[n-1][ngram] += min(count, max_ref_counts[ngram])
                    total_counts[n-1][ngram] += count
        
        bp = self._brevity_penalty(total_ref_length, total_hyp_length)
        
        precisions = []
        for n in range(max_n):
            numerator = sum(clipped_counts[n].values())
            denominator = sum(total_counts[n].values())
            
            if denominator > 0:
                precision = numerator / denominator
            else:
                precision = 0.0
            
            if smoothing and precision == 0.0:
                precision = 1e-7
            
            precisions.append(precision)
        
        # Calculate weighted geometric mean
        log_sum = 0.0
        for p, w in zip(precisions, weights[:max_n]):
            if w > 0 and p > 0:
                log_sum += w * math.log(p)
        
        geo_mean = math.exp(log_sum)
        
        bleu = bp * geo_mean
        return bleu


if __name__ == "__main__":
    print("="*70)
    print("BLEU Metric Test - 3 Sample Sentences")
    print("="*70)
    
    # Sample 1: Perfect match
    print("\n[Sample 1] Perfect Match:")
    refs1 = ["the cat is on the mat"]
    hyps1 = ["the cat is on the mat"]
    print(f"Reference:  {refs1[0]}")
    print(f"Hypothesis: {hyps1[0]}")
    
    bleu1 = BLEUMetric(max_n=4, smoothing=False)
    scores1 = bleu1.compute(refs1, hyps1)
    print("\nScores:")
    for k, v in sorted(scores1.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    # Sample 2: Partial match
    print("\n" + "-"*70)
    print("[Sample 2] Partial Match:")
    refs2 = ["hello world how are you"]
    hyps2 = ["hello world"]
    print(f"Reference:  {refs2[0]}")
    print(f"Hypothesis: {hyps2[0]}")
    
    bleu2 = BLEUMetric(max_n=4, smoothing=True)
    scores2 = bleu2.compute(refs2, hyps2)
    print("\nScores:")
    for k, v in sorted(scores2.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    # Sample 3: Different sentence
    print("\n" + "-"*70)
    print("[Sample 3] Different Words:")
    refs3 = ["machine translation is amazing"]
    hyps3 = ["machine learning is great"]
    print(f"Reference:  {refs3[0]}")
    print(f"Hypothesis: {hyps3[0]}")
    
    bleu3 = BLEUMetric(max_n=4, smoothing=True)
    scores3 = bleu3.compute(refs3, hyps3)
    print("\nScores:")
    for k, v in sorted(scores3.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    # Batch test with all 3 samples
    print("\n" + "="*70)
    print("[Batch Test] All 3 Samples Together:")
    refs_batch = refs1 + refs2 + refs3
    hyps_batch = hyps1 + hyps2 + hyps3
    
    bleu_batch = BLEUMetric(max_n=4, smoothing=True)
    scores_batch = bleu_batch.compute(refs_batch, hyps_batch)
    print("\nCorpus-level Scores:")
    for k, v in sorted(scores_batch.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    print("\n" + "="*70)
    print("Test completed!")