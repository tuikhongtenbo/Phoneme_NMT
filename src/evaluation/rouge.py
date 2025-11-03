"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric implementation

Reference:
    Lin (2004) "ROUGE: A Package for Automatic Evaluation of Summaries"
    https://aclanthology.org/W04-1013.pdf
"""
from typing import List, Dict, Union
import numpy as np

from .base_metric import BaseMetric


class ROUGEMetric(BaseMetric):
    """
    ROUGE-L metric for machine translation evaluation.
    
    ROUGE-L measures the longest common subsequence (LCS) 
    between reference and hypothesis translations.
    """
    
    def __init__(self, beta: float = 1.0):

        super().__init__(name="ROUGE")
        self.beta = beta
    
    def compute(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]], **kwargs) -> Dict[str, float]:
        """
        Calculate ROUGE-L scores.
        
        Args:
            references (Union[str, List[str]]): Single reference string or list of reference strings
            hypotheses (Union[str, List[str]]): Single hypothesis string or list of hypothesis strings
            **kwargs: Additional arguments
                - beta: Beta parameter for ROUGE-L (default: 1.0)
        
        Returns:
            Dict[str, float]: Dictionary with ROUGE-L scores
        """
        beta = kwargs.get('beta', self.beta)
        
        if isinstance(references, str):
            references = [references]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        
        if isinstance(references[0], str):
            reference_tokens = [ref.split() for ref in references]
            hypothesis_tokens = [hyp.split() for hyp in hypotheses]
        else:
            reference_tokens = references
            hypothesis_tokens = hypotheses
        
        if len(hypothesis_tokens) == 1:
            precision, recall, f_score = self._sentence_rouge_l(
                reference_tokens[0],
                hypothesis_tokens[0],
                beta
            )
        else:
            precisions = []
            recalls = []
            f_scores = []
            
            for ref_tokens, hyp_tokens in zip(reference_tokens, hypothesis_tokens):
                p, r, f = self._sentence_rouge_l(ref_tokens, hyp_tokens, beta)
                precisions.append(p)
                recalls.append(r)
                f_scores.append(f)
            
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f_score = np.mean(f_scores)
        
        return {
            'rouge_l_p': precision,
            'rouge_l_r': recall,
            'rouge_l_f': f_score
        }
    
    @staticmethod
    def _lcs_length(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
        """
        Calculate the length of Longest Common Subsequence (LCS).
        
        Args:
            ref_tokens (List[str]): List of reference tokens
            hyp_tokens (List[str]): List of hypothesis tokens
        
        Returns:
            int: Length of LCS
        """
        m = len(ref_tokens)
        n = len(hyp_tokens)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == hyp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _sentence_rouge_l(self, ref_tokens: List[str], hyp_tokens: List[str], beta: float = 1.0) -> tuple:
        """
        Calculate ROUGE-L for a single sentence.
        
        Args:
            ref_tokens (List[str]): List of reference tokens
            hyp_tokens (List[str]): List of hypothesis tokens
            beta (float): Beta parameter for ROUGE-L (default: 1.0)
        
        Returns:
            Tuple[float, float, float]: Tuple containing the precision, recall, and F-score
        """
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0, 0.0, 0.0
        
        lcs_len = self._lcs_length(ref_tokens, hyp_tokens)
        
        precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if precision + recall > 0:
            f_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
        else:
            f_score = 0.0
        
        return precision, recall, f_score


if __name__ == "__main__":
    print("="*70)
    print("ROUGE-L Metric Test - 3 Sample Sentences")
    print("="*70)
    
    # Sample 1: Perfect match
    print("\n[Sample 1] Perfect Match:")
    refs1 = ["the cat is on the mat"]
    hyps1 = ["the cat is on the mat"]
    print(f"Reference:  {refs1[0]}")
    print(f"Hypothesis: {hyps1[0]}")
    
    rouge1 = ROUGEMetric(beta=1.0)
    scores1 = rouge1.compute(refs1, hyps1)
    print("\nScores:")
    for k, v in sorted(scores1.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    # Sample 2: Partial match (LCS test)
    print("\n" + "-"*70)
    print("[Sample 2] Partial Match:")
    refs2 = ["the cat is on the mat"]
    hyps2 = ["the cat on mat"]
    print(f"Reference:  {refs2[0]}")
    print(f"Hypothesis: {hyps2[0]}")
    
    rouge2 = ROUGEMetric(beta=1.0)
    scores2 = rouge2.compute(refs2, hyps2)
    print("\nScores:")
    for k, v in sorted(scores2.items()):
        print(f"  {k:12s}: {v:.4f}")
    print("  Note: LCS = 'the cat on mat' (4 words)")
    
    # Sample 3: Different word order
    print("\n" + "-"*70)
    print("[Sample 3] Different Word Order:")
    refs3 = ["the quick brown fox"]
    hyps3 = ["the brown quick fox"]
    print(f"Reference:  {refs3[0]}")
    print(f"Hypothesis: {hyps3[0]}")
    
    rouge3 = ROUGEMetric(beta=1.0)
    scores3 = rouge3.compute(refs3, hyps3)
    print("\nScores:")
    for k, v in sorted(scores3.items()):
        print(f"  {k:12s}: {v:.4f}")
    print("  Note: LCS = 'the brown fox' or 'the quick fox' (3 words)")
    
    # Batch test with all 3 samples
    print("\n" + "="*70)
    print("[Batch Test] All 3 Samples Together:")
    refs_batch = refs1 + refs2 + refs3
    hyps_batch = hyps1 + hyps2 + hyps3
    
    rouge_batch = ROUGEMetric(beta=1.0)
    scores_batch = rouge_batch.compute(refs_batch, hyps_batch)
    print("\nCorpus-level Scores:")
    for k, v in sorted(scores_batch.items()):
        print(f"  {k:12s}: {v:.4f}")
    
    print("\n" + "="*70)
    print("Test completed!")