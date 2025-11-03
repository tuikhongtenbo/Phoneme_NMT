"""
METEOR (Metric for Evaluation of Translation with Explicit ORdering) implementation.

Reference:
    Banerjee and Lavie (2005) "METEOR: An Automatic Metric for MT Evaluation 
    with Improved Correlation with Human Judgments"
    https://aclanthology.org/W05-0909.pdf
"""
from typing import List, Dict, Union, Set, Tuple
import numpy as np

from .base_metric import BaseMetric


class METEORMetric(BaseMetric):
    """
    METEOR metric for machine translation evaluation.
    
    METEOR considers:
    - Unigram matching (exact matching in this implementation)
    - Word order through chunk-based penalty
    - Harmonic mean of precision and recall
    """
    
    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):

        super().__init__(name="METEOR")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]], **kwargs) -> Dict[str, float]:
        """
        Calculate METEOR scores
        
        Args:
            references (Union[str, List[str]]): Single reference string or list of reference strings
            hypotheses (Union[str, List[str]]): Single hypothesis string or list of hypothesis strings
            **kwargs: Additional arguments
                - alpha: Alpha parameter for METEOR (default: 0.9)
                - beta: Beta parameter for METEOR (default: 3.0)
                - gamma: Gamma parameter for METEOR (default: 0.5)
        
        Returns:
            Dict[str, float]: Dictionary with METEOR scores
        """
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        gamma = kwargs.get('gamma', self.gamma)
        
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
            meteor, precision, recall, penalty = self._sentence_meteor(
                reference_tokens[0],
                hypothesis_tokens[0],
                alpha, beta, gamma
            )
        else:
            meteors = []
            precisions = []
            recalls = []
            penalties = []
            
            for ref_tokens, hyp_tokens in zip(reference_tokens, hypothesis_tokens):
                m, p, r, pen = self._sentence_meteor(
                    ref_tokens, hyp_tokens, alpha, beta, gamma
                )
                meteors.append(m)
                precisions.append(p)
                recalls.append(r)
                penalties.append(pen)
            
            meteor = np.mean(meteors)
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            penalty = np.mean(penalties)
        
        return {
            'meteor': meteor,
            'meteor_p': precision,
            'meteor_r': recall,
            'meteor_penalty': penalty
        }
    
    @staticmethod
    def _exact_match_alignment(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Create alignment based on exact string matching.
        Args:
            ref_tokens (List[str]): List of reference tokens
            hyp_tokens (List[str]): List of hypothesis tokens
        
        Returns:
            List[Tuple[int, int]]: List of (ref_idx, hyp_idx) alignment pairs
        """
        alignments = []
        ref_matched = set()
        hyp_matched = set()
        
        # Greedy alignment: for each hypothesis word, find first unmatched reference word
        for i, hyp_token in enumerate(hyp_tokens):
            for j, ref_token in enumerate(ref_tokens):
                if j not in ref_matched and i not in hyp_matched and hyp_token == ref_token:
                    alignments.append((j, i))
                    ref_matched.add(j)
                    hyp_matched.add(i)
                    break
        
        return alignments
    
    @staticmethod
    def _count_chunks(alignments: List[Tuple[int, int]]) -> int:
        """
        Count the number of chunks in the alignment.
        
        A chunk is a contiguous sequence of matched words. Words are in the same chunk
        if they are adjacent in both the reference and hypothesis.
        
        Args:
            alignments (List[Tuple[int, int]]): List of (ref_idx, hyp_idx) alignment pairs
        
        Returns:
            int: Number of chunks in the alignment
        """
        if len(alignments) == 0:
            return 0
        
        # Sort alignments by hypothesis index
        sorted_alignments = sorted(alignments, key=lambda x: x[1])
        
        chunks = 1
        for i in range(1, len(sorted_alignments)):
            prev_ref_idx, prev_hyp_idx = sorted_alignments[i-1]
            curr_ref_idx, curr_hyp_idx = sorted_alignments[i]
            
            # New chunk if not adjacent in both sequences
            if curr_ref_idx != prev_ref_idx + 1 or curr_hyp_idx != prev_hyp_idx + 1:
                chunks += 1
        
        return chunks
    
    def _sentence_meteor(
        self,
        ref_tokens: List[str],
        hyp_tokens: List[str],
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5
    ) -> Tuple[float, float, float, float]:
        """
        Calculate METEOR for a single sentence.
        Args:
            ref_tokens (List[str]): List of reference tokens
            hyp_tokens (List[str]): List of hypothesis tokens
            alpha (float): Alpha parameter for METEOR (default: 0.9)
            beta (float): Beta parameter for METEOR (default: 3.0)
            gamma (float): Gamma parameter for METEOR (default: 0.5)
        
        Returns:
            Tuple[float, float, float, float]: Tuple containing the METEOR score, precision, recall, and penalty
        """
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Get alignment pairs
        alignments = self._exact_match_alignment(ref_tokens, hyp_tokens)
        matches = len(alignments)
        
        precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
        recall = matches / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        if matches == 0:
            penalty = 0.0
            f_mean = 0.0
        else:
            # Count chunks using alignment pairs
            chunks = self._count_chunks(alignments)
            fragmentation = chunks / matches if matches > 0 else 1.0
            penalty = gamma * (fragmentation ** beta)
            
            if precision + recall > 0:
                f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
            else:
                f_mean = 0.0
        
        meteor = f_mean * (1 - penalty)
        
        return meteor, precision, recall, penalty


if __name__ == "__main__":
    print("="*70)
    print("METEOR Metric Test - 3 Sample Sentences")
    print("="*70)
    
    # Sample 1: Perfect match
    print("\n[Sample 1] Perfect Match:")
    refs1 = ["the cat is on the mat"]
    hyps1 = ["the cat is on the mat"]
    print(f"Reference:  {refs1[0]}")
    print(f"Hypothesis: {hyps1[0]}")
    
    meteor1 = METEORMetric(alpha=0.9, beta=3.0, gamma=0.5)
    scores1 = meteor1.compute(refs1, hyps1)
    print("\nScores:")
    for k, v in sorted(scores1.items()):
        print(f"  {k:16s}: {v:.4f}")
    
    # Sample 2: Word order matters
    print("\n" + "-"*70)
    print("[Sample 2] Different Word Order (Tests Chunking):")
    refs2 = ["the quick brown fox"]
    hyps2 = ["the fox brown quick"]
    print(f"Reference:  {refs2[0]}")
    print(f"Hypothesis: {hyps2[0]}")
    
    meteor2 = METEORMetric(alpha=0.9, beta=3.0, gamma=0.5)
    scores2 = meteor2.compute(refs2, hyps2)
    print("\nScores:")
    for k, v in sorted(scores2.items()):
        print(f"  {k:16s}: {v:.4f}")
    print("  Note: All words match but fragmented (4 chunks)")
    
    # Sample 3: Partial match
    print("\n" + "-"*70)
    print("[Sample 3] Partial Match:")
    refs3 = ["machine translation is amazing"]
    hyps3 = ["machine learning is great"]
    print(f"Reference:  {refs3[0]}")
    print(f"Hypothesis: {hyps3[0]}")
    
    meteor3 = METEORMetric(alpha=0.9, beta=3.0, gamma=0.5)
    scores3 = meteor3.compute(refs3, hyps3)
    print("\nScores:")
    for k, v in sorted(scores3.items()):
        print(f"  {k:16s}: {v:.4f}")
    print("  Note: 'machine' and 'is' match (2/4 words)")
    
    # Batch test with all 3 samples
    print("\n" + "="*70)
    print("[Batch Test] All 3 Samples Together:")
    refs_batch = refs1 + refs2 + refs3
    hyps_batch = hyps1 + hyps2 + hyps3
    
    meteor_batch = METEORMetric(alpha=0.9, beta=3.0, gamma=0.5)
    scores_batch = meteor_batch.compute(refs_batch, hyps_batch)
    print("\nCorpus-level Scores:")
    for k, v in sorted(scores_batch.items()):
        print(f"  {k:16s}: {v:.4f}")
    
    print("\n" + "="*70)
    print("Test completed!")