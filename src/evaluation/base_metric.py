"""
Base class for evaluation metrics
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union


class BaseMetric(ABC):
    def __init__(self, name: str):
        """
        Initialize the base metric.

        Args:
            name (str)L Name of the metric
        """
        self.name = name

    @abstractmethod
    def compute(
        self,
        references: Union[str, List[str]],
        hypotheses: Union[str, List[str]],
         **kwargs
    ) -> Dict[str, float]:
        """
        Compute metric score.

        Args:
            references (Union[str, List[str]]): The reference translations.
            hypotheses (Union[str, List[str]]): The hypothesis translations.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        pass

    def __call__(
        self,
        references: Union[str, List[str]],
        hypotheses: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Callable interface for computing metric.
        
        Args:
            references: Reference translations
            hypotheses: Hypothesis translations
            **kwargs: Additional arguments
        
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        return self.compute(references, hypotheses, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"