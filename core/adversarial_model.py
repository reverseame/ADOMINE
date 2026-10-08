from abc import ABC, abstractmethod
from typing import List, Any

class AdversarialModel(ABC):
    """
    Abstract base class for all adversarial Domain Generation Algorithms (DGAs).
    
    This class enforces the implementation of essential methods required by 
    the framework to orchestrate and evaluate different adversarial models.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the adversarial model.
        
        Args:
            name (str): The identifier for this specific model instance.
            **kwargs: Additional model-specific configuration parameters.
        """
        self.name = name
        self.config = kwargs
        self.is_fitted = False
        self.is_malicious = True
        self.training_time = None
        self.generation_time = None
        self.trained_from_scratch = None

    @abstractmethod
    def fit(self) -> None:
        """
        Train or configure the adversarial model based on its internal state or predefined data.
        
        This method should be self-contained; the model should autonomously read its
        own dataset or load its pre-trained weights without relying on the orchestrator
        to feed it training data.
        """
        pass

    @abstractmethod
    def generate_domain(self) -> str:
        """
        Generate a single adversarial domain.
        
        Returns:
            str: A generated adversarial domain string.
        """
        pass

    def generate_domains(self, n: int) -> List[str]:
        """
        Generate a specific number of adversarial domains.
        
        By default, this method repeatedly calls `generate_domain`. 
        Subclasses can override this method if batch generation is more efficient.
        
        Args:
            n (int): Number of domains to generate.
            
        Returns:
            List[str]: A list of generated adversarial domains.
        """
        return [self.generate_domain() for _ in range(n)]

