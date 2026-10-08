from abc import ABC, abstractmethod
from typing import List, Tuple

class Detector(ABC):
    """
    Abstract Base Class for all Evasion Detectors.
    Each detector must be entirely self-contained (managing its own
    paths, models, and datasets natively).
    """
    
    def __init__(self, name: str):
        self.name = name
        self.training_time = None
        self.trained_from_scratch = None
        self.inference_time = None
        self.inference_domains = 0

    @abstractmethod
    def fit(self) -> None:
        """
        Train or initialize the detector.
        Must take no arguments. The detector class natively handles 
        all configuration, data retrieval, and modeling details internally.
        """
        pass

    @abstractmethod
    def detect(self, domains: List[str]) -> List[Tuple[str, bool]]:
        """
        Takes a list of domains and classifies them.
        Returns a list of tuples containing the domain and a boolean value.
        True denotes benign, False denotes malicious.
        """
        pass
