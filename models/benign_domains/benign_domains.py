import logging
import random
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D3_BENIGN, load_tranco

logger = logging.getLogger(__name__)


class BenignDomainsModel(AdversarialModel):
    """Benign control model loading from the D3 Tranco slice."""

    def __init__(self, name: str = "BenignDomains"):
        super().__init__(name=name)
        self.is_malicious = False
        self.domains = []

    def fit(self) -> None:
        self.domains = load_tranco(TRANCO_D3_BENIGN)
        if not self.domains:
            raise ValueError("No domains available in the D3 benign control slice.")
        self.is_fitted = True

    def generate_domain(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
        return random.choice(self.domains)

    def generate_domains(self, n: int) -> List[str]:
        """Draw n distinct domains from the D3 benign slice, like MaliciousDGAModel does."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
        if n > len(self.domains):
            logger.warning(
                f"[!] D3 benign slice has only {len(self.domains)} domains, {n} requested. "
                "Returning the whole slice shuffled."
            )
            return random.sample(self.domains, len(self.domains))
        return random.sample(self.domains, n)
