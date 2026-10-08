import logging
import random
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import DGARCHIVE_D3_MALICIOUS, load_dgarchive

logger = logging.getLogger(__name__)


class MaliciousDGAModel(AdversarialModel):
    """Malicious control model loading per-family AGDs from the D3 DGArchive slice."""

    def __init__(self, name: str = "MaliciousDGA", family: str = "emotet_dga.csv"):
        super().__init__(name=name)
        self.family = family
        self.domains = []

    def fit(self) -> None:
        self.domains = load_dgarchive(self.family, DGARCHIVE_D3_MALICIOUS)
        if not self.domains:
            raise ValueError(
                f"No domains available in the D3 malicious slice for family '{self.family}'."
            )
        self.is_fitted = True

    def generate_domain(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
        return random.choice(self.domains)

    def generate_domains(self, n: int) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
        if n > len(self.domains):
            logger.warning(
                f"[!] Family '{self.family}' D3 slice has only {len(self.domains)} "
                f"domains, {n} requested. Returning the whole slice shuffled."
            )
            return random.sample(self.domains, len(self.domains))
        return random.sample(self.domains, n)
