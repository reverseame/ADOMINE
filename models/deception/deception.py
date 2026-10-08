import logging
import random

from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco

logger = logging.getLogger(__name__)


class DeceptionModel(AdversarialModel):
    """
    Deception DGA model adapted from deception2.py (adversarial DGAs).
    Models the trigram probabilities from legitimate Alexa domains distribution,
    regardless of their position in the domain name.
    """
    def __init__(self, name: str = "Deception"):
        super().__init__(name=name)
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]
        self.min_domain_length = 6

        self.d_len = {}
        self.d2 = {}
        self.d3 = {}
        self.d3_subsets = {}

    def fit(self) -> None:
        domains = []
        for full_domain in load_tranco(TRANCO_D1_BENIGN):
            sld = full_domain.split('.')[0]
            # Discard IDN domains or abnormal chars
            if all(ord(c) < 128 for c in sld):
                domains.append(sld)
        if not domains:
            raise ValueError("No domains available in the D1 benign slice after filtering.")

        # Calculate distributions
        self.d_len = {}
        self.d2 = {}
        self.d3 = {}

        for d in domains:
            # Length distribution
            length = len(d)
            self.d_len[length] = self.d_len.get(length, 0) + 1

            # First 2 chars distribution
            if len(d) > 1:
                first2 = d[:2]
                self.d2[first2] = self.d2.get(first2, 0) + 1

            # 3-grams distribution
            if len(d) > 2:
                for i in range(len(d) - 2):
                    trigram = d[i:i+3]
                    self.d3[trigram] = self.d3.get(trigram, 0) + 1

        # Precompute subsets for faster generation
        self.d3_subsets = {}
        for tri, count in self.d3.items():
            prefix = tri[:2]
            if prefix not in self.d3_subsets:
                self.d3_subsets[prefix] = {}
            self.d3_subsets[prefix][tri] = count
            
        self.is_fitted = True

    def generate_domain(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
            
        length = 0
        while length < self.min_domain_length:
            length = random.choices(list(self.d_len.keys()), weights=list(self.d_len.values()))[0]
            
        if not self.d2:
            logger.warning(f"[{self.name}] No trigram statistics available. Returning a random fallback.")  # SAFETY GUARD
            return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + "." + random.choice(self.tlds)  # SAFETY GUARD: random SLD fallback

        # Generate the first two characters
        domain = random.choices(list(self.d2.keys()), weights=list(self.d2.values()))[0]

        # Iteratively append chars using 3-grams (Markov Chain representation)
        while len(domain) < length:
            prefix = domain[-2:]
            subdict = self.d3_subsets.get(prefix, None)
            
            if subdict:
                chosen = random.choices(list(subdict.keys()), weights=list(subdict.values()))[0][-1]
                domain += chosen
            else:
                # Back-off mechanism similar to what my_dga might do if path is cold
                domain += random.choice('abcdefghijklmnopqrstuvwxyz')

        tld_value = random.choice(self.tlds)
        return domain + '.' + tld_value
