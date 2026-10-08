import logging
import random
import numpy as np
import os
import h5py
import tensorflow as tf
from typing import List
from core.adversarial_model import AdversarialModel
from core.data_splits import load_malicious_seeds
from detectors.cnn.cnn import CNNDetector

logger = logging.getLogger(__name__)

class CLETerModel(AdversarialModel):
    """
    CLETer: A Character-level Evasion Technique Against Deep Learning DGA Classifiers.
    Implements black-box character substitution based on influence scores.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Target Models: Default CNN architecture: Embedding(128) → Conv1D(128, k=5) → 
    GlobalMaxPooling → Dense(64) → Dropout(0.5) → Dense(1, sigmoid). Optimizer: Adam (LR=0.001), 
    loss: BCE, batch=64.
    2. CIS Score Lambda: Strictly set to 1.0.
    3. Tie-breaking: Uses original character positional order (Left to Right) if influence 
    scores are identical.
    4. Substitution Search: Iterates over all 37 valid characters per position evaluated to 
    find the lowest probability.
    5. TLD Handling: The original TLD/SLD is concatenated back to the modified domain.
    6. Max Substitutions: Bounded to 5 character modifications per domain.
    7. Seeds: the whole D1 malicious split (SLDs longer than 3 characters), one seed per
    generated domain, because the substitution search is deterministic.
    """
    def __init__(self, name: str = "CLETer", target_detector=None, scoring_method: str = "CIS"):
        super().__init__(name=name)
        # The targeted model to evade (must be already fitted)
        self.target_detector = target_detector 
        self.max_substitutions = 5
        self.scoring_method = scoring_method
        
        # Valid character set C (37 tokens: 26 letters, 10 digits, 1 hyphen)
        self.valid_chars = list("abcdefghijklmnopqrstuvwxyz0123456789-")
        self.base_domains = []
        self.domain_idx = 0

        # File management paths for caching weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_path = os.path.join(self.weights_dir, "cleter_cache.h5")

    def _load_agd_domains(self) -> List[tuple]:
        """
        Auxiliary function to obtain (sld, tld) pairs of real AGDs through core.data_splits.load_malicious_seeds.
        The whole D1 split is used: the search is deterministic, so every generated domain needs its own seed.
        """
        agd_domains = []
        try:
            for domain_str in load_malicious_seeds():
                parts = domain_str.split('.', 1)
                domain_only = parts[0] # TLD isolation
                tld = parts[1] if len(parts) > 1 else "com"
                if domain_only and len(domain_only) > 3:
                    agd_domains.append((domain_only, tld))
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs for CLETer: {e}")

        if not agd_domains:
            logger.warning(f"[{self.name}] No AGD seeds loaded. Using random base SLDs.")  # SAFETY GUARD
            agd_domains = [
                ("".join(random.choices(self.valid_chars[:26], k=random.randint(8, 15))), "com")  # SAFETY GUARD: random SLD fallback
                for _ in range(100000)
            ]

        return agd_domains

    def fit(self) -> None:
        """
        Loads base DGA domains that CLETer will attempt to modify.
        Caches the loaded domains in a .h5 file to speed up future executions.
        """

        # Try to load from cache first
        if os.path.exists(self.weights_path):
            logger.info(f"[*] CLETer found cached weights. Skipping training.")
            try:
                with h5py.File(self.weights_path, 'r') as f:
                    slds = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['slds'][()]]
                    tlds = [t.decode('utf-8') if isinstance(t, bytes) else t for t in f['tlds'][()]]
                    self.base_domains = list(zip(slds, tlds))
                
                if not self.target_detector:
                    logger.info("[*] No target detector provided. Instantiating default CNN detector...")
                    self.target_detector = CNNDetector()
                    self.target_detector.fit()

                self.is_fitted = True
                return
            except Exception as e:
                logger.warning(f"[*] Could not load cache: {e}. Fitting normally.")

        logger.info(f"[*] No cached weights found for CLETer. Training...")

        if not self.target_detector:
            logger.info("[*] No target detector provided. Instantiating and fitting default CNN detector...")
            self.target_detector = CNNDetector()
            self.target_detector.fit()
            
        # Load a sample of malicious domains to act as our starting point
        self.base_domains = self._load_agd_domains()
        random.shuffle(self.base_domains)
        self.is_fitted = True

        # Cache artifacts for future executions
        try:
            os.makedirs(self.weights_dir, exist_ok=True)
            with h5py.File(self.weights_path, 'w') as f:
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('slds', data=np.array([d[0] for d in self.base_domains], dtype=object).astype(dt))
                f.create_dataset('tlds', data=np.array([d[1] for d in self.base_domains], dtype=object).astype(dt))
        except Exception as e:
            logger.warning(f"[*] Could not save cache: {e}")

    def _oracle_fn(self):
        """
        Returns a compiled forward pass of the target CNN. The oracle is queried
        in batches: one call per scoring pass and one per substitution round,
        instead of one Keras predict() per string, whose fixed overhead would
        dominate the cost. The algorithm and the evaluation order are those of
        the paper; only the grouping of independent queries differs.
        """
        if getattr(self, "_oracle", None) is None:
            model = self.target_detector.model
            self._oracle = tf.function(
                lambda x: model(x, training=False),
                input_signature=[tf.TensorSpec([None, None], tf.int32)],
            )
        return self._oracle

    def _get_malicious_probs(self, domain_strs: List[str]) -> List[float]:
        """
        Queries the targeted deep learning model for a batch of strings and
        returns their malicious probabilities, in order.
        In our framework, 1 = Benign, 0 = Malicious. Thus, P(Malicious) = 1.0 - P(Benign).
        Empty strings are not sent to the oracle and score 0.0.
        """
        probs = [0.0] * len(domain_strs)
        if not self.target_detector or not self.target_detector.model:
            return probs
        idx = [i for i, s in enumerate(domain_strs) if s]
        if not idx:
            return probs
        encoded = self.target_detector._encode_domains([domain_strs[i] for i in idx])
        prob_benign = self._oracle_fn()(tf.constant(encoded, dtype=tf.int32)).numpy()[:, 0]
        for k, i in enumerate(idx):
            probs[i] = 1.0 - float(prob_benign[k])
        return probs

    def _get_malicious_prob(self, domain_str: str) -> float:
        """Single-string black-box query. See _get_malicious_probs."""
        return self._get_malicious_probs([domain_str])[0]

    def _compute_scores(self, domain_str: str) -> List[float]:
        """
        Calculates the influence score for every character in the domain.
        All the prefixes, suffixes and deletions the scores depend on are
        independent of each other, so they go to the oracle in a single batch.
        """
        n = len(domain_str)
        if self.scoring_method == "CIS":
            # HIS(i) = F(x[:i+1]) - F(x[:i]); TIS(i) = F(x[i:]) - F(x[i+1:]).
            # Consecutive positions share their prefix and suffix, so the batch
            # holds the n+1 prefixes and the n+1 suffixes once each.
            prefixes = [domain_str[:i] for i in range(n + 1)]
            suffixes = [domain_str[i:] for i in range(n + 1)]
            probs = self._get_malicious_probs(prefixes + suffixes)
            p_pre, p_suf = probs[:n + 1], probs[n + 1:]
            # Combined Influence Score (CIS) with lambda = 1
            return [(p_pre[i + 1] - p_pre[i]) + 1.0 * (p_suf[i] - p_suf[i + 1]) for i in range(n)]

        if self.scoring_method == "OIS":
            # Overall Influence Score (OIS): F(x) - F(x without character i)
            probs = self._get_malicious_probs([domain_str] + [domain_str[:i] + domain_str[i+1:] for i in range(n)])
            return [probs[0] - probs[i + 1] for i in range(n)]

        return [0.0] * n

    def generate_domain(self) -> str:
        """
        Generates a single adversarial domain by modifying an existing DGA domain.
        """
        if not self.is_fitted:
            self.fit()
            
        # 1. Select an existing DGA domain
        original_domain, tld = self.base_domains[self.domain_idx % len(self.base_domains)]
        self.domain_idx += 1
        x_adv = original_domain

        # 2. Evaluate every character's influence
        scores = self._compute_scores(x_adv)

        # 3. Sort by descending score. Tie-breaker: original positional order (Left to Right)
        # Using -score for descending, and index (i) for ascending tie-breaking
        ranked_indices = [i for i, s in sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))]

        # 4. Perform character-level transformations
        current_prob = self._get_malicious_prob(x_adv)
        for j in range(min(self.max_substitutions, len(ranked_indices))):
            # If successfully classified as benign (prob < 0.5), evasion is complete
            if current_prob < 0.5:
                break

            target_idx = ranked_indices[j]

            # Candidate substitutions over the 37 valid characters, in the
            # character set order. Hyphens cannot be at the beginning or the end of a domain.
            candidates = [
                c for c in self.valid_chars
                if c != x_adv[target_idx]
                and not (c == '-' and (target_idx == 0 or target_idx == len(x_adv) - 1))
            ]
            x_news = [x_adv[:target_idx] + c + x_adv[target_idx+1:] for c in candidates]

            # One oracle call for the whole round. Keeping the first strict
            # minimum reproduces the sequential search and its tie-breaking.
            best_char = x_adv[target_idx]
            min_prob = current_prob
            for c, new_prob in zip(candidates, self._get_malicious_probs(x_news)):
                if new_prob < min_prob:
                    min_prob = new_prob
                    best_char = c

            # Apply the best substitution for this iteration. Its probability is
            # already known, so the next round does not query it again.
            x_adv = x_adv[:target_idx] + best_char + x_adv[target_idx+1:]
            current_prob = min_prob

        return x_adv + "." + tld
