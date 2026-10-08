"""Framework-wide TLD convention for models whose paper generates only the SLD.

Several adversarial DGAs (WGAN-GP DeepDGA, CDGA, GWDGA, WGAN-based DGA, PKDGA,
WordDGA, TLVDGA, SADGA) are defined on the second-level label alone. The
detectors are trained on full domains and the lexical metrics change with the
TLD, so every model must return a full domain for the comparison to be fair.
Models whose paper or seed data fixes the TLD keep their own rule; the others
draw one here, from the empirical public-suffix distribution of the D1 Tranco
slice, so that the TLD is never a confounder between generators.
"""

import random
from collections import Counter
from typing import List, Optional, Tuple

from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from core.sld import extract_suffix

TLD_DISTRIBUTION_TOP_K = 50

_distribution: Optional[Tuple[List[str], List[float]]] = None


def tld_distribution(top_k: int = TLD_DISTRIBUTION_TOP_K) -> Tuple[List[str], List[float]]:
    """Return (suffixes, weights): the top_k public suffixes of the D1 Tranco slice
    with their normalised frequencies. Computed once per process."""
    global _distribution
    if _distribution is None:
        counts = Counter(extract_suffix(d) for d in load_tranco(TRANCO_D1_BENIGN))
        counts.pop("", None)
        top = counts.most_common(top_k)
        total = sum(c for _, c in top)
        _distribution = ([s for s, _ in top], [c / total for _, c in top])
    return _distribution


def sample_tld(rng: Optional[random.Random] = None) -> str:
    """Draw one public suffix from the D1 distribution.

    Uses the global `random` module (seeded by core.seeding.seed_everything)
    unless a private RNG is given.
    """
    suffixes, weights = tld_distribution()
    r = rng if rng is not None else random
    return r.choices(suffixes, weights=weights, k=1)[0]


def with_tld(sld: str, rng: Optional[random.Random] = None) -> str:
    """Return sld + "." + a suffix drawn with sample_tld()."""
    return f"{sld}.{sample_tld(rng)}"
