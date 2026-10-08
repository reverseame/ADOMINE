import random

import numpy as np

_CURRENT_SEED = 42  # last value passed to seed_everything(); read with get_seed()


def get_seed() -> int:
    """Return the seed given to the last seed_everything() call (42 if it was never called)."""
    return _CURRENT_SEED


def seed_everything(seed: int) -> None:
    """
    Seed the Python, NumPy, TensorFlow, PyTorch, and SentencePiece global RNGs for a
    reproducible run, and record the seed for libraries that have no global RNG
    (gensim: Word2Vec takes its own `seed`, read back through get_seed()).
    """
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow is optional in this repo, seed it only when importable.
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch is optional in this repo, seed it only when importable.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Gensim has no global RNG (gensim.utils.RANDOM_SEED is not read by Word2Vec).
    # Models pass get_seed() to Word2Vec(seed=..., workers=1) instead.
    global _CURRENT_SEED
    _CURRENT_SEED = seed

    # SentencePiece is optional in this repo, seed it only when importable.
    try:
        import sentencepiece as spm
        spm.set_random_generator_seed(seed)
    except ImportError:
        pass
