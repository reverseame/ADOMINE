import os
import json
import random
import h5py
import numpy as np
from collections import defaultdict
from typing import List, Any

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN


class HMMBasedDGAModel(AdversarialModel):
    """
    HMM-based DGA model proposed by Fu et al. (2017) in "Stealthy Domain Generation Algorithms".
    Generates stealthy domains character-by-character utilizing learned transition probabilities
    from a benign training dataset (Tranco D1).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Architecture Simplification: The paper discusses a "zero-knowledge HMM inference" 
       but lacks mathematical specifications. We implement a visible Markov Model (N-grams) 
       conditioned on history length L, which behaves equivalently.
    2. Context Initialization: A padding token '^' is assumed to bootstrap the initial context.
    3. Zero-Probability Handling: Laplace smoothing (+1) is applied to prevent dead-ends 
       and execution blocks when encountering unseen states.
    4. Dynamic Lengths: If bounds are omitted, 5th and 95th empirical percentiles are 
       assumed to mimic benign length distribution and avoid detection shortcuts.
    5. TLD Extension: A static list of 22 common TLDs is assumed to output resolvable domains.
    """

    def __init__(self, name: str = "HMMBasedDGA", L: int = 4, min_len: int = 3, max_len: int = 10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.L = L
        self.min_len = min_len
        self.max_len = max_len
        
        # Pool of TLDs
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]
        
        # N-gram counts
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
        # Unique characters observed
        self.vocab = set()
        self.vocab_size = 0
        
        # Cache directory and path for weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'hmm_weights.h5')
        
        self.trained_from_scratch = False

    def fit(self) -> None:
        """
        Train the model using Tranco D1 Benign domains.
        If cached weights exist (as .h5), load them; otherwise train and save.
        """
        if os.path.exists(self.weights_path):
            self._load_weights()
            self.trained_from_scratch = False
            self.is_fitted = True
            return

        try:
            domains = load_tranco(TRANCO_D1_BENIGN)
        except Exception as e:
            raise ValueError("No domains available in the D1 benign slice.")

        empirical_lengths = []
        
        for domain in domains:
            name = domain.split('.')[0].lower()
            empirical_lengths.append(len(name))
            
            # Initialize the beginning of words using an artificial start token '^' repeated L times.
            # This allows the model to learn which characters are statistically favored to start a domain name.
            padded_name = ("^" * self.L) + name
            
            for i in range(len(name)):
                context = padded_name[i : i + self.L]
                next_char = padded_name[i + self.L]
                
                self.ngram_counts[context][next_char] += 1
                self.context_counts[context] += 1
                self.vocab.add(next_char)

        # If lengths are None, use empirical percentiles (5th and 95th) instead of a simple mean.
        if self.min_len is None or self.max_len is None:
            self.min_len = max(3, int(np.percentile(empirical_lengths, 5)))
            self.max_len = min(20, int(np.percentile(empirical_lengths, 95)))

        # Ensure vocab is a sorted list for consistent random choices
        self.vocab = sorted(list(self.vocab))
        self.vocab_size = len(self.vocab)
        
        # Save weights to HDF5
        self._save_weights()
        
        self.is_fitted = True
        self.trained_from_scratch = True

    def _save_weights(self) -> None:
        """
        Serializes internal state and saves it to an HDF5 file.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Convert defaultdicts to standard dicts for JSON serialization
        ngram_dict = {k: dict(v) for k, v in self.ngram_counts.items()}
        context_dict = dict(self.context_counts)
        
        dt = h5py.string_dtype(encoding='utf-8')
        
        with h5py.File(self.weights_path, 'w') as f:
            f.create_dataset('vocab', data=self.vocab, dtype=dt)
            # Store JSON serialized dictionaries as scalar strings
            f.create_dataset('ngram_counts', data=json.dumps(ngram_dict), dtype=dt)
            f.create_dataset('context_counts', data=json.dumps(context_dict), dtype=dt)
            
            # Save hyperparameters
            f.create_dataset('min_len', data=self.min_len)
            f.create_dataset('max_len', data=self.max_len)
            f.create_dataset('L', data=self.L)

    def _load_weights(self) -> None:
        """
        Loads weights and hyperparameters from the HDF5 cache.
        """
        if not os.path.exists(self.weights_path):
            raise RuntimeError(f"Weights file not found at {self.weights_path}.")
        
        with h5py.File(self.weights_path, 'r') as f:
            self.vocab = [x.decode('utf-8') if isinstance(x, bytes) else x for x in f['vocab'][:]]
            self.vocab_size = len(self.vocab)
            
            self.min_len = int(f['min_len'][()])
            self.max_len = int(f['max_len'][()])
            self.L = int(f['L'][()])
            
            ngram_dict = json.loads(f['ngram_counts'][()].decode('utf-8') if isinstance(f['ngram_counts'][()], bytes) else f['ngram_counts'][()])
            context_dict = json.loads(f['context_counts'][()].decode('utf-8') if isinstance(f['context_counts'][()], bytes) else f['context_counts'][()])
            
        # Reconstruct defaultdicts
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in ngram_dict.items():
            self.ngram_counts[k].update(v)
            
        self.context_counts = defaultdict(int, context_dict)

    def generate_domain(self) -> str:
        """
        Generates a single domain using the learned Markov probabilities.
        """
        if not self.is_fitted:
            self._load_weights()
            self.is_fitted = True

        # Determine length
        target_len = random.randint(self.min_len, self.max_len)
        
        current_context = "^" * self.L
        generated_name = ""

        for _ in range(target_len):
            # Apply Laplace smoothing (+1) over the entire vocabulary.
            probs = []
            context_total = self.context_counts[current_context] + self.vocab_size
            
            for char in self.vocab:
                # Count with smoothing
                count = self.ngram_counts[current_context].get(char, 0) + 1
                probs.append(count / context_total)
            
            # Choose next character based on probabilities
            next_char = random.choices(self.vocab, weights=probs, k=1)[0]
            generated_name += next_char
            
            # Update context
            if self.L > 1:
                current_context = current_context[1:] + next_char
            else:
                current_context = next_char

        # Append random TLD.
        tld_value = random.choice(self.tlds)
        return generated_name + '.' + tld_value