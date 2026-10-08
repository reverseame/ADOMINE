import os
import re
import random
import h5py
import numpy as np
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN


class PCFGBasedDGAModel(AdversarialModel):
    """
    PCFG-based DGA model proposed by Fu et al. (2017) in "Stealthy Domain Generation Algorithms".
    Generates stealthy domains by modelling the syllable structure of benign domains
    using a Probabilistic Context-Free Grammar (PCFG).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Grammar: Implements S -> B; B -> A B C (p=0.25) | A C (p=0.75) to avoid empty strings.
    2. Length: Constrained to 2-4 syllables and 3-10 characters. Enforced via rejection sampling.
    3. Syllabification: Implements a native regex-based heuristic to split benign domains (Tranco D1) 
       into syllables, avoiding external dependencies. Split 50/50 into disjoint sets A and C.
    4. Numeric Injection: Numbers 0-2000 are injected into terminal C, each repeated 20 times.
    5. TLD Extension: A static pool of 22 common TLDs is assumed to construct the domains.
    """

    def __init__(self, name: str = "PCFGBasedDGA", min_chars: int = 3, max_chars: int = 10,
                 min_syllables: int = 2, max_syllables: int = 4, **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_syllables = min_syllables
        self.max_syllables = max_syllables
        
        # Pool of TLDs
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]
        
        # Terminal lists (Disjoint sets A and C) - These are the "weights" of the model
        self.list_A = []
        self.list_C = []
        
        # Cache directory and path for weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'pcfg_weights.h5')
        
        # Flag to indicate if the model was trained from scratch or loaded from cache
        self.trained_from_scratch = False

    def _extract_syllables(self, word: str) -> List[str]:
        """
        Native heuristic to split a word into syllables (consonant(s) + vowel(s) chunks).
        Replaces the need for external dependencies like pyphen.
        """
        # Find chunks of optional consonants followed by vowels (e.g., 'fa', 'ce', 'boo')
        chunks = re.findall(r'[^aeiouy]*[aeiouy]+', word)
        
        if not chunks:
            return [word] if word.isalpha() else []
            
        # Append any trailing consonants to the last chunk (e.g., 'k' in 'facebook')
        reconstructed = "".join(chunks)
        tail = word[len(reconstructed):]
        if tail:
            chunks[-1] += tail
            
        return chunks

    def fit(self) -> None:
        """
        Train the model using Tranco D1 Benign domains.
        If cached weights exist (as .h5), load them; otherwise, extract syllables,
        build lists A and C, and save the weights to the cache directory.
        """
        # Check if cached weights already exist
        if os.path.exists(self.weights_path):
            self._load_weights()
            self.trained_from_scratch = False
            self.is_fitted = True
            return

        # Otherwise, train from scratch
        domains = load_tranco(TRANCO_D1_BENIGN)

        all_syllables = set()
        
        for domain in domains:
            # Strip TLD and dots
            name = domain.split('.')[0].lower()
            
            # Extract syllables natively
            syllables = self._extract_syllables(name)
            
            for syl in syllables:
                if syl.isalpha():
                    all_syllables.add(syl)

        all_syllables_list = list(all_syllables)
        random.shuffle(all_syllables_list)

        # Split 50/50 into disjoint sets A and C
        mid = len(all_syllables_list) // 2
        self.list_A = all_syllables_list[:mid]
        self.list_C = all_syllables_list[mid:]

        # Inject numeric dictionary into set C (0-2000, each repeated 20 times)
        numerics = [str(i) for i in range(2001)] * 20
        self.list_C.extend(numerics)

        # Ensure the weights directory exists
        os.makedirs(self.weights_dir, exist_ok=True)

        dt = h5py.string_dtype(encoding='utf-8')

        # Save weights to the cache directory
        with h5py.File(self.weights_path, 'w') as f:
            f.create_dataset('list_A', data=self.list_A, dtype=dt)
            f.create_dataset('list_C', data=self.list_C, dtype=dt)

        self.is_fitted = True
        self.trained_from_scratch = True

    def _load_weights(self) -> None:
        """
        Helper to load weights from the HDF5 cache file.
        """
        if not os.path.exists(self.weights_path):
            raise RuntimeError(f"Weights file not found at {self.weights_path}. Must call fit() first.")
        
        with h5py.File(self.weights_path, 'r') as f:
            self.list_A = [x.decode('utf-8') if isinstance(x, bytes) else x for x in f['list_A'][:]]
            self.list_C = [x.decode('utf-8') if isinstance(x, bytes) else x for x in f['list_C'][:]]
            
        if not self.list_A or not self.list_C:
            raise ValueError("Loaded weights are empty. Refit the model.")
            
        self.is_fitted = True

    def _expand_B(self) -> List[str]:
        """
        Recursive function simulating the grammar rules:
        B -> A B C (p = 0.25)
        B -> A C   (p = 0.75)
        """
        if random.random() > 0.25:
            A = random.choice(self.list_A)
            C = random.choice(self.list_C)
            return [A, C]
        else:
            A = random.choice(self.list_A)
            C = random.choice(self.list_C)
            return [A] + self._expand_B() + [C]

    def generate_domain(self) -> str:
        """
        Generates a single domain adhering to the PCFG grammar and length constraints.
        """
        if not self.is_fitted:
            self._load_weights()

        while True:
            generated_terminals = self._expand_B()
            
            # Syllable count
            num_syllables = len(generated_terminals)
            if not (self.min_syllables <= num_syllables <= self.max_syllables):
                continue
            
            # Combine to form SLD
            sld = "".join(generated_terminals)
            
            # Character length
            if not (self.min_chars <= len(sld) <= self.max_chars):
                continue
            
            # Append TLD
            tld = random.choice(self.tlds)
            return f"{sld}.{tld}"