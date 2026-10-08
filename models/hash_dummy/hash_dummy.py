import random
import hashlib
from typing import Any
from core.adversarial_model import AdversarialModel

class HashDummyModel(AdversarialModel):
    """
    A dummy DGA model that generates domains using a pseudo-hash approach.
    Length of the resulting domain part is between 4 and 10 characters.
    """
    def __init__(self, name: str = "HashDummy"):
        super().__init__(name=name)
        self.counter = 0
        self.tld = ".org"

    def fit(self) -> None:
        self.is_fitted = True

    def generate_domain(self) -> str:
        # We use a varying string to hash
        seed_string = f"seed_value_{self.counter}".encode('utf-8')
        hash_digest = hashlib.md5(seed_string).hexdigest()
        self.counter += 1
        
        # Take a random slice of the hash of length 4 to 10
        length = random.randint(4, 10)
        start_idx = random.randint(0, len(hash_digest) - length)
        domain = hash_digest[start_idx:start_idx+length]
        
        return domain + self.tld
