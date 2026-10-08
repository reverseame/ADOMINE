import random
import string
from typing import Any
from core.adversarial_model import AdversarialModel

class RandomDummyModel(AdversarialModel):
    """
    A dummy DGA model that generates completely random alphanumeric domains.
    Length is randomly chosen between 6 and 20 characters.
    """
    def __init__(self, name: str = "RandomDummy"):
        super().__init__(name=name)
        self.tld = ".com"

    def fit(self) -> None:
        self.is_fitted = True

    def generate_domain(self) -> str:
        length = random.randint(6, 20)
        chars = string.ascii_lowercase + string.digits
        domain = ''.join(random.choice(chars) for _ in range(length))
        return domain + self.tld
