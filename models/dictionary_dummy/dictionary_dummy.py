import random
from typing import Any
from core.adversarial_model import AdversarialModel

class DictionaryDummyModel(AdversarialModel):
    """
    A dummy DGA model that generates domains by concatenating exactly 2 words 
    from a predefined dictionary of 30 words.
    """
    def __init__(self, name: str = "DictionaryDummy"):
        super().__init__(name=name)
        self.tld = ".net"
        self.words = [
            "apple", "banana", "cherry", "date", "elderberry", 
            "fig", "grape", "honeydew", "kiwi", "lemon", 
            "mango", "nectarine", "orange", "papaya", "quince", 
            "raspberry", "strawberry", "tangerine", "ugli", "vanilla", 
            "watermelon", "xigua", "yam", "zucchini", "apricot", 
            "blackberry", "coconut", "dragonfruit", "guava", "lychee"
        ]

    def fit(self) -> None:
        self.is_fitted = True

    def generate_domain(self) -> str:
        word1 = random.choice(self.words)
        word2 = random.choice(self.words)
        return f"{word1}{word2}{self.tld}"
