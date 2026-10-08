import random

from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco


class CharBotModel(AdversarialModel):
    """
    CharBot DGA model adapted from CharBot.ipynb.
    Randomly alters 2 characters of a legitimate Alexa domain and appends a random TLD.
    """
    def __init__(self, name: str = "CharBot"):
        super().__init__(name=name)
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]
        self.url_chars = 'abcdefghijklmnopqrstuvwxyz0123456789-'.encode('utf-8')
        self.min_domain_length = 6
        self.alexa_domains = []

    def fit(self) -> None:
        slds = []
        for full_domain in load_tranco(TRANCO_D1_BENIGN):
            sld = full_domain.split('.')[0]
            if len(sld) >= self.min_domain_length:
                slds.append(sld)
        self.alexa_domains = slds
        if not self.alexa_domains:
            raise ValueError("No domains available in the D1 benign slice after filtering.")
        self.is_fitted = True

    def generate_domain(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating domains")
            
        domain_index = random.randint(0, len(self.alexa_domains) - 1)
        domain = list(self.alexa_domains[domain_index])
        current = domain.copy()

        chars_to_change = 2
        valid_indices = range(len(current))
        if len(current) < chars_to_change:
            chars_to_change = len(current)

        if chars_to_change > 0:
            char_index = random.sample(valid_indices, chars_to_change)
            char_value = random.sample(range(len(self.url_chars)), chars_to_change)
            
            for i in range(chars_to_change):
                while current[char_index[i]] == chr(self.url_chars[char_value[i]]):
                    char_index[i] = random.randint(0, len(current) - 1)
                    char_value[i] = random.randint(0, len(self.url_chars) - 1)
                current[char_index[i]] = chr(self.url_chars[char_value[i]])

        tld_value = random.choice(self.tlds)
        result = ''.join(current) + '.' + tld_value
        return result
