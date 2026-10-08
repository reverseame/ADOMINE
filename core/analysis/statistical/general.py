from typing import List, Dict, Any
from core.sld import extract_sld
from .shannon_entropy import calculate_shannon_entropy
from .vowel_ratio import calculate_vowel_ratio
from .consonant_ratio import calculate_consonant_ratio
from .domain_length import calculate_domain_length
from .digit_ratio import calculate_digit_ratio
from .unique_chars import calculate_unique_char_ratio
from .consecutive_consonants import calculate_consecutive_consonants

def evaluate_statistical(domains: List[str], sld_only: bool = False) -> Dict[str, Any]:
    """
    Orchestrates all statistical analyses on a list of domains.
    Returns a unified dictionary with general averages and detailed
    per-sample metrics.
    """
    if not domains:
        return {}

    samples_data = {}
    sum_entropy = 0.0
    sum_vowel = 0.0
    sum_consonant_rat = 0.0
    sum_length = 0
    sum_digit = 0.0
    sum_unique = 0.0
    sum_consonants = 0

    for domain in domains:
        analyzed_string = extract_sld(domain) if sld_only else domain
        
        entropy = calculate_shannon_entropy(analyzed_string)
        vowel_rat = calculate_vowel_ratio(analyzed_string)
        consonant_rat = calculate_consonant_ratio(analyzed_string)
        length = calculate_domain_length(analyzed_string)
        digit_rat = calculate_digit_ratio(analyzed_string)
        unique_rat = calculate_unique_char_ratio(analyzed_string)
        max_cons = calculate_consecutive_consonants(analyzed_string)

        sum_entropy += entropy
        sum_vowel += vowel_rat
        sum_consonant_rat += consonant_rat
        sum_length += length
        sum_digit += digit_rat
        sum_unique += unique_rat
        sum_consonants += max_cons

        samples_data[domain] = {
            "statistics": {
                "shannon_entropy": entropy,
                "vowel_ratio": vowel_rat,
                "consonant_ratio": consonant_rat,
                "length": length,
                "digit_ratio": digit_rat,
                "unique_char_ratio": unique_rat,
                "consecutive_consonants": max_cons
            }
        }

    n = len(domains)
    general_data = {
        "statistics": {
            "shannon_entropy": sum_entropy / n,
            "vowel_ratio": sum_vowel / n,
            "consonant_ratio": sum_consonant_rat / n,
            "length": sum_length / n,
            "digit_ratio": sum_digit / n,
            "unique_char_ratio": sum_unique / n,
            "consecutive_consonants": sum_consonants / n
        }
    }

    return {
        "general": general_data,
        "samples": samples_data
    }
