import math
from collections import Counter
from typing import List, Dict

def calculate_shannon_entropy(domain: str) -> float:
    """
    Calculates the Shannon Entropy of a domain name.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        float: The Shannon entropy value.
    """
    if not domain:
        return 0.0
    
    length = len(domain)
    char_counts = Counter(domain)
    entropy = 0.0
    
    for count in char_counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
        
    return entropy

def analyze_shannon_entropy(domains: List[str]) -> Dict[str, float]:
    """
    Analyzes a list of domains and returns their entropy.
    
    Args:
        domains (List[str]): List of domains.
        
    Returns:
        Dict[str, float]: Mapping of domain to its entropy score.
    """
    results = {}
    for d in domains:
        results[d] = calculate_shannon_entropy(d)
    return results
