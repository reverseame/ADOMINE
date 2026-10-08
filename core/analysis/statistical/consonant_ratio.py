from typing import List, Dict

def calculate_consonant_ratio(domain: str) -> float:
    """
    Calculates the ratio of consonants to total letters in a domain name.
    Ignores numbers and special characters. If there are no letters, returns 0.0.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        float: Consonant ratio (0.0 to 1.0).
    """
    if not domain:
        return 0.0
        
    vowels = set("aeiouAEIOU")
    letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    consonant_count = 0
    total_letters = 0
    
    for char in domain:
        if char in letters:
            total_letters += 1
            if char not in vowels:
                consonant_count += 1
                
    if total_letters == 0:
        return 0.0
        
    return consonant_count / total_letters

def analyze_consonant_ratio(domains: List[str]) -> Dict[str, float]:
    """
    Analyzes a list of domains and returns their consonant ratio.
    
    Args:
        domains (List[str]): List of domains.
        
    Returns:
        Dict[str, float]: Mapping of domain to its consonant ratio.
    """
    results = {}
    for d in domains:
        results[d] = calculate_consonant_ratio(d)
    return results
