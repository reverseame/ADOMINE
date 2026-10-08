from typing import List, Dict

def calculate_vowel_ratio(domain: str) -> float:
    """
    Calculates the ratio of vowels to total letters in a domain name.
    Ignores numbers and special characters.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        float: Vowel ratio (0.0 to 1.0). Returns 0.0 if no letters.
    """
    if not domain:
        return 0.0
        
    vowels = set("aeiouAEIOU")
    letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    vowel_count = 0
    total_letters = 0
    
    for char in domain:
        if char in letters:
            total_letters += 1
            if char in vowels:
                vowel_count += 1
                
    if total_letters == 0:
        return 0.0
        
    return vowel_count / total_letters

def analyze_vowel_ratio(domains: List[str]) -> Dict[str, float]:
    """
    Analyzes a list of domains and returns their vowel ratio.
    
    Args:
        domains (List[str]): List of domains.
        
    Returns:
        Dict[str, float]: Mapping of domain to its vowel ratio.
    """
    results = {}
    for d in domains:
        results[d] = calculate_vowel_ratio(d)
    return results
