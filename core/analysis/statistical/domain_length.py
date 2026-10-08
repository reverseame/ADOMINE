from typing import List, Dict

def calculate_domain_length(domain: str) -> int:
    """
    Calculates the length of the domain.
    
    Args:
        domain (str): The domain string.
        
    Returns:
        int: The length.
    """
    return len(domain)

def analyze_domain_length(domains: List[str]) -> Dict[str, int]:
    """
    Analyzes a list of domains and returns their length.
    
    Args:
        domains (List[str]): List of domains.
        
    Returns:
        Dict[str, int]: Mapping of domain to its length.
    """
    results = {}
    for d in domains:
        results[d] = calculate_domain_length(d)
    return results
