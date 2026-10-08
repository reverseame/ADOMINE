def calculate_unique_char_ratio(domain: str) -> float:
    """
    Calculates the ratio of unique characters to total length.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        float: Unique character ratio (0.0 to 1.0). Returns 0.0 if empty.
    """
    if not domain:
        return 0.0
        
    unique_chars = len(set(domain))
    return unique_chars / len(domain)
