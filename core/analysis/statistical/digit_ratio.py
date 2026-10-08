def calculate_digit_ratio(domain: str) -> float:
    """
    Calculates the ratio of digits to total characters in a domain name.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        float: Digit ratio (0.0 to 1.0). Returns 0.0 if empty.
    """
    if not domain:
        return 0.0
        
    digit_count = sum(1 for char in domain if char.isdigit())
    return digit_count / len(domain)
