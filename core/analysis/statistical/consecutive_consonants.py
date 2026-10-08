def calculate_consecutive_consonants(domain: str) -> int:
    """
    Calculates the maximum number of consecutive consonants in the domain.
    
    Args:
        domain (str): The domain string to analyze.
        
    Returns:
        int: Maximum length of substring consisting entirely of consonants.
    """
    if not domain:
        return 0
        
    vowels = set("aeiouAEIOU")
    letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    max_consecutive = 0
    current_consecutive = 0
    
    for char in domain:
        if char in letters and char not in vowels:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 0
            
    return max_consecutive
