import math
import time
from typing import List, Dict, Any
from core.detector import Detector

def evaluate_detection(domains: List[str], detector: Detector, are_malicious: bool = True) -> Dict[str, Any]:
    """
    Evaluate a list of domains against one detector and return classification metrics.

    The detector labels each domain benign (True) or malicious (False). Scoring
    depends on the ground truth of the input set, given by `are_malicious`:

    - are_malicious=True (default): the set is adversarial (DGA). A domain the
      detector calls benign is a successful evasion (False Negative); a domain it
      calls malicious is caught (True Positive). TN and FP stay 0.
    - are_malicious=False: the set is legitimate (benign control). A domain the
      detector calls benign is correct (True Negative); a domain it calls
      malicious is a false alarm (False Positive). TP and FN stay 0. Metrics that
      need both ground-truth classes (Precision, Recall, TPR, F1, MCC, Kappa) are
      degenerate for a single-class set and report 0; the meaningful outputs are
      Accuracy, FPR, TN, and FP.

    `evaded_domains` and `evasion_rate_percent` count, in both cases, the domains
    the detector passed as benign. For a malicious set that is the evasion count
    (= FN); for a benign set it is the count the detector correctly left alone
    (= TN).

    Args:
        domains (List[str]): Domains to test.
        detector (Detector): A fitted detector.
        are_malicious (bool): Ground truth of the whole set. True if all domains
            are malicious (DGA), False if all are benign.

    Returns:
        Dict with the confusion-matrix counts, the derived classification
        metrics, and the wall time of the detector.detect call
        (inference_time_seconds).
    """
    if not domains:
        # No data: rates and metrics are undefined (null), not 0.0, so an empty set
        # cannot be read as a 0% evasion rate. Counts stay 0.
        return {
            "detector_name": detector.name,
            "total_domains": 0,
            "evaded_domains": 0,
            "evasion_rate_percent": None,
            "Accuracy": None, "Precision": None, "Recall": None, "F1 score": None,
            "FPR": None, "TPR": None, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
            "MCC": None, "Kappa": None, "inference_time_seconds": 0.0
        }

    inference_start = time.time()
    results = detector.detect(domains)
    inference_time = time.time() - inference_start
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    passed_as_benign = 0

    for _, is_benign in results:
        classified_as_malicious = not is_benign
        if is_benign:
            passed_as_benign += 1

        if are_malicious: # Ground Truth is Malicious
            if classified_as_malicious:
                tp += 1
            else:
                fn += 1
        else: # Ground Truth is Benign
            if is_benign:
                tn += 1
            else:
                fp += 1
                
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = recall
    
    evasion_rate = passed_as_benign / total if total > 0 else 0.0
    
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator != 0 else 0.0
    
    p0 = (tn + tp) / total if total > 0 else 0.0
    pa = ((tn + fp) / total) * ((tn + fn) / total) if total > 0 else 0.0
    pb = ((fn + tp) / total) * ((fp + tp) / total) if total > 0 else 0.0
    pe = pa + pb
    kappa = (p0 - pe) / (1 - pe) if pe != 1 else 0.0
    
    return {
        "detector_name": detector.name,
        "total_domains": total,
        "evaded_domains": passed_as_benign,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1_score,
        "FPR": fpr,
        "TPR": tpr,
        "MCC": mcc,
        "Kappa": kappa,
        "evasion_rate_percent": round(evasion_rate * 100.0, 2),
        "inference_time_seconds": inference_time
    }
