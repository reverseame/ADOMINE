from typing import Dict, Any, Optional

def evaluate_time(training_time: Optional[float], generation_time: Optional[float], num_domains: int, existing_data: Optional[Dict[str, Any]] = None, trained_from_scratch: Optional[bool] = None) -> Dict[str, Any]:
    """
    Orchestrates time evaluation metrics returning a structured dictionary
    similar to evaluate_statistical, encompassing total times and average per sample.
    Updates existing data if provided, so we only overwrite the values that actually changed.
    """
    if existing_data is None:
        existing_data = {"general": {"time": {}}}
        
    if "general" not in existing_data:
        existing_data["general"] = {}
    if "time" not in existing_data["general"]:
        existing_data["general"]["time"] = {}
        
    time_data = existing_data["general"]["time"]
    
    real_time_recorded = (
        "training_time_seconds" in time_data
        and time_data.get("trained_from_scratch") is True
    )
    if trained_from_scratch is False and real_time_recorded:
        # Cache-hit run: keep the real training time and flag from the prior run.
        pass
    else:
        if training_time is not None:
            time_data["training_time_seconds"] = training_time
        if trained_from_scratch is not None:
            time_data["trained_from_scratch"] = trained_from_scratch
        
    if generation_time is not None:
        time_data["generation_time_seconds"] = generation_time
        if num_domains > 0:
            time_data["generation_time_per_domain_seconds"] = generation_time / num_domains
        time_data["total_domains_generated"] = num_domains

    return existing_data


def evaluate_detector_time(training_time: Optional[float], inference_time: Optional[float], num_domains: int, existing_data: Optional[Dict[str, Any]] = None, trained_from_scratch: Optional[bool] = None) -> Dict[str, Any]:
    """
    Detector counterpart of evaluate_time: training cost plus inference cost
    (accumulated over every detect call of the run). Applies the same
    cache-hit protection to the training time.
    """
    if existing_data is None:
        existing_data = {"general": {"time": {}}}

    if "general" not in existing_data:
        existing_data["general"] = {}
    if "time" not in existing_data["general"]:
        existing_data["general"]["time"] = {}

    time_data = existing_data["general"]["time"]

    real_time_recorded = (
        "training_time_seconds" in time_data
        and time_data.get("trained_from_scratch") is True
    )
    if trained_from_scratch is False and real_time_recorded:
        # Cache-hit run: keep the real training time and flag from the prior run.
        pass
    else:
        if training_time is not None:
            time_data["training_time_seconds"] = training_time
        if trained_from_scratch is not None:
            time_data["trained_from_scratch"] = trained_from_scratch

    if inference_time is not None:
        time_data["inference_time_seconds"] = inference_time
        if num_domains > 0:
            time_data["inference_time_per_domain_seconds"] = inference_time / num_domains
        time_data["total_domains_scored"] = num_domains

    return existing_data
