import os
import json
import logging
from typing import Dict, List, Any
import time
from .adversarial_model import AdversarialModel
from .detector import Detector
from .analysis.statistical.general import evaluate_statistical
from .analysis.time.general import evaluate_time, evaluate_detector_time
from .analysis.detection.general import evaluate_detection
from .run_metadata import collect_run_metadata
from .sld import extract_sld

logger = logging.getLogger(__name__)

class Framework:
    """
    The main orchestrator class for the adversarial DGA evaluation framework.
    
    This class is responsible for managing multiple adversarial models, coordinating
    the generation of their domains, saving samples, and running evaluations.
    """

    def __init__(self, workspace_dir: str = "workspace", seed: int = 42, sld_only: bool = False):
        """
        Initialize the framework with an empty registry of adversarial models.

        Args:
            workspace_dir (str): Base directory where all samples and analysis will be stored.
            seed (int): RNG seed for the run, stamped into every output JSON.
            sld_only (bool): Initial analysis mode; see set_analysis_mode for
                flipping it mid-run.
        """
        self.models: Dict[str, AdversarialModel] = {}
        self.detectors: Dict[str, Detector] = {}
        self.workspace_dir = workspace_dir
        self.seed = seed
        self.run_metadata = collect_run_metadata(seed)
        self.analyze_sld_only = sld_only
        
        # Ensure base directories exist
        self.samples_dir = os.path.join(self.workspace_dir, "samples")
        self.analysis_dir = os.path.join(self.workspace_dir, "analysis")
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

    def register_model(self, model: AdversarialModel) -> None:
        """
        Register a new adversarial model into the framework.
        """
        if model.name in self.models:
            logger.warning(f"Warning: Model with name '{model.name}' is already registered and will be overwritten.")
        
        self.models[model.name] = model
        logger.info(f"Model '{model.name}' successfully registered.")

    def register_detector(self, detector: Detector) -> None:
        """
        Register an evasion detector into the framework.
        """
        if detector.name in self.detectors:
            logger.warning(f"Warning: Detector '{detector.name}' is already registered and will be overwritten.")
        
        self.detectors[detector.name] = detector
        logger.info(f"Detector '{detector.name}' successfully registered.")

    def set_analysis_mode(self, sld_only: bool) -> None:
        """
        Configure whether the statistical and detection analyses run on the
        full domain name (SLD.TLD) or strictly on the Second-Level Domain (SLD).

        In SLD-only mode the detection results are written to
        *_evasion_sld.json, next to the full-domain *_evasion.json, so both
        modes can coexist in the workspace.

        Args:
            sld_only (bool): If True, isolates the SLD before analysis.
        """
        self.analyze_sld_only = sld_only
        mode_str = "SLD-only" if sld_only else "Full TLD"
        logger.info(f"Analysis mode set to: {mode_str}")

    def generate_all(self, n: int) -> Dict[str, List[str]]:
        """
        Generate 'n' domains from every registered adversarial model and save them 
        to the workspace samples directory.
        
        Args:
            n (int): Number of domains to generate per model.
            
        Returns:
            Dict[str, List[str]]: Mapping of model names to their generated domains.
        """
        results = {}
        for name, model in self.models.items():
            logger.info(f"Generating {n} domains from model: {name}...")
            start_time = time.time()
            domains = model.generate_domains(n)
            model.generation_time = time.time() - start_time
            results[name] = domains
            
            # Save the samples to file
            sample_file = os.path.join(self.samples_dir, f"{name.replace(' ', '_')}_samples.txt")
            with open(sample_file, "w") as f:
                for domain in domains:
                    f.write(f"{domain}\n")
            logger.info(f"  -> Saved {len(domains)} samples to {sample_file}")
            
        return results

    def run_statistical_analysis(self, domains_dict: Dict[str, List[str]]) -> None:
        """
        Runs statistical evaluation on the provided domains and saves the results 
        inside the workspace's analysis/statistical/ directory.
        
        Args:
            domains_dict (Dict[str, List[str]]): Domains to analyze.
        """
        stat_dir = os.path.join(self.analysis_dir, "statistical")
        os.makedirs(stat_dir, exist_ok=True)
        
        logger.info(f"\nRunning statistical analysis. Results will be saved to {stat_dir}")
        for model_name, domains in domains_dict.items():
            # Run the general statistical evaluation over the domains
            results = evaluate_statistical(domains, sld_only=self.analyze_sld_only)
            results["run_metadata"] = self.run_metadata
            
            # Saving the comprehensive output for the model:
            safe_name = model_name.replace(" ", "_")
            # SLD-only results go to *_statistical_eval_sld.json so both modes coexist (as for detection)
            mode_suffix = "_statistical_eval_sld.json" if self.analyze_sld_only else "_statistical_eval.json"
            out_file = os.path.join(stat_dir, f"{safe_name}{mode_suffix}")
            
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"  -> Statistical results for '{model_name}' saved to {out_file}")

    def run_time_analysis(self, domains_dict: Dict[str, List[str]]) -> None:
        """
        Runs time evaluation on the generated domains and saves the results 
        inside the workspace's analysis/time/ directory.
        
        Args:
            domains_dict (Dict[str, List[str]]): Domains to analyze.
        """
        time_dir = os.path.join(self.analysis_dir, "time")
        os.makedirs(time_dir, exist_ok=True)
        
        logger.info(f"\nRunning time analysis. Results will be saved to {time_dir}")
        for model_name, domains in domains_dict.items():
            model = self.models.get(model_name)
            if not model:
                continue
                
            safe_name = model_name.replace(" ", "_")
            out_file = os.path.join(time_dir, f"{safe_name}_time_eval.json")
            
            # Load existing evaluation to preserve unmodified times
            existing_data = None
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing {out_file}, overwriting it: {e}")

            # Run the general time evaluation, merging with existing data
            results = evaluate_time(
                training_time=model.training_time,
                generation_time=model.generation_time,
                num_domains=len(domains),
                existing_data=existing_data,
                trained_from_scratch=model.trained_from_scratch
            )
            results["run_metadata"] = self.run_metadata
            
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
                
            logger.info(f"  -> Time evaluation results for '{model_name}' saved to {out_file}")

    def run_detection_analysis(self, domains_dict: Dict[str, List[str]]) -> None:
        """
        Runs the evasion detection evaluation against all registered detectors 
        and saves results inside workspace/analysis/detection/
        """
        if not self.detectors:
            logger.warning("\n[!] No detectors registered for analysis.")
            return

        detection_dir = os.path.join(self.analysis_dir, "detection")
        os.makedirs(detection_dir, exist_ok=True)

        if self.analyze_sld_only:
            domains_dict = {
                name: [extract_sld(d) for d in domains]
                for name, domains in domains_dict.items()
            }

        logger.info(f"\nRunning detection analysis across {len(self.detectors)} detectors. Results will be saved to {detection_dir}")
        for detector_name, detector in self.detectors.items():
            logger.info(f"\n[*] Evaluating evasion against: {detector_name}")
            detector_safe_name = detector_name.replace(" ", "_").lower()
            detector_out_dir = os.path.join(detection_dir, detector_safe_name)
            os.makedirs(detector_out_dir, exist_ok=True)
            
            for model_name, domains in domains_dict.items():
                model = self.models.get(model_name)
                are_malicious = model.is_malicious if model is not None else True
                results = evaluate_detection(domains, detector, are_malicious=are_malicious)
                # Timing is machine-dependent: it stays out of the (seed-deterministic)
                # detection JSON and accumulates in the detector's running totals.
                inference_time = results.pop("inference_time_seconds", 0.0)
                detector.inference_time = (detector.inference_time or 0.0) + inference_time
                detector.inference_domains += len(domains)
                results["run_metadata"] = self.run_metadata
                results["sld_only"] = self.analyze_sld_only

                model_safe_name = model_name.replace(" ", "_").lower()
                mode_suffix = "_evasion_sld.json" if self.analyze_sld_only else "_evasion.json"
                out_file = os.path.join(detector_out_dir, f"{model_safe_name}{mode_suffix}")
                
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)

                logger.info(f"  -> '{model_name}': {results['evaded_domains']}/{results['total_domains']} evaded ({results['evasion_rate_percent']}%) -> Saved to {out_file}")

        self._write_detector_time_analysis()

    def _write_detector_time_analysis(self) -> None:
        """
        Write one analysis/time/<Detector>_time_eval.json per registered
        detector: training cost from fit(), inference cost accumulated over
        every detect call of the run.
        """
        time_dir = os.path.join(self.analysis_dir, "time")
        os.makedirs(time_dir, exist_ok=True)

        for detector_name, detector in self.detectors.items():
            safe_name = detector_name.replace(" ", "_")
            out_file = os.path.join(time_dir, f"{safe_name}_time_eval.json")

            existing_data = None
            if os.path.exists(out_file):
                try:
                    with open(out_file, "r") as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing {out_file}, overwriting it: {e}")

            results = evaluate_detector_time(
                training_time=detector.training_time,
                inference_time=detector.inference_time,
                num_domains=detector.inference_domains,
                existing_data=existing_data,
                trained_from_scratch=detector.trained_from_scratch
            )
            results["run_metadata"] = self.run_metadata

            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"  -> Time evaluation results for '{detector_name}' saved to {out_file}")

