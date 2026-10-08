from core.framework import Framework
from core.seeding import seed_everything
from models.deepdga.deep_dga import DeepDGAModel
from models.charbot.charbot import CharBotModel
from models.deception.deception import DeceptionModel
from models.maskdga.maskdga import MaskDGAModel
from models.hmm_based_dga.hmm_based_dga import HMMBasedDGAModel
from models.pcfg_based_dga.pcfg_based_dga import PCFGBasedDGAModel
from models.fgsm_based_dga.fgsm_based_dga import FGSMBasedDGAModel
from models.gadga.gadga import GADGAModel
from models.wgan_gp_deepdga.wgan_gp_deepdga import WGANGPDeepDGAModel
from models.khaos.khaos import KhaosModel
from models.dngan.dngan import DnGANModel
from models.shadowdga.shadowdga import ShadowDGAModel
from models.ndg.ndg import NDGModel
from models.cleter.cleter import CLETerModel
from models.geometric_perturbation_dga.geometric_perturbation_dga import GeometricPerturbationDGAModel
from models.cdga.cdga import CDGAModel
from models.gwdga.gwdga import GWDGAModel
from models.wgan_based_dga.wgan_based_dga import WGANBasedDGAModel
from models.replacedga.replacedga import ReplaceDGAModel
from models.pkdga.pkdga import PKDGAModel
from models.worddga.worddga import WordDGAModel
from models.tlvdga.tlvdga import TLVDGAModel
from models.titan_dga.titan_dga import TITANDGAModel
from models.sadga.sadga import SADGAModel
from detectors.lstm.lstm import LSTMDetector
from detectors.cnn.cnn import CNNDetector
from detectors.woodbridge_lstm.woodbridge_lstm import WoodbridgeLSTM
from detectors.yu_lstm.yu_lstm import YuLSTM
from detectors.tweet2vec.tweet2vec import Tweet2Vec
from detectors.tweet2vec2.tweet2vec2 import Tweet2Vec2
from detectors.dbd.dbd import DBD
from detectors.zhang.zhang import Zhang
from detectors.yang_cnn.yang_cnn import YangCNN
from models.malicious_dga.malicious_dga import MaliciousDGAModel
from models.benign_domains.benign_domains import BenignDomainsModel
import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

SEED = 42

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    seed_everything(SEED)

    # 1. Initialize the framework with a default workspace directory
    fw = Framework(workspace_dir="my_eval_workspace", seed=SEED)
    
    # 2. Instantiate the adversarial and control models
    deepdga = DeepDGAModel(name="DeepDGA")
    charbot = CharBotModel(name="CharBot")
    deception = DeceptionModel(name="Deception")
    maskdga = MaskDGAModel(name="MaskDGA")
    pcfg_based_dga = PCFGBasedDGAModel(name="PCFGBasedDGA")
    hmm_based_dga = HMMBasedDGAModel(name="HMMBasedDGA")
    fgsm_based_dga = FGSMBasedDGAModel(name="FGSMBasedDGA")
    gadga = GADGAModel(name="GADGA")
    wgan_gp_deepdga = WGANGPDeepDGAModel(name="WGANGPDeepDGA")
    khaos = KhaosModel(name="Khaos")
    dngan = DnGANModel(name="DnGAN")
    shadowdga = ShadowDGAModel(name="ShadowDGA")
    ndg = NDGModel(name="NDG")
    cleter = CLETerModel(name="CLETer")
    geometric_perturbation_dga = GeometricPerturbationDGAModel(name="GeometricPerturbationDGA")
    cdga = CDGAModel(name="CDGA")
    gwdga = GWDGAModel(name="GWDGA")
    wgan_based_dga = WGANBasedDGAModel(name="WGANBasedDGA")
    replacedga = ReplaceDGAModel(name="ReplaceDGA")
    pkdga = PKDGAModel(name="PKDGA")
    worddga = WordDGAModel(name="WordDGA")
    tlvdga = TLVDGAModel(name="TLVDGA")
    titan_dga = TITANDGAModel(name="TITANDGA")
    sadga = SADGAModel(name="SADGA")
    malicious_dga_hash = MaliciousDGAModel(name="Dyre", family="dyre_dga.csv")
    malicious_dga_dict = MaliciousDGAModel(name="Suppobox", family="suppobox_dga.csv")
    malicious_dga_art1 = MaliciousDGAModel(name="Qakbot", family="qakbot_dga.csv")
    malicious_dga_art2 = MaliciousDGAModel(name="Rovnix", family="rovnix_dga.csv")
    malicious_dga_art3 = MaliciousDGAModel(name="Conficker", family="conficker_dga.csv")
    malicious_dga_art4 = MaliciousDGAModel(name="Cryptolocker", family="cryptolocker_dga.csv")
    malicious_dga_perm = MaliciousDGAModel(name="Banjori", family="banjori_dga.csv")
    malicious_dga_pron = MaliciousDGAModel(name="Symmi", family="symmi_dga.csv")
    malicious_dga_dict2 = MaliciousDGAModel(name="Gozi", family="gozi_dga.csv")
    malicious_dga_hash2 = MaliciousDGAModel(name="Bamital", family="bamital_dga.csv")
    benign_domains = BenignDomainsModel(name="Benign Domains")
    
    # Trigger internal training or configuration for each model
    for model in [deepdga, charbot, deception, maskdga, pcfg_based_dga, hmm_based_dga, fgsm_based_dga, gadga, wgan_gp_deepdga, khaos, dngan, shadowdga, ndg, cleter, geometric_perturbation_dga, cdga, gwdga, wgan_based_dga, replacedga, pkdga, worddga, tlvdga, titan_dga, sadga, malicious_dga_hash, malicious_dga_dict, malicious_dga_art1, malicious_dga_art2, malicious_dga_art3, malicious_dga_art4, malicious_dga_perm, malicious_dga_pron, malicious_dga_dict2, malicious_dga_hash2, benign_domains]:
        logger.info(f"[*] Fitting {model.name}...")
        start_t = time.time()
        model.fit()
        if model.training_time is None:
            model.training_time = time.time() - start_t
    
    # 3. Register models
    fw.register_model(deepdga)
    fw.register_model(charbot)
    fw.register_model(deception)
    fw.register_model(maskdga)
    fw.register_model(pcfg_based_dga)
    fw.register_model(hmm_based_dga)
    fw.register_model(fgsm_based_dga)
    fw.register_model(gadga)
    fw.register_model(wgan_gp_deepdga)
    fw.register_model(khaos)
    fw.register_model(dngan)
    fw.register_model(shadowdga)
    fw.register_model(ndg)
    fw.register_model(cleter)
    fw.register_model(geometric_perturbation_dga)
    fw.register_model(cdga)
    fw.register_model(gwdga)
    fw.register_model(wgan_based_dga)
    fw.register_model(replacedga)
    fw.register_model(pkdga)
    fw.register_model(worddga)
    fw.register_model(tlvdga)
    fw.register_model(titan_dga)
    fw.register_model(sadga)
    fw.register_model(malicious_dga_hash)
    fw.register_model(malicious_dga_dict)
    fw.register_model(malicious_dga_art1)
    fw.register_model(malicious_dga_art2)
    fw.register_model(malicious_dga_art3)
    fw.register_model(malicious_dga_art4)
    fw.register_model(malicious_dga_perm)
    fw.register_model(malicious_dga_pron)
    fw.register_model(malicious_dga_dict2)
    fw.register_model(malicious_dga_hash2)
    fw.register_model(benign_domains)
    
    # 4. Generate 100000 domains for each model and save them to the workspace
    logger.info("\n[+] Stage 1: Generating and saving samples")
    domains_dict = {}
    num_domains = 100000
    generation_report = {}  # per model: requested vs obtained, written next to the samples
    for name, model in fw.models.items():
        sample_file = os.path.join(fw.samples_dir, f"{name.replace(' ', '_')}_samples.txt")
        cached = None
        if os.path.exists(sample_file):
            with open(sample_file, "r") as f:
                cached = [line.strip() for line in f if line.strip()]
            # A short cache (left by a model that could not fill its quota) is not reused
            if len(cached) < num_domains:
                logger.warning(f"[!] Cached samples for {name} hold {len(cached)} < {num_domains} domains. Regenerating.")
                cached = None
        if cached is not None:
            logger.info(f"[*] Loading pre-generated samples for {name} from {sample_file}...")
            domains_dict[name] = cached
            generation_report[name] = {"requested": num_domains, "obtained": len(cached), "source": "cache", "short": False}
        else:
            logger.info(f"[*] Generating {num_domains} domains for {name}...")
            start_t = time.time()
            domains = model.generate_domains(num_domains)
            model.generation_time = time.time() - start_t
            domains_dict[name] = domains
            
            with open(sample_file, "w") as f:
                for d in domains:
                    f.write(f"{d}\n")
            short = len(domains) < num_domains
            generation_report[name] = {"requested": num_domains, "obtained": len(domains), "source": "generated", "short": short}
            if short:
                logger.warning(f"[!] {name} produced {len(domains)} of {num_domains} requested domains. Its results rest on a short sample.")
            logger.info(f"  -> Saved {len(domains)} samples to {sample_file}")

    # Written record of the generation stage (a short sample changes how its results must be read)
    report_file = os.path.join(fw.samples_dir, "generation_report.json")
    with open(report_file, "w") as f:
        json.dump(generation_report, f, indent=4)
    short_models = [n for n, r in generation_report.items() if r["short"]]
    if short_models:
        logger.warning(f"[!] Models below the requested sample size: {short_models}. See {report_file}.")
    else:
        logger.info(f"[*] All models reached {num_domains} samples. Report: {report_file}")
    
    # 5. Run statistical analysis and save outputs
    logger.info("\n[+] Stage 2: Running analysis")
    fw.run_statistical_analysis(domains_dict)
    fw.run_time_analysis(domains_dict)

    # 6. Evaluate generated domains against our Detectors
    logger.info("\n[+] Stage 3: Evaluating Detectors")
    # The two original detectors (LSTM, CNN) and the seven best models of the RAMPAGE
    # evaluation (Pelayo-Benedet et al.), all trained on D2 and cached under detectors/<x>/weights.
    detectors = [
        LSTMDetector(),
        CNNDetector(),
        WoodbridgeLSTM(),
        YuLSTM(),
        Tweet2Vec(),
        Tweet2Vec2(),
        DBD(),
        Zhang(),
        YangCNN(),
    ]
    for detector in detectors:
        logger.info(f"[*] Fitting {detector.name}...")
        detector.fit()

    # Register the detectors to the framework and execute evaluation automatically
    for detector in detectors:
        fw.register_detector(detector)
    fw.run_detection_analysis(domains_dict)

    # 7. Repeat the statistical and detection analyses on the SLD alone. Results go to
    #    *_statistical_eval_sld.json and *_evasion_sld.json next to the full-domain files.
    logger.info("\n[+] Stage 4: SLD-only analysis")
    fw.set_analysis_mode(sld_only=True)
    fw.run_statistical_analysis(domains_dict)
    fw.run_detection_analysis(domains_dict)

if __name__ == "__main__":
    main()
