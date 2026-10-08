import os
import re
import random
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# Framework imports
from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from core.sld import extract_sld
from detectors.cnn.cnn import CNNDetector
from .generator import Generator
from .rollout import Rollout

logger = logging.getLogger(__name__)

class PKDGAModel(AdversarialModel):
    """
    Implementation of PKDGA (Partial Knowledge-Based DGA), proposed by Nie et al. (2023).
    It uses Reinforcement Learning to generate adversarial SLDs that evade a target detector
    by learning from feedback.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Network architecture: The policy network is an LSTM with the same hyperparameters as the 
    original repository.
    2. Pre-training: Uses Maximum Likelihood Estimation (MLE) on benign domains from Tranco D1.
    3. Adversarial training: Uses Monte Carlo rollouts with the detector as the reward oracle.
    4. Generation: Generated domains are second-level domains (SLDs) only, without TLD.
    5. Epochs: 1 epoch of MLE pre-training and 60 batches of adversarial training.
    6. Target detector: If no target detector is provided, we use a default CNN-based detector.
    """

    def __init__(self, name: str = "PKDGA", target_detector=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_detector = target_detector

        # Hyperparameters
        self.batch_size = 64
        self.g_sequence_len = 65          # maximum length of generated sequences
        self.vocab_size = 39              # alphabet size (space, letters, digits, hyphen)
        self.g_emb_dim = 32               # embedding dimension
        self.g_hidden_dim = 64            # LSTM hidden dimension
        self.num_mc = 20                  # number of Monte Carlo searches
        self.total_batch = 60              # adversarial training cycles (adjustable)
        self.pre_epoch_num = 1            # number of MLE pre-training epochs
        self.lr = 0.001                   # learning rate
        self.num_layers = 1               # number of LSTM layers

        # Alphabet and mappings (same as original repository)
        self.alphabet = [' ', 'g', 'o', 'l', 'e', 'y', 'u', 't', 'b', 'm', 'a', 'i', 'd', 'q', 's', 'h', 'f', 'c', 'k', '3', '6',
                         '0', 'j', 'z', 'n', 'w', 'p', 'r', 'x', 'v', '1', '8', '7', '2', '9', '-', '5', '4', '_']
        self.int_to_char = dict(enumerate(self.alphabet))
        self.char_to_int = {c: i for i, c in enumerate(self.alphabet)}

        # Directory for cached weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.generator_path = os.path.join(self.weights_dir, "generator.trc")
        self.generator = None

    def fit(self) -> None:
        """
        Self-contained training or initialization. Checks for cached weights first.
        If no weights are found, it trains the generator using MLE followed by adversarial RL
        (Reinforcement Learning).
        """
        start_time = time.time()
        os.makedirs(self.weights_dir, exist_ok=True)

        # 1. Check for cached models
        if os.path.exists(self.generator_path):
            logger.info(f"[*] {self.name} found cached weights. Skipping training.")
            self.generator = Generator(self.vocab_size, self.g_emb_dim, self.g_hidden_dim, use_cuda=False,
                                       num_layers=self.num_layers)
            self.generator.load_state_dict(torch.load(self.generator_path, map_location='cpu'))
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            return

        logger.info(f"[*] No cached weights found for {self.name}. Training...")

        # 2. Setup Target Detector for Feedback
        if self.target_detector is None:
            logger.info("[*] No target detector provided. Instantiating default CNN detector...")
            self.target_detector = CNNDetector()
        if not getattr(self.target_detector, 'is_fitted', False):
            logger.info("[*] Fitting the target detector...")
            self.target_detector.fit()

        # 3. Prepare Training Data (SLDs only)
        # 3a. Read Benign Domains from Tranco D1 and extract SLDs
        benign_raw = load_tranco(TRANCO_D1_BENIGN)
        benign_slds = []
        for d in benign_raw:
            sld = extract_sld(d)
            # Keep only domains with characters present in our alphabet
            if all(c in self.char_to_int for c in sld):
                benign_slds.append(sld)
        if not benign_slds:
            raise ValueError("No valid benign SLDs found for training PKDGA.")

        # 3b. Encode sequences for MLE pre-training (use a subset for speed)
        train_samples = benign_slds[:10000]   # can be increased if needed
        sequences = []
        for sld in train_samples:
            seq = [self.char_to_int[c] for c in sld if c in self.char_to_int]
            if len(seq) > self.g_sequence_len:
                seq = seq[:self.g_sequence_len]
            else:
                seq += [0] * (self.g_sequence_len - len(seq))   # 0 is the padding token (space)
            sequences.append(seq)
        X_train = torch.LongTensor(sequences)

        # 4. Initialize generator and optimizer
        self.generator = Generator(self.vocab_size, self.g_emb_dim, self.g_hidden_dim, use_cuda=False,
                                   num_layers=self.num_layers)
        gen_criterion = nn.NLLLoss(reduction='sum')
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

        # 5. MLE Pre-training
        logger.info("[*] Pre-training with MLE...")
        self.generator.train()
        for epoch in range(self.pre_epoch_num):
            total_loss = 0.0
            for i in range(0, len(X_train), self.batch_size):
                batch = X_train[i:i+self.batch_size]
                # Input: all tokens except the last; target: all tokens except the first
                inputs = Variable(batch[:, :-1])
                targets = Variable(batch[:, 1:].contiguous().view(-1))
                self.generator.zero_grad()
                pred = self.generator.forward(inputs)
                loss = gen_criterion(pred, targets)
                loss.backward()
                gen_optimizer.step()
                total_loss += loss.item()
            logger.info(f"  MLE epoch {epoch+1}: loss = {total_loss / len(X_train):.4f}")

        # 6. Adversarial Training with Reinforcement Learning
        logger.info("[*] Adversarial RL training...")
        rollout = Rollout(self.generator, update_rate=0.7)
        gen_gan_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

        # Reward function: detector returns (domain, is_benign) -> 1 for evasion, 0 otherwise
        def compute_rewards(domains):
            results = self.target_detector.detect(domains)
            return [1 if is_benign else 0 for _, is_benign in results]

        for batch_idx in range(self.total_batch):
            # Sample a batch from the generator
            samples = self.generator.sample(self.batch_size, self.g_sequence_len)
            # Obtain rewards via Monte Carlo rollouts
            rewards = rollout.get_reward(samples, self.num_mc, self.target_detector,
                                         self.alphabet, model_name="", use="", cuda=False)
            rewards = torch.Tensor(rewards).reshape(-1)

            # Prepare inputs and targets for the generator
            zeros = torch.zeros((self.batch_size, 1)).long()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view(-1)

            # Forward pass and reward-weighted loss
            prob = self.generator.forward(inputs)
            loss = -torch.sum(prob.gather(1, targets.view(-1, 1)).squeeze() * rewards)

            gen_gan_optimizer.zero_grad()
            loss.backward()
            gen_gan_optimizer.step()

            logger.info(f"  Adversarial batch {batch_idx+1}: loss = {loss.item():.4f}")
            rollout.update_params()

        # 7. Save trained weights
        torch.save(self.generator.state_dict(), self.generator_path)

        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time

    @staticmethod
    def _is_valid_sld(sld: str) -> bool:
        """
        RFC 1034/1035 label check: 1 to 63 chars, only a-z, 0-9 and '-', no leading or trailing hyphen.
        """
        return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", sld))

    def generate_domain(self) -> str:
        """
        Produce a single adversarial SLD.
        """
        domains = self.generate_domains(1)
        if domains:
            return domains[0]
        logger.warning(f"[{self.name}] No valid SLD produced. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"

    def generate_domains(self, n: int) -> list:
        """
        Produce n adversarial SLDs using the trained policy. The generator is loaded from cache
        if not already available. The 39-token alphabet of the authors' code includes '_', which
        is not a valid hostname character, so every sample is checked as an RFC label and invalid
        ones are rejected and resampled.
        """
        if not self.is_fitted or self.generator is None:
            self.fit()

        self.generator.eval()
        domains = []
        # SAFETY GUARD: bounded rejection sampling that scales with n; allows up to 75% of candidates
        # to be discarded before returning a partial list with a warning
        max_batches = max(200, 4 * -(-n // self.batch_size))
        with torch.no_grad():
            for _batch in range(max_batches):
                if len(domains) >= n:
                    break
                samples = self.generator.sample(self.batch_size, self.g_sequence_len)
                for seq in samples.cpu().numpy():
                    chars = [self.int_to_char.get(int(idx), '') for idx in seq if int(idx) != 0]
                    sld = ''.join(chars).strip('-')
                    if self._is_valid_sld(sld):
                        domains.append(f"{sld}.{sample_tld()}")  # paper: "appending a top-level domain at the end"
                        if len(domains) == n:
                            break
        if len(domains) < n:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(domains)} of {n} valid SLDs after {max_batches} batches. Returning a partial list.")
        return domains