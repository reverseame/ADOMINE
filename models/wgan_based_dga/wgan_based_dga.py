import os
import re
import time
import logging
import random
import numpy as np
from typing import List
from core.adversarial_model import AdversarialModel
from core.data_splits import load_malicious_seeds
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, ReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import HeNormal, GlorotUniform

logger = logging.getLogger(__name__)

class WGANBasedDGAModel(AdversarialModel):
    """
    WGAN-based Malicious Domain Name Generator as proposed by Zhang et al. (2022).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Learning rate: Set to 0.00005 for the RMSProp optimizer.
    2. Weight clipping: Range of [-0.01, 0.01] applied to the discriminator layers.
    3. Update frequency (n_critic): 5 discriminator updates per generator update.
    4. Batch size: Set to 100 samples per batch.
    5. Input noise distribution: Standard Normal (Gaussian) distribution.
    6. Weight initialization: He Normal for hidden layers (ReLU) and Glorot Uniform for 
    output layers.
    7. Generation scope: The model generates only the host name label, not the full domain.
    """

    def __init__(self, name: str = "WGANBasedDGA", **kwargs):
        super().__init__(name, **kwargs)
        
        # Hyperparameters and constraints derived from assumptions and the paper
        self.max_len = 30
        self.noise_dim = 100
        self.batch_size = 100
        self.iterations = 10000
        self.n_critic = 5   # Discriminator updates per Generator update
        self.clip_value = 0.01
        self.learning_rate = 0.00005
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
        
        self.generator = None
        self.discriminator = None

        # Paths for caching the generator model weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.generator_weights_path = os.path.join(self.weights_dir, "generator.weights.h5")

    def _load_malicious_domains(self) -> List[str]:
        """
        Loads the training data (real AGD host names) from the D1 malicious split through
        core.data_splits.load_malicious_seeds, isolating the host name.
        """
        agd_domains = []
        try:
            seeds = load_malicious_seeds()
            agd_domains = [d.split('.')[0] for d in seeds if len(d.split('.')[0]) > 3]  # TLD isolation
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs for WGANBasedDGA: {e}")
            
        # Fallback if loading fails
        if not agd_domains:
            logger.warning(f"[*] Could not load malicious domains from the D1 malicious split. Using fallback.")
            agd_domains = ["".join(random.choices(self.chars[:26], k=random.randint(10, 20))) for _ in range(10000)]

        return agd_domains

    def _encode_domain(self, domain: str) -> np.ndarray:
        """
        Encodes a domain name string into a normalized numeric vector.
        Follows Table 1: Encoder implement from ASCII to New Code.
        """
        vector = np.zeros(self.max_len, dtype=np.float32)
        # Process up to max_len characters
        for i, char in enumerate(domain[:self.max_len]):
            ascii_val = ord(char)
            new_code = 0
            
            if '0' <= char <= '9':
                new_code = ascii_val - 47
            elif 'a' <= char <= 'z':
                new_code = ascii_val - 86
            elif 'A' <= char <= 'Z':
                new_code = ascii_val - 28
            elif char == '-':
                new_code = ascii_val + 18
            
            # Map element to [0,1] interval by dividing by 63, keeping 8 decimals
            vector[i] = round(new_code / 63.0, 8)
            
        return vector

    def _decode_domain(self, tensor: np.ndarray) -> str:
        """
        Decodes a tensor back into a domain name string.
        Follows Table 2: Decoder implement from New Code to ASCII.
        """
        chars = []
        for val in tensor:
            # Reverse the mapping
            a_i = int(round(val * 63))
            
            if a_i == 0:
                continue # Padding character
            elif 1 <= a_i <= 10:
                chars.append(chr(a_i + 47))
            elif 11 <= a_i <= 36:
                chars.append(chr(a_i + 86))
            elif 37 <= a_i <= 62:
                chars.append(chr(a_i + 28))
            elif a_i == 63:
                chars.append(chr(a_i - 18))
                
        # DNS names are case-insensitive. Codes 37-62 (A-Z) belong to the paper's Table 1 but never
        # occur in the lowercase training hostnames, so the decoded text is folded to lowercase.
        return "".join(chars).lower()

    @staticmethod
    def _is_valid_sld(sld: str) -> bool:
        """
        RFC 1034/1035 label check: 1 to 63 chars, only a-z, 0-9 and '-', no leading or trailing hyphen.
        """
        return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", sld))

    def _build_generator(self) -> Sequential:
        """
        Builds the 4-layer fully connected generative network. Hidden layers 
        use ReLU, output uses Sigmoid. Weight initialization based on the assumption handling.
        """
        model = Sequential(name="WGAN_Generator")
        model.add(Input(shape=(self.noise_dim,)))
        
        # Hidden layers (100 and 128 nodes)
        model.add(Dense(100, kernel_initializer=HeNormal()))
        model.add(ReLU())
        
        model.add(Dense(128, kernel_initializer=HeNormal()))
        model.add(ReLU())
        
        # Output layer (30 nodes, representing the domain vector dimension)
        model.add(Dense(self.max_len, activation='sigmoid', kernel_initializer=GlorotUniform()))
        return model

    def _build_discriminator(self) -> Sequential:
        """
        Builds the 4-layer fully connected discrimination network. Hidden layers use 
        ReLU, output layer is linear (Wasserstein distance).
        """
        model = Sequential(name="WGAN_Discriminator")
        model.add(Input(shape=(self.max_len,)))
        
        # Hidden layers (100 and 128 nodes)
        model.add(Dense(100, kernel_initializer=HeNormal()))
        model.add(ReLU())
        
        model.add(Dense(128, kernel_initializer=HeNormal()))
        model.add(ReLU())
        
        # Output layer for regression task
        model.add(Dense(1, kernel_initializer=GlorotUniform()))
        return model

    def fit(self) -> None:
        """
        Trains the WGAN-based DGA model or loads cached weights.
        """

        start_t = time.time()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Try to load everything from cache
        if os.path.exists(self.generator_weights_path):
            logger.info(f"[*] Cached weights found for {self.name}. Skipping training.")
            self.generator.load_weights(self.generator_weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        logger.info(f"[*] No cached weights found for {self.name}. Training...")
        os.makedirs(self.weights_dir, exist_ok=True)

        # Load malicious domains
        real_domains = self._load_malicious_domains()
           
        encoded_real_data = np.array([self._encode_domain(d) for d in real_domains])
        dataset = tf.data.Dataset.from_tensor_slices(encoded_real_data).shuffle(len(encoded_real_data)).batch(self.batch_size, drop_remainder=True).repeat()
        dataset_iter = iter(dataset)

        # Optimizers (RMSProp) with assumed learning rate
        g_optimizer = RMSprop(learning_rate=self.learning_rate)
        d_optimizer = RMSprop(learning_rate=self.learning_rate)

        # Training loop
        for iteration in range(self.iterations):
            # Train Discriminator (n_critic times per generator update)
            for _ in range(self.n_critic):
                real_batch = next(dataset_iter)
                # Assumed Gaussian Noise Z ~ N(0, I_100)
                noise = tf.random.normal([self.batch_size, self.noise_dim])
                
                with tf.GradientTape() as d_tape:
                    fake_batch = self.generator(noise, training=True)
                    real_logits = self.discriminator(real_batch, training=True)
                    fake_logits = self.discriminator(fake_batch, training=True)
                    
                    # Wasserstein Loss for Discriminator: mean(D(fake)) - mean(D(real))
                    d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
                
                # Weight Clipping for Discriminator
                for var in self.discriminator.trainable_variables:
                    var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))

            # Train Generator
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            with tf.GradientTape() as g_tape:
                fake_batch = self.generator(noise, training=True)
                fake_logits = self.discriminator(fake_batch, training=True)
                
                # Wasserstein Loss for Generator: -mean(D(fake))
                g_loss = -tf.reduce_mean(fake_logits)
                
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

            if (iteration + 1) % 1000 == 0:
                logger.info(f"[{self.name}] Iteration {iteration + 1}/{self.iterations} | D_loss: {d_loss.numpy():.4f} | G_loss: {g_loss.numpy():.4f}")

        # Save weights
        self.generator.save_weights(self.generator_weights_path)
        logger.info(f"[{self.name}] Training complete. Weights saved to {self.generator_weights_path}")
        
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def generate_domain(self) -> str:
        """
        Generates a single domain name utilizing the trained generator.
        """
        if not self.is_fitted:
            self.fit()
            
        max_attempts = 100  # SAFETY GUARD: bounded rejection sampling
        for _attempt in range(max_attempts):
            noise = tf.random.normal([1, self.noise_dim])
            generated_tensor = self.generator(noise, training=False)[0].numpy()
            domain = self._decode_domain(generated_tensor)
            if self._is_valid_sld(domain):
                return f"{domain}.{sample_tld()}"
        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"

    def generate_domains(self, n: int) -> List[str]:
        """
        Generates a batch of domain names efficiently.
        """
        if not self.is_fitted:
            self.fit()

        domains = []
        # SAFETY GUARD: bounded rejection sampling that scales with n; allows up to 75% of candidates
        # to be discarded before returning a partial list with a warning
        max_batches = max(200, 4 * -(-n // self.batch_size))
        for _batch in range(max_batches):
            if len(domains) >= n:
                break
            noise = tf.random.normal([self.batch_size, self.noise_dim])
            generated_tensors = self.generator(noise, training=False).numpy()
            for tensor in generated_tensors:
                domain = self._decode_domain(tensor)
                # Only RFC-valid labels are kept; invalid outputs are resampled
                if self._is_valid_sld(domain):
                    domains.append(f"{domain}.{sample_tld()}")
                    if len(domains) == n:
                        break
        if len(domains) < n:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(domains)} of {n} valid domains after {max_batches} batches. Returning a partial list.")
        return domains