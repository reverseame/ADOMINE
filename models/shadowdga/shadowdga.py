import os
import time
import random
import numpy as np
import logging
import re
import tensorflow as tf
from tensorflow.keras import layers, Model

from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from core.sld import extract_sld

logger = logging.getLogger(__name__)

class ShadowDGAModel(AdversarialModel):
    """
    ShadowDGA based on the Zheng et al. (2021) paper, using 1D CNNs, Residual Blocks, and WGAN-GP.
    
    REPRODUCIBILITY ASSUMPTIONS:
    1. Architecture and Dimensions: Assumes a 100-dimensional latent space for the noise vector 
    and 128 filters (channels) for the Conv1D layers with a kernel size of 3 and "same" padding.
    2. Vocabulary and Tensors: Assumes a one-hot encoding with a static vocabulary of 38 characters
    (including the special padding token).
    3. Text Differentiability: To allow gradient backpropagation, assumes a continuous relaxation
    during training, passing the generator's continuous probability matrix (Softmax) directly to
    the discriminator.
    4. Residual Blocks: Maintains a constant width of 128 channels across all internal 
    convolutions to allow direct residual addition without projections, and omits any 
    normalization in the discriminator to prevent interference with WGAN-GP.
    5. WGAN-GP Configuration: Assumes a standard gradient penalty coefficient of lambda = 10.0 
    and utilizes a strictly linear activation in the discriminator's final dense layer.
    6. Training and Validation: During generation, discretized domains are validated via regular 
    expressions to ensure RFC compatibility and are randomly assigned a TLD from a predefined list.
    """

    def __init__(self, name: str = "ShadowDGA", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.latent_dim = 100
        self.min_len = 6        # Generated domains minimum length
        self.max_len = 16       # Generated domains maximum length
        self.channels = 128
        self.batch_size = 64
        self.iterations = 5000  # JS divergence stabilizes at 5000 iterations
        self.n_critic = 10      # Discriminator updates per Generator update
        self.lambda_gp = 10.0   # Gradient penalty coefficient
        
        self.PAD = 0            # Padding character index
        chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
        
        # Character mapping: index 1..37 for valid chars, index 0 reserved for PAD
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.idx2char[self.PAD] = ""
        
        self.vocab_size = len(self.idx2char)  # Total 38 characters
        self.tlds = ['com', 'net', 'org', 'info', 'biz', 'ru', 'in', 'cc']

        # Build the Generator and Discriminator models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Adam Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
            
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.g_weights_path = os.path.join(self.weights_dir, "generator.weights.h5")
        self.d_weights_path = os.path.join(self.weights_dir, "discriminator.weights.h5")

    def _residual_block(self, x_in: tf.Tensor) -> tf.Tensor:
        """
        Builds a 1D residual block.
        H(x) = F(x) + x. Requires equal channel dimensions on input and output.
        """
        # First Conv1D
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x_in)
        x = layers.ReLU()(x)
        # Second Conv1D
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x)
        # Residual connection
        return layers.Add()([x_in, x])

    def _build_generator(self) -> tf.keras.Model:
        """
        Constructs the Generator network. Maps continuous noise (100-dim) to a sequence of 
        softmax probability distributions.
        """
        noise_input = layers.Input(shape=(self.latent_dim,))
        
        # Project noise to the required matrix shape (Batch, Length, Channels)
        x = layers.Dense(self.max_len * self.channels)(noise_input)
        x = layers.Reshape((self.max_len, self.channels))(x)

        # 5 Residual Blocks
        for _ in range(5):
            x = self._residual_block(x)

        # Output continuous probabilities to allow gradient backpropagation
        # Using 1x1 Conv to map channels to vocab_size
        output = layers.Conv1D(self.vocab_size, kernel_size=1, activation='softmax', padding='same')(x)
        
        return Model(noise_input, output, name="Generator")

    def _build_discriminator(self) -> tf.keras.Model:
        """
        Constructs the Discriminator (Critic) network. Takes a sequence of continuous 
        probabilities (or one-hot real data) and outputs a scalar critique.
        """
        seq_input = layers.Input(shape=(self.max_len, self.vocab_size))
        
        # Project input depth to target channel dimension
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(seq_input)

        # 5 Residual Blocks (No Batch Normalization applied to preserve Gradient Penalty)
        for _ in range(5):
            x = self._residual_block(x)

        x = layers.Flatten()(x)
        
        # Linear activation to output Wasserstein distance
        output = layers.Dense(1, activation='linear')(x)
        
        return Model(seq_input, output, name="Discriminator")

    def _encode_domains(self, domains: list) -> np.ndarray:
        """
        Converts a list of SLDs into one-hot encoded tensors.
        """
        encoded = np.zeros((len(domains), self.max_len, self.vocab_size), dtype=np.float32)
        
        for i, domain in enumerate(domains):
            for j, char in enumerate(domain):
                if j >= self.max_len:
                    break
                idx = self.char2idx.get(char, self.PAD)
                encoded[i, j, idx] = 1.0
            
            # Fill the rest of the sequence with PAD token
            for j in range(min(len(domain), self.max_len), self.max_len):
                encoded[i, j, self.PAD] = 1.0
                
        return encoded

    @tf.function
    def _train_step(self, real_domains: tf.Tensor):
        """
        A single training step enforcing WGAN-GP logic.
        """
        batch_size = tf.shape(real_domains)[0]

        # 1. Train the Discriminator (Critic)
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            
            with tf.GradientTape() as d_tape:
                fake_domains_probs = self.generator(noise, training=True)
                
                fake_logits = self.discriminator(fake_domains_probs, training=True)
                real_logits = self.discriminator(real_domains, training=True)

                # Wasserstein Distance Loss
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

                # Gradient Penalty Computation
                alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
                interpolated = alpha * real_domains + (1.0 - alpha) * fake_domains_probs
                
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    inter_logits = self.discriminator(interpolated, training=True)
                    
                grads = gp_tape.gradient(inter_logits, interpolated)
                norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
                gp = tf.reduce_mean((norm - 1.0) ** 2)

                # Total Discriminator Loss
                d_loss = d_cost + self.lambda_gp * gp

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # 2. Train the Generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as g_tape:
            fake_domains_probs = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake_domains_probs, training=True)
            
            # Generator wants to maximize the discriminator's output
            g_loss = -tf.reduce_mean(fake_logits)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return d_loss, g_loss

    def fit(self) -> None:
        """
        Trains the GAN model from scratch or loads pre-trained weights.
        """

        start_t = time.time()

        # Try loading cached weights first
        if os.path.exists(self.g_weights_path) and os.path.exists(self.d_weights_path):
            logger.info(f"[*] {self.name} found cached weights. Skipping training...")
            self.generator.load_weights(self.g_weights_path)
            self.discriminator.load_weights(self.d_weights_path)
            self.trained_from_scratch = False
            self.is_fitted = True
            self.training_time = time.time() - start_t
            return

        logger.info(f"[*] No cached weights found for {self.name}. Training...")
        
        # Load and preprocess benign domains (D1 partition)
        benign_full_domains = load_tranco(TRANCO_D1_BENIGN)
        benign_slds = [extract_sld(dom).lower() for dom in benign_full_domains]
        
        # Filter valid characters and length constraints
        valid_slds = []
        for sld in benign_slds:
            if self.min_len <= len(sld) <= self.max_len and all(c in self.char2idx for c in sld):
                valid_slds.append(sld)
                
        logger.info(f"[*] {self.name} filtered {len(valid_slds)} valid SLDs for training.")
        
        X_train = self._encode_domains(valid_slds)
        dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(10000).batch(self.batch_size, drop_remainder=True).repeat()
        dataset_iter = iter(dataset)

        # Training Loop
        for i in range(1, self.iterations + 1):
            real_batch = next(dataset_iter)
            d_loss, g_loss = self._train_step(real_batch)
            
            if i % 500 == 0:
                logger.info(f"[*] Iteration {i}/{self.iterations} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

        # Save weights after training
        os.makedirs(self.weights_dir, exist_ok=True)
        self.generator.save_weights(self.g_weights_path)
        self.discriminator.save_weights(self.d_weights_path)
        logger.info(f"[*] {self.name} training complete. Weights saved.")

        self.trained_from_scratch = True
        self.is_fitted = True
        self.training_time = time.time() - start_t

    def _validate_domain(self, sld: str) -> bool:
        """
        Validates if the generated SLD meets the length constraints and RFC format.
        """
        if not (self.min_len <= len(sld) <= self.max_len):
            return False
        
        # Starts and ends with alphanumeric, allows hyphens in between
        if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", sld):
            return False
            
        return True

    def generate_domain(self) -> str:
        """
        Generates a single valid domain name.
        """
        if not self.is_fitted:
            self.fit()
            
        max_attempts = 100  # SAFETY GUARD: bounded loop (a degenerate generator would otherwise hang)
        for _attempt in range(max_attempts):
            noise = tf.random.normal([1, self.latent_dim])
            probs = self.generator(noise, training=False).numpy()
            char_indices = np.argmax(probs, axis=-1)[0]
            
            domain_chars = []
            for idx in char_indices:
                if idx == self.PAD:
                    break
                domain_chars.append(self.idx2char[idx])
            
            sld = "".join(domain_chars)
            
            if self._validate_domain(sld):
                tld = random.choice(self.tlds)
                return f"{sld}.{tld}"

        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + "." + random.choice(self.tlds)

    def generate_domains(self, n: int) -> list:
        """
        Batch generation of valid domains. Converts continuous probabilities to discrete characters.
        """
        if not self.is_fitted:
            self.fit()

        generated_domains = []
        
        # SAFETY GUARD: bounded loop that scales with n (batches are capped at 500 candidates below);
        # allows up to 75% of candidates to be discarded before returning a partial list with a warning
        max_batches = max(200, 4 * -(-n // 500))
        for _batch in range(max_batches):
            if len(generated_domains) >= n:
                break
            needed = n - len(generated_domains)
            # Fetch a slightly larger batch size to account for invalid domains that get discarded
            batch_size = min(max(needed * 2, 10), 500) 
            
            noise = tf.random.normal([batch_size, self.latent_dim])
            probs = self.generator(noise, training=False).numpy()
            char_indices = np.argmax(probs, axis=-1)
            
            for indices in char_indices:
                domain_chars = []
                for idx in indices:
                    if idx == self.PAD:
                        break
                    domain_chars.append(self.idx2char[idx])
                
                sld = "".join(domain_chars)
                
                # Check generated SLD against constraints
                if self._validate_domain(sld):
                    tld = random.choice(self.tlds)
                    generated_domains.append(f"{sld}.{tld}")
                    
                    if len(generated_domains) == n:
                        break

        if len(generated_domains) < n:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(generated_domains)} of {n} valid domains after {max_batches} batches. Returning a partial list.")
        return generated_domains