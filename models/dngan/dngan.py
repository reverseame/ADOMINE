import os
import time
import random
import logging
import re
from collections import deque
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, LeakyReLU, RepeatVector, Embedding, TimeDistributed
from tensorflow.keras.optimizers import Adam

from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco, load_malicious_seeds
from core.sld import extract_sld


logger = logging.getLogger(__name__)

class DnGANModel(AdversarialModel):
    """
    Implementation of the DnGAN model based on Cao et al. (2020).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Latent space: noise vector Z dimension set to 100.
    2. Architecture: generator and discriminator use two LSTM layers with 128 units each, 
    followed by a dense layer with 64 units and LeakyReLU (α=0.2). The generator outputs a
    softmax distribution over 128 ASCII characters (indices 0–127), not a sigmoid.
    3. Optimizer: Adam with learning rate 1e-4 and β₁=0.5.
    4. Batch size: 32 for adversarial training and pre-training.
    5. Training: discriminator is pre-trained for 5 epochs on a balanced set of 5,000 
    benign and 5,000 DGA domains. For training 100 epochs are assumed.
    6. Historical buffer: a FIFO buffer of capacity 10,000 stores past generator outputs; 
    each discriminator batch mixes 50% real, 25% new fakes, and 25% historical fakes.
    7. Continuous representation: generator outputs a softmax distribution, which is projected 
    to an embedding space via a fixed embedding matrix (shared with the real-data
    embedding) to feed the discriminator.
    8. Decoding: domains are sampled from the softmax distribution to encourage diversity. 
    Invalid characters (not a-z, 0-9, or '-') are replaced randomly, and hyphens
    are stripped from the ends. Length is chosen uniformly between 1 and max_len (30).
    9. Label smoothing: real labels are set to 0.9 instead of 1.0 to prevent the discriminator 
    from becoming overconfident and to mitigate mode collapse.
    """
    def __init__(self, name: str = "DnGAN", max_len: int = 30, **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Maximum length for the domains
        self.max_len = max_len
        
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]       

        self.epochs = 100                     # GAN training rounds
        self.g_steps = 7                      # generator steps per round
        self.d_steps = 5                      # discriminator steps per round
        self.batch_size = 32                  # batch size in the GAN phase
        self.latent_dim = 100                 # dimension of the noise vector Z
        self.lstm_units = 128                 # units in the LSTM layers
        self.dense_units = 64                 # intermediate dense layer units
        self.learning_rate = 0.0001           # learning rate of the Adam optimizer
        self.beta_1 = 0.5                     # beta_1 of the Adam optimizer
        self.leaky_alpha = 0.2                # slope of the LeakyReLU
        self.embedding_dim = 128              # output dimension of the embedding and the continuous matrix

        # Pre‑training parameters for the discriminator
        self.pretrain_epochs = 5
        self.pretrain_batch_size = 32    
        self.pretrain_samples_real = 5000
        self.pretrain_samples_dga = 5000

        # Historical FIFO buffer
        self.fifo_capacity = 10000

        # Reading DGA data (cut-off values)
        self.dga_total_samples = 100000

        # Batch size for mass generation (inference)
        self.gen_batch_size = 512

        self.generator = None
        self.discriminator = None
        self.gan = None
        self.pre_discriminator = None
        self.char_embedding = None
        self.embed_model = None          # auxiliary model to extract embeddings
        self.projector = None            # projects softmax distribution to embedding space

        # Path configuration for weights persistence
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_path_g = os.path.join(self.weights_dir, "generator.weights.h5")
        self.weights_path_d = os.path.join(self.weights_dir, "discriminator.weights.h5")

    def _clean_sld(self, sld: str) -> str:
        """
        Cleans the SLD by applying character filtering rules:
        - Keep only the ASCII range [33, 127].
        - Discard control characters [0, 32].
        - Trim hyphens at the beginning and end of the domain.
        """
        cleaned_chars = [c for c in sld if 33 <= ord(c) <= 127]
        cleaned_str = "".join(cleaned_chars)
        return cleaned_str.strip('-')

    def _encode_sld(self, sld: str) -> np.ndarray:
        """
        Encodes an SLD domain into a numerical vector:
        - Truncates if it exceeds the maximum length.
        - Applies post-padding with the value 0 if it is shorter.
        """
        cleaned = self._clean_sld(sld)
        truncated = cleaned[:self.max_len]
        encoded = [ord(c) for c in truncated]
        
        # Post-padding with 0
        while len(encoded) < self.max_len:
            encoded.append(0)
            
        return np.array(encoded, dtype=np.int32)

    def _decode_matrix(self, matrix: np.ndarray) -> str:
        """
        Decodes a softmax distribution (max_len, embedding_dim) by sampling
        from the distribution at each position. The length of the generated string
        is chosen randomly between 1 and self.max_len to mimic variable-length domains.
        Invalid characters are replaced with a random lowercase letter, and hyphens
        are stripped from the ends.
        """
        # Choose a random length between 1 and max_len (inclusive)
        target_len = random.randint(1, self.max_len)
        chars = []
        for i in range(target_len):
            probs = matrix[i]  # only use the first target_len positions
            idx = np.random.choice(self.embedding_dim, p=probs)
            if 33 <= idx <= 127:
                char = chr(idx)
            else:
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
            # Ensure character is valid: lowercase letter, digit, or hyphen
            if not re.match(r'[a-z0-9-]', char):
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
            chars.append(char)
        sld = ''.join(chars).strip('-')

        # Fallback if the string becomes empty after stripping hyphens
        if not sld:
            sld = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(1, self.max_len)))
        return sld

    def _build_models(self):
        """
        Builds the GAN. Compilation is deferred to fit() to avoid optimizer variable conflicts.
        """
        # 1. GENERATOR (Latent Space Z -> LSTM -> LSTM -> Dense -> Dense)
        # Outputs a softmax distribution over characters (ASCII indices)
        gen_input = Input(shape=(self.latent_dim,), name="Z_noise")
        x_g = RepeatVector(self.max_len)(gen_input)
        x_g = LSTM(self.lstm_units, return_sequences=True)(x_g)
        x_g = LSTM(self.lstm_units, return_sequences=True)(x_g)
        x_g = Dense(self.dense_units)(x_g)
        x_g = LeakyReLU(negative_slope=self.leaky_alpha)(x_g)
        # Softmax over the character dimension to produce a probability distribution
        gen_output = Dense(self.embedding_dim, activation='softmax', name="Char_Distribution")(x_g)
        self.generator = Model(gen_input, gen_output, name="Generator")


        # 2. DISCRIMINATOR (Continuous representation self.max_len x 128 -> LSTM -> LSTM -> Dense -> Dense -> Classification)
        # Takes continuous embedding vectors (max_len, embedding_dim)
        disc_input = Input(shape=(self.max_len, self.embedding_dim), name="Continuous_Input")
        x_d = LSTM(self.lstm_units, return_sequences=True)(disc_input)
        x_d = LSTM(self.lstm_units, return_sequences=False)(x_d)
        x_d = Dense(self.dense_units)(x_d)
        x_d = LeakyReLU(negative_slope=self.leaky_alpha)(x_d)
        disc_output = Dense(1, activation='sigmoid', name="Real_Fake_Class")(x_d)
        self.discriminator = Model(disc_input, disc_output, name="Discriminator")

        # 3. SHARED EMBEDDING LAYER FOR REAL DATA (Indexed)
        # This layer maps integer indices to continuous embeddings
        self.char_embedding = Embedding(input_dim=self.embedding_dim, output_dim=self.embedding_dim,
                                        name="char_embedding")

        # 4. PROJECTION LAYER: converts softmax distribution (sequence) to continuous representation
        # Applies a Dense layer (without bias) to each time step using TimeDistributed
        projection_input = Input(shape=(self.max_len, self.embedding_dim))
        projection_dense = TimeDistributed(Dense(self.embedding_dim, use_bias=False, name="projection_dense"))
        projected = projection_dense(projection_input)
        self.projector = Model(projection_input, projected, name="Projector")

        # 5. Model for pre-training the discriminator on indexed domains
        pre_input = Input(shape=(self.max_len,))
        embedded_real = self.char_embedding(pre_input)
        pre_disc_output = self.discriminator(embedded_real)
        self.pre_discriminator = Model(pre_input, pre_disc_output, name="Pre_Discriminator")

        # Auxiliary model to extract embeddings from indexed domains (same as pre_input -> embedded_real)
        self.embed_model = Model(pre_input, embedded_real, name="Embed_Model")

    def fit(self) -> None:
        """
        Trains the generator and discriminator. Implements pre-training and adversarial 
        optimization loop under a 7:5 ratio.
        """

        start_time = time.time()
        self._build_models()

        # Check for pre-existing weights to skip training
        if os.path.exists(self.weights_path_g) and os.path.exists(self.weights_path_d):
            logger.info(f"[*] {self.name} found cached weights. Skipping training...")
            self.generator.load_weights(self.weights_path_g)
            self.discriminator.load_weights(self.weights_path_d)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            return

        logger.info(f"[*] No cached weights found for {self.name}. Training...")

        # Load real and DGA samples to optimally initialize the Discriminator
        # 1. READ BENIGN DOMAINS from the D1 Tranco slice
        real_domains = load_tranco(TRANCO_D1_BENIGN)[:self.pretrain_samples_real]
        
        # 2. READ MALICIOUS DOMAINS
        agd_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py (TLD isolated below)
            seeds = load_malicious_seeds(total=self.dga_total_samples)
            agd_domains = [d.split('.')[0] for d in seeds if d.split('.')[0]]
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs: {e}")
            
        if not agd_domains:
            agd_domains = ["".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(10, 20))) for _ in range(len(real_domains))]

        dga_domains = agd_domains[:self.pretrain_samples_dga]

        # Extract SLD and perform numerical encoding
        real_slds = [extract_sld(d) for d in real_domains]
        dga_slds = [extract_sld(d) for d in dga_domains]

        X_real_idx = np.array([self._encode_sld(s) for s in real_slds])
        X_dga_idx = np.array([self._encode_sld(s) for s in dga_slds])

        # Balanced mix of sets for pre-training
        X_pre = np.vstack([X_real_idx, X_dga_idx])
        y_pre = np.hstack([np.ones(len(X_real_idx)), np.zeros(len(X_dga_idx))])

        shuffle_indices = np.arange(len(X_pre))
        np.random.shuffle(shuffle_indices)
        X_pre = X_pre[shuffle_indices]
        y_pre = y_pre[shuffle_indices]

        # DISCRIMINATOR PRE-TRAINING
        # Compile only the pre-discriminator for the pre-training phase
        opt_pre = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        self.pre_discriminator.compile(loss='binary_crossentropy', optimizer=opt_pre, metrics=['accuracy'])

        logger.info(f"[*] {self.name} discriminator pre-training phase ({self.pretrain_epochs} epochs)...")
        self.pre_discriminator.fit(X_pre, y_pre, epochs=self.pretrain_epochs,
                                   batch_size=self.pretrain_batch_size, verbose=0)

        # PREPARATION FOR GAN PHASE
        # Set the projector weights
        td_layer = self.projector.layers[1]  # TimeDistributed layer
        dense_layer = td_layer.layer        # Dense layer inside
        embedding_weights = self.char_embedding.get_weights()[0]  # shape (embedding_dim, embedding_dim)
        dense_layer.set_weights([embedding_weights])
        self.projector.trainable = False

        # Compile the discriminator and GAN with fresh optimizers
        opt_d = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        opt_g = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt_d, metrics=['accuracy'])

        # Build GAN: generator -> projector -> discriminator
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        gen_dist = self.generator(gan_input)
        gen_embed = self.projector(gen_dist)          # convert to continuous representation
        gan_output = self.discriminator(gen_embed)
        self.gan = Model(gan_input, gan_output, name="DnGAN_Combined")
        self.gan.compile(loss='binary_crossentropy', optimizer=opt_g)

        # Obtain the continuous embeddings of real domains (used as real samples during GAN training)
        real_embeddings = self.embed_model.predict(X_real_idx, verbose=0)

        # FIFO buffer with historical samples
        fifo_buffer = deque(maxlen=self.fifo_capacity)
        
        rounds = self.config.get("epochs", self.epochs)
        logger.info(f"[*] {self.name} starting alternating adversarial training for {rounds} rounds...")

        batch = self.batch_size
        half_batch = batch // 2
        quarter_batch = batch // 4
        # Label smoothing for real samples to prevent discriminator overconfidence
        real_label = 0.9

        for r in range(rounds):
            # A) GENERATOR UPDATE
            self.discriminator.trainable = False
            g_loss = 0.0
            for _ in range(self.g_steps):
                noise = np.random.normal(0, 1, (batch, self.latent_dim))
                # Use smoothed label for generator target
                g_loss = self.gan.train_on_batch(noise, np.ones((batch, 1)) * real_label)
                
                # Get continuous output (projected) and add it to the historical FIFO
                # We store the softmax distributions, not the projected embeddings
                gen_samples = self.generator.predict(noise, verbose=0)
                for sample in gen_samples:
                    fifo_buffer.append(sample)

            # B) DISCRIMINATOR UPDATE
            self.discriminator.trainable = True
            d_loss = [0.0]
            for _ in range(self.d_steps):
                # half_batch real samples
                real_sel = np.random.choice(len(real_embeddings), half_batch, replace=False)
                batch_real = real_embeddings[real_sel]

                # quarter_batch newly generated samples (project them)
                noise_new = np.random.normal(0, 1, (quarter_batch, self.latent_dim))
                batch_new_dist = self.generator.predict(noise_new, verbose=0)
                batch_new_fake = self.projector.predict(batch_new_dist, verbose=0)

                # quarter_batch random historical samples from FIFO (project them)
                if len(fifo_buffer) >= quarter_batch:
                    hist_samples = random.sample(list(fifo_buffer), quarter_batch)
                    hist_dist = np.array(hist_samples)
                    batch_hist_fake = self.projector.predict(hist_dist, verbose=0)
                else:
                    noise_fallback = np.random.normal(0, 1, (quarter_batch, self.latent_dim))
                    batch_fallback_dist = self.generator.predict(noise_fallback, verbose=0)
                    batch_hist_fake = self.projector.predict(batch_fallback_dist, verbose=0)

                # Consolidate training batch
                X_batch = np.vstack([batch_real, batch_new_fake, batch_hist_fake])
                y_batch = np.vstack([np.ones((half_batch, 1)) * real_label,
                                     np.zeros((half_batch, 1))])

                d_loss = self.discriminator.train_on_batch(X_batch, y_batch)

            if (r + 1) % 10 == 0 or r == 0:
                logger.info(f"  Round {r + 1}/{rounds} | G Loss: {g_loss:.4f} | D Loss: {d_loss[0]:.4f} | FIFO Size: {len(fifo_buffer)}")

        # 4. TRAINED WEIGHTS PERSISTENCE
        os.makedirs(self.weights_dir, exist_ok=True)
        self.generator.save_weights(self.weights_path_g)
        self.discriminator.save_weights(self.weights_path_d)
        
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time

    def generate_domain(self) -> str:
        """
        Generates a single valid adversarial domain.
        """
        if not self.is_fitted:
            self.fit()

        noise = np.random.normal(0, 1, (1, self.latent_dim))
        pred_dist = self.generator.predict(noise, verbose=0)[0]
        sld = self._decode_matrix(pred_dist)
        tld = random.choice(self.tlds)
        return f"{sld}.{tld}"

    def generate_domains(self, n: int) -> List[str]:
        """
        Generates 'n' valid domains using optimized parallel inference.
        """
        if not self.is_fitted:
            self.fit()

        domains = []
        batch_size = self.gen_batch_size
        for i in range(0, n, batch_size):
            chunk_size = min(batch_size, n - i)
            noise = np.random.normal(0, 1, (chunk_size, self.latent_dim))
            dists = self.generator.predict(noise, verbose=0)
            for dist in dists:
                sld = self._decode_matrix(dist)
                tld = random.choice(self.tlds)
                domains.append(f"{sld}.{tld}")
        return domains