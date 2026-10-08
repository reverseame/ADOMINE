import os
import re
import time
import logging
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras import layers, Model
from collections import Counter
from typing import List
from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco 

logger = logging.getLogger(__name__)

class KhaosModel(AdversarialModel):
    """
    Khaos model proposed by Yun et al. (2020) in "Khaos: An Adversarial Neural Network DGA
    With High Anti-Detection Ability".
    Generates stealthy domains by using an WGAN-GP approach, combined with n-grams.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Optimizers & Hyperparameters: Used Adam optimizer with learning rate 0.0001 
       (beta_1=0.5, beta_2=0.9). Batch size is set to 64.
    2. WGAN-GP Dynamics: The gradient penalty coefficient (lambda) is set to 10.0. 
       The discriminator (critic) is updated 5 times for every generator update.
    3. Weight Initialization: All Convolutional and Dense layers use a Random Normal 
       initializer (mean=0.0, stddev=0.02) to prevent early gradient explosion/vanishing.
    4. Normalization Layers: Batch Normalization is used in the Generator, but strictly 
       replaced with Layer Normalization in the Discriminator to preserve the validity 
       of the gradient penalty calculation.
    5. Architecture Dimensions: Zero-padding (padding='same') is applied to all Conv1D 
       layers to maintain the spatial dimension (ml) constant across ResBlocks. The 
       Generator's input is mapped via a Dense layer (with ReLU) and reshaped to (ml, 64).
       ml represents the maximum number of n-gram tokens (default 15, ensuring that even
       with 4-character n-grams the SLD stays within the 63-character limit).
    6. Dictionary Size (ds): Set exactly to 5001 (the top 5,000 frequent n-grams plus 
       1 special token reserved exclusively for right-padding).
    7. Continuous to Discrete Mapping: The Discriminator is designed to process 
       continuous floating-point matrices. Real domains are One-Hot encoded, while fake 
       domains use the continuous probabilities from the Generator's Softmax output.
    8. Decoding Mechanism: During inference, an Argmax operation is applied across the 
       last axis of the generator's output tensor to select the most probable n-gram 
       indices, ignoring the padding token to reconstruct the final domain string.
    9. Data Cleaning: Domains are strictly filtered using regex to comply with RFC 
       standards. For training, we use the domains in the 95th percentile of length. Generated 
       domains with < 3 characters are padded with random n-grams.
    10. TLD Addition: All generated SLDs are appended with '.com'.
    """
    def __init__(self, name="Khaos", ml=15, gp_weight=10.0, d_steps=5, epochs=100, batch_size=64, **kwargs):
        super().__init__(name, **kwargs)
        
        # Hyperparameters defined by the paper and theoretical assumptions
        self.ml = ml  # Maximum number of n-gram tokens (15 keeps SLDs under 63 characters even with 4-character n-grams)
        self.max_char_len = None  # Maximum character length
        self.ds = 5001  # Dictionary size (5000 n-grams + 1 padding token)
        self.latent_dim = 64
        self.gp_weight = gp_weight
        self.d_steps = d_steps
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # Adam optimizers for WGAN-GP
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        
        # Vocabulary structures
        self.ngram_to_index = {}
        self.index_to_ngram = {}
        self.pad_token_index = 5000  # Special token reserved for padding
        
        # Models will be built in fit() after ml is determined
        self.generator = None
        self.discriminator = None
        
        # File management paths for caching weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.gen_weights_path = os.path.join(self.weights_dir, "khaos_generator.weights.h5")
        self.disc_weights_path = os.path.join(self.weights_dir, "khaos_discriminator.weights.h5")
        self.dict_path = os.path.join(self.weights_dir, "khaos_dictionary.h5") 

    # Network Architecture (ResBlocks & GAN)
    
    def _resblock(self, x, filters, kernel_size, norm_type):
        """
        Pre-activation residual block.
        """
        shortcut = x
        
        # First convolutional layer
        if norm_type == 'layer':
            x = layers.LayerNormalization()(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer=self.weight_init)(x)
        
        # Second convolutional layer
        if norm_type == 'layer':
            x = layers.LayerNormalization()(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_initializer=self.weight_init)(x)
        
        # Residual connection
        return layers.Add()([shortcut, x])

    def _build_generator(self):
        """
        Builds the Generator network (uses BatchNormalization).
        """
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Dense layer and Reshape to match spatial dimensions
        x = layers.Dense(self.ml * 64, activation='relu', kernel_initializer=self.weight_init)(noise)
        x = layers.Reshape((self.ml, 64))(x)
        
        # 3 Residual Blocks
        for _ in range(3):
            x = self._resblock(x, filters=64, kernel_size=3, norm_type='batch')
            
        # Output convolutional layer and Softmax activation
        x = layers.Conv1D(self.ds, kernel_size=3, padding='same', activation='linear', kernel_initializer=self.weight_init)(x)
        out = layers.Softmax(axis=-1)(x)
        
        return Model(noise, out, name="Generator")

    def _build_discriminator(self):
        """
        Builds the Discriminator/Critic network (uses LayerNormalization).
        """
        img = layers.Input(shape=(self.ml, self.ds))
        
        # Initial convolutional layer
        x = layers.Conv1D(64, kernel_size=3, padding='same', kernel_initializer=self.weight_init)(img)
        
        # 3 Residual Blocks
        for _ in range(3):
            x = self._resblock(x, filters=64, kernel_size=3, norm_type='layer')
            
        x = layers.Flatten()(x)
        out = layers.Dense(1, kernel_initializer=self.weight_init)(x)
        
        return Model(img, out, name="Discriminator")

    # Loss Functions and Training Step
    
    @tf.function
    def _train_step(self, real_domains):
        """
        Custom WGAN-GP training step.
        """
        batch_size = tf.shape(real_domains)[0]
        
        # Train Discriminator (Critic)
        for _ in range(self.d_steps):
            # Sample seed from uniform distribution U(-500, 500)
            noise = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-500.0, maxval=500.0)
            
            with tf.GradientTape() as d_tape:
                fake_domains = self.generator(noise, training=True)
                
                fake_logits = self.discriminator(fake_domains, training=True)
                real_logits = self.discriminator(real_domains, training=True)
                
                # Wasserstein Loss
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                # Gradient Penalty calculation
                alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
                interpolated = alpha * real_domains + (1 - alpha) * fake_domains
                
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    interp_logits = self.discriminator(interpolated, training=True)
                    
                grads = gp_tape.gradient(interp_logits, interpolated)
                norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
                gp = tf.reduce_mean((norm - 1.0) ** 2)
                
                d_loss = d_cost + self.gp_weight * gp
                
            d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
            
        # Train Generator
        noise = tf.random.uniform(shape=(batch_size, self.latent_dim), minval=-500.0, maxval=500.0)
        with tf.GradientTape() as g_tape:
            fake_domains = self.generator(noise, training=True)
            gen_logits = self.discriminator(fake_domains, training=True)
            g_loss = -tf.reduce_mean(gen_logits)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss

    # Text Processing & Framework Integration
    
    def _clean_and_filter(self, domains):
        """
        Cleans the dataset and applies filtering rules based on length and regex.
        """
        cleaned = set()
        regex = re.compile(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$')
        for d in domains:
            # Extract SLD, convert to lowercase and strip whitespace
            d = str(d).lower().strip().split('.')[0]
            if self.max_char_len is not None and len(d) > self.max_char_len:
                continue
            if regex.match(d):
                cleaned.add(d)
        return list(cleaned)

    def _save_dictionary(self):
        """
        Serializes the n-grams dictionary and hyperparameters into an .h5 file.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        with h5py.File(self.dict_path, 'w') as f:
            # Save ngram_to_index as two parallel arrays
            ngrams = list(self.ngram_to_index.keys())
            indices = list(self.ngram_to_index.values())
            # Convert arrays into bytes
            f.create_dataset('ngrams', data=np.array(ngrams, dtype=h5py.string_dtype(encoding='utf-8')))
            f.create_dataset('ngram_indices', data=np.array(indices, dtype=np.int32))
            
            # Save index_to_ngram
            idx_keys = list(self.index_to_ngram.keys())
            idx_vals = list(self.index_to_ngram.values())
            f.create_dataset('idx_keys', data=np.array(idx_keys, dtype=np.int32))
            f.create_dataset('idx_vals', data=np.array(idx_vals, dtype=h5py.string_dtype(encoding='utf-8')))
            
            # Save hyperparameters
            f.attrs['pad_token_index'] = self.pad_token_index
            f.attrs['ml'] = self.ml
            f.attrs['max_char_len'] = self.max_char_len
            f.attrs['ds'] = self.ds

    def _load_dictionary(self):
        """
        Loads the n-grams dictionary and hyperparameters from an .h5 file.
        """
        with h5py.File(self.dict_path, 'r') as f:
            ngrams = [x.decode('utf-8') for x in f['ngrams'][:]]
            indices = f['ngram_indices'][:].tolist()
            self.ngram_to_index = dict(zip(ngrams, indices))
            
            idx_keys = f['idx_keys'][:].tolist()
            idx_vals = [x.decode('utf-8') for x in f['idx_vals'][:]]
            self.index_to_ngram = dict(zip(idx_keys, idx_vals))
            
            self.pad_token_index = int(f.attrs['pad_token_index'])
            self.ml = int(f.attrs['ml'])
            self.max_char_len = int(f.attrs['max_char_len']) if f.attrs['max_char_len'] != 0 else None
            self.ds = int(f.attrs['ds'])

    def _build_dictionary(self, domains):
        """
        Extracts the top 5000 frequent n-grams (n=1..4).
        """
        ngram_counts = Counter()
        for d in domains:
            length = len(d)
            for n in range(1, 5):
                for i in range(length - n + 1):
                    ngram_counts[d[i:i+n]] += 1
                    
        top_ngrams = [n for n, c in ngram_counts.most_common(5000)]
        if not top_ngrams:
            raise RuntimeError("No n-grams extracted from training domains. Dictionary is empty.")
        self.ngram_to_index = {ng: i for i, ng in enumerate(top_ngrams)}
        self.index_to_ngram = {i: ng for ng, i in self.ngram_to_index.items()}
        
        # Special token reserved for padding
        self.index_to_ngram[self.pad_token_index] = ""

    def _tokenize_forward(self, domain):
        """
        Forward Maximum Matching (FMM) tokenization.
        """
        tokens = []
        idx = 0
        while idx < len(domain):
            match = None
            for n in range(4, 0, -1):
                if idx + n <= len(domain):
                    ngram = domain[idx:idx+n]
                    if ngram in self.ngram_to_index:
                        match = ngram
                        break
            if match:
                tokens.append(self.ngram_to_index[match])
                idx += len(match)
            else:
                # Fallback: single character (1-gram) should always exist
                # but if not, skip
                if domain[idx] in self.ngram_to_index:
                    tokens.append(self.ngram_to_index[domain[idx]])
                idx += 1
        return tokens

    def _tokenize_backward(self, domain):
        """
        Backward Maximum Matching (BMM) tokenization.
        """
        tokens = []
        idx = len(domain)
        while idx > 0:
            match = None
            for n in range(4, 0, -1):
                if idx - n >= 0:
                    ngram = domain[idx-n:idx]
                    if ngram in self.ngram_to_index:
                        match = ngram
                        break
            if match:
                tokens.insert(0, self.ngram_to_index[match])
                idx -= len(match)
            else:
                # Fallback: single character
                if domain[idx-1] in self.ngram_to_index:
                    tokens.insert(0, self.ngram_to_index[domain[idx-1]])
                idx -= 1
        return tokens

    def _tokenize_domain(self, domain):
        """
        Tokenizes domain using Bi-direction Maximum Matching.
        Chooses the best result between FMM and BMM.
        """
        fwd_tokens = self._tokenize_forward(domain)
        bwd_tokens = self._tokenize_backward(domain)
        
        # Choose the one with fewer tokens
        if len(fwd_tokens) != len(bwd_tokens):
            tokens = fwd_tokens if len(fwd_tokens) < len(bwd_tokens) else bwd_tokens
        else:
            # If equal number of tokens and they are identical, choose either
            if fwd_tokens == bwd_tokens:
                tokens = fwd_tokens
            else:
                tokens = fwd_tokens
                
        # Right padding
        while len(tokens) < self.ml:
            tokens.append(self.pad_token_index)
            
        return tokens[:self.ml]

    def fit(self):
        """
        Trains the model from scratch using the D1 Benign split dataset,
        or loads cached weights and dictionary if they exist.
        """
        start_time = time.time()
        
        # Check for cached artifacts
        if (os.path.exists(self.gen_weights_path) and 
            os.path.exists(self.disc_weights_path) and 
            os.path.exists(self.dict_path)):
            logger.info(f"[*] Khaos found cached weights and dictionary. Skipping training.")
            self._load_dictionary()
            self.generator = self._build_generator()
            self.discriminator = self._build_discriminator()
            self.generator.load_weights(self.gen_weights_path)
            self.discriminator.load_weights(self.disc_weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            return

        logger.info(f"[*] No cached weights or dictionary found for Khaos. Training...")

        # No cache: train from scratch
        raw_domains = load_tranco(TRANCO_D1_BENIGN)
        if not raw_domains:
            raise RuntimeError("No domains loaded from Tranco D1 split. Training cannot proceed.")
        
        # Compute max character length from data (95th percentile, capped at 63)
        char_lengths = [len(d.split('.')[0]) for d in raw_domains]
        self.max_char_len = min(63, int(np.percentile(char_lengths, 95)))
            
        clean_domains = self._clean_and_filter(raw_domains)
        if not clean_domains:
            raise RuntimeError("After cleaning, no valid domains remain. Check filtering rules.")
        
        # Train/Val/Test Split (80/10/10)
        np.random.shuffle(clean_domains)
        train_size = int(len(clean_domains) * 0.8)
        train_domains = clean_domains[:train_size]
        
        self._build_dictionary(train_domains)
        if not self.ngram_to_index:
            raise RuntimeError("Dictionary is empty after building. Cannot train.")
        
        # Build models
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Keep token indices only and expand to one-hot per batch inside the
        # pipeline: the whole training set as a one-hot tensor (N x ml x ds floats)
        # does not fit in memory.
        X_train = np.array([self._tokenize_domain(d) for d in train_domains], dtype=np.int32)
        dataset = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .shuffle(10000)
            .batch(self.batch_size, drop_remainder=True)
            .map(lambda tokens: tf.cast(tf.one_hot(tokens, depth=self.ds), tf.float32))
        )
        
        for epoch in range(self.epochs):
            for batch in dataset:
                self._train_step(batch)
            logger.info(f"Epoch {epoch+1}/{self.epochs} completed.")
            
        # Save trained artifacts
        os.makedirs(self.weights_dir, exist_ok=True)
        self.generator.save_weights(self.gen_weights_path)
        self.discriminator.save_weights(self.disc_weights_path)
        self._save_dictionary()
        
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time

    def generate_domain(self) -> str:
        """
        Generates a single adversarial domain directly from the generator model.
        The SLD is truncated to 63 characters to comply with DNS label length limits
        and needs to have a minimun length of 3 characters.
        """
        if not self.is_fitted:
            self.fit()
            
        # Sample noise for a single instance from a uniform distribution U(-500, 500)
        noise = tf.random.uniform(shape=(1, self.latent_dim), minval=-500.0, maxval=500.0)
        
        # Predict probability distributions for the sequence
        prob_tensors = self.generator.predict(noise, verbose=0)
        
        # Extract the single generated sequence (batch index 0)
        seq = prob_tensors[0]
        
        # Apply argmax to get the most likely token indices in the matrix
        token_indices = np.argmax(seq, axis=-1)
        
        # Decode indices back to n-grams, omitting the padding token
        domain_str = "".join([
            self.index_to_ngram.get(idx, "") 
            for idx in token_indices 
            if idx != self.pad_token_index
        ])

        # Strip leading/trailing hyphens
        domain_str = domain_str.strip('-')

        # Pad with random n-grams to comply with the minimum length
        valid_upper_bound = len(self.ngram_to_index)
        while len(domain_str) < 3:
            rand_idx = np.random.randint(0, valid_upper_bound)
            rand_ngram = self.index_to_ngram.get(rand_idx, "")
            if rand_ngram:
                domain_str += rand_ngram
        
        # Enforce 63-character limit for SLD (DNS label maximum)
        if len(domain_str) > 63:
            domain_str = domain_str[:63]
        
        # Append TLD
        return domain_str + ".com"

    def generate_domains(self, n: int) -> List[str]:
        """
        Generates n adversarial domains directly from the generator model in a single batch, 
        more efficient. Each SLD is truncated to 63 characters to comply with DNS label length 
        limits and needs to have a minimun length of 3 characters.
        """
        if not self.is_fitted:
            self.fit()

        noise = tf.random.uniform((n, self.latent_dim), minval=-500.0, maxval=500.0)
        # Predict in chunks and keep only the argmax: the full probability tensor
        # (n x ml x ds floats) does not fit in memory for large n.
        chunk = 2048
        indices = np.concatenate([
            np.argmax(self.generator.predict(noise[i:i + chunk], verbose=0), axis=-1)
            for i in range(0, n, chunk)
        ], axis=0)
        domains = []
        valid_upper_bound = len(self.ngram_to_index)

        for seq in indices:
            domain = "".join([self.index_to_ngram.get(idx, "") for idx in seq if idx != self.pad_token_index])
            
            # Strip leading/trailing hyphens
            domain = domain.strip('-')

            # Pad with random n-grams to comply with the minimum length
            while len(domain) < 3:
                rand_idx = np.random.randint(0, valid_upper_bound)
                rand_ngram = self.index_to_ngram.get(rand_idx, "")
                if rand_ngram:
                    domain += rand_ngram

            # Enforce 63-character limit
            if len(domain) > 63:
                domain = domain[:63]
            domains.append(domain + ".com")  # Append TLD
        return domains