import os
import time
import random
import logging
import re
import numpy as np
import h5py
from collections import Counter
from typing import List, Dict, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical


from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import load_tranco, TRANCO_D1_BENIGN

logger = logging.getLogger(__name__)

class SelfAttention1D(layers.Layer):
    """
    1D Self-Attention layer for SADGA. Calculates attention maps and applies a 
    learnable scale parameter (gamma).
    """
    def __init__(self, channels: int, **kwargs):
        super(SelfAttention1D, self).__init__(**kwargs)
        self.channels = channels
        # f(x) and g(x) reduce channels to C/8 = 16
        self.f = layers.Conv1D(channels // 8, kernel_size=1, padding='same')
        self.g = layers.Conv1D(channels // 8, kernel_size=1, padding='same')
        # h(x) reduces to C/2 = 64
        self.h = layers.Conv1D(channels // 2, kernel_size=1, padding='same')
        # v(o) restores channels to C = 128
        self.v = layers.Conv1D(channels, kernel_size=1, padding='same')

    def build(self, input_shape):
        # Gamma is a learnable scalar initialized to 0
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        super(SelfAttention1D, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, L, C)
        f_x = self.f(x)  # (batch, L, C/8)
        g_x = self.g(x)  # (batch, L, C/8)
        h_x = self.h(x)  # (batch, L, C/2)

        # Attention map: s = f(x) * g(x)^T
        s = tf.matmul(f_x, g_x, transpose_b=True)  # (batch, L, L)
        beta = tf.nn.softmax(s, axis=-1)           # (batch, L, L)

        # Output projection: o = beta * h(x)
        o = tf.matmul(beta, h_x)                   # (batch, L, C/2)
        v_o = self.v(o)                            # (batch, L, C)

        # Residual connection scaled by gamma
        return x + self.gamma * v_o


class WGANGP(tf.keras.Model):
    """
    Custom WGAN-GP training loop designed to maintain differentiability by 
    handling continuous Softmax probabilities natively in the Discriminator.
    """
    def __init__(self, generator, discriminator, latent_dim=128, n_critic=5, gp_weight=10.0):
        super(WGANGP, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super(WGANGP, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """
        Calculates the gradient penalty.
        """
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # Train Discriminator (n_critic times)
        for _ in range(self.n_critic):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                fake_data = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_data, training=True)
                real_logits = self.discriminator(real_data, training=True)
                
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                d_loss = d_cost + gp * self.gp_weight

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train Generator (1 time)
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_data = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_logits)
            
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}


class SADGAModel(AdversarialModel):
    """
    Implementation of SADGA (Self-Attention GAN based DGA) model, proposed by Luo et al. (2026).

    REPRODUCIBILITY ASSUMPTIONS:
    1. External Data: Lean Domain Search (LDS) frequent words are locally loaded from 
    dataset/fixes.md, obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103.
    If the file is missing, it falls back to extracting the most frequent 5-7 character words from the 
    training dataset.
    2. BiMM: chooses the tokenization with fewer tokens, between FMM and BMM. In case of a tie, 
    it favors FMM.
    3. Architecture & Latent Space: Assumes a 128-dimensional standard normal noise vector, 
    reshaped to a 3D tensor (Batch, 20, 128) via a Dense layer. Base channels are set to 128, 
    using Conv1D with kernel_size=3 and padding='same'.
    4. Self-Attention: Applies channel reduction for computational viability (f(x) and g(x) 
    reduce to C/8, h(x) to C/2). The learnable scalar gamma is initialized to 0.
    5. Differentiable Training: The Generator outputs continuous post-Softmax probabilities 
    during training. The argmax function is used strictly during inference.
    6. Sequence Length & Padding: Maximum sequence length of 20 tokens. Index 0 is reserved 
    for the <PAD> token, increasing the vocabulary size to 6674.
    7. WGAN-GP Hyperparameters: Assumes Adam optimizer (learning rate = 0.0001, beta 1 = 0.0, 
    beta 2 = 0.9), batch size 64, 100 epochs, and 5 discriminator updates per generator update.
    8. Discriminator Output: Incorporates a GlobalAveragePooling1D layer before the final Dense(1) layer.
    9. Filtering Rules: Keeps generated domains with length >= 7 that comply with RFC constraints.
    10. Training Data: Utilizes the top 10,000 domains from the Tranco list (including TLDs).
    """
    def __init__(self, name: str = "SADGA", **kwargs):
        super().__init__(name=name, **kwargs)

        # Hyperparameters
        self.latent_dim = 128
        self.max_len = 20
        self.channels = 128
        self.vocab_size = 6674  # 6673 tokens + 1 <PAD> at index 0
        self.batch_size = 64
        self.epochs = 100
        self.n_critic = 5   # Discriminator updates per Generator update
        self.gp_weight = 10.0
        self.learning_rate = 0.0001
        self.beta_1 = 0.0
        self.beta_2 = 0.9

        self.generator = None
        self.discriminator = None
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

        # Caching paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.g_weights_path = os.path.join(self.weights_dir, "sadga_generator.weights.h5")
        self.vocab_path = os.path.join(self.weights_dir, "sadga_vocab.h5")

    def _build_generator(self) -> tf.keras.Model:
        """
        Builds the Generator using Conv1D, ResBlocks, and Self-Attention.
        Output is a continuous Softmax probability matrix.
        """
        inputs = layers.Input(shape=(self.latent_dim,))
        
        # Project noise to C * max_len and reshape (Dense)
        x = layers.Dense(self.channels * self.max_len, activation='relu')(inputs)
        x = layers.Reshape((self.max_len, self.channels))(x)
        
        # Res-Block
        shortcut = x
        x = layers.ReLU()(x)
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x)
        x = layers.Add()([shortcut, x])
        
        # Self-Attn
        x = SelfAttention1D(self.channels)(x)
        
        # Output layer matching vocab size (Conv1d -> Softmax)
        outputs = layers.Conv1D(self.vocab_size, kernel_size=3, padding='same', activation='softmax')(x)
        
        return models.Model(inputs, outputs, name="SADGA_Generator")

    def _build_discriminator(self) -> tf.keras.Model:
        """
        Builds the Discriminator receiving continuous matrices (Batch, L, Vocab).
        Output is an unconstrained scalar (Wasserstein score).
        """
        inputs = layers.Input(shape=(self.max_len, self.vocab_size))
        
        # Initial Conv1D to map vocab dimensions to channel dimensions
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(inputs)
        
        # Self-Attn
        x = SelfAttention1D(self.channels)(x)
        
        # Res-Block
        shortcut = x
        x = layers.ReLU()(x)
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(self.channels, kernel_size=3, padding='same')(x)
        x = layers.Add()([shortcut, x])
        
        # Global collapse and final scoring (Dense)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        return models.Model(inputs, outputs, name="SADGA_Discriminator")

    def _build_sadict(self, domains: List[str], lds_words: Optional[List[str]] = None):
        """
        Extracts n-grams and high-frequency words to build SADict.
        Filters words to keep only those with length <= 7.
        """
        logger.info(f"[*] Building SADict...")
        
        # 1. Extract top 5000 n-grams (1-gram to 4-gram) from domains
        ngram_counter = Counter()
        for d in domains:
            for n in range(1, 5):
                for i in range(len(d) - n + 1):
                    ngram_counter[d[i:i+n]] += 1
                    
        top_ngrams = [item[0] for item in ngram_counter.most_common(5000)]
        
        # 2. Extract high-frequency words: use LDS if provided, else try to load them, else fallback;
        # filtering words to length <= 7
        if lds_words is None:
            lds_words = self._load_lds_words()
        if lds_words:
            # Filter words to length <= 7
            top_words = [w for w in lds_words if len(w) <= 7][:5000]
        else:
            # Fallback: most frequent words (length > 4 and <= 7) from training data
            logger.info(f"[*] Most frequent words couldn't be extracted. Using fallback.")
            word_counter = Counter(d for d in domains if 4 < len(d) <= 7)
            top_words = [item[0] for item in word_counter.most_common(5000)]

        # 3. Combine and deduplicate
        combined_elements = list(dict.fromkeys(top_ngrams + top_words))[:self.vocab_size - 1]

        # 4. Sort by overall frequency in training dataset
        total_counter = Counter()
        for d in domains:
            for elem in combined_elements:
                if elem in d:
                    total_counter[elem] += 1

        sorted_elements = sorted(combined_elements, key=lambda x: total_counter[x], reverse=True)
        
        # Initialize mappings with zero reserved for padding
        self.vocab = {"<PAD>": 0}
        self.inv_vocab = {0: "<PAD>"}
        
        # Populate the translation dictionaries
        for idx, token in enumerate(sorted_elements, start=1):
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

        logger.info(f"[*] SADict built with {len(self.vocab)} tokens.")

        # Cache vocabulary
        with h5py.File(self.vocab_path, 'w') as f:
            keys = list(self.vocab.keys())
            vals = list(self.vocab.values())
            f.create_dataset('keys', data=np.array(keys, dtype=h5py.string_dtype(encoding='utf-8')))
            f.create_dataset('values', data=np.array(vals, dtype=np.int32))

    def _load_lds_words(self) -> Optional[List[str]]:
        """
        Load the top 5000 prefixes/suffixes from Lean Domain Search (LDS).
        Expects a file 'dataset/fixes.md' with one token per line prefixed by '* '.
        Returns a list of the alphabetic parts (e.g., 'aa') or None if the file is not found.
        """
        lds_path = os.path.join("dataset", "fixes.md")
        if not os.path.exists(lds_path):
            logger.warning(
                f"[!] LDS word list not found at {lds_path}. "
                "Using fallback (most frequent words from training data). "
                "To follow the paper, place the LDS list (obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103) there."
            )
            return None

        words = []
        with open(lds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip heading lines
                if line.startswith("#"):
                    continue
                # Only process the lines that start with "* "
                if not line.startswith("* "):
                    continue
                # Extract token after "* "
                token = line[2:].strip()
                # Discard hyphens (at the beginning or the end)
                clean = token.replace("-", "")
                if clean:
                    words.append(clean)
                if len(words) >= 5000:
                    break
        return words

    def _fmm(self, domain: str) -> List[int]:
        """
        Forward Maximum Matching (FMM) tokenization.
        """
        tokens = []
        i = 0
        while i < len(domain):
            match_found = False
            # Check up to 7 chars ahead to match the vocabulary building logic
            for window in range(7, 0, -1):
                if i + window <= len(domain):
                    subword = domain[i:i+window]
                    if subword in self.vocab:
                        tokens.append(self.vocab[subword])
                        i += window
                        match_found = True
                        break
            if not match_found:
                # If character completely unknown (rare), ignore or pad
                i += 1
        return tokens

    def _bmm(self, domain: str) -> List[int]:
        """
        Backward Maximum Matching (BMM) tokenization.
        """
        tokens = []
        i = len(domain)
        while i > 0:
            match_found = False
            # Check up to 7 chars backward to match the vocabulary building logic
            for window in range(7, 0, -1):
                if i - window >= 0:
                    subword = domain[i-window:i]
                    if subword in self.vocab:
                        tokens.insert(0, self.vocab[subword])
                        i -= window
                        match_found = True
                        break
            if not match_found:
                # If character completely unknown (rare), ignore or pad
                i -= 1
        return tokens

    def _tokenize(self, domain: str) -> List[int]:
        """
        Bidirectional Maximum Matching (BiMM) tokenization.
        """
        fmm_tokens = self._fmm(domain)
        bmm_tokens = self._bmm(domain)
        
        # BiMM logic: juxtapose the two outcomes and derive the final result.
        # Selects the one with fewer tokens; in case of a tie, favors FMM.
        if len(bmm_tokens) < len(fmm_tokens):
            tokens = bmm_tokens
        else:
            tokens = fmm_tokens
                
        # Truncate and post-pad with 0s to max_len
        tokens = tokens[:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return tokens

    def fit(self, lds_words: Optional[List[str]] = None) -> None:
        """
        Trains SADGA or loads pre-trained weights and vocabulary from .h5 files.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        start_t = time.time()
        
        # Caching logic
        if os.path.exists(self.g_weights_path) and os.path.exists(self.vocab_path):
            logger.info(f"[*] {self.name} found cached weights and vocab. Skipping training.")
            with h5py.File(self.vocab_path, 'r') as f:
                keys = [k.decode('utf-8') for k in f['keys'][:]]
                vals = f['values'][:].tolist()
                self.vocab = dict(zip(keys, vals))
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
                
            self.generator = self._build_generator()
            self.generator.load_weights(self.g_weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        logger.info(f"[*] No cached weights or vocab found for {self.name}. Training...")
        
        # Load Tranco D1 and build vocab
        domains = load_tranco(TRANCO_D1_BENIGN)[:10000] # Limit to 10k as specified in the paper
        self._build_sadict(domains, lds_words=lds_words)
        
        # Encode dataset and yield one-hot matrices
        def data_generator():
            for dom in domains:
                seq = self._tokenize(dom)
                # Convert to One-Hot representation (continuous representation equivalent)
                yield to_categorical(seq, num_classes=self.vocab_size)
                
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=tf.TensorSpec(shape=(self.max_len, self.vocab_size), dtype=tf.float32)
        )
        dataset = dataset.shuffle(10000).batch(self.batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        wgan = WGANGP(
            generator=self.generator,
            discriminator=self.discriminator,
            latent_dim=self.latent_dim,
            n_critic=self.n_critic,
            gp_weight=self.gp_weight
        )
        
        wgan.compile(
            d_optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2),
            g_optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        )
        
        logger.info(f"[*] {self.name} fitting WGAN-GP model...")
        steps_per_epoch = len(domains) // self.batch_size
        wgan.fit(dataset, epochs=self.epochs, steps_per_epoch=steps_per_epoch)
        
        # Save artifacts
        self.generator.save_weights(self.g_weights_path)

        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def generate_domain(self) -> str:
        """
        Generate a single strictly validated domain dynamically. Filters invalid RFC outputs.
        Applies mathematical Argmax during inference to collapse probabilities.
        """
        if not self.is_fitted:
            self.fit()
            
        max_attempts = 100  # SAFETY GUARD: bounded loop (a degenerate generator would otherwise hang)
        for _attempt in range(max_attempts):
            # Generate noise
            noise = tf.random.normal((1, self.latent_dim))
            
            # Predict continuous probabilities (1, L, Vocab)
            prob_matrix = self.generator(noise, training=False)
            
            # Non-differentiable Argmax exclusively for inference
            token_matrix = tf.argmax(prob_matrix, axis=-1).numpy()
            
            # Extract the generated sequence
            seq = token_matrix[0]
            
            domain_parts = []
            for token_id in seq:
                # Break upon first <PAD> token
                if token_id == 0:
                    break
                domain_parts.append(self.inv_vocab.get(token_id, ""))
                
            raw_domain = "".join(domain_parts)
            
            # Filter strictly by RFC constraints
            filtered = re.sub(r'[^a-z0-9-]', '', raw_domain.lower())
            
            # Discard if length <= 6 or starts/ends with hyphen
            if len(filtered) > 6 and not filtered.startswith('-') and not filtered.endswith('-'):
                return f"{filtered}.{sample_tld()}"

        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"

    def generate_domains(self, n: int) -> List[str]:
        """
        Generates n domains dynamically. Filters invalid RFC outputs.
        Applies mathematical Argmax during inference to collapse probabilities.
        """
        if not self.is_fitted:
            self.fit()

        generated_domains = []
        target_count = n
        
        max_batches = 200  # SAFETY GUARD: bounded loop; returns a partial list with a warning if the generator is degenerate
        for _batch in range(max_batches):
            if len(generated_domains) >= target_count:
                break
            # Oversample to account for RFC and length filtering discards
            batch_size = min(1000, (target_count - len(generated_domains)) * 3)
            noise = tf.random.normal((batch_size, self.latent_dim))
            
            # Predict continuous probabilities (B, L, Vocab)
            prob_matrix = self.generator(noise, training=False)
            
            # Non-differentiable Argmax exclusively for inference
            token_matrix = tf.argmax(prob_matrix, axis=-1).numpy()
            
            for seq in token_matrix:
                domain_parts = []
                for token_id in seq:
                    # Break upon first <PAD> token
                    if token_id == 0:
                        break
                    domain_parts.append(self.inv_vocab.get(token_id, ""))
                    
                raw_domain = "".join(domain_parts)
                
                # Filter strictly by RFC constraints
                filtered = re.sub(r'[^a-z0-9-]', '', raw_domain.lower())
                
                # Discard if length <= 6 or starts/ends with hyphen
                if len(filtered) > 6 and not filtered.startswith('-') and not filtered.endswith('-'):
                    generated_domains.append(f"{filtered}.{sample_tld()}")
                    if len(generated_domains) == target_count:
                        break
                        
        if len(generated_domains) < target_count:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(generated_domains)} of {target_count} valid domains after {max_batches} batches. Returning a partial list.")
        return generated_domains