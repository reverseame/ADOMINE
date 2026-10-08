import os
import time
import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, initializers
import h5py
from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import load_tranco, TRANCO_D1_BENIGN

logger = logging.getLogger(__name__)

class GCNNLayer(layers.Layer):
    """
    Gated Convolutional Neural Network (GCNN) layer with residual skip connection.
    Implements O = E + ((E * W + b) * sigmoid(E * V + c))
    """
    def __init__(self, filters: int = 128, kernel_size: int = 3, **kwargs):
        super(GCNNLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        init = initializers.GlorotNormal()
        
        # Parallel convolutions for the gating mechanism
        self.conv_w = layers.Conv1D(filters, kernel_size, padding="same", kernel_initializer=init)
        self.conv_v = layers.Conv1D(filters, kernel_size, padding="same", kernel_initializer=init)

    def call(self, inputs):
        # Linear feature extraction
        A = self.conv_w(inputs)
        # Gating control via sigmoid activation
        B = self.conv_v(inputs)
        gated = A * tf.math.sigmoid(B)
        
        # Apply residual connection to mitigate vanishing gradients
        return inputs + gated

class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a domain.
    Implements the Reparameterization Trick: z = μ + σ * ε
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample random noise from standard normal distribution
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Transform noise using the learned mean and variance
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class GWDGAVAE(models.Model):
    """
    Variational Autoencoder configured with ELBO and KL Annealing mechanism.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(GWDGAVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Beta parameter controls the weight of the KL divergence loss
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=False):
        # Required for Keras model subclassing when using validation_split
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_loss(self, data):
        """
        Centralized loss computation for train and test steps.
        """
        # 1. Forward pass through encoder to get latent space parameters
        z_mean, z_log_var, z = self.encoder(data)
        
        # 2. Reconstruct the input data from the sampled latent vector
        reconstruction = self.decoder(z)

        # 3. Calculate reconstruction loss (sum over sequence length, mean over batch)
        scce = tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction)
        loss_reconstruction = tf.reduce_mean(tf.reduce_sum(scce, axis=1))

        # 4. Calculate KL divergence
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # 5. Combine losses using the annealing beta weight
        total_loss = loss_reconstruction + self.beta * kl_loss
        return total_loss, loss_reconstruction, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, loss_reconstruction, kl_loss = self._compute_loss(data)

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "loss": total_loss, 
            "reconstruction_loss": loss_reconstruction, 
            "kl_loss": kl_loss
        }

    def test_step(self, data):
        # Required for Keras validation during model.fit()
        total_loss, loss_reconstruction, kl_loss = self._compute_loss(data)
        return {
            "loss": total_loss, 
            "reconstruction_loss": loss_reconstruction, 
            "kl_loss": kl_loss
        }

class KLAnnealingCallback(callbacks.Callback):
    """
    Linearly increases beta from 0.0 to 1.0 over the first 10 epochs.
    """
    def __init__(self, target_epochs=10):
        super().__init__()
        self.target_epochs = target_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate new beta and ensure it caps at 1.0
        new_beta = min(1.0, epoch / self.target_epochs)
        tf.keras.backend.set_value(self.model.beta, new_beta)

class GWDGAModel(AdversarialModel):
    """
    GWDGA model proposed by Shu et al. (2022), based on statistical language models, GCNN and VAE.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Training Hyperparameters: Adam optimizer (lr=0.001), batch size of 128, maximum 50 
    epochs, 20% validation split, and Early Stopping (patience=5).
    2. Reconstruction Loss: Sparse Categorical Cross-Entropy is utilized for the VAE 
    reconstruction loss.
    3. KL Annealing: Linear annealing of the KL divergence weight (β) from 0.0 to 1.0
    over the first 10 epochs to prevent posterior collapse.  
    4. Reparameterization Trick: The encoder's variance layer predicts log σ² instead
    of raw variance.
    5. GCNN Architecture: Convolutional layers use 128 filters with padding="same", followed
    by a GlobalAveragePooling1D layer.  
    6. Vocabulary & Padding: Index 0 is strictly reserved for <PAD>, aligning perfectly with 
    the 7487 output dimensions of the Softmax layer.  
    7. Decoding Strategy: Deterministic generation using Greedy Search (argmax), without 
    stochastic temperature sampling.  
    8. BiMM Tokenizer: Forward Maximum Matching (FMM) is strictly prioritized over BMM in 
    tie-breaking scenarios.  
    9. Weights Initialization: Glorot/Xavier Normal initialization for Conv1D and Dense layers;
    random uniform/normal for the Embedding layer.  
    10. External Data: Lean Domain Search (LDS) frequent words are locally loaded from 
    dataset/fixes.md, obtained from https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103.  
    11. Output Filtering: Generated subdomains are strictly constrained to lengths between 5 and 
    63 characters. 
    """
    def __init__(self, name: str = "GWDGA", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.vocab_size = 7487 # GWDict size after combining n-gram and word lists
        self.max_length = 75
        self.embedding_dim = 128
        self.batch_size = 128
        self.epochs = 50
        self.learning_rate = 0.001
        
        self.word2idx = {}
        self.idx2word = {}
        self.vae_model = None
        self.encoder = None
        self.decoder = None

        # Cache directory and paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.vocab_path = os.path.join(self.weights_dir, "vocab.h5")
        self.weights_path = os.path.join(self.weights_dir, "model.weights.h5")

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

    def _save_cache(self):
        """
        Saves the vocabulary and model weights to disk.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
                
        # Ensure model is built before saving weights
        if not self.vae_model.built:
            self.vae_model(tf.zeros((1, self.max_length)))
        
        # Save vocabulary
        with h5py.File(self.vocab_path, 'w') as f:
            # Save word2idx as two arrays: words (strings) and indices (int)
            words = list(self.word2idx.keys())
            indices = list(self.word2idx.values())
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('words', data=np.array(words, dtype=object), dtype=dt)
            f.create_dataset('indices', data=np.array(indices, dtype=np.int32))
            # Save idx2word
            idx_keys = list(self.idx2word.keys())
            idx_vals = list(self.idx2word.values())
            f.create_dataset('idx_keys', data=np.array(idx_keys, dtype=np.int32))
            f.create_dataset('idx_vals', data=np.array(idx_vals, dtype=object), dtype=dt)
            
        # Save the trained neural network weights
        self.vae_model.save_weights(self.weights_path)

    def _load_cache(self) -> None:
        """
        Attempts to load vocabulary and model weights from disk.
        """
        
        # Load vocabulary from HDF5
        with h5py.File(self.vocab_path, 'r') as f:
            words = [x.decode('utf-8') if isinstance(x, bytes) else x for x in f['words'][:]]
            indices = f['indices'][:].tolist()
            self.word2idx = dict(zip(words, indices))
            
            idx_keys = f['idx_keys'][:].tolist()
            idx_vals = [x.decode('utf-8') if isinstance(x, bytes) else x for x in f['idx_vals'][:]]
            self.idx2word = dict(zip(idx_keys, idx_vals))
        
        # Reconstruct the model graph before loading weights
        self._build_architecture()
        
        # Initialize model graph to allow loading weights in subclassed model
        dummy_input = tf.zeros((1, self.max_length), dtype=tf.int32)
        self.vae_model(dummy_input)
        
        self.vae_model.load_weights(self.weights_path)

    def _build_gwdict(self, domains: List[str], lds_words: Optional[List[str]] = None):
        """
        Extracts n-grams and high-frequency words to build GWDict.
        """
        logger.info(f"[*] Building GWDict...")
        
        # Extract SLDs from full domains
        slds = [d.split('.')[0] for d in domains]
        
        # 1. Extract top 5000 n-grams (1-gram to 4-gram) from SLDs
        ngram_counter = Counter()
        for sld in slds:
            for n in range(1, 5):
                for i in range(len(sld) - n + 1):
                    ngram_counter[sld[i:i+n]] += 1
                    
        top_ngrams = [item[0] for item in ngram_counter.most_common(5000)]
        
        # 2. Extract high-frequency words: use LDS if provided, else try to load, else fallback
        if lds_words is None:
            lds_words = self._load_lds_words()
        if lds_words:
            top_words = lds_words[:5000]
        else:
            # Fallback: most frequent words (length > 4) from training data
            logger.info(f"[*] Most frequent words couldn't be extracted. Using fallback.")
            word_counter = Counter(d for d in slds if len(d) > 4)
            top_words = [item[0] for item in word_counter.most_common(5000)]

        # 3. Combine and deduplicate
        combined_elements = list(dict.fromkeys(top_ngrams + top_words))[:self.vocab_size - 1]

        # 4. Sort by overall frequency in training dataset
        total_counter = Counter()
        for sld in slds:
            for elem in combined_elements:
                if elem in sld:
                    total_counter[elem] += 1

        sorted_elements = sorted(combined_elements, key=lambda x: total_counter[x], reverse=True)
        
        # Initialize mappings with zero reserved for padding
        self.word2idx = {"<PAD>": 0}
        self.idx2word = {0: "<PAD>"}
        
        # Populate the translation dictionaries
        for idx, token in enumerate(sorted_elements, start=1):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            
        logger.info(f"[*] GWDict built with {len(self.word2idx)} tokens.")

    def _bimm_segmentation(self, domain: str) -> List[str]:
        """
        Bi-direction Maximum Matching Algorithm (BiMM) for tokenization.
        """
        
        # Sub-function for Forward Maximum Matching
        def fmm(d: str) -> List[str]:
            res = []
            while d:
                for i in range(len(d), 0, -1):
                    sub = d[:i]
                    if sub in self.word2idx or i == 1:
                        res.append(sub)
                        d = d[i:]
                        break
            return res

        # Sub-function for Backward Maximum Matching
        def bmm(d: str) -> List[str]:
            res = []
            while d:
                for i in range(0, len(d)):
                    sub = d[i:]
                    if sub in self.word2idx or i == len(d) - 1:
                        res.insert(0, sub)
                        d = d[:i]
                        break
            return res

        fmm_res = fmm(domain)
        bmm_res = bmm(domain)
        
        # Count unassigned single characters in both segmentations
        fmm_single = sum(1 for w in fmm_res if len(w) == 1)
        bmm_single = sum(1 for w in bmm_res if len(w) == 1)
        
        # Tie-breaking logic to select the most optimal segmentation
        if len(fmm_res) < len(bmm_res): return fmm_res
        if len(bmm_res) < len(fmm_res): return bmm_res
        if fmm_single < bmm_single: return fmm_res
        if bmm_single < fmm_single: return bmm_res
        
        # Default fallback
        return fmm_res

    def _tokenize_and_pad(self, domains: List[str]) -> np.ndarray:
        """
        Segments domains and pads them to fixed length L=75.
        """
        tokenized = []
        for d in domains:
            # Extract SLD
            sld = d.split('.')[0]
            # Segment the SLD into known tokens
            segments = self._bimm_segmentation(sld)
            # Map tokens to numerical IDs
            seq = [self.word2idx.get(w, 0) for w in segments]
            
            # Enforce maximum sequence length
            seq = seq[:self.max_length]
            # Pad with zeroes if the sequence is too short
            seq += [0] * (self.max_length - len(seq))
            tokenized.append(seq)
            
        return np.array(tokenized, dtype=np.int32)

    def _build_architecture(self):
        """
        Constructs the Encoder and Decoder architecture.
        """
        init = initializers.GlorotNormal()
        
        # ENCODER
        encoder_inputs = layers.Input(shape=(self.max_length,))
        # Map IDs to continuous vectors
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, embeddings_initializer=initializers.RandomUniform())(encoder_inputs)
        # Apply gated convolutions for sequence context extraction
        x = GCNNLayer(filters=128, kernel_size=3)(x)
        x = GCNNLayer(filters=128, kernel_size=3)(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Generate latent distribution parameters
        z_mean = layers.Dense(128, activation='linear', kernel_initializer=init, name="dense_mean")(x)
        z_log_var = layers.Dense(128, activation='linear', kernel_initializer=init, name="dense_var")(x)
        
        # Sample latent vector z
        z = Sampling()([z_mean, z_log_var])
        self.encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # DECODER
        latent_inputs = layers.Input(shape=(128,))
        # Project latent vector back to sequence dimensions
        x = layers.Dense(self.max_length * 128, activation="relu", kernel_initializer=init, name="dense_1")(latent_inputs)
        x = layers.Reshape((self.max_length, 128))(x)
        
        # Reconstruct structural context
        x = GCNNLayer(filters=128, kernel_size=3)(x)
        
        # Predict probability distribution over the vocabulary for each timestep
        decoder_outputs = layers.Dense(self.vocab_size, activation="softmax", kernel_initializer=init, name="dense_2")(x)
        self.decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

        # VAE ASSEMBLY
        self.vae_model = GWDGAVAE(self.encoder, self.decoder)
        self.vae_model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

    def fit(self, lds_words: Optional[List[str]] = None) -> None:
        """
        Trains the GWDGA model or loads it from cache if available.
        """

        start_t = time.time()

        # 1. Try to load everything from cache
        if os.path.exists(self.vocab_path) and os.path.exists(self.weights_path):
            logger.info(f"[*] {self.name} found cached artifacts. Skipping training.")
            self._load_cache()
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        # 2. If no cache is found, initialize the training sequence
        logger.info(f"[*] No cached artifacts found for {self.name}. Training...")
        
        # Fetch the dataset and construct the dictionary
        train_domains = load_tranco(TRANCO_D1_BENIGN)
        self._build_gwdict(train_domains, lds_words=lds_words)
        
        # Preprocess the text data into numerical sequences
        x_data = self._tokenize_and_pad(train_domains)

        # Assemble neural network topology
        self._build_architecture()
        
        # Define monitoring and dynamic adjustment callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
            KLAnnealingCallback(target_epochs=10)
        ]

        # 3. Execute model training
        self.vae_model.fit(
            x_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2, # Monitor performance on 20% unseen data
            callbacks=callbacks_list,
            verbose=1
        )

        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t
        
        # 4. Save the final trained state to prevent retraining next time
        self._save_cache()
        
    def generate_domain(self) -> str:
        """
        Generates a single domain (SLD only).
        """
        if not self.is_fitted:
            self.fit()
            
        max_attempts = 100  # SAFETY GUARD: bounded loop (a degenerate generator would otherwise hang)
        for _attempt in range(max_attempts):
            # 1. Sample a random noise vector representing a target latent state
            z_sample = np.random.normal(0, 1, size=(1, 128))
            
            # 2. Predict token probabilities using the decoder
            predicted_probs = self.decoder.predict(z_sample, verbose=0)[0]
            
            # 3. Apply Greedy Decoding (select the highest probability token ID)
            predicted_indices = np.argmax(predicted_probs, axis=-1)
            
            # 4. Map numerical IDs back to string segments
            domain_parts = []
            for idx in predicted_indices:
                if idx == 0: continue # Ignore padding tokens
                domain_parts.append(self.idx2word.get(idx, ""))
                
            domain = "".join(domain_parts)
            
            # 5. Apply RFC structural filters to ensure the domain is valid
            if 4 < len(domain) <= 63 and "--" not in domain and not domain.startswith("-") and not domain.endswith("-"):
                return f"{domain}.{sample_tld()}"

        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"

    def generate_domains(self, n: int) -> List[str]:
        """
        Optimized batch generation (SLDs only).
        """
        if not self.is_fitted:
            self.fit()

        start_time = time.time()
        
        valid_domains = set()
        # Scale batch size for better GPU utilization during prediction
        batch_size = max(1000, self.batch_size)

        # Loop until the requested amount of valid domains is collected
        max_batches = 200  # SAFETY GUARD: bounded loop; returns a partial list with a warning if the generator is degenerate
        for _batch in range(max_batches):
            if len(valid_domains) >= n:
                break
            # Predict multiple latent samples concurrently
            z_samples = np.random.normal(0, 1, size=(batch_size, 128))
            predicted_probs = self.decoder.predict(z_samples, verbose=0)
            predicted_indices = np.argmax(predicted_probs, axis=-1)
            
            # Reconstruct and filter each prediction in the batch
            for indices in predicted_indices:
                domain = "".join([self.idx2word.get(idx, "") for idx in indices if idx != 0])
                if 4 < len(domain) <= 63 and "--" not in domain and not domain.startswith("-") and not domain.endswith("-"):
                    valid_domains.add(domain)
                    # Stop early if the target number is reached
                    if len(valid_domains) == n:
                        break

        if len(valid_domains) < n:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(valid_domains)} of {n} valid domains after {max_batches} batches. Returning a partial list.")
        self.generation_time = time.time() - start_time
        return [f"{d}.{sample_tld()}" for d in valid_domains]