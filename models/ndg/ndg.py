import os
import re
import random
import time
import logging
import numpy as np
import tensorflow as tf
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN

logger = logging.getLogger(__name__)

# Configuration Constants
MAX_LEN = 30
VOCAB_SIZE = 39
EMBEDDING_DIM = 16
LATENT_DIM = 16
GCNN_FILTERS = 128
KERNEL_SIZE = 3

# Character set: 26 lowercase letters + 10 digits + '-' + '.' + '<PAD>'
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
PAD_IDX = len(CHARS)  # 38
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
IDX_TO_CHAR[PAD_IDX] = "" # Padding is ignored in final output


class Sampling(tf.keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a domain.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GCNNBlock(tf.keras.layers.Layer):
    """
    Gated Convolutional Neural Network block.
    Computes: h(X) = X + (X * W + b) (*) sigmoid(X * V + d)
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(GCNNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        # Main linear convolution
        self.conv_linear = tf.keras.layers.Conv1D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same"
        )
        # Gate convolution (Sigmoid activation)
        self.conv_gate = tf.keras.layers.Conv1D(
            filters=self.filters, kernel_size=self.kernel_size, padding="same", activation="sigmoid"
        )
        # Residual projection (1x1 Conv) if input channels mismatch filters
        self.shortcut_proj = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1, padding="same")

    def call(self, inputs):
        # Apply 1x1 projection for the residual connection if dimensions do not match
        if inputs.shape[-1] != self.filters:
            shortcut = self.shortcut_proj(inputs)
        else:
            shortcut = inputs
            
        linear_out = self.conv_linear(inputs)
        gate_out = self.conv_gate(inputs)
        
        # Element-wise product and residual connection
        return shortcut + (linear_out * gate_out)

    def build(self, input_shape):
        super(GCNNBlock, self).build(input_shape)


class NDG_VAE(tf.keras.Model):
    """
    Variational Autoencoder for NDG.
    Architecture based on stacked GCNN layers for the encoder and decoder.
    """
    def __init__(self, **kwargs):
        super(NDG_VAE, self).__init__(**kwargs)
        
        # Encoder architecture
        self.embedding = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)
        self.enc_gcnn_1 = GCNNBlock(filters=GCNN_FILTERS, kernel_size=KERNEL_SIZE)
        self.enc_gcnn_2 = GCNNBlock(filters=GCNN_FILTERS, kernel_size=KERNEL_SIZE)
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.enc_dense_hidden = tf.keras.layers.Dense(GCNN_FILTERS, activation="relu")
        self.z_mean_layer = tf.keras.layers.Dense(LATENT_DIM, name="z_mean")
        self.z_log_var_layer = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var")
        self.sampling = Sampling()

        # Decoder architecture
        self.dec_dense_proj = tf.keras.layers.Dense(MAX_LEN * GCNN_FILTERS, activation="relu")
        self.dec_reshape = tf.keras.layers.Reshape((MAX_LEN, GCNN_FILTERS))
        self.dec_gcnn_1 = GCNNBlock(filters=GCNN_FILTERS, kernel_size=KERNEL_SIZE)
        self.dec_output = tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")

        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def build(self, input_shape):
        super(NDG_VAE, self).build(input_shape)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, data):
        x = self.embedding(data)
        x = self.enc_gcnn_1(x)
        x = self.enc_gcnn_2(x)
        x = self.gap(x)
        x = self.enc_dense_hidden(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var

    def decode(self, z):
        x = self.dec_dense_proj(z)
        x = self.dec_reshape(x)
        x = self.dec_gcnn_1(x)
        reconstruction = self.dec_output(x)
        return reconstruction

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling((z_mean, z_log_var))
        return self.decode(z)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sampling((z_mean, z_log_var))
            reconstruction = self.decode(z)
            
            # Reconstruction loss
            one_hot_data = tf.one_hot(data, VOCAB_SIZE)
            cce = tf.keras.losses.categorical_crossentropy(one_hot_data, reconstruction)
            reconstruction_loss = tf.reduce_sum(cce, axis=1)
            
            # KL divergence base
            kl_loss_base = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            
            # Total VAE loss
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss_base)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss_base)) 
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var = self.encode(data)
        z = self.sampling((z_mean, z_log_var))
        reconstruction = self.decode(z)
        
        # One-hot encoding
        one_hot_data = tf.one_hot(data, VOCAB_SIZE)
        cce = tf.keras.losses.categorical_crossentropy(one_hot_data, reconstruction)
        reconstruction_loss = tf.reduce_sum(cce, axis=1)
        
        kl_loss_base = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss_base)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(tf.reduce_mean(kl_loss_base))
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class NDGModel(AdversarialModel):
    """
    NDG (Neural Domain Generation) adversarial model.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Network Architecture: Uses 128 filters for GCNN blocks, a 128-unit Dense layer 
    with ReLU for hidden representations, and Linear outputs for latent mu and log_var.
    2. Residual Connections: Applies a 1x1 Conv projection to match channel dimensions 
    (projections from 16 to 128) during residual additions.
    3. Text Preprocessing & Vocabulary: Converts domains to lowercase, extracts SLDs, 
    and applies post-padding/truncation to a fixed length of 30 characters (Vocabulary size = 39).
    4. Decoder Projection: Projects latent vector z (dim 16) through a Dense layer to (30 * 128) 
    followed by a Reshape layer before feeding into the decoder GCNN.
    5. Loss Formulation: Reconstruction loss uses Categorical Cross-Entropy summed over the 
    sequence length (30 steps) and averaged across the batch to balance KL divergence.
    6. Training & Early Stopping: Trained using Adam (lr=0.0005, batch_size=512) with Early 
    Stopping monitoring validation loss (`val_loss`, patience=10) on a 10% validation split.
    7. Inference & Generation: Generates domains by sampling z ~ N(0, I), decoding via deterministic
    argmax over Softmax outputs, and appending a static '.com' suffix.
    """
    def __init__(self, name: str = "NDG", **kwargs):
        super().__init__(name=name, **kwargs)
        self.model = None
        self.epochs = 100
        self.batch_size = 512
        self.learning_rate = 0.0005
        # Cache directory and weights path
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'model.weights.h5')

    def _preprocess_domain(self, domain: str) -> List[int]:
        """
        Convert a domain string to a padded sequence of integer indices. Removes TLD and keeps 
        only the SLD.
        """
        domain = domain.lower()
        # Extract SLD
        sld = domain.split('.')[0]
        seq = [CHAR_TO_IDX.get(c, PAD_IDX) for c in sld]
        
        # Truncate or pad to MAX_LEN (Post-padding)
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        else:
            seq = seq + [PAD_IDX] * (MAX_LEN - len(seq))
        return seq

    def _postprocess_sequence(self, sequence: np.ndarray) -> str:
        """
        Convert a sequence of integer indices back to a string, stripping padding.
        """
        chars = [IDX_TO_CHAR.get(idx, "") for idx in sequence if idx != PAD_IDX]
        return "".join(chars)

    def fit(self) -> None:
        """
        Train the NDG model natively on the benign dataset (Tranco).
        """
            
        start_time = time.time()
        
        # Build the model architecture
        self.model = NDG_VAE()
        
        # Dummy forward pass to build the complete graph
        dummy_input = tf.zeros((1, MAX_LEN), dtype=tf.int32)
        self.model(dummy_input)

        # Check for cached weights
        if os.path.exists(self.weights_path):
            logger.info(f"[*] NDG found cached weights. Skipping training.")
            self.model.load_weights(self.weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            return

        logger.info(f"[*] No cached weights found for NDG. Training...")
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=None)
        
        # 1. Load and preprocess benign dataset (D1 Tranco split)
        raw_domains = load_tranco(TRANCO_D1_BENIGN)
        # Paper Sec. 5.4: duplicates produced by TLD removal are eliminated before training
        unique_slds = list(dict.fromkeys(d.lower().split('.')[0] for d in raw_domains))
        logger.info(f"[*] NDG training on {len(unique_slds)} distinct SLDs (from {len(raw_domains)} domains).")
        encoded_domains = np.array([self._preprocess_domain(d) for d in unique_slds])
        
        # Create validation split implicitly to monitor accuracy for early stopping
        
        # 2. Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # 3. Train the VAE
        self.model.fit(
            encoded_domains,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,  # Reserving 10% for early stopping evaluation
            callbacks=[early_stopping],
            verbose=1
        )
        
        # 4. Save trained weights
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_weights(self.weights_path)
        
        self.training_time = time.time() - start_time
        self.trained_from_scratch = True
        self.is_fitted = True

    @staticmethod
    def _is_valid_sld(sld: str) -> bool:
        """
        RFC 1034/1035 label check applied after padding truncation (paper Sec. 5.4 post-processing):
        1 to 63 chars, only a-z, 0-9 and '-', no leading or trailing hyphen. Rejects the empty
        string and any '.' the vocabulary allows but a label cannot hold.
        """
        return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", sld))

    def _decode_batch(self, n: int) -> List[str]:
        """
        Sample n latent vectors z ~ N(0, I), decode with deterministic argmax and strip padding.
        """
        z_samples = tf.random.normal(shape=(n, LATENT_DIM))
        reconstructions = self.model.decode(z_samples)
        predicted_indices = tf.argmax(reconstructions, axis=-1).numpy()
        return [self._postprocess_sequence(seq) for seq in predicted_indices]

    def generate_domain(self) -> str:
        """
        Generate a single domain using deterministic argmax. Invalid SLDs are rejected and z is
        resampled, up to a bounded number of attempts.
        """
        if not self.is_fitted:
            self.fit()

        max_attempts = 100  # SAFETY GUARD: bounded rejection sampling
        for _attempt in range(max_attempts):
            sld = self._decode_batch(1)[0]
            if self._is_valid_sld(sld):
                return sld + ".com"
        logger.warning(f"[{self.name}] No valid SLD after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + ".com"

    def generate_domains(self, n: int) -> List[str]:
        """
        Generate a batch of domains efficiently.
        """
        if not self.is_fitted:
            self.fit()

        generated_domains: List[str] = []
        max_batches = 200  # SAFETY GUARD: bounded rejection sampling; partial list with a warning if the decoder is degenerate
        for _batch in range(max_batches):
            needed = n - len(generated_domains)
            if needed <= 0:
                break
            # Oversample to absorb rejected outputs, capped to keep memory bounded
            for sld in self._decode_batch(min(max(needed * 2, 10), 1000)):
                if self._is_valid_sld(sld):
                    generated_domains.append(sld + ".com")
                    if len(generated_domains) == n:
                        break
        if len(generated_domains) < n:  # SAFETY GUARD
            logger.warning(f"[{self.name}] Only {len(generated_domains)} of {n} valid domains after {max_batches} batches. Returning a partial list.")
        return generated_domains