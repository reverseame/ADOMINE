import os
import time
import h5py
import random
import logging
import numpy as np
import tensorflow as tf
from collections import Counter
from typing import List
from tensorflow.keras.layers import (
    Input, Dense, LSTM, MultiHeadAttention, LayerNormalization, 
    Dropout, Embedding, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from core.seeding import get_seed
from core.sld import extract_sld

# NLTK and Gensim for n-grams and Word2Vec
import nltk
from nltk.util import ngrams
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    """
    Callback to implement S-type KL annealing strategy.
    Formula: beta = 1 / (1 + exp(-k * (t - t0)))
    """
    def __init__(self, k=0.05, t0=50):
        super(KLAnnealingCallback, self).__init__()
        self.k = k
        self.t0 = t0

    def on_epoch_begin(self, epoch, logs=None):
        beta = 1.0 / (1.0 + np.exp(-self.k * (epoch - self.t0)))
        tf.keras.backend.set_value(self.model.beta, float(beta))
        logger.info(f" [*] Epoch {epoch + 1}: KL Beta set to {beta:.4f}")

class TLVDGAModel(AdversarialModel):
    """
    Implementation of TLVDGA model proposed by Nangong and Wu (2025). Includes 
    Transformer Encoder, LSTM Decoder, and Word2Vec embeddings.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Sequences: Maximum length is set to 20 tokens. The vocabulary size is 5004, 
    explicitly including control tokens (<PAD>, <UNK>, <SOS>, and <EOS>).
    2. Dimensionality: Word2Vec embedding dimension is configured to 128. 
    The latent space vector is mapped to 64 dimensions.
    3. Encoder Architecture: The Transformer consists of 2 layers, 4 attention heads, 
    and a 512 feed-forward dimension. Positional encoding and Global Average Pooling 
    are included to manage sequential order and representation.
    4. Decoder Architecture: Uses a single unidirectional LSTM layer with 256 hidden units.
    5. Training and Loss: Uses the Adam optimizer (learning rate 0.001) and a batch size of 128. 
    Sparse Categorical Cross-Entropy is applied for the reconstruction loss. Training is set to 
    100 epochs with the KL annealing central point at epoch 50.
    6. BiMM Tie-Resolution: When bidirectional matchings differ, the segmentation with fewer total
    tokens is prioritized to maximize longer n-grams. If the token count is identical, Forward 
    Maximum Matching (FMM) is selected by default.
    7. Data Strategy: Training relies on benign domains sourced from the Tranco list. An 80/20 
    internal split is applied to separate training and validation subsets.
    """
    def __init__(self, name: str = "TLVDGA"):
        super().__init__(name=name)
        
        # Model hyperparameters based on assumptions
        self.max_len = 20
        self.vocab_size = 5004  # 5000 n-grams + PAD, UNK, SOS, EOS
        self.embed_dim = 128
        self.latent_dim = 64
        self.transformer_heads = 4
        self.transformer_ff_dim = 512
        self.transformer_layers = 2
        self.lstm_units = 256
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 100
        self.t0 = 50    # Central time step of annealing
        self.k = 0.05   # To control the growth rate during Sigmoid KL annealing strategy
        
        # Special tokens
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.special_tokens = {"<PAD>": self.PAD, "<UNK>": self.UNK, "<SOS>": self.SOS, "<EOS>": self.EOS}
        self.reverse_vocab = {}
        self.vocab = {}
        
        # Component placeholders
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.w2v_model = None

        # Paths for caching
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_path = os.path.join(self.weights_dir, "vae.weights.h5")
        self.vocab_path = os.path.join(self.weights_dir, "tlvdga_vocab.h5")
        os.makedirs(self.weights_dir, exist_ok=True)

    def _build_dictionary(self, domains: List[str]):
        """
        Extract top 5000 n-grams (n=1,2,3,4) from domains to build vocabulary, using NLTK library.
        """
        logger.info("[*] TLVDGA building n-gram dictionary...")
        all_ngrams = []
        for d in domains:
            for n in range(1, 5):
                all_ngrams.extend([''.join(gram) for gram in ngrams(d, n)])
                
        counter = Counter(all_ngrams)
        most_common = counter.most_common(5000)
        
        self.vocab = self.special_tokens.copy()
        for idx, (gram, _) in enumerate(most_common):
            self.vocab[gram] = idx + 4
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        with h5py.File(self.vocab_path, 'w') as f:
            tokens = np.array(list(self.vocab.keys()), dtype='S')
            indices = np.array(list(self.vocab.values()), dtype=int)
            f.create_dataset('tokens', data=tokens)
            f.create_dataset('indices', data=indices)

    def _fmm(self, domain: str) -> List[str]:
        """
        Forward Maximum Matching.
        """
        tokens = []
        i = 0
        while i < len(domain):
            match = False
            for length in range(4, 0, -1):
                if i + length <= len(domain):
                    gram = domain[i:i+length]
                    if gram in self.vocab:
                        tokens.append(gram)
                        i += length
                        match = True
                        break
            if not match:
                tokens.append("<UNK>")
                i += 1
        return tokens

    def _bmm(self, domain: str) -> List[str]:
        """
        Backward Maximum Matching.
        """
        tokens = []
        i = len(domain)
        while i > 0:
            match = False
            for length in range(4, 0, -1):
                if i - length >= 0:
                    gram = domain[i-length:i]
                    if gram in self.vocab:
                        tokens.insert(0, gram)
                        i -= length
                        match = True
                        break
            if not match:
                tokens.insert(0, "<UNK>")
                i -= 1
        return tokens

    def _bimm_segment(self, domain: str) -> List[str]:
        """
        Bi-direction Maximum Matching with tie-breaking rules.
        """
        f_tokens = self._fmm(domain)
        b_tokens = self._bmm(domain)
        
        # Tie-breaker: choose the one with fewer tokens (maximizes longer n-grams)
        if len(f_tokens) < len(b_tokens):
            return f_tokens
        elif len(b_tokens) < len(f_tokens):
            return b_tokens
        else:
            return f_tokens # Default to FMM on equal length

    def _transformer_encoder(self):
        """
        Builds the Multi-head Self-Attention Transformer Encoder with Positional Encoding.
        """
        inputs = Input(shape=(self.max_len, self.embed_dim))
        
        # Positional Encoding
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        pos_emb = Embedding(input_dim=self.max_len, output_dim=self.embed_dim)(positions)
        x = inputs + pos_emb
        
        for _ in range(self.transformer_layers):
            # Multi-head attention
            attn_output = MultiHeadAttention(num_heads=self.transformer_heads, key_dim=self.embed_dim)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed-forward
            ff_output = Dense(self.transformer_ff_dim, activation="relu")(x)
            ff_output = Dense(self.embed_dim)(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
            
        x = GlobalAveragePooling1D()(x)
        
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        
        return Model(inputs, [z_mean, z_log_var], name="encoder")

    def _lstm_decoder(self):
        """
        Builds the LSTM Decoder.
        """
        latent_inputs = Input(shape=(self.latent_dim,))
        
        # Map latent space to higher dimensions for LSTM processing
        x = Dense(self.max_len * self.embed_dim, activation="relu")(latent_inputs)
        x = Reshape((self.max_len, self.embed_dim))(x)
        
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        
        # Softmax over vocabulary size
        outputs = Dense(self.vocab_size, activation="softmax")(x)
        
        return Model(latent_inputs, outputs, name="decoder")

    class VAE(Model):
        """
        Keras Subclass for VAE to handle custom KL loss and annealing.
        """
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        def call(self, inputs):
            z_mean, z_log_var = self.encoder(inputs)
            # Reparameterization trick: z = mu + sigma * epsilon
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            reconstructed = self.decoder(z)
            
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(self.beta * kl_loss)
            
            return reconstructed

    def fit(self) -> None:
        """
        Trains or loads the TLVDGA VAE model.
        """
        start_time = time.time()
        
        if os.path.exists(self.weights_path) and os.path.exists(self.vocab_path):
            logger.info("[*] TLVDGA found cached vocabulary and weights. Skipping training.")
            with h5py.File(self.vocab_path, 'r') as f:
                tokens = [t.decode('utf-8') for t in f['tokens'][:]]
                indices = [int(i) for i in f['indices'][:]]
                self.vocab = dict(zip(tokens, indices))
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            # Build models to load weights
            self.encoder = self._transformer_encoder()
            self.decoder = self._lstm_decoder()
            self.vae = self.VAE(self.encoder, self.decoder)
            
            # Dummy call to initialize variables before loading weights
            dummy_input = tf.zeros((1, self.max_len, self.embed_dim))
            self.vae(dummy_input)
            self.vae.load_weights(self.weights_path)
            
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            self.is_fitted = True
            return

        logger.info("[*] No cached weights or vocabulary found for TLVDGA. Training...")
        
        # Load Benign domains from Tranco list
        benign_domains = load_tranco(TRANCO_D1_BENIGN)
        # Keep only the Second-Level Domain (SLD)
        train_domains = [extract_sld(d) for d in benign_domains]
        
        self._build_dictionary(train_domains)
        
        # Segment domains using BiMM
        logger.info("[*] TLVDGA segmenting domains via BiMM...")
        tokenized_corpus = [self._bimm_segment(d) for d in train_domains]
        
        # Train Word2Vec Skip-gram model
        logger.info("[*] TLVDGA training Word2Vec embeddings...")
        # gensim has no global RNG; seed comes from seed_everything() and workers=1 makes it deterministic
        self.w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=self.embed_dim, window=5, sg=1, min_count=1,
                                  seed=get_seed(), workers=1)
        
        # Prepare datasets
        logger.info("[*] TLVDGA encoding sequences for VAE...")
        X_data = np.zeros((len(tokenized_corpus), self.max_len, self.embed_dim))
        Y_data = np.zeros((len(tokenized_corpus), self.max_len))
        
        for i, seq in enumerate(tokenized_corpus):
            for j, token in enumerate(seq[:self.max_len-1]):
                idx = self.vocab.get(token, self.UNK)
                Y_data[i, j] = idx
                if token in self.w2v_model.wv:
                    X_data[i, j] = self.w2v_model.wv[token]
            Y_data[i, min(len(seq), self.max_len - 1)] = self.EOS
            
        # 80/20 Train/Validation Split
        split_idx = int(0.8 * len(X_data))
        X_train, X_val = X_data[:split_idx], X_data[split_idx:]
        Y_train, Y_val = Y_data[:split_idx], Y_data[split_idx:]

        # Build and compile VAE
        self.encoder = self._transformer_encoder()
        self.decoder = self._lstm_decoder()
        self.vae = self.VAE(self.encoder, self.decoder)
        
        # Categorial Cross-Entropy for Reconstruction Loss
        self.vae.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                         loss=SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE))
        
        kl_annealing = KLAnnealingCallback(k=self.k, t0=self.t0)
        
        logger.info("[*] TLVDGA starting VAE training...")
        self.vae.fit(X_train, Y_train, 
                     validation_data=(X_val, Y_val),
                     batch_size=self.batch_size, 
                     epochs=self.epochs, 
                     callbacks=[kl_annealing],
                     verbose=1)
                     
        self.vae.save_weights(self.weights_path)
        
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time
        self.is_fitted = True

    def _is_valid_domain(self, domain: str) -> bool:
        """
        Filters generated domains based on RFC and length heuristics.
        """
        if len(domain) < 4 or len(domain) > 64:
            return False
        if "--" in domain:
            return False
        if domain.startswith("-") or domain.endswith("-"):
            return False
        if domain.isdigit():
            return False
        return True

    def generate_domain(self) -> str:
        """
        Generates a single adversarial domain from the latent space.
        """
        if not self.is_fitted:
            self.fit()

        max_attempts = 100  # SAFETY GUARD: bounded loop (a degenerate generator would otherwise hang)
        for _attempt in range(max_attempts):
            # Sample z ~ N(0, 1)
            z_sample = np.random.normal(size=(1, self.latent_dim))
            
            # Decode
            predicted_probs = self.decoder.predict(z_sample, verbose=0)[0]
            
            # Reconstruct string
            domain_tokens = []
            for prob in predicted_probs:
                token_idx = np.argmax(prob)
                if token_idx == self.EOS or token_idx == self.PAD:
                    break
                token = self.reverse_vocab.get(token_idx, "")
                if token not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]:
                    domain_tokens.append(token)
                    
            domain = "".join(domain_tokens)
            
            if self._is_valid_domain(domain):
                return f"{domain}.{sample_tld()}"
        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"
