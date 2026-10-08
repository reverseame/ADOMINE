import os
import time
import random
import logging
import numpy as np
import tensorflow as tf
import h5py
from typing import List, Callable
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D, Concatenate, LeakyReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# NLP features extraction and embeddings libraries mentioned in the paper
import wordninja

from detectors.cnn.cnn import CNNDetector
from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import TRANCO_D1_BENIGN, load_tranco, load_malicious_seeds
from core.sld import extract_sld

logger = logging.getLogger(__name__)


class WordDGAModel(AdversarialModel):
    """
    WordDGA proposed by Selvaraj and Panjanathan (2024). A cWGAN-based Adversarial DGA with 
    Gumbel-Softmax, RCNN-BiLSTM feature extraction, and Wasserstein Gradient Penalty.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Domain length: Sequence fixed at 64 characters.
    2. Feature Extraction (RCNN-BiLSTM): one branch with a Conv1D of 128 filters and
    window h=3 (the h x d filter of the paper), BiLSTM (64 units per direction, 128 total),
    and FastText projected to 100 dimensions.
    3. cWGAN Architecture: Progressive Generator (256 to 1024 neurons) and inverse Critic 
    (1024 to 64) with LeakyReLU (alpha 0.2). Discriminator update ratio fixed at 1:1.
    4. Optimizers and Penalties: Adam (learning rate 0.0002, beta1 0.5, beta2 0.9), WGAN-GP 
    penalty (lambda 10.0), and conditional loss weight of 1.0.
    5. Gumbel Softmax: Initial temperature of 1.0 with exponential decay down to a minimum limit of 0.1.
    6. Statistical Variables and TLDs: Use of ReLU to reduce statistical variables (11 to 8). 
    Exclusive processing on Second-Level Domains (SLD).
    7. Differentiability (Soft-Approximation): ELMo simulation and verbal heuristics approximation 
    using raw probabilities to keep the model end-to-end differentiable.
    8. Epochs: Training bounded to 100 epochs for computational simplicity.
    9. Generation and Detection: Maximum of 50 attempts to generate domains (Algorithm 1) and
    evaluation using a default CNN detector. A candidate whose composed SLD exceeds the
    63-character RFC label limit is discarded like a collision.
    """
    
    def __init__(self, name: str = "WordDGA", target_detector=None, **kwargs):
        super().__init__(name, **kwargs)
        self.target_detector = target_detector
        
        # Architecture parameters
        self.seq_len = 64  # RFC standard for max DNS label length (63 + 1)
        self.noise_dim = 100
        self.batch_size = 128
        self.epochs = 100
        self.n_critic = 1
        self.lambda_gp = 10.0
        self.lambda_cls = 1.0  # weight for conditional classification loss (assumed)
        self.vocab = 'abcdefghijklmnopqrstuvwxyz0123456789-'
        self.vocab_size = len(self.vocab) + 1 # +1 for padding
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(self.vocab)}
        
        # Gumbel-Softmax Temperature Annealing
        self.init_temp = 1.0
        self.min_temp = 0.1
        
        # Optimizers (Adam with Beta_1=0.5, Beta_2=0.9 as assumed)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        
        self.generator = self._build_generator()
        self.critic = self._build_critic()

        # Cache directory and path for weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        self.weights_path = os.path.join(self.weights_dir, 'worddga.weights.h5')
        self.known_dga_path = os.path.join(self.weights_dir, 'known_dga.h5')
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Will hold known malicious domains for collision avoidance
        self.known_dga = None
        # Base DGA SLDs x(i) of Algorithm 1, one per generated domain, in a shuffled cycle
        self.base_domains = []
        self.domain_idx = 0
        self._oracle = None
        
        # N-gram model for statistical feature
        self.ngram_model = None  # dictionary: ngram -> log10 prob

    def _extract_statistical_features(self, domain: str) -> np.ndarray:
        """
        Extracts 11 statistical features as specified in the paper.
        """
        words = wordninja.split(domain)
        no_of_words = len(words)
        
        # 1. DomainLength: Number of tokens/characters
        domain_length = len(domain)
        # 2. NoOfDomainWords: Number of words in the domain
        no_of_domain_words = no_of_words
        # 3. NoOfHypen: Number of hyphens in the domain
        no_of_hyphen = domain.count('-')
        # 8. Vowel_freq: Number of vowels in the domain
        vowel_freq = sum(1 for c in domain.lower() if c in 'aeiou')
        
        # Process word-level heuristics
        rare_count = 0
        common_count = 0
        misspelled_count = 0
        random_count = 0
        jj_count = 0
        
        adjective_suffixes = ('able', 'ible', 'al', 'ial', 'ic', 'ical', 'ish', 'ive', 'ous', 'ful', 'less')
        
        for w in words:
            w_len = len(w)
            w_vowels = sum(1 for c in w if c in 'aeiou')
            v_ratio = w_vowels / w_len if w_len > 0 else 0
            
            # 4, 5, 6, 7. Heuristic for common, rare, misspelled and random based on vowel distributions
            if v_ratio < 0.2 or v_ratio > 0.8:
                rare_count += 1
                random_count += 1
                misspelled_count += 1
            else:
                common_count += 1
                
            # 11. Adjective ratio heuristic via suffixes
            if w.endswith(adjective_suffixes):
                jj_count += 1

        # 4. RareRatio: Ratio of rare words in domain
        rare_ratio = (rare_count / no_of_words) if no_of_words > 0 else 0.0
        # 5. CommonRatio: Ratio of common words in domain
        common_ratio = (common_count / no_of_words) if no_of_words > 0 else 0.0
        # 6. RatioMisspelled: Ratio of misspelled words in the domain name
        ratio_misspelled = (misspelled_count / no_of_words) if no_of_words > 0 else 0.0
        # 7. NoOfRandom: Number of random domain words
        no_of_random = float(random_count)
        # 11. JJ_Ratio: Rate of adjectives in a word behind the substantial word distribution
        jj_ratio = (jj_count / no_of_words) if no_of_words > 0 else 0.0
        
        # 9. Reputation: Reputation score of words (Approximated via ngram average log10 prob)
        reputation = self._compute_ngram_prob(domain)
        
        # 10. Words_I: Real integer of words after distribution
        words_i = float(no_of_words)
        
        features = [
            domain_length, no_of_domain_words, no_of_hyphen, rare_ratio, common_ratio,
            ratio_misspelled, no_of_random, vowel_freq, reputation, words_i, jj_ratio
        ]
        return np.array(features, dtype=np.float32)

    def _compute_ngram_prob(self, domain: str) -> float:
        """
        Compute average log10-probability of n-grams (n=3..7) using the ngram model.
        Start/end markers '#' are added for word-hashing sequence processing.
        """
        if self.ngram_model is None or not domain:
            return 0.0
            
        marked_domain = f"#{domain}#"
        total_log_prob = 0.0
        count = 0
        for n in range(3, 8):
            for i in range(len(marked_domain) - n + 1):
                ngram = marked_domain[i:i+n]
                log_prob = self.ngram_model.get(ngram, -15.0)  # penalty for unseen
                total_log_prob += log_prob
                count += 1
        return total_log_prob / count if count > 0 else 0.0

    def _build_ngram_model(self, domains: List[str]) -> None:
        """
        Train a character-level n-gram model (n=3..7) with Laplace smoothing.
        Values are calculated using log10 frequency per the paper design.
        """
        from collections import defaultdict
        ngram_counts = defaultdict(int)
        total_ngrams = 0
        
        for dom in domains:
            marked_dom = f"#{dom.lower()}#"
            for n in range(3, 8):
                for i in range(len(marked_dom) - n + 1):
                    ngram = marked_dom[i:i+n]
                    ngram_counts[ngram] += 1
                    total_ngrams += 1
                    
        vocab_size_est = 26**7  # upper bound estimate
        self.ngram_model = {}
        for ngram, cnt in ngram_counts.items():
            # Use log10 based smoothing as specified in the reputation score details
            self.ngram_model[ngram] = np.log10((cnt + 1) / (total_ngrams + vocab_size_est))

    def _build_generator(self) -> Model:
        """
        Builds the Generator G: three progressive hidden layers (256, 512, 1024)
        projected to a sequence of character logits.
        """
        noise_input = Input(shape=(self.noise_dim,), name="g_noise")
        label_input = Input(shape=(1,), name="g_label")
        
        # Conditional concatenation for cWGAN
        x = Concatenate()([noise_input, label_input])
        
        x = Dense(256, name="g_dense_256")(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dense(512, name="g_dense_512")(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dense(1024, name="g_dense_1024")(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        
        # Project to sequence length * vocab size
        x = Dense(self.seq_len * self.vocab_size, name="g_out")(x)
        output = tf.keras.layers.Reshape((self.seq_len, self.vocab_size))(x)
        
        return Model([noise_input, label_input], output, name="Generator")

    def _gumbel_softmax(self, logits: tf.Tensor, temperature: float) -> tf.Tensor:
        """
        Applies Gumbel-Softmax to get differentiable discrete outputs.
        """
        u = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
        gumbel_noise = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        return tf.nn.softmax(y / temperature, axis=-1)

    def _build_critic(self) -> Model:
        """
        Builds the Critic/Discriminator D. Fuses ELMo (1024->128), FastText->RCNN-BiLSTM 
        (128-d), Statistical (11->8), Word Dist (128), and conditional label for cWGAN.
        """
        # 1. Sequence Input (for ELMo & FastText logic)
        seq_input = Input(shape=(self.seq_len, self.vocab_size), name="c_seq_input")
        
        # Feature 1: FastText -> RCNN-BiLSTM (Section 4.2.1, Eqs. 5 to 7)
        # Project to 100 dimensions representing FastText CBOW embeddings
        fasttext_proj = Dense(100, use_bias=False, trainable=True, name="fasttext_proj")(seq_input)
        # RCNN: Conv1D with 128 filters of size h x d, h = 3
        rcnn_conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(fasttext_proj)
        # BiLSTM with 64 units per direction producing the 128-d sequence feature
        bilstm_merged = Bidirectional(LSTM(64, return_sequences=False), name="bilstm")(rcnn_conv)
        
        # Feature 2: ELMo Contextual Projection (Soft approximation for gradient graph 1024->128)
        # Note: TF Hub ELMo requires string tensors. In a fully differentiable pipeline, 
        # we soft-approximate the 1024 dimension structure during continuous Gumbel training.
        elmo_sim = Dense(1024, activation='relu', name="elmo_soft_approx")(seq_input)
        elmo_pool = GlobalAveragePooling1D()(elmo_sim)
        elmo_feat = Dense(128, activation='relu')(elmo_pool)
        
        # Feature 3: Statistical Features (11 -> 8 using ReLU)
        stat_input = Input(shape=(11,), name="c_stat_input")
        stat_feat = Dense(8, activation='relu')(stat_input)
        
        # Feature 4: Word Distribution (128-d feature layer)
        word_dist_pool = GlobalAveragePooling1D()(seq_input)
        word_dist_feat = Dense(128, activation='relu')(word_dist_pool)
        
        # Conditional Input for cWGAN
        label_input = Input(shape=(1,), name="c_label")
        
        # Feature Fusion (Concatenate axis=-1)
        merged = Concatenate(axis=-1)([bilstm_merged, elmo_feat, stat_feat, word_dist_feat, label_input])
        
        # Critic specific hidden layers: Inverse progressive process (1024, 512, 256)
        x = Dense(1024)(merged)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(negative_slope=0.2)(x)

        # Two fully connected layers for final feature fusion prior to classifier (Section 4.2.1)
        x = Dense(128)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dense(64)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        
        # Output: linear (Wasserstein score) + classification logits for conditional loss
        validity = Dense(1, name="critic_score")(x)
        cls_logits = Dense(1, name="cls_logits")(x)
        
        return Model(inputs=[seq_input, stat_input, label_input], outputs=[validity, cls_logits], name="Critic")

    def _compute_statistics_from_probs(self, probs: tf.Tensor) -> tf.Tensor:
        """
        Compute the 11 statistical features directly from the probability distribution
        over characters in a differentiable way, using soft-approximations for 
        word-level heuristics to maintain theoretical gradient flow fidelity.
        """
        
        # 1. DomainLength: expected length = sum over positions of (1 - prob(padding))
        pad_idx = 0
        pad_probs = probs[:, :, pad_idx]
        domain_length = tf.reduce_sum(1.0 - pad_probs, axis=1)
        
        # 3. NoOfHypen
        hyphen_idx = self.char2idx.get('-', -1)
        if hyphen_idx != -1:
            no_of_hyphen = tf.reduce_sum(probs[:, :, hyphen_idx], axis=1)
        else:
            no_of_hyphen = domain_length * 0.0
            
        # 8. Vowel_freq
        vowels = set('aeiou')
        vowel_indices = [self.char2idx[c] for c in vowels if c in self.char2idx]
        vowel_probs = tf.gather(probs, vowel_indices, axis=-1)
        vowel_freq = tf.reduce_sum(vowel_probs, axis=[1, 2])
        
        # Differentiable soft-approximations for word-level heuristics
        pseudo_words = no_of_hyphen + 1.0
        no_of_domain_words = pseudo_words
        rare_ratio = (vowel_freq / (domain_length + 1e-5)) * 0.5 
        common_ratio = 1.0 - rare_ratio
        ratio_misspelled = rare_ratio * 0.8
        no_of_random = rare_ratio * pseudo_words
        reputation = tf.reduce_mean(probs, axis=[1, 2]) * -10.0 # Proxy approximation
        words_i = pseudo_words
        jj_ratio = rare_ratio * 0.2
        
        stats = tf.stack([
            domain_length, no_of_domain_words, no_of_hyphen, rare_ratio, common_ratio,
            ratio_misspelled, no_of_random, vowel_freq, reputation, words_i, jj_ratio
        ], axis=1)  # shape [batch, 11]
        
        return stats

    def _gradient_penalty(self, real_seq, fake_seq, real_stats, fake_stats, labels):
        """
        Calculates WGAN-GP gradient penalty.
        """
        alpha = tf.random.uniform([tf.shape(real_seq)[0], 1, 1], 0.0, 1.0)
        alpha_stat = tf.random.uniform([tf.shape(real_stats)[0], 1], 0.0, 1.0)
        
        interpolated_seq = alpha * real_seq + (1 - alpha) * fake_seq
        interpolated_stat = alpha_stat * real_stats + (1 - alpha_stat) * fake_stats
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated_seq, interpolated_stat])
            score, _ = self.critic([interpolated_seq, interpolated_stat, labels], training=True)
            
        grads = gp_tape.gradient(score, [interpolated_seq, interpolated_stat])
        norm_seq = tf.sqrt(tf.reduce_sum(tf.square(grads[0]), axis=[1, 2]))
        norm_stat = tf.sqrt(tf.reduce_sum(tf.square(grads[1]), axis=[1]))
        
        gp = tf.reduce_mean((norm_seq - 1.0) ** 2) + tf.reduce_mean((norm_stat - 1.0) ** 2)
        return gp

    @tf.function
    def _train_step(self, real_seq, real_stats, labels, temperature):
        """
        Single training step adhering strictly to Algorithm 1 (WGAN-GP) + conditional loss.
        """
        batch_sz = tf.shape(real_seq)[0]
        
        # 1. Train Critic (n_critic times)
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_sz, self.noise_dim])
            
            with tf.GradientTape() as tape:
                fake_logits = self.generator([noise, labels], training=True)
                fake_seq = self._gumbel_softmax(fake_logits, temperature)
                
                # Compute real statistical features for fake samples (differentiable)
                fake_stats = self._compute_statistics_from_probs(fake_seq)
                
                real_score, real_cls_logits = self.critic([real_seq, real_stats, labels], training=True)
                fake_score, fake_cls_logits = self.critic([fake_seq, fake_stats, labels], training=True)
                
                # Wasserstein loss
                w_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
                
                # Gradient penalty
                gp = self._gradient_penalty(real_seq, fake_seq, real_stats, fake_stats, labels)
                
                # Conditional classification loss (Equation 11)
                real_cls_loss = tf.keras.losses.binary_crossentropy(labels, tf.sigmoid(real_cls_logits))
                fake_cls_loss = tf.keras.losses.binary_crossentropy(labels, tf.sigmoid(fake_cls_logits))
                cls_loss = tf.reduce_mean(real_cls_loss + fake_cls_loss)
                
                c_loss = w_loss + self.lambda_gp * gp + self.lambda_cls * cls_loss
                
            c_grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))
            
        # 2. Train Generator
        noise = tf.random.normal([batch_sz, self.noise_dim])
        with tf.GradientTape() as tape:
            fake_logits = self.generator([noise, labels], training=True)
            fake_seq = self._gumbel_softmax(fake_logits, temperature)
            fake_stats = self._compute_statistics_from_probs(fake_seq)
            
            fake_score, fake_cls_logits = self.critic([fake_seq, fake_stats, labels], training=True)
            
            # Wasserstein generator loss
            g_w_loss = -tf.reduce_mean(fake_score)
            
            # Conditional classification loss for generator
            g_cls_loss = tf.keras.losses.binary_crossentropy(labels, tf.sigmoid(fake_cls_logits))
            g_cls_loss = tf.reduce_mean(g_cls_loss)
            
            g_loss = g_w_loss + self.lambda_cls * g_cls_loss
            
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return c_loss, g_loss

    def _load_known_dga(self):
        """
        Load known_dga from HDF5 file.
        """
        if os.path.exists(self.known_dga_path):
            with h5py.File(self.known_dga_path, 'r') as f:
                if 'known_dga' in f:
                    data = f['known_dga'][:]
                    # Convert bytes to strings if needed
                    if data.dtype == object:
                        self.known_dga = set([x.decode('utf-8') if isinstance(x, bytes) else x for x in data])
                    else:
                        self.known_dga = set(data)
                else:
                    self.known_dga = set()
        else:
            self.known_dga = set()

    def _save_known_dga(self):
        """
        Save known_dga to HDF5 file.
        """
        with h5py.File(self.known_dga_path, 'w') as f:
            dt = h5py.string_dtype(encoding='utf-8')
            # Convert set to list of strings
            data = list(self.known_dga)
            f.create_dataset('known_dga', data=data, dtype=dt)

    def fit(self) -> None:
        """
        Trains the WordDGA model or loads cached weights. Operates strictly on SLDs.
        """
        start_t = time.time()

        # Try to load everything from cache
        if os.path.exists(self.weights_path) and os.path.exists(self.known_dga_path):
            logger.info("[*] WordDGA found cached weights. Skipping training.")
            self.generator.load_weights(self.weights_path)
            self._load_known_dga()
            self._set_base_domains()
            if not self.target_detector:
                logger.info("[*] No target detector provided for WordDGA. Instantiating default CNN detector...")
                self.target_detector = CNNDetector()
                self.target_detector.fit()
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        # If only weights exist but not known_dga, we force re-training to avoid inconsistency
        if os.path.exists(self.weights_path) and not os.path.exists(self.known_dga_path):
            # Delete weights to ensure clean training
            os.remove(self.weights_path)

        logger.info("[*] No cached weights found for WordDGA. Training...")
        if not self.target_detector:
            logger.info("[*] No target detector provided for WordDGA. Instantiating default CNN detector...")
            self.target_detector = CNNDetector()
            self.target_detector.fit()
                    
        # Load malicious datasets
        malicious_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py (SLD extracted below)
            malicious_domains = load_malicious_seeds(total=100000)
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs: {e}")
            
        # Load Tranco dataset for benign domains
        benign_domains = load_tranco(TRANCO_D1_BENIGN)
        
        # Use SLDs only and assign conditional labels (1.0 for DGA, 0.0 for benign)
        dataset_malicious = [(extract_sld(d).lower(), 1.0) for d in malicious_domains if isinstance(d, str)]
        dataset_benign = [(extract_sld(d).lower(), 0.0) for d in benign_domains if isinstance(d, str)]
        
        dataset = dataset_malicious + dataset_benign
        dataset = [d for d in dataset if len(d[0]) > 0 and len(d[0]) <= self.seq_len]
        
        # Save known malicious domains for Algorithm 1 collision checks
        self.known_dga = set([d[0] for d in dataset_malicious])
        self._set_base_domains()
        
        # Train n-gram model on all domains (benign + malicious) for feature #11
        all_domains = [d[0] for d in dataset]
        self._build_ngram_model(all_domains)
        
        # Data pre-processing
        X_seq = np.zeros((len(dataset), self.seq_len, self.vocab_size), dtype=np.float32)
        X_stat = np.zeros((len(dataset), 11), dtype=np.float32)
        Y_label = np.zeros((len(dataset), 1), dtype=np.float32)
        
        for i, (dom, label) in enumerate(dataset):
            X_stat[i] = self._extract_statistical_features(dom)
            Y_label[i, 0] = label
            
            for j, char in enumerate(dom):
                if char in self.char2idx:
                    X_seq[i, j, self.char2idx[char]] = 1.0
            # Pad
            for j in range(len(dom), self.seq_len):
                X_seq[i, j, 0] = 1.0
                
        dataset_tf = tf.data.Dataset.from_tensor_slices((X_seq, X_stat, Y_label))
        dataset_tf = dataset_tf.shuffle(10000).batch(self.batch_size, drop_remainder=True)
        
        # Training Loop. The Gumbel-Softmax temperature decays exponentially
        # from init_temp to min_temp over the epochs.
        temperature = self.init_temp
        decay_rate = np.exp(np.log(self.min_temp / self.init_temp) / self.epochs)
        logger.info(f"[*] Training {self.epochs} epochs over {len(dataset)} SLDs")
        
        for epoch in range(self.epochs):
            for real_seq, real_stats, labels in dataset_tf:
                c_loss, g_loss = self._train_step(real_seq, real_stats, labels, tf.constant(temperature, dtype=tf.float32))
            
            # Anneal temperature
            temperature = max(self.min_temp, temperature * decay_rate)
            logger.info(f"[*] Epoch {epoch+1}/{self.epochs} c_loss={float(c_loss):.4f} "
                        f"g_loss={float(g_loss):.4f} temp={temperature:.3f} "
                        f"elapsed={time.time() - start_t:.0f}s")

        # Save weights and known_dga
        self.generator.save_weights(self.weights_path)
        self._save_known_dga()
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def _set_base_domains(self) -> None:
        """
        Fixes the cycle of base DGA SLDs x(i) that Algorithm 1 modifies, one per
        generated domain. Sorted before shuffling so the order depends only on
        the framework seed, not on set iteration order.
        """
        self.base_domains = sorted(self.known_dga) if self.known_dga else []
        random.shuffle(self.base_domains)
        self.domain_idx = 0

    def _malicious_probs(self, slds: List[str]) -> List[float]:
        """
        Black-box query of the target classifier M(.) for a batch of SLDs.
        Returns P(malicious) = 1 - P(benign) in order. One compiled call per
        batch instead of one Keras predict() per attempt, whose fixed overhead
        would dominate the cost of Algorithm 1.
        """
        if self._oracle is None:
            model = self.target_detector.model
            self._oracle = tf.function(
                lambda x: model(x, training=False),
                input_signature=[tf.TensorSpec([None, None], tf.int32)],
            )
        encoded = self.target_detector._encode_domains(slds)
        prob_benign = self._oracle(tf.constant(encoded, dtype=tf.int32)).numpy()[:, 0]
        return [1.0 - float(p) for p in prob_benign]

    def generate_domain(self, target_classifier: Callable = None) -> str:
        """
        Generates a single SLD domain using the trained Generator (Algorithm 1).
        Draws up to max_attempts candidate words W from G, substitutes one
        segment of the base DGA SLD x(i) with each, discards repeated words and
        collisions with known DGA domains, and returns the first substitution
        that the target classifier M(.) labels as benign.

        The attempts are evaluated in order, as in the sequential loop of the
        paper, but generated and scored in one batch each: the outcome is the
        same and the oracle is queried once per domain instead of once per attempt.
        """
        if not self.is_fitted:
            self.fit()

        max_attempts = 50

        # Ensure known_dga is loaded
        if self.known_dga is None:
            self._load_known_dga()
            if self.known_dga is None:
                self.known_dga = set()
        if not self.base_domains and self.known_dga:
            self._set_base_domains()

        # Base DGA SLD x(i): a different one for every generated domain, cycling
        # over the shuffled known DGA SLDs
        if self.base_domains:
            base_domain = self.base_domains[self.domain_idx % len(self.base_domains)]
            self.domain_idx += 1
        else:
            logger.warning(f"[{self.name}] No known DGA domains loaded. Using a random base SLD.")  # SAFETY GUARD
            base_domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))  # SAFETY GUARD: random SLD fallback

        # Wordninja segmentation s(.) of the base domain
        segmented_base = wordninja.split(base_domain)
        if not segmented_base:
            segmented_base = [base_domain]  # SAFETY GUARD: unsegmentable base, keep it as a single word (it is replaced below)

        # All the attempts' words W come from G in one call, label 1.0 (malicious)
        noise = tf.random.normal([max_attempts, self.noise_dim])
        target_label = tf.ones([max_attempts, 1], dtype=tf.float32)
        fake_logits = self.generator([noise, target_label], training=False)
        fake_seq = self._gumbel_softmax(fake_logits, temperature=self.min_temp)
        char_indices = tf.argmax(fake_seq, axis=-1).numpy()

        tried_substitutions = set()
        candidates = []
        for row in char_indices:
            generated_chars = ""
            for idx in row:
                if idx == 0:  # Padding token
                    break
                generated_chars += self.idx2char.get(int(idx), "")

            # Filter invalid bounding hyphens per RFC
            generated_chars = generated_chars.strip('-')

            # P(W|z)=0: exclude tested/failed substitutions to avoid repetition
            if not generated_chars or generated_chars in tried_substitutions:
                continue
            tried_substitutions.add(generated_chars)

            # R(.) rule: replace one random segment of the base to keep its structure
            segments = list(segmented_base)
            segments[random.randint(0, len(segments) - 1)] = generated_chars
            domain = "".join(segments)

            # RFC 1035 label limit, the paper's reason for the 64-position sequence
            if len(domain) > 63:
                continue

            # Verify Collision Rate
            if self.known_dga and domain in self.known_dga:
                continue
            candidates.append(domain)

        if not candidates:
            logger.warning(f"[{self.name}] No valid candidate in {max_attempts} attempts. Using a random fallback.")  # SAFETY GUARD
            domain = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))  # SAFETY GUARD: random SLD fallback
            return f"{domain}.{sample_tld()}"

        # Evaluate evasion against the target classifier M(.): first candidate,
        # in attempt order, that is not classified as DGA
        if self.target_detector is not None and self.target_detector.model is not None:
            for domain, score in zip(candidates, self._malicious_probs(candidates)):
                if score <= 0.5:
                    return f"{domain}.{sample_tld()}"
            # Every attempt was detected: return the last one, as the bounded loop did
            return f"{candidates[-1]}.{sample_tld()}"

        return f"{candidates[0]}.{sample_tld()}"
