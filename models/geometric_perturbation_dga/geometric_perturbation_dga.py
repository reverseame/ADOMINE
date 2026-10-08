import os
import time
import logging
import random
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN, load_malicious_seeds
from core.sld import extract_sld
from typing import List, Tuple

logger = logging.getLogger(__name__)

class GeometricPerturbationDGAModel(AdversarialModel):
    """
    Implementation of the Geometric Perturbation DGA algorithm from Liu, Yu, et al. 2021.
    Generates adversarial samples by adding geometric vectors from benign domains to DGA domains.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Network Architectures: Generator G uses an LSTM-based topology, and target network 
    D uses a CNN-based architecture for binary classification.
    2. Differentiability: the discrete modulo operation is computed on a continuous
    approximation of X' (expected index under the softmax), and D consumes one-hot rows
    through a bias-free Dense layer (a differentiable Embedding). A Straight-Through
    Estimator (STE) rounds the perturbed indices in the forward pass while the backward
    pass flows through a soft one-hot, so L_D reaches the generator's parameters
    (Table 1, step 6 of the paper).
    3. Loss functions: L_G is the target network's evaluation of X' (Eq. 12), minimised so
    that the adversarial legal sample crosses the decision boundary (Eq. 10); L_D is the
    cross-entropy that pushes D(M') towards the benign class (Eq. 13). L = beta*L_G + L_D
    (Eq. 16) with beta = 0.5, Adam for both networks.
    4. Training Scheme: The target network D is pre-trained and subsequently frozen during 
    the adversarial training loop to prevent forgetting.
    5. Domain Constraints & TLD Handling: Geometric perturbations are strictly applied to 
    the Second-Level Domain (SLD) with a max length of 63 and vocabulary size |V|=38. The
    adversarial legal sample X' keeps the length of X (Eq. 18: positions beyond |X| are
    the padding symbol), and labels with a leading or trailing hyphen are redrawn; the 
    original TLD is preserved and re-appended.
    """
    
    def __init__(self, name: str = "GeometricPerturbationDGA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = 63  # DNS label limit, applied to SLD only
        
        # Valid character dictionary (V) for SLD characters (a-z, 0-9, '-')
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789-'
        self.valid_chars = {x: idx + 1 for idx, x in enumerate(chars)}
        self.idx_to_char = {idx + 1: x for idx, x in enumerate(chars)}
        self.idx_to_char[0] = ''  # Padding character
        self.vocab_size = len(self.valid_chars) + 1  # 38 (37 chars + padding)
        
        # Hyperparameters
        self.beta = 0.5  # Balance coefficient for the target loss function
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 50          # Adversarial training epochs
        self.mle_epochs = 20       # MLE pre-training epochs for G
        self.d_pretrain_epochs = 10 # Pre-training epochs for D
        
        # Framework caching paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.g_weights_path = os.path.join(self.weights_dir, f"G_network.weights.h5")
        self.d_weights_path = os.path.join(self.weights_dir, f"D_network.weights.h5")
        
        self.G = None
        self.D = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Cached data for generation (SLD only, TLD stored separately)
        self.benign_slds = []
        self.benign_tlds = []
        self.dga_slds = []
        self.dga_tlds = []

    def _encode_domains(self, domains: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Encodes SLD to index sequences and extracts TLDs.
        Returns:
            sequences: np.array of shape (n, max_len)
            tlds: list of TLD strings
        """
        sequences = []
        tlds = []
        for dom in domains:
            sld = extract_sld(dom)
            tld = dom[len(sld):]
            seq = [self.valid_chars.get(c, 0) for c in sld.lower() if c in self.valid_chars]
            sequences.append(seq)
            tlds.append(tld)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post'), tlds

    def _decode_domains(self, sequences: np.ndarray, tlds: List[str]) -> List[str]:
        """
        Decodes index sequences back to domains, appending the original TLDs.
        """
        domains = []
        for seq, tld in zip(sequences, tlds):
            sld = "".join([self.idx_to_char.get(int(i), '') for i in seq if int(i) != 0])
            domains.append(sld + tld)
        return domains

    def _build_target_network_D(self):
        """
        Builds the Target Network D (CNN-based classifier).
        """
        # One-hot input and a bias-free Dense layer are equivalent to an Embedding lookup
        # but keep the input differentiable, so the perturbation loss can reach G.
        model = Sequential(name="Target_Network_D")
        model.add(Input(shape=(self.max_len, self.vocab_size)))
        model.add(Dense(64, use_bias=False, name="embedding_matrix"))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def _build_generator_G(self):
        """
        Builds the ATN Network G to generate adversarial samples X'.
        """
        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=self.vocab_size, output_dim=64)(inputs)
        x = LSTM(128, return_sequences=True)(x)
        outputs = Dense(self.vocab_size, activation='softmax')(x)
        return Model(inputs, outputs, name="Generator_G")

    def _pre_train_generator_mle(self, X_legit: np.ndarray) -> None:
        """
        Pre-trains G using Maximum Likelihood Estimation (MLE) on legitimate domains, to
        predict next character at each position.
        """
        logger.info(f"[{self.name}] MLE Pre-training G on {len(X_legit)} benign domains...")
        # Prepare input: sequence without last char, output: shifted sequence
        X_in = X_legit[:, :-1]  # remove last position
        X_in = np.pad(X_in, ((0,0), (1,0)), constant_values=0)  # left-pad to keep max_len
        X_out = X_legit[:, 1:]  # remove first position
        X_out = np.pad(X_out, ((0,0), (0,1)), constant_values=0)  # right-pad
        
        # Convert to one-hot for categorical crossentropy (output shape: batch, max_len, vocab_size)
        y_onehot = np.eye(self.vocab_size)[X_out.astype(int)]
        
        self.G.compile(optimizer=Adam(learning_rate=self.learning_rate),
                       loss='categorical_crossentropy')
        self.G.fit(X_in, y_onehot, epochs=self.mle_epochs, batch_size=self.batch_size, verbose=1)

    def _ste_one_hot(self, continuous_idx: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        """
        Straight-through one-hot of a continuous index: the forward pass uses the rounded
        index, the backward pass flows through a softmax over the distance to each symbol.
        """
        vocab_indices = tf.range(self.vocab_size, dtype=tf.float32)
        soft = tf.nn.softmax(-tf.square(continuous_idx[..., None] - vocab_indices) / temperature, axis=-1)
        hard_idx = tf.clip_by_value(tf.round(continuous_idx), 0, self.vocab_size - 1)
        hard = tf.one_hot(tf.cast(hard_idx, tf.int32), self.vocab_size)
        return tf.stop_gradient(hard - soft) + soft

    def _train_step(self, X_batch, M_batch):
        """
        Custom training step to minimize the joint loss function L (Table 1 of the paper).
        """
        # Positions of the legal domain X (Eq. 18): x_j = 0 beyond |X|. The adversarial
        # sample X' keeps the length of X, so G's output at padded positions is forced to
        # the padding symbol. Without this mask G can fill the tail with a symbol
        # that D scores as benign (for example hyphen runs up to 63 characters).
        X_mask = tf.cast(X_batch > 0, tf.float32)
        pad_one_hot = tf.one_hot(tf.zeros_like(X_batch, dtype=tf.int32), self.vocab_size)

        with tf.GradientTape() as tape:
            # 1. G generates adversarial sample X' (soft one-hot rows), padded like X
            G_probs = self.G(X_batch, training=True)
            G_probs = X_mask[..., None] * G_probs + (1.0 - X_mask[..., None]) * pad_one_hot

            # Continuous approximation of X' (expected index) for the modulo arithmetic
            vocab_indices = tf.range(self.vocab_size, dtype=tf.float32)
            X_prime_continuous = tf.reduce_sum(G_probs * vocab_indices, axis=-1)

            X_float = tf.cast(X_batch, tf.float32)
            M_float = tf.cast(M_batch, tf.float32)

            # 2. Perturbation function R(M, Z), Z = X - X'
            # M' = |2X' - M + Z| % |V|  ->  |X' + X - M| % |V|
            M_prime_continuous = tf.abs(X_prime_continuous + X_float - M_float)
            M_prime_continuous = tf.math.floormod(M_prime_continuous, float(self.vocab_size))

            # 3. Losses
            # Perturbation loss LG (Eq. 12): D's evaluation of X'. Minimising it drives the
            # adversarial legal sample to the malicious side of the boundary (Eq. 10).
            X_prime_hard = tf.one_hot(tf.argmax(G_probs, axis=-1, output_type=tf.int32), self.vocab_size)
            X_prime_ste = tf.stop_gradient(X_prime_hard - G_probs) + G_probs
            LG = tf.reduce_mean(self.D(X_prime_ste, training=False))

            # Object loss LD (Eq. 13): make D classify the perturbed DGA M' as benign
            preds = self.D(self._ste_one_hot(M_prime_continuous), training=False)
            LD = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(preds), preds))

            # Target loss L (Eq. 16)
            loss = self.beta * LG + LD

        grads = tape.gradient(loss, self.G.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.G.trainable_variables))
        return loss

    def _load_malicious_domains(self, target_count: int) -> List[str]:
        """
        Loads malicious domains through core.data_splits.load_malicious_seeds. Keeps the full domain 
        to preserve TLD information. Returns a list of full domain strings, approximately
        target_count in size.
        """
        all_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py (full domains, TLD kept).
            all_domains = load_malicious_seeds(total=target_count)
        except Exception as e:
            logger.warning(f"Warning reading AGDs for GeometricPerturbationDGA: {e}")

        # Fallback: generate random domains if not enough
        if len(all_domains) < target_count:
            fallbacks = [
                f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(8, 15)))}.com"
                for _ in range(target_count - len(all_domains))
            ]
            all_domains.extend(fallbacks)
        else:
            # If we have more than target_count, take a random sample
            if len(all_domains) > target_count:
                all_domains = random.sample(all_domains, target_count)

        return all_domains

    def fit(self) -> None:
        """
        Trains or loads the Geometric Perturbation networks.
        """
        start_t = time.time()
        
        self.D = self._build_target_network_D()
        self.G = self._build_generator_G()
        
        # Load benign domains (SLD only) from D1 slice
        benign_full = load_tranco(TRANCO_D1_BENIGN)
        # Optionally sample a subset for training speed (e.g., 100k)
        benign_full = benign_full[:100000]
        X_legit, _ = self._encode_domains(benign_full)  # X_legit are SLD sequences
        self.benign_slds, self.benign_tlds = self._encode_domains(benign_full)
        
        # Load malicious domains (SLD only) from DGArchive
        dga_full = self._load_malicious_domains(len(benign_full))
        self.dga_slds, self.dga_tlds = self._encode_domains(dga_full)
        
        # Try to load weights from cache
        if os.path.exists(self.g_weights_path) and os.path.exists(self.d_weights_path):
            logger.info(f"[*] {self.name} found existing weights. Skipping training.")
            self.G.load_weights(self.g_weights_path)
            self.D.load_weights(self.d_weights_path)
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return
            
        logger.info(f"[*] No cached weights found for {self.name}. Training...")
        
        # Ensure balanced dataset sizes
        min_len = min(len(X_legit), len(self.dga_slds))
        X_legit = X_legit[:min_len]
        M_encoded = self.dga_slds[:min_len]
        X_legit_tlds = self.benign_tlds[:min_len]  # not used for training, but kept for consistency
        
        # 1. MLE Pre-training of Generator G
        self._pre_train_generator_mle(X_legit)
        
        # 2. Pre-train Object Network D (classifier)
        logger.info(f"[{self.name}] Pre-training Target Network D...")
        X_train_D = np.vstack((X_legit, M_encoded)).astype(np.int32)
        y_train_D = np.array([1] * len(X_legit) + [0] * len(M_encoded), dtype=np.float32)
        idx = np.arange(len(X_train_D))
        np.random.shuffle(idx)
        # One-hot rows are built per batch (the full matrix would take GBs)
        d_dataset = (tf.data.Dataset.from_tensor_slices((X_train_D[idx], y_train_D[idx]))
                     .batch(self.batch_size)
                     .map(lambda x, y: (tf.one_hot(x, self.vocab_size), y)))
        self.D.fit(d_dataset, epochs=self.d_pretrain_epochs, verbose=1)
        
        # Freeze D for G training
        self.D.trainable = False
        
        # 3. Train Generator G through the geometric perturbation joint loss
        logger.info(f"[{self.name}] Adversarial training of Generator G...")
        dataset = tf.data.Dataset.from_tensor_slices((X_legit, M_encoded)).batch(self.batch_size)
        train_step = tf.function(self._train_step)  # graph mode

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            for X_batch, M_batch in dataset:
                loss = train_step(X_batch, M_batch)
                epoch_loss += loss.numpy()
                batches += 1
            if (epoch+1) % 10 == 0:
                logger.info(f"  -> Epoch {epoch+1}/{self.epochs}, Joint Loss: {epoch_loss/batches:.4f}")
            
        # Save weights in cache
        self.G.save_weights(self.g_weights_path)
        self.D.save_weights(self.d_weights_path)
        
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def _perturb_pair(self) -> str:
        """
        Generates a single adversarial DGA domain by geometrically perturbing a 
        malicious domain using vectors extracted from a benign domain.
        The TLD of the malicious domain is preserved.
        """
        # Pick random benign and DGA samples (preserving their TLDs)
        idx_benign = np.random.randint(0, len(self.benign_slds))
        idx_dga = np.random.randint(0, len(self.dga_slds))
        X_seq = self.benign_slds[idx_benign:idx_benign+1]
        M_seq = self.dga_slds[idx_dga:idx_dga+1]
        tld = self.dga_tlds[idx_dga] if self.dga_tlds[idx_dga] else ".com"
        
        # Calculate adversarial sample X' for the legal domain, padded like X (Eq. 18)
        G_probs = self._g_forward(tf.constant(X_seq)).numpy()
        X_prime = np.argmax(G_probs, axis=-1) * (X_seq > 0)
        
        # Calculate the geometric perturbation vector Z = X - X'
        # Add perturbation to DGA domain: M' = |2X' - M + Z| % |V|
        # Simplified to |X' + X - M| % |V|
        M_prime = np.abs(X_prime + X_seq - M_seq) % self.vocab_size
        
        # Decode SLD and append original TLD
        return self._decode_domains(M_prime, [tld])[0]

    def _g_forward(self, X_seq):
        """Compiled forward pass of G (avoids the per-call overhead of predict())."""
        if getattr(self, "_g_fn", None) is None:
            G = self.G
            self._g_fn = tf.function(lambda x: G(x, training=False),
                                     input_signature=[tf.TensorSpec([None, self.max_len], tf.int32)])
        return self._g_fn(X_seq)

    def generate_domain(self) -> str:
        """
        Generates one adversarial domain. The modulo of Eq. 15 can place a hyphen at
        either end of the label, which RFC 1034/1035 forbid, so pairs are redrawn a
        bounded number of times until the label is valid.
        """
        for _ in range(20):
            domain = self._perturb_pair()
            sld = extract_sld(domain)
            if re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$", sld):
                return domain
        logger.warning(f"[{self.name}] No RFC-valid label in 20 pairs. Trimming the hyphens of the last one.")  # SAFETY GUARD
        tld = domain[len(sld):]
        sld = sld.strip('-') or "".join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(8, 15)))  # SAFETY GUARD
        return sld + tld