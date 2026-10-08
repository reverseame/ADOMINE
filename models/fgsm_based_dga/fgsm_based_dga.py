from typing import List
import logging
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import h5py

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN, load_malicious_seeds

logger = logging.getLogger(__name__)

class FGSMBasedDGAModel(AdversarialModel):
    """
    Gradient-based Adversarial DGA Model proposed by Yilmaz et al. (2020).
    Uses a target LSTM classifier and FGSM-like continuous noise injection,
    followed by cosine similarity projection to generate discrete text mutations.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Hidden Dimensions of LSTM: Assumed 128 units per layer.
    2. Sequence Control and Padding: Standardized maximum length of 63 characters (SLD).
       Post-padding with <PAD> token (0).
    3. Adversarial Attack Direction: Evasion (untargeted) maximizing BCE loss with respect to y=1.
    4. Perturbation Magnitude: Fixed epsilon = 1.0.
    5. Valid Projection: Strictly restricted to [a-z0-9-].
    6. TLD Isolation: FGSM is applied only on the SLD. Seeds are stored without their
       TLD and every generated domain gets .com appended.
    7. Data Source: Trained with TRANCO_D1_BENIGN and real AGD from DGArchive CSVs
       (fallback to dynamically generated domains if CSVs are missing), ensuring 50/50 class balance.
    8. Persistence Mechanism (.h5 cache): Exports Keras weights and the
       malicious seed list (as a dataset) in two different .h5 files.
    9. Seed Supply: The model uses the real malicious domains loaded from DGArchive during
       training as seeds for adversarial perturbation. generate_domains(n) walks a shuffled
       copy of the seed list without replacement (reshuffling only if n exceeds the number of
       seeds), because the perturbation is deterministic per seed and drawing with replacement
       repeats outputs. generate_domain() alone still draws one seed at random.
    """

    def __init__(self, name: str = "FGSMBasedDGA", epsilon: float = 1.0, base_dga=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.base_dga = base_dga  # Injected generator (only used as fallback)
        self.max_len = 63
        
        # Valid DNS vocabulary (26 lower-case letters + 10 digits + 1 hyphen = 37 characters)
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
        self.vocab_size = len(self.chars) + 1  # +1 to reserve index 0 for the <PAD> token
        
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}
        self.idx2char[0] = ""  # Padding does not contribute text
        
        # Cache directory and paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        
        # The cache name includes the base_dga name to avoid mixing classifiers
        # trained on different malware families.
        base_name = self.base_dga.name if self.base_dga else "default"
        self.weights_path = os.path.join(self.weights_dir, f'fgsm_model_{base_name}.weights.h5')
        self.seeds_path = os.path.join(self.weights_dir, f'fgsm_seeds.h5')
        
        # Build classifier architecture
        self.classifier = self._build_classifier()
        self.malicious_seeds = []   # list of SLDs (strings) used as seeds for generation
        self.trained_from_scratch = False

    def _build_classifier(self):
        """
        Builds the LSTM classifier architecture described in Yilmaz (2020).
        """
        inputs = Input(shape=(self.max_len,), dtype='int32', name='input_sequence')
        
        x = Embedding(input_dim=self.vocab_size, output_dim=256, mask_zero=True, name='embedding')(inputs)
        x = LSTM(128, return_sequences=True, name='lstm_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = LSTM(128, return_sequences=False, name='lstm_2')(x)
        x = Dropout(0.5, name='dropout_2')(x)
        
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss=BinaryCrossentropy(), 
                      metrics=['accuracy'])
        return model

    def _encode_domain(self, domain_sld: str) -> np.ndarray:
        """
        Converts a string into an integer tensor with post-padding.
        """
        seq = [self.char2idx.get(c, 0) for c in domain_sld[:self.max_len]]
        padded = seq + [0] * (self.max_len - len(seq))
        return np.array(padded, dtype=np.int32)

    def fit(self) -> None:
        """
        Loads or trains the target model and caches the malicious seed list.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Try to load cached model and seeds from the .h5 files
        if os.path.exists(self.weights_path) and os.path.exists(self.seeds_path):
            try:
                logger.info(f"[*] FGSM-based DGA found cached weights and seeds. Skipping all training phases.")
                self.classifier.load_weights(self.weights_path)
                # Read the seeds dataset
                with h5py.File(self.seeds_path, 'r') as f:
                    if 'seeds' in f:
                        seeds_data = f['seeds'][:]
                        # Decode bytes to strings
                        self.malicious_seeds = [x.decode('utf-8') if isinstance(x, bytes) else x for x in seeds_data]
                    else:
                        raise KeyError("seeds dataset not found in cache")
                self.trained_from_scratch = False
                self.is_fitted = True
                return
            except Exception as e:
                logger.error(f"[!] Error loading cache: {e}")

        logger.info(f"[*] No cached weights or seeds. Training...")
        # If not cached or loading failed, train from scratch
        # Load Benign Domains (Tranco)
        benign_domains = load_tranco(TRANCO_D1_BENIGN)
        benign_pool = [d.split('.')[0] for d in benign_domains]

        # Load Malicious Domains (AGD) from CSV files (DGArchive)
        agd_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py (TLD isolated below)
            seeds = load_malicious_seeds(total=100000)
            agd_domains = [d.split('.')[0] for d in seeds if d.split('.')[0]]
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs: {e}")

        # Fallback if no CSVs are found to avoid crashing
        if not agd_domains:
            logger.warning("[*] No AGD domains found in CSVs. Falling back to dynamic generation.")
            if self.base_dga:
                if not getattr(self.base_dga, 'is_fitted', True):
                    self.base_dga.fit()
                agd_domains = [self.base_dga.generate_domain().split('.')[0] for _ in range(50000)]
            else:
                agd_domains = ["".join(random.choices(self.chars[:26], k=random.randint(12, 22))) for _ in range(50000)]

        # Store the malicious seeds for later generation
        self.malicious_seeds = agd_domains[:]

        # Balance the dataset (50/50 split)
        num_samples = min(len(benign_pool), len(agd_domains), 50000)
        
        X_texts = benign_pool[:num_samples] + agd_domains[:num_samples]
        y_labels = [0] * num_samples + [1] * num_samples
        
        # Shuffle and encode
        combined = list(zip(X_texts, y_labels))
        random.shuffle(combined)
        
        X_encoded = np.array([self._encode_domain(d) for d, _ in combined])
        y_encoded = np.array([l for _, l in combined])
        
        # Build and Train the Classifier
        self.classifier = self._build_classifier()
        self.classifier.fit(X_encoded, y_encoded, batch_size=128, epochs=6, validation_split=0.1, verbose=1)
        
        # Save model weights and seeds in two different .h5 files
        self.classifier.save_weights(self.weights_path)
        # Now append the seeds dataset
        with h5py.File(self.seeds_path, 'a') as f:
            # Delete existing seeds if any
            if 'seeds' in f:
                del f['seeds']
            # Create string dataset
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('seeds', data=np.array(self.malicious_seeds, dtype=object), dtype=dt)
        
        self.trained_from_scratch = True
        self.is_fitted = True

    @tf.function
    def _compute_adversarial_step(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        FGSM mathematical core in Keras. Computes gradients and returns the
        cosine similarity matrix with respect to valid characters.
        """
        embedding_layer = self.classifier.get_layer('embedding')
        
        with tf.GradientTape() as tape:
            embeddings = embedding_layer(input_tensor)
            tape.watch(embeddings)
            
            x = self.classifier.get_layer('lstm_1')(embeddings)
            x = self.classifier.get_layer('dropout_1')(x)
            x = self.classifier.get_layer('lstm_2')(x)
            x = self.classifier.get_layer('dropout_2')(x)
            predictions = self.classifier.get_layer('output')(x)
            
            # Untargeted attack: push prediction away from 1 (malicious)
            target_labels = tf.ones_like(predictions)
            bce = tf.keras.losses.binary_crossentropy(target_labels, predictions)
        
        gradients = tape.gradient(bce, embeddings)
        signed_grad = tf.sign(gradients)
        perturbed_embeddings = embeddings + (self.epsilon * signed_grad)
        
        vocab_embeddings = embedding_layer.embeddings 
        
        pert_norm = tf.nn.l2_normalize(perturbed_embeddings, axis=-1)
        vocab_norm = tf.nn.l2_normalize(vocab_embeddings, axis=-1)
        
        similarity = tf.matmul(pert_norm, vocab_norm, transpose_b=True)
        return similarity

    def generate_domains(self, n: int) -> List[str]:
        """
        Generates n domains using each cached seed at most once per pass. The FGSM step is
        deterministic for a given seed, so sampling with replacement would repeat outputs.
        """
        if not self.is_fitted:
            self.fit()
        if not self.malicious_seeds:
            return [self.generate_domain() for _ in range(n)]
        domains = []
        while len(domains) < n:
            pool = self.malicious_seeds[:]
            random.shuffle(pool)
            for seed in pool[:n - len(domains)]:
                domains.append(self._perturb_seed(seed))
        return domains

    def generate_domain(self) -> str:
        """
        Generates a single domain by mutating a malicious seed drawn at random.
        """
        if not self.is_fitted:
            self.fit()
        if self.malicious_seeds:
            return self._perturb_seed(random.choice(self.malicious_seeds))
        return self._perturb_seed(None)

    def _perturb_seed(self, seed) -> str:
        """
        Applies the FGSM perturbation to one seed SLD. A None seed triggers the fallback.
        """
        # 1. Base AGD name; seeds are cached without TLD, so .com is appended
        if seed is not None:
            if '.' not in seed:
                seed += '.com'
        else:
            # Fallback
            logger.warning("[!] No malicious seeds available. Using fallback generation.")
            if self.base_dga:
                seed = self.base_dga.generate_domain()
            else:
                length = random.randint(12, 22)
                seed = "".join(random.choices(self.chars[:26], k=length)) + ".com"
            
        parts = seed.split('.')
        sld = parts[0]
        tld = parts[1] if len(parts) > 1 else "com"
        original_length = len(sld)
        
        # 2. Encode and convert to tensor
        x_tensor = tf.convert_to_tensor([self._encode_domain(sld)], dtype=tf.int32)
        
        # 3. Execute FGSM attack
        similarity_matrix = self._compute_adversarial_step(x_tensor)
        
        # 4. Discrete projection using restricted argmax (exclude padding index 0)
        mask = tf.one_hot(0, depth=self.vocab_size, on_value=-1e9, off_value=0.0)
        similarity_matrix = similarity_matrix + mask
        
        best_indices = tf.argmax(similarity_matrix[0], axis=-1).numpy()
        
        # 5. Reconstruct the string domain (keep original length)
        adversarial_sld = "".join([self.idx2char[idx] for idx in best_indices[:original_length]])
        
        # 6. Sanitize SLD to comply with DNS format (no leading/trailing hyphen)
        if original_length > 0:
            # Replace leading hyphen with the original first character
            if adversarial_sld[0] == '-':
                adversarial_sld = sld[0] + adversarial_sld[1:]
            # Replace trailing hyphen with the original last character
            if adversarial_sld[-1] == '-':
                adversarial_sld = adversarial_sld[:-1] + sld[-1]
            # Edge case: if the whole domain became empty
            if not adversarial_sld:
                adversarial_sld = sld  # fallback to the original seed
        else:
            adversarial_sld = sld  # fallback
        
        return f"{adversarial_sld}.{tld}"