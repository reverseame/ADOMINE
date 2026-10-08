import logging
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Embedding
from tensorflow.keras.models import Model
from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco, load_malicious_seeds
import random

logger = logging.getLogger(__name__)

class MaskDGAModel(AdversarialModel):
    """
    MaskDGA Model based on the Sidi et al. paper.
    It uses a substitute model to compute the Jacobian saliency matrix and 
    modify base AGD names to evade detection.
    """
    def __init__(self, name: str = "MaskDGA", base_dga=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = 63
        # 38 symbols: 26 letters + 10 digits + '-' + padding symbol (e.g., 0)
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
        self.char_indices = {c: i+1 for i, c in enumerate(self.chars)}  # +1 to reserve 0 for padding
        self.indices_char = {i+1: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1 # 38 + 1 = 39
        
        # The base DGA to be evaded (if none is provided, we simulate one)
        self.base_dga = base_dga
        
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self._build_substitute_model()

    def _build_substitute_model(self):
        """
        Builds the substitute model based on the Invincea CNN architecture as described in the paper:
        63x38 one-hot max length, CNN filters (2,3,4,5), 256 filters, 128 units concatenated,
        followed by two dense layers (1024 units), and output sigmoid.
        """
        inp = Input(shape=(self.max_len, self.vocab_size), name="input_onehot")
        
        convs = []
        for kernel_size in [2, 3, 4, 5]:
            # 256 filters total for 4 kernel sizes implies 64 filters per kernel
            c = Conv1D(filters=64, kernel_size=kernel_size, activation='relu')(inp)
            c = GlobalMaxPooling1D()(c)
            # The paper says 128 units, we can use a dense layer for projection if needed, 
            # or just rely on concatenation. We'll use a Dense to match "128 units".
            c = Dense(128, activation='relu')(c)
            convs.append(c)
            
        x = concatenate(convs)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        out = Dense(1, activation='sigmoid', name="output_sigmoid")(x)
        
        self.substitute_model = Model(inp, out, name="SubstituteModel")
        
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.substitute_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    def fit(self) -> None:
        """
        Trains the substitute model. Ideally, trained with benign domains and AGD samples.
        To maintain decoupling, it reads from the dataset as do other models, 
        and simulates AGD samples for binary classification learning.
        """
        start_t = time.time()
        substitute_weights_path = os.path.join(self.weights_dir, "substitute_model.weights.h5")
        if os.path.exists(substitute_weights_path):
            logger.info("[*] MaskDGA found cached substitute model weights. Skipping training phase.")
            self.substitute_model.load_weights(substitute_weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return
            
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # 1. READ BENIGN DOMAINS from the D1 Tranco slice.
        benign_domains_pool = [d.split('.')[0] for d in load_tranco(TRANCO_D1_BENIGN)]
        benign_domains = random.sample(benign_domains_pool, min(100000, len(benign_domains_pool)))
            
        # 2. READ MALICIOUS DOMAINS
        agd_domains = []
        try:
            # Real AGDs come from the shared, row-bounded seed draw in core/data_splits.py (TLD isolated below)
            agd_domains = [d.split('.')[0] for d in load_malicious_seeds(total=100000) if d.split('.')[0]]
        except Exception as e:
            logger.warning(f"[*] Warning reading AGDs: {e}")
            
        if not agd_domains:
            agd_domains = ["".join(random.choices(self.chars[:26], k=random.randint(10, 20))) for _ in range(len(benign_domains))]
        
        # Prepare Data
        X_str = benign_domains + agd_domains
        y = np.array([0]*len(benign_domains) + [1]*len(agd_domains))
        
        X_ohe = np.zeros((len(X_str), self.max_len, self.vocab_size), dtype=np.float32)
        for i, domain in enumerate(X_str):
            for t, char in enumerate(domain[:self.max_len]):
                if char in self.char_indices:
                    X_ohe[i, t, self.char_indices[char]] = 1.0
                
        # Shuffle
        indices = np.arange(len(X_str))
        np.random.shuffle(indices)
        X_ohe = X_ohe[indices]
        y = y[indices]
        
        self.substitute_model.fit(X_ohe, y, epochs=25, batch_size=64)  # paper: 25 epochs
        self.substitute_model.save_weights(substitute_weights_path)
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def _encode_onehot(self, domain: str) -> np.ndarray:
        X = np.zeros((1, self.max_len, self.vocab_size), dtype=np.float32)
        for t, char in enumerate(domain[:self.max_len]):
            if char in self.char_indices:
                X[0, t, self.char_indices[char]] = 1.0
        return X

    def get_gradients(self, X_input: np.ndarray) -> np.ndarray:
        """
        Calculates the Jacobian saliency matrix against the benign class (0).
        Using Binary Crossentropy loss.
        """
        X_tensor = tf.convert_to_tensor(X_input, dtype=tf.float32)
        target_labels = tf.constant([[0.0]]) # Try to make it predictable as benign class
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.substitute_model(X_tensor, training=False)
            loss = tf.keras.losses.binary_crossentropy(target_labels, predictions)
            
        gradients = tape.gradient(loss, X_tensor)
        return gradients.numpy()[0] # Return the first sample gradient matrix

    def maskDGA(self, X: np.ndarray, S: np.ndarray, original_domain_len: int) -> str:
        """
        Modifies the base AGD name according to MaskDGA policy.
        X: (max_len, vocab_size) One-hot
        S: (max_len, vocab_size) Gradients
        """
        # We only consider the actual characters of the domain, avoiding padding modifications
        # If we didn't slice, we might inject random letters at the end
        X_valid = X[:original_domain_len]
        S_valid = S[:original_domain_len]
        
        # Calculate maximum gradient per character position
        scores = np.max(S_valid, axis=1)
        # We calculate median over valid positions to restrict changing too many characters
        if len(scores) == 0:
            median_score = 0
        else:
            median_score = np.percentile(scores, 50)
            
        adversarial = ""
        for i, (pos_ohe, pos_gradients) in enumerate(zip(X_valid, S_valid)):
            original_symbol = np.argmax(pos_ohe)
            new_symbol = np.argmax(pos_gradients)
            
            if np.max(pos_gradients) > median_score:
                # Use the new adversarial symbol if its position importance > median
                sym_to_add = new_symbol
            else:
                sym_to_add = original_symbol
                
            # If symbol is 0 (padding) or out of bounds, skip or fallback a valid char
            if sym_to_add in self.indices_char:
                adversarial += self.indices_char[sym_to_add]
            else:
                # fallback to original if something corrupted
                orig_char = self.indices_char.get(original_symbol, 'a')
                adversarial += orig_char
                
        return adversarial

    def generate_domain(self) -> str:
        """
        Generates an adversarial domain by executing a base DGA and then modifying it.
        """
        if not self.is_fitted:
            self.fit()
            
        # 1. Provide a base AGD name
        if self.base_dga:
            base_domain = self.base_dga.generate_domain().split('.')[0]
        else:
            # Simulate a standard DGA
            length = random.randint(12, 22)
            base_domain = "".join(random.choices(self.chars[:26], k=length))
            
        domain_len = len(base_domain)
        if domain_len == 0:
            logger.warning(f"[{self.name}] Empty base domain. Returning a random fallback.")  # SAFETY GUARD
            return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + ".com"  # SAFETY GUARD: random SLD fallback
            
        # 2. Translate base domain to one-hot layout X
        X_ohe_batch = self._encode_onehot(base_domain)
        X_single = X_ohe_batch[0]
        
        # 3. Compute Gradients S
        S_single = self.get_gradients(X_ohe_batch)
        
        # 4. Apply Modification Mask (MaskDGA technique)
        adversarial_domain = self.maskDGA(X_single, S_single, domain_len)
        
        return adversarial_domain.strip('-') + ".com"

# Quick test if run standalone
if __name__ == "__main__":
    model = MaskDGAModel()
    model.fit()
    print("MaskDGA test generation:", model.generate_domain())
