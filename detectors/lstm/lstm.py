import logging
import os
import time
import numpy as np
from typing import List, Tuple
from core.detector import Detector
from core.data_splits import (
    TRANCO_D2_BENIGN,
    load_d2_malicious_training,
    load_tranco,
)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    Sequential = None

logger = logging.getLogger(__name__)

class LSTMDetector(Detector):
    """
    Real implementation of the LSTM-based DGA detector proposed by Woodbridge et al. (2016).

    Original detector of the framework (39-character alphabet, rmsprop, 100 epochs), display
    name "LSTM". The RAMPAGE implementation of the same paper is detectors/woodbridge_lstm/.
    """
    def __init__(self, name: str = "LSTM"):
        super().__init__(name=name)
        self.weights_path = os.path.join(os.path.dirname(__file__), "weights", "lstm.weights.h5")

        # Hyperparameters for Woodbridge Architecture
        self.max_len = 75  # covers full domains (SLD plus TLD); a single DNS label caps at 63
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789-_.'
        self.valid_chars = {x: idx + 1 for idx, x in enumerate(chars)}
        self.vocab_size = len(self.valid_chars) + 1  # 39 chars (26 letters, 10 digits, '-_.') + 1 padding = 40
        self.model = None

    def _build_model(self):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras is required to run this detector.")

        model = Sequential()
        model.add(Input(shape=(self.max_len,)))
        model.add(Embedding(input_dim=self.vocab_size, output_dim=128))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def _encode_domains(self, domains: List[str]) -> np.ndarray:
        sequences = []
        for dom in domains:
            dom = dom.lower()
            seq = [self.valid_chars.get(c, 0) for c in dom]
            sequences.append(seq)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

    def fit(self) -> None:
        """
        Trains the Woodbridge architecture LSTM model.
        Extracts balanced datasets from multiple DGA files and Alexa top 1M.
        """
        if Sequential is None:
            raise RuntimeError("TensorFlow is required to train this detector.")

        start_t = time.time()
        self.model = self._build_model()

        # Check if model has already been trained
        if os.path.exists(self.weights_path):
            logger.info(f"[LSTM] Found existing weights at {self.weights_path}. Loading...")
            self.model.load_weights(self.weights_path)
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        logger.info("[LSTM] No pre-trained weights found. Training Woodbridge LSTM from scratch...")

        # 1. Load the equal-per-family malicious draw from the D2 slices.
        dga_domains = load_d2_malicious_training()

        target_count = len(dga_domains)
        logger.info(f"[LSTM] Extracted {target_count} DGA domains in total.")

        # 2. Load Benign domains from the D2 Tranco slice, capped to DGA count.
        benign_domains = load_tranco(TRANCO_D2_BENIGN)[:target_count]
        logger.info(f"[LSTM] Extracted {len(benign_domains)} benign domains.")

        if not benign_domains or not dga_domains:
            logger.warning("[LSTM] Missing datasets. Skipping training process.")
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        # 3. Prepare the final dataset (X and y)
        X_str = benign_domains + dga_domains
        # Labels: 1 = Benign, 0 = DGA
        y = np.array([1]*len(benign_domains) + [0]*len(dga_domains))

        # Shuffle the samples
        indices = np.arange(len(X_str))
        np.random.shuffle(indices)
        X_str = [X_str[i] for i in indices]
        y = y[indices]

        X = self._encode_domains(X_str)

        logger.info(f"[LSTM] Balanced dataset prepared. Training with {len(X)} total instances...")
        # 100 epochs: training stopped improving beyond this in practice
        self.model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

        try:
            # Ensure the weights directory exists before saving
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            self.model.save_weights(self.weights_path)
            logger.info(f"[LSTM] Training complete. Saved weights to {self.weights_path}")
        except Exception as e:
            logger.warning(f"[LSTM] Could not save weights properly: {e}")

        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def detect(self, domains: List[str]) -> List[Tuple[str, bool]]:
        """
        Classifies domains into True (Benign) or False (Malicious/DGA).
        """
        if self.model is None:
            if Sequential is None:
                raise RuntimeError("TensorFlow is required to run detect().")
            raise RuntimeError("Model must be fitted before calling detect().")

        if not domains:
            return []

        X = self._encode_domains(domains)
        preds = self.model.predict(X, verbose=0)

        results = []
        for idx, dom in enumerate(domains):
            is_benign = bool(preds[idx][0] >= 0.5)
            results.append((dom, is_benign))

        return results
