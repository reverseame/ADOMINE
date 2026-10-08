import logging
import os
import time
from typing import List, Tuple

import numpy as np

from core.detector import Detector
from core.data_splits import (
    TRANCO_D2_BENIGN,
    load_d2_malicious_training,
    load_tranco,
)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Activation
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    Sequential = None

logger = logging.getLogger(__name__)

# Fixed permutation of the D2 training set: the seven RAMPAGE detectors share the same
# train/validation split regardless of the order in which they are fitted.
SPLIT_SEED = 42


class DBD(Detector):
    """
    DBD detector of Vinayakumar et al. (2019), "DBD: Deep Learning DGA-Based Botnet Detection", as
    implemented in RAMPAGE (classifiers/Vinayakumar/DBD.py): the "DBD (Vinayakumar et al., 2019)"
    row of its tables.

    Architecture: Embedding(256, 128), Conv1D(64, kernel 5), MaxPooling1D(4), LSTM(70, unrolled),
    Dense(1), sigmoid.
    Optimizer: Adam with Keras defaults, binary cross-entropy.
    Training: D2 (100,000 benign and 100,000 malicious), batch 50, up to 500 epochs with early
    stopping on val_accuracy (patience 10, best weights restored), 20 percent validation.
    Label 1 = DGA, as in RAMPAGE; detect() returns True (benign) when P(DGA) <= 0.5.
    """

    def __init__(self, name: str = "DBD"):
        super().__init__(name=name)
        self.weights_path = os.path.join(os.path.dirname(__file__), "weights", "dbd.weights.h5")

        # RAMPAGE hyperparameters (classifiers/common/commonData.py and the model file)
        self.max_len = 75  # RAMPAGE uses 70; 75 covers every domain of D2, D3 and the samples without truncation
        self.embedding_input_dim = 256  # characters are encoded as ord(c) - 33
        self.batch_size = 50
        self.max_epochs = 500
        self.patience = 10  # early stopping on val_accuracy, best weights restored
        self.validation_split = 0.2
        self.model = None

    def _build_model(self):
        if Sequential is None:
            raise ImportError("TensorFlow/Keras is required to run this detector.")

        model = Sequential(name="DBD")
        model.add(Input(shape=(self.max_len,)))
        model.add(Embedding(input_dim=self.embedding_input_dim, output_dim=128))
        model.add(Conv1D(filters=64, kernel_size=5))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(70, unroll=True))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def _encode_domains(self, domains: List[str]) -> np.ndarray:
        """RAMPAGE encoding: each character maps to ord(c) - 33 ('!' is the first printable ASCII character)."""
        sequences = []
        for dom in domains:
            codes = [ord(c) - 33 for c in dom.lower()]
            # SAFETY GUARD: a character outside the embedding alphabet falls to index 0 (padding) instead of crashing
            sequences.append([c if 0 <= c < self.embedding_input_dim else 0 for c in codes])
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def _load_training_data(self) -> Tuple[List[str], np.ndarray]:
        """
        D2 training set, balanced: the equal-per-family malicious draw and the same number of
        Tranco domains, the same wiring as the original detectors. Label 1 = DGA, as in RAMPAGE.
        """
        dga_domains = load_d2_malicious_training()
        benign_domains = load_tranco(TRANCO_D2_BENIGN)[:len(dga_domains)]
        X_str = benign_domains + dga_domains
        y = np.array([0] * len(benign_domains) + [1] * len(dga_domains), dtype=np.float32)
        return X_str, y

    def fit(self) -> None:
        """
        Trains the network on D2 or loads the cached weights. Up to max_epochs epochs with
        early stopping on val_accuracy, as in the RAMPAGE evaluation.
        """
        if Sequential is None:
            raise RuntimeError("TensorFlow is required to train this detector.")

        start_t = time.time()
        self.model = self._build_model()

        if os.path.exists(self.weights_path):
            logger.info(f"[{self.name}] Found existing weights at {self.weights_path}. Loading...")
            self.model.load_weights(self.weights_path)
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        logger.info(f"[{self.name}] No pre-trained weights found. Training from scratch...")
        X_str, y = self._load_training_data()
        if len(X_str) == 0 or not y.any() or y.all():
            logger.warning(f"[{self.name}] Missing datasets. Skipping training process.")
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return
        logger.info(f"[{self.name}] Training set: {int((y == 1).sum())} DGA and {int((y == 0).sum())} benign domains.")

        # Shared seeded permutation; Keras takes the last validation_split fraction as validation
        indices = np.random.RandomState(SPLIT_SEED).permutation(len(X_str))
        X = self._encode_domains([X_str[i] for i in indices])
        y = y[indices]

        early_stopping = EarlyStopping(monitor="val_accuracy", patience=self.patience,
                                       restore_best_weights=True, verbose=1)
        history = self.model.fit(X, y, epochs=self.max_epochs, batch_size=self.batch_size,
                                 validation_split=self.validation_split,
                                 callbacks=[early_stopping], verbose=2)
        epochs_run = len(history.history["loss"])
        best_val_acc = max(history.history["val_accuracy"])
        logger.info(f"[{self.name}] Trained for {epochs_run} epochs. Best val_accuracy {best_val_acc:.4f}.")

        try:
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            self.model.save_weights(self.weights_path)
            logger.info(f"[{self.name}] Training complete. Saved weights to {self.weights_path}")
        except Exception as e:
            logger.warning(f"[{self.name}] Could not save weights properly: {e}")

        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def detect(self, domains: List[str]) -> List[Tuple[str, bool]]:
        """
        Classifies domains into True (benign) or False (malicious/DGA).
        The network outputs P(DGA); as in RAMPAGE, a domain is DGA when P(DGA) > 0.5.
        """
        if self.model is None:
            if Sequential is None:
                raise RuntimeError("TensorFlow is required to run detect().")
            raise RuntimeError("Model must be fitted before calling detect().")

        if not domains:
            return []

        X = self._encode_domains(domains)
        p_dga = self.model.predict(X, batch_size=2048, verbose=0)[:, 0]
        return [(dom, bool(p <= 0.5)) for dom, p in zip(domains, p_dga)]
