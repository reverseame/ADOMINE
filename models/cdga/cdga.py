import os
import random
import time
import logging
import h5py
import numpy as np
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Add, Activation, Flatten, GlobalAveragePooling1D, Embedding
from tensorflow.keras.models import Model
import sentencepiece as spm
from gensim.models import Word2Vec

from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from core.seeding import get_seed
from core.sld import extract_sld

logger = logging.getLogger(__name__)

class CDGAModel(AdversarialModel):
    """
    Implementation of CDGA proposed by Zhai et al. (2022). The algorithm is based on 
    WGAN-GP and Sequence Autoencoder architectures.

    REPRODUCIBILITY ASSUMPTIONS:
    1. NLP Processing: Uses SentencePiece (Unigram model, vocab size 5000) and Word2Vec 
       (continuous skip-gram, embedding dim 128, window 5, min_count 1).
    2. Autoencoder Architecture: Comprises 2 LSTM layers with 256 hidden units. The context 
       vector is calculated using GlobalAveragePooling1D from the encoder's hidden states 
       and initializes both LSTM layers in the decoder. Optimized with Adam (lr=0.001).
    3. WGAN-GP Architecture: Latent noise vector Z has a dimension of 100. The Generator is 
       a single Dense layer (256 units, linear activation). The Critic uses 3 1D Residual 
       Blocks (kernel size 3, ReLU). Training uses lambda=10.0, 5 critic updates per 
       generator update, and Adam optimizers (lr=0.0001, beta_1=0.0, beta_2=0.9).
    4. Training Hyperparameters: Batch size is 64. The Autoencoder is trained for 50 epochs, 
       and the WGAN-GP for 200 epochs.
    5. Domain Handling: Models strictly on Second-Level Domains (SLDs) with TLDs removed. 
       Domain lengths are bounded between 4 and 63 characters, right-padded with ID 0.
    6. Generation Logic: Invalid tokens (PAD, UNK, BOS) are strictly masked (probability 0). 
       If a valid/unique domain is not generated after 100 attempts, a fallback string is returned.
    7. Seed Sychronization: A fixed global seed is used for experimental reproducibility 
       instead of the time-dependent seed described in the paper.
    """
    def __init__(self, name: str = "CDGA", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Hyperparameters
        self.vocab_size = 5000
        self.embedding_dim = 128
        self.lstm_layers = 2
        self.hidden_units = 256
        self.z_dim = 100
        self.min_len = 4
        self.max_len = 63
        self.batch_size = 64
        self.ae_epochs = 50
        self.gan_epochs = 200
        self.n_critic = 5 # Number of critic updates per generator update
        self.lambda_gp = 10.0
        
        # Prefix for temporary NLP models and caching
        self.spm_prefix = "sentencepiece_model"

        # Caching paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.spm_model_file = os.path.join(self.weights_dir, f"{self.spm_prefix}.model")
        self.cache_file = os.path.join(self.weights_dir, "cdga_cache.weights.h5")
        self.encoder_weights_file = os.path.join(self.weights_dir, "cdga_encoder.weights.h5")
        self.decoder_weights_file = os.path.join(self.weights_dir, "cdga_decoder.weights.h5")
        self.generator_weights_file = os.path.join(self.weights_dir, "cdga_generator.weights.h5")
        self.critic_weights_file = os.path.join(self.weights_dir, "cdga_critic.weights.h5")
        
        # Models
        self.sp = None
        self.w2v = None
        self.encoder = None
        self.decoder = None
        self.generator = None
        self.critic = None
        self.inference_decoder = None  # Model for step‑by‑step generation with state reuse

        # Set for repetition control (persisted in cache)
        self.generated_set = set()

    def _train_nlp_components(self, domains: List[str]):
        """
        Trains SentencePiece and Word2Vec models as defined in the assumptions.
        """
        os.makedirs(self.weights_dir, exist_ok=True)

        logger.info(f"[*] {self.name} training SentencePiece (Unigram, vocab={self.vocab_size})...")
        temp_txt = "cdga_temp_domains.txt"
        with open(temp_txt, "w", encoding="utf-8") as f:
            for d in domains:
                f.write(f"{d}\n")
                
        spm.SentencePieceTrainer.train(
            input=temp_txt, 
            model_prefix=os.path.join(self.weights_dir, self.spm_prefix), 
            vocab_size=self.vocab_size,
            model_type='unigram',
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
        self.sp = spm.SentencePieceProcessor(model_file=self.spm_model_file)
        os.remove(temp_txt)

        logger.info(f"[*] {self.name} training Word2Vec (skip-gram, dim={self.embedding_dim})...")
        tokenized_domains = [self.sp.encode(d, out_type=str) for d in domains]
        
        self.w2v = Word2Vec(
            sentences=tokenized_domains, 
            vector_size=self.embedding_dim, 
            window=5, 
            min_count=1, 
            sg=1, # Continuous skip-gram
            seed=get_seed(),  # gensim has no global RNG; workers=1 makes the embedding deterministic
            workers=1
        )
        
        # Create embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for i in range(self.vocab_size):
            token = self.sp.id_to_piece(i)
            if token in self.w2v.wv:
                self.embedding_matrix[i] = self.w2v.wv[token]

    def _encode_dataset(self, domains: List[str]) -> np.ndarray:
        """
        Tokenizes and pads domains to max_len.
        """
        X = []
        for d in domains:
            tokens = self.sp.encode(d, out_type=int)
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                tokens = tokens + [0] * (self.max_len - len(tokens)) # Right pad with <pad> ID (0)
            X.append(tokens)
        return np.array(X)

    def _build_autoencoder(self):
        """
        Builds the LSTM Autoencoder with 2 layers and 256 hidden units. z (context) is 
        computed as the average of encoder hidden states (Eq. 3). Decoder is autoregressive: 
        uses teacher forcing during training. Also builds a separate inference model for 
        step‑by‑step generation with state reuse.
        """
        # Encoder
        enc_input = Input(shape=(self.max_len,), name="enc_input")
        emb = Embedding(self.vocab_size, self.embedding_dim, 
                        weights=[self.embedding_matrix], trainable=False, name="enc_embedding")(enc_input)
        
        # First LSTM layer, return sequences for next layer
        h1, state_h1, state_c1 = LSTM(self.hidden_units, return_sequences=True, return_state=True, name="enc_lstm1")(emb)
        # Second LSTM layer, return sequences to compute average context
        h2, state_h2, state_c2 = LSTM(self.hidden_units, return_sequences=True, return_state=True, name="enc_lstm2")(h1)
        
        # Compute context z as average of hidden states (Eq. 3) using GlobalAveragePooling1D
        z = GlobalAveragePooling1D()(h2)  # shape: (batch, hidden_units)
        self.encoder = Model(enc_input, z, name="Encoder")

        # Decoder: autoregressive, input is shifted sequence (BOS + tokens[:-1])
        dec_input = Input(shape=(self.max_len,), name="dec_input")
        dec_emb = Embedding(self.vocab_size, self.embedding_dim, 
                            weights=[self.embedding_matrix], trainable=False, name="dec_embedding")(dec_input)
        
        # Context vector z is used to initialize both LSTM layers' states
        z_input = Input(shape=(self.hidden_units,), name="z_input")
        
        # First LSTM layer, initialized with z
        dec_lstm1 = LSTM(self.hidden_units, return_sequences=True, return_state=True, name="dec_lstm1")
        dec_lstm1_out, dec_state_h1, dec_state_c1 = dec_lstm1(dec_emb, initial_state=[z_input, z_input])
        
        # Second LSTM layer, initialized with z (also use z as initial state)
        dec_lstm2 = LSTM(self.hidden_units, return_sequences=True, return_state=True, name="dec_lstm2")
        dec_lstm2_out, dec_state_h2, dec_state_c2 = dec_lstm2(dec_lstm1_out, initial_state=[z_input, z_input])
        
        outputs = Dense(self.vocab_size, activation="softmax", name="dec_output")(dec_lstm2_out)
        
        self.decoder = Model([dec_input, z_input], outputs, name="Decoder")
        
        # Full Autoencoder: encoder -> decoder (with shifted input)
        # During training, we will provide dec_input separately, so we build a separate model
        # that takes both enc_input and dec_input.
        ae_enc_out = self.encoder(enc_input)
        ae_out = self.decoder([dec_input, ae_enc_out])
        self.ae = Model([enc_input, dec_input], ae_out, name="Autoencoder")
        self.ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                        loss="sparse_categorical_crossentropy")

        # Build inference model for step‑by‑step generation
        # Inputs: token (scalar), current states (h1,c1,h2,c2)
        token_in = Input(shape=(1,), dtype='int32', name="token_in")
        state_h1_in = Input(shape=(self.hidden_units,), name="state_h1_in")
        state_c1_in = Input(shape=(self.hidden_units,), name="state_c1_in")
        state_h2_in = Input(shape=(self.hidden_units,), name="state_h2_in")
        state_c2_in = Input(shape=(self.hidden_units,), name="state_c2_in")

        # Embed token using the same embedding layer as the decoder
        token_emb = self.decoder.get_layer('dec_embedding')(token_in)

        # First LSTM: use the same layer as in decoder (weights are shared)
        lstm1 = self.decoder.get_layer('dec_lstm1')
        lstm1_out, new_h1, new_c1 = lstm1(token_emb, initial_state=[state_h1_in, state_c1_in])

        # Second LSTM: use the same layer as in decoder
        lstm2 = self.decoder.get_layer('dec_lstm2')
        lstm2_out, new_h2, new_c2 = lstm2(lstm1_out, initial_state=[state_h2_in, state_c2_in])

        # Output projection (shared with decoder)
        dense = self.decoder.get_layer('dec_output')
        logits = dense(lstm2_out)  # shape (1, vocab_size)

        self.inference_decoder = Model(
            inputs=[token_in, state_h1_in, state_c1_in, state_h2_in, state_c2_in],
            outputs=[logits, new_h1, new_c1, new_h2, new_c2],
            name="InferenceDecoder"
        )

    def _build_wgan(self):
        """
        Builds the WGAN-GP components: MLP Generator and ResNet Critic.
        """
        # Generator (1 Dense Layer)
        gen_input = Input(shape=(self.z_dim,))
        x = Dense(self.hidden_units, activation="linear")(gen_input)
        self.generator = Model(gen_input, x, name="Generator")

        # Critic (3 1D Residual Blocks)
        critic_input = Input(shape=(self.hidden_units,))
        x_reshaped = tf.keras.layers.Reshape((self.hidden_units, 1))(critic_input)
        
        x = x_reshaped
        for _ in range(3):
            residual = x
            x = Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
            x = Conv1D(1, kernel_size=3, padding="same")(x)
            x = Add()([x, residual])
            x = Activation("relu")(x)
            
        x = Flatten()(x)
        critic_output = Dense(1)(x)
        self.critic = Model(critic_input, critic_output, name="Critic")
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)

    @tf.function
    def _critic_train_step(self, real_z, batch_size):
        noise = tf.random.normal([batch_size, self.z_dim])
        with tf.GradientTape() as critic_tape:
            fake_z = self.generator(noise, training=True)
            
            fake_logits = self.critic(fake_z, training=True)
            real_logits = self.critic(real_z, training=True)
            
            c_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            
            # Gradient Penalty
            alpha = tf.random.uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            interpolated = alpha * real_z + (1 - alpha) * fake_z
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                inter_logits = self.critic(interpolated, training=True)
            grads = gp_tape.gradient(inter_logits, interpolated)
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]) + 1e-12)
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            
            critic_loss = c_cost + self.lambda_gp * gp
            
        c_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_variables))
        return critic_loss

    @tf.function
    def _generator_train_step(self, batch_size):
        noise = tf.random.normal([batch_size, self.z_dim])
        with tf.GradientTape() as gen_tape:
            fake_z = self.generator(noise, training=True)
            fake_logits = self.critic(fake_z, training=True)
            gen_loss = -tf.reduce_mean(fake_logits)
            
        g_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return gen_loss

    # ---------- Caching helpers ----------
    def _save_cache(self) -> None:
        """
        Saves trained embedding matrix, model weights, and generated set to disk.
        """
        os.makedirs(self.weights_dir, exist_ok=True)
        try:
            with h5py.File(self.cache_file, "w") as f:
                f.create_dataset("embedding_matrix", data=self.embedding_matrix)
                # Save generated_set as an array of strings
                if self.generated_set:
                    # Convert to list of bytes for HDF5 string storage
                    gen_list = list(self.generated_set)
                    dt = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset("generated_set", data=np.array(gen_list, dtype=object), dtype=dt)
                else:
                    # Create empty dataset
                    f.create_dataset("generated_set", data=np.array([], dtype=object), dtype=h5py.string_dtype(encoding='utf-8'))
                
            self.encoder.save_weights(self.encoder_weights_file)
            self.decoder.save_weights(self.decoder_weights_file)
            self.generator.save_weights(self.generator_weights_file)
            self.critic.save_weights(self.critic_weights_file)

        except Exception as e:
            logger.error(f"[*] Error saving CDGA cache: {e}")

    def _load_cache(self) -> None:
        """
        Loads cached SentencePiece model, embedding matrix, neural network weights, and generated_set.
        """
        self.sp = spm.SentencePieceProcessor(model_file=self.spm_model_file)
        
        with h5py.File(self.cache_file, "r") as f:
            self.embedding_matrix = f["embedding_matrix"][:]
            # Load generated_set if present
            if "generated_set" in f:
                gen_data = f["generated_set"][:]
                # Decode bytes to strings if necessary
                if gen_data.dtype == object:
                    self.generated_set = set(
                        x.decode('utf-8') if isinstance(x, bytes) else x
                        for x in gen_data
                    )
                else:
                    self.generated_set = set(gen_data)
            else:
                self.generated_set = set()
            
        self._build_autoencoder()
        self._build_wgan()
        
        self.encoder.load_weights(self.encoder_weights_file)
        self.decoder.load_weights(self.decoder_weights_file)
        self.generator.load_weights(self.generator_weights_file)
        self.critic.load_weights(self.critic_weights_file)

    def fit(self) -> None:
        """
        Trains the full CDGA pipeline: NLP -> AE -> WGAN-GP or loads from cache if available.
        """
        start_time = time.time()
        
        cache_files_exist = (
            os.path.exists(self.cache_file) and
            os.path.exists(self.spm_model_file) and
            os.path.exists(self.encoder_weights_file) and
            os.path.exists(self.decoder_weights_file) and
            os.path.exists(self.generator_weights_file) and
            os.path.exists(self.critic_weights_file)
        )

        # Try to load everything from cache
        if cache_files_exist:
            logger.info(f"[*] {self.name} found cached model components. Skipping training.")
            self._load_cache()
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            return

        logger.info(f"[*] No cached model components found for {self.name}. Training...")
        raw_domains = load_tranco(TRANCO_D1_BENIGN)
        if not raw_domains:
            return
            
        # Extract only SLD labels (removes TLDs and subdomains) and deduplicate
        extracted_domains = set()
        for d in raw_domains:
            sld = extract_sld(d)
            if sld:
                extracted_domains.add(sld)
        domains = list(extracted_domains)
            
        self._train_nlp_components(domains)
        X = self._encode_dataset(domains)  # shape: (N, max_len)
        
        self._build_autoencoder()
        
        # Prepare decoder input (shifted) and target (original)
        # Shift right: [BOS] + tokens[:-1]
        dec_input = np.full_like(X, 0)  # pad with 0
        dec_input[:, 0] = 2  # BOS token ID
        dec_input[:, 1:] = X[:, :-1]
        
        # Target is original X (we predict original tokens)
        target = X
        
        logger.info(f"[*] {self.name} training Autoencoder for {self.ae_epochs} epochs...")
        self.ae.fit([X, dec_input], target, 
                    batch_size=self.batch_size, 
                    epochs=self.ae_epochs, 
                    verbose=1)
        
        logger.info(f"[*] {self.name} extracting Latent Context Vectors (z)...")
        real_z_dataset = self.encoder.predict(X, batch_size=self.batch_size)
        
        self._build_wgan()
        
        logger.info(f"[*] {self.name} training WGAN-GP for {self.gan_epochs} epochs...")
        dataset = tf.data.Dataset.from_tensor_slices(real_z_dataset).shuffle(10000).batch(self.batch_size, drop_remainder=True)
        
        for epoch in range(self.gan_epochs):
            for batch_z in dataset:
                for _ in range(self.n_critic):
                    self._critic_train_step(batch_z, self.batch_size)
                self._generator_train_step(self.batch_size)
                
        # Save cache
        self._save_cache()
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time

    def _is_valid_prefix(self, domain: str) -> bool:
        """
        Validates a prefix (partial domain) during generation.
        A prefix must:
        - Not be empty.
        - Have length <= max_len (63).
        - Contain only alphanumeric characters or hyphens.
        - Not start with a hyphen (a leading hyphen can never be fixed later).
        - Not enforce minimum length, and not check trailing hyphen (since it can be fixed later).
        """
        # Remove SentencePiece special underscore character and whitespace
        domain = domain.replace('_', '').replace(' ', '')
        if not domain:
            return False
        if len(domain) > self.max_len:
            return False
        # First character cannot be hyphen (invalid prefix)
        if domain[0] == '-':
            return False
        for char in domain:
            if not (char.isalnum() or char == '-'):
                return False
        return True

    def _is_valid_full_domain(self, domain: str) -> bool:
        """
        Validates a complete domain name according to RFC 1034/1035 and the paper's requirements.
        A full domain must:
        - Not be empty.
        - Have length between 4 and 63 (inclusive).
        - Contain only alphanumeric characters or hyphens.
        - Not start or end with a hyphen.
        """
        # Remove SentencePiece special underscore character and whitespace
        domain = domain.replace('_', '').replace(' ', '')
        if not domain:
            return False
        if len(domain) < self.min_len or len(domain) > self.max_len:
            return False
        if domain[0] == '-' or domain[-1] == '-':
            return False
        for char in domain:
            if not (char.isalnum() or char == '-'):
                return False
        return True

    def generate_domain(self) -> str:
        """
        Generates a single domain using Controllable Text Generation (Alg 1). Dynamically 
        applies masking to force P=0 on invalid tokens and ensures no repetition. Uses the 
        inference decoder with state reuse for O(n) generation.
        """
        if not self.is_fitted:
            self.fit()

        max_attempts = 100
        for attempt in range(max_attempts):
            noise = tf.random.normal([1, self.z_dim])
            z = self.generator(noise, training=False)
            # z is a tensor of shape (1, hidden_units); convert to numpy for convenience
            z_np = z.numpy()[0]  # shape (hidden_units,)

            # Initial states: both h and c are initialized with z (as in training)
            state_h1 = z_np.copy()
            state_c1 = z_np.copy()
            state_h2 = z_np.copy()
            state_c2 = z_np.copy()

            # Start with BOS token
            current_token = np.array([[2]], dtype=np.int32)  # BOS id = 2
            generated_ids = []

            for step in range(self.max_len):
                # Run one step of the inference decoder
                logits, new_h1, new_c1, new_h2, new_c2 = self.inference_decoder.predict_on_batch(
                    [current_token, 
                     np.expand_dims(state_h1, axis=0), 
                     np.expand_dims(state_c1, axis=0),
                     np.expand_dims(state_h2, axis=0),
                     np.expand_dims(state_c2, axis=0)]
                )
                # logits shape: (1, 1, vocab_size) -> squeeze to (vocab_size,)
                probs = tf.nn.softmax(logits[0, 0]).numpy()

                # Mask out PAD, UNK, BOS
                probs[0] = 0.0
                probs[1] = 0.0
                probs[2] = 0.0

                # Try tokens in descending probability order; pick first that keeps the prefix valid
                sorted_indices = np.argsort(probs)[::-1]
                chosen = None
                eos_reached = False
                for idx in sorted_indices:
                    if probs[idx] == 0:
                        continue

                    if idx == 3:  # EOS -> we can stop, but we don't add it
                        current_domain = self.sp.decode(generated_ids)
                        # Check if the final domain is valid
                        if self._is_valid_full_domain(current_domain):
                            eos_reached = True
                            break # Finish generation
                        else:
                            # If not valid, treat EOS as an invalid token
                            continue

                    candidate_tokens = generated_ids + [int(idx)]
                    candidate_domain = self.sp.decode(candidate_tokens)
                    if self._is_valid_prefix(candidate_domain):
                        chosen = int(idx)
                        break

                if eos_reached:
                    break

                if chosen is None:
                    # No valid token found; restart with new noise
                    break
                else:
                    generated_ids.append(chosen)
                    # Update states for next step
                    state_h1 = new_h1[0]
                    state_c1 = new_c1[0]
                    state_h2 = new_h2[0]
                    state_c2 = new_c2[0]
                    # Prepare next token
                    current_token = np.array([[chosen]], dtype=np.int32)

            # Decode final domain and clean it (remove SentencePiece special chars and spaces)
            raw_domain = self.sp.decode(generated_ids)
            final_domain = raw_domain.replace('_', '').replace(' ', '')

            # Check final validity (full domain) and repetition
            if self._is_valid_full_domain(final_domain) and final_domain not in self.generated_set:
                self.generated_set.add(final_domain)
                return f"{final_domain}.{sample_tld()}"
            # else, try again with new noise

        # Fallback: if all attempts fail, generate a random string
        logger.warning(f"[{self.name}] Failed to generate a unique valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        fallback = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))  # SAFETY GUARD: random SLD fallback
        self.generated_set.add(fallback)
        return f"{fallback}.{sample_tld()}"