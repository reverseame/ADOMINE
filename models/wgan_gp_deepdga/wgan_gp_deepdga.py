import os
import random
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Conv1D, Concatenate
from tensorflow.keras.models import Model
from core.adversarial_model import AdversarialModel
from core.tld import sample_tld  # framework TLD convention: the paper generates only the SLD
from core.data_splits import TRANCO_D1_BENIGN, load_tranco

logger = logging.getLogger(__name__)

class WGANGPDeepDGAModel(AdversarialModel):
    """
    WGAN-GP DeepDGA Model implementation based on Gould (2020).
    
    REPRODUCIBILITY ASSUMPTIONS:
    - Batch Size: 128
    - LSTM Units: 64
    - Conv1D Filters: 32 per branch (parallel branches with kernel sizes 2 and 3)
    - Noise Dimension: 63 integers (values from 0 to 37)
    - Dropout Rate: 0.5 (applied in Critic after LSTM, to prevent overfitting)
    - Optimizer: Adam with learning rate = 0.0001, beta_1 = 0.0, beta_2 = 0.9
    - Critic/Generator update ratio (n_critic): 5
    - Gradient Penalty Lambda: 10
    - Embedding Dimension: 32
    - Hidden Activations: LeakyReLU (alpha = 0.2)
    - Generator Output Activation: Softmax
    - Critic Output Activation: Linear (returns the Realness Score)
    - Weight Initialization: He_normal (he_normal)
    - Conv1D Padding: Same
    - Total Epochs: 100
    - Padding Strategy: Post-padding with a special token at index 37
    - Domain format: Second-Level Domains (SLDs) only, no Top-Level Domains (TLDs) appended
    - Dataset: Tranco D1 benign slice used for training, one-hot encoded to shape (63, 38)
    - Validity of DNS domains: generated domains that are not valid are rejected.
    """

    def __init__(self, name: str = "WGAN-GP DeepDGA", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = 63
        self.vocab_size = 38  # 36 alphanumeric characters + 1 hyphen + 1 padding token
        self.embedding_dim = 32
        self.lstm_units = 64
        self.conv_filters = 32
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.epochs = 100
        self.n_critic = 5
        self.gp_lambda = 10.0
        self.lr = 0.0001
        self.beta_1 = 0.0
        self.beta_2 = 0.9
        
        # Vocabulary configurations
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.pad_idx = 37  # Designated token index for post-padding
        
        self.generator = None
        self.critic = None
        
        # File management paths for caching weights
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.gen_weights_path = os.path.join(self.weights_dir, "wgan_gp_generator.weights.h5")
        self.crit_weights_path = os.path.join(self.weights_dir, "wgan_gp_critic.weights.h5")

    def _build_generator(self) -> Model:
        """
        Builds the Generator network accepting integer sequences as noise.
        Architecture: Embedding -> LSTM -> parallel Conv1D (k=2, k=3) -> Concatenate -> Dense(softmax)
        """
        inputs = Input(shape=(self.max_len,))
        
        # 1. Embedding layer
        x = Embedding(
            input_dim=self.vocab_size, 
            output_dim=self.embedding_dim, 
            embeddings_initializer='he_normal'
        )(inputs)
        
        # 2. LSTM layer (returns sequences)
        x = LSTM(self.lstm_units, return_sequences=True, kernel_initializer='he_normal')(x)
        
        # 3. Parallel Conv1D branches (kernel sizes 2 and 3)
        branch1 = Conv1D(filters=self.conv_filters, kernel_size=2, padding='same', kernel_initializer='he_normal')(x)
        branch1 = tf.keras.layers.LeakyReLU(negative_slope=0.2)(branch1)
        
        branch2 = Conv1D(filters=self.conv_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        branch2 = tf.keras.layers.LeakyReLU(negative_slope=0.2)(branch2)
        
        # 4. Concatenate branches
        x = Concatenate(axis=-1)([branch1, branch2])
        
        # 5. Output layer: softmax over vocabulary at each time step
        outputs = Dense(self.vocab_size, activation='softmax', kernel_initializer='he_normal')(x)
        
        return Model(inputs, outputs, name="Generator")

    def _build_critic(self) -> Model:
        """
        Builds the Critic network.
        Architecture: input (63,38) -> parallel Conv1D (k=2, k=3) -> Concatenate -> LSTM -> Dropout -> Dense -> Realness Score
        """
        inputs = Input(shape=(self.max_len, self.vocab_size))
        
        # Parallel Conv1D branches (kernel sizes 2 and 3)
        branch1 = Conv1D(filters=self.conv_filters, kernel_size=2, padding='same', kernel_initializer='he_normal')(inputs)
        branch1 = tf.keras.layers.LeakyReLU(negative_slope=0.2)(branch1)
        
        branch2 = Conv1D(filters=self.conv_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
        branch2 = tf.keras.layers.LeakyReLU(negative_slope=0.2)(branch2)
        
        x = Concatenate(axis=-1)([branch1, branch2])
        
        # LSTM layer (returns only last output)
        x = LSTM(self.lstm_units, return_sequences=False, kernel_initializer='he_normal')(x)
        
        # Dropout for regularization
        x = Dropout(self.dropout_rate)(x)
        
        # Linear output (Realness Score)
        outputs = Dense(1, activation=None, kernel_initializer='he_normal')(x)
        
        return Model(inputs, outputs, name="Critic")

    @tf.function
    def _train_critic_step(self, real_images, c_optimizer):
        """
        Executes a single critic training step in graph mode.
        """
        current_batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.uniform(
            shape=(current_batch_size, self.max_len), 
            minval=0, 
            maxval=self.vocab_size, 
            dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(fake_images, training=True)
            real_logits = self.critic(real_images, training=True)
            
            c_was_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            
            alpha = tf.random.uniform([current_batch_size, 1, 1], 0.0, 1.0)
            interpolated = real_images * alpha + fake_images * (1.0 - alpha)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.critic(interpolated, training=True)
            grads = gp_tape.gradient(pred, [interpolated])[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-12)
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            
            critic_loss = c_was_loss + self.gp_lambda * gp
            
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        c_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

    @tf.function
    def _train_generator_step(self, current_batch_size, g_optimizer):
        """
        Executes a single generator training step in graph mode.
        """
        random_latent_vectors = tf.random.uniform(
            shape=(current_batch_size, self.max_len), 
            minval=0, 
            maxval=self.vocab_size, 
            dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(fake_images, training=True)
            gen_loss = -tf.reduce_mean(fake_logits)
            
        gen_grad = tape.gradient(gen_loss, self.generator.trainable_variables)
        g_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

    def fit(self) -> None:
        """
        Trains the WGAN-GP architecture or loads pre-existing weights.
        """
        start_time = time.time()
        self.generator = self._build_generator()
        self.critic = self._build_critic()
        
        # Short-circuit setup if pre-existing weights are discovered
        if os.path.exists(self.gen_weights_path) and os.path.exists(self.crit_weights_path):
            logger.info(f"[*] WGANGPDeepDGA found cached weights. Skipping all training phases.")
            self.generator.load_weights(self.gen_weights_path)
            self.critic.load_weights(self.crit_weights_path)
            self.trained_from_scratch = False
            self.training_time = time.time() - start_time
            self.is_fitted = True
            return

        raw_domains = load_tranco(TRANCO_D1_BENIGN)
        
        processed_data = []
        for domain in raw_domains:
            sld = domain.split('.')[0].lower()
            sld = sld[:self.max_len]
            
            indices = []
            for char in sld:
                if char in self.char_to_idx:
                    indices.append(self.char_to_idx[char])
            
            # Post-padding configuration logic
            while len(indices) < self.max_len:
                indices.append(self.pad_idx)
                
            processed_data.append(indices)
            
        X_train_indices = np.array(processed_data, dtype=np.int32)
        X_train_one_hot = tf.one_hot(X_train_indices, depth=self.vocab_size).numpy()
        
        # Optimization configurations
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        
        dataset = tf.data.Dataset.from_tensor_slices(X_train_one_hot)
        dataset = dataset.shuffle(buffer_size=10000).batch(self.batch_size, drop_remainder=True)
                
        # Calls the compiled graphs
        for epoch in range(self.epochs):
            for real_images in dataset:
                current_batch_size = tf.shape(real_images)[0]
                
                # 1. CRITIC OPTIMIZATION PHASE
                for _ in range(self.n_critic):
                    self._train_critic_step(real_images, c_optimizer)
                    
                # 2. GENERATOR OPTIMIZATION PHASE
                self._train_generator_step(current_batch_size, g_optimizer)

            logger.info(f"Epoch {epoch + 1}/{self.epochs} finalized.")
            
        # Serialize artifacts
        os.makedirs(self.weights_dir, exist_ok=True)
        self.generator.save_weights(self.gen_weights_path)
        self.critic.save_weights(self.crit_weights_path)
        
        self.trained_from_scratch = True
        self.training_time = time.time() - start_time
        self.is_fitted = True
        logger.info(f"Training completed.")

    def generate_domain(self) -> str:
        """
        Generates a single Second-Level Domain (SLD) string.
        Implements rejection sampling to ensure basic DNS validity and a minimum 
        length of 5 characters.
        """
        if not self.is_fitted:
            self.fit()
        
        max_attempts = 20  # Avoids infinite loops
        
        for _ in range(max_attempts):
            random_latent_vector = tf.random.uniform(
                shape=(1, self.max_len), 
                minval=0, 
                maxval=self.vocab_size, 
                dtype=tf.int32
            )
            
            prediction = self.generator(random_latent_vector, training=False)
            char_indices = tf.argmax(prediction, axis=-1).numpy()[0]
            
            domain_chars = []
            for idx in char_indices:
                if idx == self.pad_idx:
                    # Stop at the first padding token (as we use post-padding)
                    break
                domain_chars.append(self.idx_to_char.get(idx, ''))

            sld = "".join(domain_chars)
            
            # Basic cleanup: remove hyphens at the beginning and end
            sld = sld.strip('-')
            
            # Minimum length
            if len(sld) >= 5:
                return f"{sld}.{sample_tld()}"
    
        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return f"{"".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15)))}.{sample_tld()}"  # SAFETY GUARD: random SLD fallback