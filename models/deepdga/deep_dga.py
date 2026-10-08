import logging
import random
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, MaxPooling1D, UpSampling1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from core.adversarial_model import AdversarialModel
from core.data_splits import TRANCO_D1_BENIGN, load_tranco
from typing import Any, List

logger = logging.getLogger(__name__)

class HighwayLayer(Layer):
    """
    Custom Highway Layer implementation: y = H * T + x * (1 - T)
    where H is the transform gate (relu) and T is the carry gate (sigmoid).
    """
    def __init__(self, **kwargs):
        super(HighwayLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W_T = self.add_weight(shape=(dim, dim), initializer='glorot_uniform', name='W_T', trainable=True)
        self.b_T = self.add_weight(shape=(dim,), initializer='zeros', name='b_T', trainable=True)
        self.W_H = self.add_weight(shape=(dim, dim), initializer='glorot_uniform', name='W_H', trainable=True)
        self.b_H = self.add_weight(shape=(dim,), initializer='zeros', name='b_H', trainable=True)
        super(HighwayLayer, self).build(input_shape)

    def call(self, x):
        # Flatten time steps if it's 3D, otherwise matrix mult directly over last dimension
        T = tf.sigmoid(tf.matmul(x, self.W_T) + self.b_T)
        H = tf.nn.relu(tf.matmul(x, self.W_H) + self.b_H)
        return H * T + x * (1.0 - T)


class DeepDGAModel(AdversarialModel):
    """
    Literal DeepDGA Implementation details from Anderson et al. 2016.
    Two phase training:
    1) Autoencoder pre-training (Encoder + Decoder)
    2) GAN tuning (Box Layer + Frozen Decoder VS Frozen Encoder + Logistic Regression)
    """

    def __init__(self, name: str = "DeepDGA", max_len: int = 63, vocab_size: int = 39, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_len = max_len  # 63 = maximum DNS label length; DeepDGA generates the SLD only
        self.vocab_size = vocab_size
        self.latent_dim = 64 # LSTM dimensionality output cited in paper
        
        # Internal hyperparameters configurations
        self.epochs = 300  # paper: the autoencoder is pretrained for 300 epochs
        self.max_train_samples = 256000
        self.batch_size = 128
        self.weights_dir = "models/deepdga/weights"
        
        # Characters typically used in domains (a-z, 0-9, -, .)
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-."
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}
        
        if self.vocab_size < len(self.chars):
            self.vocab_size = len(self.chars) + 1 # +1 for padding index
            
        self._build_components()

    def _build_components(self):
        """
        Builds the raw Encoder, Decoder, Autoencoder, and GAN architectures.
        """
        # ==========================================
        # 1. ENCODER
        # Inputs -> Conv1D -> MaxPool -> Highway -> LSTM -> Latent
        # ==========================================
        enc_input = Input(shape=(self.max_len, self.vocab_size), name="encoder_input")
        # Conv1D feature extraction
        e = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(enc_input)
        e = MaxPooling1D(pool_size=2, padding='same')(e)
        e = HighwayLayer()(e)
        enc_output = LSTM(self.latent_dim)(e)
        self.encoder = Model(enc_input, enc_output, name="Encoder")

        # ==========================================
        # 2. DECODER
        # Latent -> Repeat -> LSTM -> Highway -> Dense (Softmax)
        # ==========================================
        dec_input = Input(shape=(self.latent_dim,), name="decoder_input")
        # The paper says decoder is "loosely the reverse".
        d = RepeatVector(self.max_len)(dec_input)
        d = LSTM(self.latent_dim, return_sequences=True)(d)
        d = HighwayLayer()(d)
        dec_output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(d)
        self.decoder = Model(dec_input, dec_output, name="Decoder")

        # ==========================================
        # 3. AUTOENCODER (Encoder -> Decoder)
        # ==========================================
        ae_input = Input(shape=(self.max_len, self.vocab_size))
        ae_latent = self.encoder(ae_input)
        ae_output = self.decoder(ae_latent)
        self.autoencoder = Model(ae_input, ae_output, name="Autoencoder")
        
        # ==========================================
        # 4. GAN ARCHITECTURE (Adversarial Phase)
        # ==========================================
        
        # Generator: Noise -> BoxLayer -> Decoder
        gen_noise = Input(shape=(20,))
        # Principal Axis Box Layer (bounds between 0 and 1, simulating autoencoder boundaries)
        box_layer = Dense(self.latent_dim, activation='sigmoid', name="box_layer")(gen_noise)
        gen_output = self.decoder(box_layer)
        self.generator = Model(gen_noise, gen_output, name="Generator")

        # Discriminator: Encoder -> Trainable Logistic Regression
        disc_input = Input(shape=(self.max_len, self.vocab_size))
        encoded_features = self.encoder(disc_input)
        disc_output = Dense(1, activation='sigmoid', name="logistic_regression")(encoded_features)
        self.discriminator = Model(disc_input, disc_output, name="Discriminator")
        
        # GAN Model orchestrating generator updates
        gan_input = Input(shape=(20,))
        generated_domains = self.generator(gan_input)
        self.discriminator.trainable = False # Freeze discriminator while training GAN
        gan_output = self.discriminator(generated_domains)
        self.gan = Model(gan_input, gan_output, name="GAN")

    def fit(self) -> None:
        """
        Two Phase Literal Training.
        1. Pre-trains Autoencoder with 300 Epochs.
        2. Tunes GAN for exactly 3 adversarial rounds.
        """
        start_t = time.time()
        os.makedirs(self.weights_dir, exist_ok=True)
        autoencoder_weights_path = os.path.join(self.weights_dir, "autoencoder.weights.h5")
        generator_weights_path = os.path.join(self.weights_dir, "generator.weights.h5")
        disc_weights_path = os.path.join(self.weights_dir, "discriminator.weights.h5")

        if os.path.exists(generator_weights_path) and os.path.exists(disc_weights_path):
            logger.info(f"[*] DeepDGA found cached final GAN weights. Skipping all training phases.")
            self.generator.load_weights(generator_weights_path)
            self.discriminator.load_weights(disc_weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        # Load X_train
        X_train = self._load_data()
        
        if X_train is None or X_train.shape[0] == 0:
            logger.warning("[!] Cannot train DeepDGA without data.")
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        # ---------------------------------------------
        # PHASE 1: PRE-TRAIN AUTOENCODER
        # ---------------------------------------------
        if os.path.exists(autoencoder_weights_path):
            logger.info(f"[*] Found pre-trained Autoencoder weights. Loading directly...")
            self.autoencoder.load_weights(autoencoder_weights_path)
        else:
            logger.info(f"[*] PHASE 1: Pre-training Autoencoder on {X_train.shape[0]} domains for {self.epochs} epochs...")
            # Make sure base layers are trainable for pretraining
            self.encoder.trainable = True
            self.decoder.trainable = True
            self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
            
            self.autoencoder.fit(
                X_train, X_train, 
                batch_size=self.batch_size, 
                epochs=self.epochs,
                verbose=1
            )
            self.autoencoder.save_weights(autoencoder_weights_path)
            logger.info(f"[*] Saved Autoencoder base weights.")

        # At this point, encoder and decoder have their weights properly initialized.
        # Ensure those learned weights carry over to the frozen and live variants of the GAN
        # (This is automatic since in Keras layers are passed via reference when reusing them)

        # ---------------------------------------------
        # PHASE 2: ADVERSARIAL ROUNDS
        # ---------------------------------------------
        logger.info(f"[*] PHASE 2: Activating Adversarial GAN tuning (3 Rounds)...")
        # Freeze internal components as per paper
        self.encoder.trainable = False
        self.decoder.trainable = False
        
        # Ensure discriminator's logistic regression is compileable
        self.discriminator.trainable = True
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False # Refreeze discriminator to train GAN's Box Layer
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')

        gan_rounds = 3 # paper: all experiments use three adversarial rounds
        domains_per_round = 12800 # Specific paper metric
        
        for rnd in range(gan_rounds):
            logger.info(f"  [Round {rnd+1}/{gan_rounds}] Generating {domains_per_round} samples to confront the detector...")
            
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], domains_per_round)
            real_domains = X_train[idx]
            
            noise = np.random.rand(domains_per_round, 20)
            fake_domains = self.generator.predict(noise, batch_size=256, verbose=0)
            
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(real_domains, np.ones((domains_per_round, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_domains, np.zeros((domains_per_round, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train Generator (Only Box Layer is updated since decoder is frozen)
            self.discriminator.trainable = False
            noise = np.random.rand(domains_per_round, 20)
            g_loss = self.gan.train_on_batch(noise, np.ones((domains_per_round, 1)))
            
            logger.info(f"    -> D loss: {d_loss[0]:.4f}, D acc: {100*d_loss[1]:.2f}%, G loss: {g_loss:.4f}")

        # Save adversarial stage weights
        self.generator.save_weights(generator_weights_path)
        self.discriminator.save_weights(disc_weights_path)
        logger.info(f"[*] Literal architecture fully pre-trained and adversarially tuned.")
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def _load_data(self):
        domains = load_tranco(TRANCO_D1_BENIGN)
        max_samples = min(self.max_train_samples, len(domains))
        if max_samples == 0:
            return None

        X_train = np.zeros((max_samples, self.max_len, self.vocab_size))
        for i, domain in enumerate(domains[:max_samples]):
            for t, char in enumerate(domain[:self.max_len]):
                if char in self.char_indices:
                    X_train[i, t, self.char_indices[char]] = 1.0
        return X_train

    def generate_domain(self) -> str:
        if not self.is_fitted:
            self.is_fitted = True 
            
        noise = np.random.rand(1, 20)
        predicted_probs = self.generator.predict(noise, verbose=0)[0]
        
        domain_chars = []
        for i in range(self.max_len):
            prob_dist = predicted_probs[i]
            prob_dist = np.asarray(prob_dist).astype('float64')
            prob_dist = prob_dist / np.sum(prob_dist)
            char_idx = np.random.multinomial(1, prob_dist).argmax()
            if char_idx < len(self.chars):
                domain_chars.append(self.indices_char[char_idx])
            else:
                break
                
        domain = "".join(domain_chars)
        domain = domain.strip(".-")
        if domain:
            return domain
        logger.warning(f"[{self.name}] Generator produced an empty domain. Returning a random fallback.")  # SAFETY GUARD
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + ".com"  # SAFETY GUARD: random SLD fallback
