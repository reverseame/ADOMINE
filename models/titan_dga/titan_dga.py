import os
import random
import time
import logging
import numpy as np
import tensorflow as tf
import sentencepiece as spm
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN
from detectors.cnn.cnn import CNNDetector
from core.sld import extract_sld

logger = logging.getLogger(__name__)

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, max_seq_len=24, embed_dim=512, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.base = 10000.0
        
        # Compute positional encodings once
        pos = np.arange(self.max_seq_len)[:, np.newaxis]
        i = np.arange(self.embed_dim)[np.newaxis, :]
        angle_rads = pos / np.power(self.base, (2 * (i // 2)) / np.float32(self.embed_dim))
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def build_transformer_block(embed_dim=512, num_heads=16, ff_dim=512, dropout_rate=0.3, norm_epsilon=1e-6):
    """
    Builds a single Transformer block (Encoder/Decoder layer). Data from paper: 
    16 heads, 512 feedforward units, 512 embedding dim, 30% dropout.
    Xavier/Glorot Uniform initialization is used natively by Keras Dense/Einsum layers.
    """
    inputs = tf.keras.Input(shape=(None, embed_dim))
    
    # Self Attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.0
    )(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)(inputs + attn_output)
    
    # Feed Forward
    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)(out1 + ffn_output)
    
    return tf.keras.Model(inputs=inputs, outputs=out2)


class Architecture(tf.keras.Model):
    """
    Architecture of TITAN DGA model, combining a Transformer Autoencoder and a GAN.
    """
    def __init__(self, vocab_size=2000, seq_len=24, embed_dim=512, **kwargs):
        super(Architecture, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Hyperparameters
        self.noise_dim = 100
        self.noise_std = 0.1
        self.num_heads = 16
        self.ff_dim = 512
        self.dropout_rate = 0.3
        self.hidden_dim = 128
        self.leaky_alpha = 0.2
        self.disc_out_dim = 1
        self.sgd_lr = 0.06
        self.gen_lr = 0.0004
        self.disc_lr = 0.0001
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.disc_loss_scale = 0.5

        # Gaussian noise params -> mean 0, std 0.1
        self.noise_layer = tf.keras.layers.GaussianNoise(self.noise_std)

        # 1. Encoder
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim)
        self.pos_encoding = PositionalEncoding(self.seq_len, self.embed_dim)
        self.enc_block1 = build_transformer_block(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
        self.enc_block2 = build_transformer_block(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
        
        # 2. Generator (MLP)
        # Data from paper: 128 neurons, LeakyReLU. Output projected to 
        # seq_len * embed_dim and reshaped. Alpha = 0.2.
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.LeakyReLU(negative_slope=self.leaky_alpha),
            tf.keras.layers.Dense(self.seq_len * self.embed_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Reshape((self.seq_len, self.embed_dim))
        ])
        
        # 3. Discriminator (MLP)
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hidden_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.LeakyReLU(negative_slope=self.leaky_alpha),
            tf.keras.layers.Dense(self.disc_out_dim, activation='sigmoid')
        ])
        
        # 4. Decoder
        self.dec_block1 = build_transformer_block(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
        self.dec_block2 = build_transformer_block(self.embed_dim, self.num_heads, self.ff_dim, self.dropout_rate)
        self.fc_out = tf.keras.layers.Dense(self.vocab_size, kernel_initializer='glorot_uniform')

        # Optimizers
        self.enc_dec_optimizer = tf.keras.optimizers.SGD(learning_rate=self.sgd_lr)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.gen_lr, beta_1=self.adam_beta1, beta_2=self.adam_beta2)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.disc_lr, beta_1=self.adam_beta1, beta_2=self.adam_beta2)

        # Metrics & Loss
        self.ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def build(self, input_shape):
        """
        Explicit build to properly initialize all sublayers.
        """
        super(Architecture, self).build(input_shape)
        if not self.embedding.built:
            self.embedding.build(input_shape)

    def encode(self, x, training=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.noise_layer(x, training=training)
        x = self.enc_block1(x, training=training)
        return self.enc_block2(x, training=training)
        
    def decode(self, latent, training=False):
        x = self.dec_block1(latent, training=training)
        x = self.dec_block2(x, training=training)
        return self.fc_out(x)
        
    @tf.function
    def train_step(self, data, kl_weight):
        real_tokens = data
        batch_size = tf.shape(real_tokens)[0]
        
        # Random noise for generator
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))
        
        with tf.GradientTape(persistent=True) as tape:
            # 1. Forward pass Autoencoder (Real Data)
            # Teacher forcing implicitly handled by passing full sequence
            latent_real = self.encode(real_tokens, training=True)
            reconstructed_logits_real = self.decode(latent_real, training=True)
            
            # 2. Forward pass GAN
            latent_fake = self.generator(noise, training=True)
            # Use fake latents in the decoder for reconstruction
            reconstructed_logits_fake = self.decode(latent_fake, training=True)
            
            # Discriminator predictions
            disc_real = self.discriminator(latent_real, training=True)
            disc_fake = self.discriminator(latent_fake, training=True)
            
            # 3. Losses
            # Cross-Entropy for reconstruction
            recon_loss_real = self.ce_loss_fn(real_tokens, reconstructed_logits_real)
            recon_loss_fake = self.ce_loss_fn(real_tokens, reconstructed_logits_fake)

            # Average both losses so that the encoder learns from both real and generated latents
            recon_loss = (recon_loss_real + recon_loss_fake) / 2.0
            
            # GAN Losses
            loss_d_real = self.bce_loss_fn(tf.ones_like(disc_real), disc_real)
            loss_d_fake = self.bce_loss_fn(tf.zeros_like(disc_fake), disc_fake)
            disc_loss = (loss_d_real + loss_d_fake) * self.disc_loss_scale
            
            gen_adv_loss = self.bce_loss_fn(tf.ones_like(disc_fake), disc_fake)
            
            # KL Divergence Proxy (simplified matching of distributions for training stability)
            # Utilizing latent MSE as a continuous proxy for feature matching/KL in sequence space
            mean_real, var_real = tf.nn.moments(latent_real, axes=[0])
            mean_fake, var_fake = tf.nn.moments(latent_fake, axes=[0])

            # Penalize the differences in both mean and variance
            mean_loss = tf.reduce_mean(tf.square(mean_real - mean_fake))
            var_loss = tf.reduce_mean(tf.square(var_real - var_fake))

            kl_loss = mean_loss + var_loss
            
            gen_total_loss = gen_adv_loss + (kl_weight * kl_loss) + recon_loss_fake
            ae_total_loss = recon_loss + (kl_weight * kl_loss)

        # Gradients and Optimization
        enc_dec_vars = self.embedding.trainable_variables + self.enc_block1.trainable_variables + \
                       self.enc_block2.trainable_variables + self.dec_block1.trainable_variables + \
                       self.dec_block2.trainable_variables + self.fc_out.trainable_variables
        
        grad_ae = tape.gradient(ae_total_loss, enc_dec_vars)
        grad_gen = tape.gradient(gen_total_loss, self.generator.trainable_variables)
        grad_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.enc_dec_optimizer.apply_gradients(zip(grad_ae, enc_dec_vars))
        self.gen_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))
        
        del tape
        return {"recon_loss": recon_loss, "disc_loss": disc_loss, "gen_loss": gen_total_loss}


class TITANDGAModel(AdversarialModel):
    """
    Implementation of TITAN DGA model proposed by Pregardier et al. (2025).

    REPRODUCIBILITY ASSUMPTIONS:
    1. Added sinusoidal positional encoding. Max sequence length is 24 tokens.
    2. GAN 128-unit layer is hidden. Added linear projection to 24x512 (alpha=0.2).
    3. Epochs: 100 (initial), 20 (retrain). Batch: 256. Noise: N(0, 0.1). KL
    annealing: 0.1 to 1.0. Discriminator loss scale: 0.5.
    4. Vocab: 2000 (SentencePiece Unigram). IDs 0-3 reserved for special tokens 
    (UNK, BOS, EOS, PAD). Reconstruction loss uses Sparse Categorical Cross-Entropy.
    5. Training uses Teacher Forcing; inference uses Greedy Decoding. Domains
    must pass RFC 1034/1035 validation or fallback after 5 failed attempts.
    6. Xavier/Glorot uniform initialization applied.
    7. Self-Augmentation uses a CNN. Evasive domains from 1000 candidates
    are appended to the legitimate dataset for retraining.
    8. Trained on 10k Tranco SLDs. Appends a random TLD from a static list composed
    by the most frequent 50 TLDs and 50 gTLDs (obtained from https://domainnamestat.com/),
    avoiding duplicates.
    """
    def __init__(self, name="TITANDGA", target_detector=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Hyperparameters
        self.vocab_size = 2000    
        self.seq_len = 24        
        self.batch_size = 256      
        self.initial_epochs = 100
        self.retrain_epochs = 20     # Using misclassified domains as benign
        self.noise_dim = 100
        self.embed_dim = 512

        # Tokenizer special token IDs
        self.unk_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 3
        self.special_tokens_max_id = 3

        # Training & Generation parameters
        self.shuffle_buffer_size = 10000
        self.kl_start_weight = 0.1
        self.kl_max_weight = 1.0
        self.num_candidates = 1000  # Generated domains evaluated by the detector for self-augmentation
        self.gen_batch_size = 512
        
        self.model = Architecture(
            vocab_size=self.vocab_size, 
            seq_len=self.seq_len, 
            embed_dim=self.embed_dim
        )
        self.sp = None
        self.target_detector = target_detector

        # Fixed list of the 50 most common traditional TLDs and the 50 most frequent gTLDs,
        # based on statistics provided by https://domainnamestat.com/ (avoiding duplicates).
        self.tlds = [
            # TLDs
            ".com", ".cn", ".de", ".tk", ".uk", ".net", ".org", ".ru", ".br", ".ga",
            ".nl", ".it", ".ws", ".ml", ".fr", ".cf", ".co", ".eu", ".in", ".au",
            ".gq", ".us", ".ca", ".ph", ".pl", ".cc", ".za", ".ch", ".es", ".se",
            ".tw", ".jp", ".me", ".be", ".ir", ".at",
            # gTLDs
            ".top", ".info", ".xyz", ".shop", ".online", ".vip", ".club", ".biz",
            ".store", ".site", ".loan", ".live", ".app", ".buzz", ".sbs", ".work",
            ".pro", ".click", ".bond", ".lol", ".wang", ".cfd", ".asia", ".life",
            ".icu", ".cloud", ".win", ".link", ".dev", ".ltd", ".world", ".space",
            ".fun", ".mobi", ".cyou", ".tech", ".today", ".digital", ".one", ".men",
            ".blog", ".bid", ".website", ".stream", ".art", ".lat", ".xin", ".page",
            ".autos", ".group",   
        ]

        # Paths for caching
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.weights_path = os.path.join(self.weights_dir, "titan_dga.weights.h5")
        self.spm_model_prefix = os.path.join(self.weights_dir, "spm_titan")
        os.makedirs(self.weights_dir, exist_ok=True)

    def _train_sentencepiece(self, domains: List[str]):
        """
        Trains SentencePiece Unigram model on legitimate domains.
        """
        temp_text = "temp_spm_corpus.txt"
        with open(temp_text, "w", encoding="utf-8") as f:
            for d in domains:
                f.write(d + "\n")
                
        spm.SentencePieceTrainer.train(
            input=temp_text, 
            model_prefix=self.spm_model_prefix, 
            vocab_size=self.vocab_size, 
            model_type='unigram',
            bos_id=self.bos_id, eos_id=self.eos_id, unk_id=self.unk_id, pad_id=self.pad_id
        )
        os.remove(temp_text)
        self.sp = spm.SentencePieceProcessor(model_file=self.spm_model_prefix + '.model')

    def _tokenize(self, domains: List[str]) -> np.ndarray:
        tokenized = []
        for d in domains:
            # Add BOS and EOS tokens
            tokens = [self.bos_id] + self.sp.encode_as_ids(d) + [self.eos_id]
            # Pad or truncate
            if len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len]
            else:
                tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))
            tokenized.append(tokens)
        return np.array(tokenized, dtype=np.int32)

    def _is_valid_sld(self, sld: str) -> bool:
        """
        Validates that the SLD (domain without TLD) complies with RFC 1034/1035.
        Rules:
        - Length between 1 and 63 characters.
        - Contains only letters, digits, and hyphens.
        - Does not start or end with a hyphen.
        """
        if not sld or len(sld) > 63:
            return False
        # Allowed characters: letters, digits, hyphen
        if not all(c.isalnum() or c == '-' for c in sld):
            return False
        if sld[0] == '-' or sld[-1] == '-':
            return False
        return True

    def _generate_fallback_sld(self) -> str:
        """
        Generates a fallback SLD using random tokens from the SentencePiece vocabulary,
        ensuring it complies with RFC 1034/1035. Falls back to random alphanumeric
        if token-based generation fails.
        """
        # Try token-based fallback up to 5 attempts
        for _ in range(5):
            # Random number of tokens (2 to 6)
            num_tokens = random.randint(2, 6)
            # Sample token IDs excluding special tokens (0-3)
            token_ids = [random.randint(self.special_tokens_max_id + 1, self.vocab_size - 1) for _ in range(num_tokens)]
            # Decode sequence
            raw = self.sp.decode_ids(token_ids).replace(" ", "")
            # Remove any disallowed characters (keep only alnum and hyphen)
            cleaned = ''.join(c for c in raw if c.isalnum() or c == '-')
            # Ensure not empty and not starting/ending with hyphen
            if cleaned and cleaned[0] != '-' and cleaned[-1] != '-' and len(cleaned) <= 63:
                return cleaned
        # Fallback to random alphanumeric string (length 6-12)
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        length = random.randint(6, 12)
        sld = ''.join(random.choice(chars) for _ in range(length))
        return sld

    def fit(self) -> None:
        """
        Trains TITAN DGA, including Targeted Self-Augmentation.
        """
        start_t = time.time()
        spm_model_file = self.spm_model_prefix + '.model'

        # Try to load everything from cache
        if os.path.exists(self.weights_path) and os.path.exists(spm_model_file):
            logger.info(f"[*] {self.name} found cached weights. Skipping training.")
            self.sp = spm.SentencePieceProcessor(model_file=spm_model_file)
            # Build model
            self.model.build(input_shape=(None, self.seq_len))
            self.model.load_weights(self.weights_path)
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return

        logger.info(f"[*] No cached weights found for {self.name}. Training ...")
        self.trained_from_scratch = True
        
        # 1. Load Benign Domains
        legit_domains = load_tranco(TRANCO_D1_BENIGN)[:10000]  # Limit to 10k for training speed
        # Remove TLDs
        legit_domains = [extract_sld(d) for d in legit_domains]
        
        # 2. Train Tokenizer
        logger.info(f"[*] {self.name} training SentencePiece model...")
        self._train_sentencepiece(legit_domains)
        
        # 3. Tokenize Dataset
        X_train = self._tokenize(legit_domains)
        dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(self.shuffle_buffer_size).batch(self.batch_size)

        # 4. Phase 1: Initial Training
        logger.info(f"[*] {self.name} starting Phase 1 Training ({self.initial_epochs} epochs)...")
        for epoch in range(self.initial_epochs):
            # KL Annealing: 0.1 to 1.0 
            kl_weight = min(self.kl_max_weight, self.kl_start_weight + (epoch / (self.initial_epochs / 2))) 
            
            for batch in dataset:
                metrics = self.model.train_step(batch, tf.constant(kl_weight, dtype=tf.float32))
                
            logger.info(f"Epoch {epoch+1}/{self.initial_epochs} - Recon: {metrics['recon_loss']:.4f} | Gen: {metrics['gen_loss']:.4f} | KLw: {kl_weight:.2f}")

        # 5. Phase 2: Targeted Self-Augmentation
        logger.info(f"[*] {self.name} starting Phase 2: Targeted Self-Augmentation...")
        
        if not self.target_detector:
            logger.info(f"[*] No target detector provided. Instantiating default CNN detector...")
            self.target_detector = CNNDetector()
            self.target_detector.fit()

        # Generate candidates to fool the detector
        candidates = self.generate_domains(self.num_candidates)
        eval_results = self.target_detector.detect(candidates)
        
        # Retain only those that evaded detection (classified as True/Benign)
        evasive_domains = [dom for dom, is_benign in eval_results if is_benign]
        logger.info(f"[*] Generated {len(candidates)} domains. {len(evasive_domains)} successfully evaded the detector.")
        
        if len(evasive_domains) > 0:
            logger.info(f"[*] {self.name} retraining with mixed dataset ({self.retrain_epochs} epochs)...")
            mixed_domains = legit_domains + evasive_domains
            X_mixed = self._tokenize(mixed_domains)
            mixed_dataset = tf.data.Dataset.from_tensor_slices(X_mixed).shuffle(self.shuffle_buffer_size).batch(self.batch_size)
            
            for epoch in range(self.retrain_epochs):
                for batch in mixed_dataset:
                    metrics = self.model.train_step(batch, tf.constant(self.kl_max_weight, dtype=tf.float32))
                logger.info(f"Epoch {epoch+1}/{self.retrain_epochs} - Recon: {metrics['recon_loss']:.4f} | Gen: {metrics['gen_loss']:.4f} | KLw: {kl_weight:.2f}")
        
        # Ensure model is built before saving
        if not self.model.built:
            self.model.build(input_shape=(None, self.seq_len))
        # Save Weights
        self.model.save_weights(self.weights_path)
        self.is_fitted = True
        self.training_time = time.time() - start_t

    def generate_domain(self) -> str:
        """
        Generates a single domain using Greedy Decoding. RFC 1034/1035 validation 
        is applied before adding the TLD.
        """
        max_attempts = 5
        for _ in range(max_attempts):
            noise = tf.random.normal(shape=(1, self.noise_dim))
            latent_fake = self.model.generator(noise, training=False)
            logits = self.model.decode(latent_fake, training=False)
            
            # Greedy decoding: argmax over vocab probabilities
            token_ids = tf.argmax(logits, axis=-1).numpy()[0]
            
            # Filter special tokens (0=unk, 1=bos, 2=eos, 3=pad)
            clean_ids = [int(tid) for tid in token_ids if tid > self.special_tokens_max_id]
            domain = self.sp.decode_ids(clean_ids).replace(" ", "")
            
            # If domain is empty, skip to next attempt
            if not domain:
                continue
                
            # Validate SLD (without TLD)
            if self._is_valid_sld(domain):
                return domain + random.choice(self.tlds)
            # else, try again
        # If all attempts fail, generate a fallback SLD using tokens from the vocabulary
        logger.warning(f"[{self.name}] No valid SLD after {max_attempts} attempts. Returning a vocabulary-based fallback.")  # SAFETY GUARD: log the fallback
        fallback_sld = self._generate_fallback_sld()
        return fallback_sld + random.choice(self.tlds)
    