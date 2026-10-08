import os
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List

from core.adversarial_model import AdversarialModel
from core.data_splits import load_tranco, TRANCO_D1_BENIGN
from core.sld import extract_sld

logger = logging.getLogger(__name__)

class BenignDomainNameModeler(nn.Module):
    """
    Implementation of the BiLSTM Benign Domain Name Modeler.
    """
    def __init__(self, vocab_size, embed_dim, lstm_units, lstm_layers, dropout_rate):
        super().__init__()
        # Padding explicitly mapped to 38
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=38)
        
        # Forward LSTM Stack
        self.lstm_fwd = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=lstm_units, 
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )
        
        # Backward LSTM Stack
        self.lstm_bwd = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=lstm_units, 
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )
        
        # Linear + Dropout Layer
        self.fc1 = nn.Linear(lstm_units * 2, lstm_units)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Softmax + Loss Layer (Outputs logits for the 38 valid characters)
        self.fc2 = nn.Linear(lstm_units, 38)

    def forward(self, x):
        # Calculate valid sequence lengths for packing (ignoring padding_idx=38)
        lengths = (x != 38).sum(dim=1).cpu()
        
        # Embedding Layer
        emb = self.embedding(x)
        
        # Pack and Forward LSTM
        packed_emb_fwd = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out_fwd, _ = self.lstm_fwd(packed_emb_fwd)
        out_fwd, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out_fwd, batch_first=True, total_length=x.size(1)
        )
        
        # Backward LSTM (flip valid sequence, feed packed, flip back)
        # Ensure padding remains at the end for pack_padded_sequence by reversing only valid chars
        idx = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        rev_idx = lengths.unsqueeze(1).to(x.device) - 1 - idx
        rev_idx = torch.where(rev_idx < 0, idx, rev_idx)
        
        emb_flipped = torch.gather(emb, 1, rev_idx.unsqueeze(-1).expand(-1, -1, emb.size(-1)))
        
        packed_emb_bwd = torch.nn.utils.rnn.pack_padded_sequence(
            emb_flipped, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out_bwd, _ = self.lstm_bwd(packed_emb_bwd)
        out_bwd, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out_bwd, batch_first=True, total_length=x.size(1)
        )
        
        # Flip backward outputs back to their original alignment
        out_bwd = torch.gather(out_bwd, 1, rev_idx.unsqueeze(-1).expand(-1, -1, out_bwd.size(-1)))
        
        # Adjust Layer: Shift forward right by 1, shift backward left by 1
        batch_size, seq_len, hidden_dim = out_fwd.size()
        zero_tensor = torch.zeros(batch_size, 1, hidden_dim, device=x.device)
        
        pad_fwd = torch.cat([zero_tensor, out_fwd[:, :-1, :]], dim=1)
        pad_bwd = torch.cat([out_bwd[:, 1:, :], zero_tensor], dim=1)
        
        # Concat Layer
        concat = torch.cat([pad_fwd, pad_bwd], dim=2)
        
        # Linear + Dropout Layer
        linear = self.fc1(concat)
        dropped = self.dropout(linear)
        
        # Outputs
        logits = self.fc2(dropped)
        return logits


class ReplaceDGAModel(AdversarialModel):
    """
    Implementation of ReplaceDGA proposed by Hu et al. (2023), utilizing a BiLSTM + Embedding 
    architecture.

    REPRODUCIBILITY ASSUMPTIONS:
    1. Learning Rate: 0.001
    2. Max Epochs: 50
    3. Early Stopping Patience: 5
    4. TLDs: Static list of common TLDs.
    5. Weight Initialization: PyTorch defaults.
    6. Invalid Character Filtering: Discard domains with any character outside the 38 allowed.
    7. Domain length: from 4 to 60 characters (inclusive).
    8. Training/Generation data split: 100,000 benign domains for training and 100,000 for 
    generation. If not enough domains, use 80/20 split.
    """
    def __init__(self, name: str = "ReplaceDGA"):
        super().__init__(name=name)
        
        # Define vocabulary: 38 optional characters mapped by frequency + padding
        # Valid chars: a-z, 0-9, '-', '_'. '_' is 0 and 'e' is 37.
        self.valid_chars = "_79685340q21jxz-wvfykbghpudmclntrsioae"
        self.char2idx = {char: idx for idx, char in enumerate(self.valid_chars)}
        self.char2idx["<PAD>"] = 38 # Padding is explicitly mapped to 38
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx) # 39
        
        # Sequence length limits
        self.min_len = 4
        self.max_len = 60
        self.training_min_len = 1
        
        # Static list of common TLDs to append during generation
        self.tlds = [
            'com', 'at', 'uk', 'pl', 'be', 'biz', 'co', 'jp', 'cz', 'de', 'eu', 'fr', 'info', 'it', 'ru', 'lv',
            'me', 'name', 'net', 'nz', 'org', 'us'
        ]
        
        # Neural Network Hyperparameters
        self.embedding_dim = 128
        self.lstm_units = 128
        self.lstm_layers = 4
        self.dropout_rate = 0.5
        self.batch_size = 512
        self.learning_rate = 0.001
        self.patience = 5   # Early Stopping patience
        self.epochs = 50
        
        self.model = None
        self.benign_domains = [] # To store domains for training
        self.gen_domains = []    # To store distinct domains for Algorithm 1 generation

        # Caching paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        self.weights_path = os.path.join(self.weights_dir, "model.weights.pth")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self) -> None:
        """
        Trains the BiLSTM network or loads cached weights from an .pth file if available.
        """
        start_t = time.time()
        
        self.model = BenignDomainNameModeler(
            vocab_size=self.vocab_size,
            embed_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            lstm_layers=self.lstm_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # 1. Dataset loading and cleaning
        raw_domains = load_tranco(TRANCO_D1_BENIGN)
        cleaned_domains = []
        
        for domain in raw_domains:
            # Extract SLD, convert to lowercase
            sld = extract_sld(domain).lower()
            
            # Enforce length boundaries and discard domains with invalid characters
            if self.training_min_len < len(sld):
                if all(c in self.valid_chars for c in sld):
                    cleaned_domains.append(sld)
                    
        # Dataset Separation: Train / Generation Split
        random.shuffle(cleaned_domains)
        
        # Try to use 100,000 domains for training and 100,000 for generation
        if len(cleaned_domains) >= 200000:
            self.benign_domains = cleaned_domains[:100000]
            self.gen_domains = cleaned_domains[100000:200000]
            logger.info(f"[*] {self.name} filtered down to {len(cleaned_domains)} valid domains.")
        else:
            # If not enough domains, use an 80/20 split
            split_idx = int(len(cleaned_domains) * 0.8)
            self.benign_domains = cleaned_domains[:split_idx]
            self.gen_domains = cleaned_domains[split_idx:]
            logger.info(f"[*] {self.name} 80/20 split: {len(self.benign_domains)} for training, {len(self.gen_domains)} for generation.")
        
        if not self.benign_domains or not self.gen_domains:
            logger.warning(f"[*] {self.name} found insufficient domains left after filtering and split! Aborting training.")
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return
        
        # Try to load model weights from cache
        if os.path.exists(self.weights_path):
            logger.info(f"[*] {self.name} found cached weights. Skipping training.")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.model.eval()
            self.is_fitted = True
            self.trained_from_scratch = False
            self.training_time = time.time() - start_t
            return
            
        logger.info(f"[*] No weights found for {self.name}. Training...")
        
        # 2. Sequence preparation
        X = []
        for dom in self.benign_domains:
            # Domain Name Encoder: Map chars to numbers without SOS/EOS
            seq = [self.char2idx[c] for c in dom]
            # Truncate if the domain exceeds max_len
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            # Padded to max_len with 38
            padded_seq = seq + [self.char2idx["<PAD>"]] * (self.max_len - len(seq))
            X.append(padded_seq)
            
        X_tensor = torch.tensor(X, dtype=torch.long)
        
        # In ReplaceDGA, the target is the exact same sequence (model shifts internally)
        Y_tensor = X_tensor.clone()
        
        # Ensure loss is only calculated on actual characters, ignoring the 38 padding
        # Set padding target to -100 for PyTorch CrossEntropyLoss ignore_index
        Y_tensor[Y_tensor == self.char2idx["<PAD>"]] = -100
        
        dataset_size = len(X_tensor)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        
        train_X, val_X = X_tensor[:train_size], X_tensor[train_size:]
        train_Y, val_Y = Y_tensor[:train_size], Y_tensor[train_size:]
        
        train_dataset = TensorDataset(train_X, train_Y)
        val_dataset = TensorDataset(val_X, val_Y)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Use Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function for predicting the next character indices (ignores padded index -100)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 3. Training dynamics        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # outputs shape: (batch_size, seq_len, 38)
                # target shape: (batch_size, seq_len)
                loss = loss_fn(outputs.view(-1, 38), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    
                    loss = loss_fn(outputs.view(-1, 38), batch_y.view(-1))
                    val_loss += loss.item() * batch_x.size(0)
                    
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f"[*] Early stopping triggered at epoch {epoch+1}")
                break
                
        # Restore best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
            self.model.eval()
        
        # 4. Cache artifacts
        try:
            torch.save(self.model.state_dict(), self.weights_path)
        except Exception as e:
            logger.warning(f"[*] {self.name} failed to save weights: {e}")
            
        self.is_fitted = True
        self.trained_from_scratch = True
        self.training_time = time.time() - start_t

    def generate_domain(self) -> str:
        """
        Generates a domain name using the Candidate Domain Name Synthesizer (Algorithm 1).
        """
        if not self.is_fitted or self.model is None:
            self.fit()
            
        # Ensure model is in eval mode and loaded to appropriate device
        self.model.eval()
        self.model.to(self.device)

        max_attempts = 10
        for _ in range(max_attempts):

            # 1. Select a SLD D with length T from SL according to r (Uses distinct separated dataset)
            domain = random.choice(self.gen_domains)
            # Truncate if the domain exceeds max_len
            if len(domain) > self.max_len:
                domain = domain[:self.max_len]
            T = len(domain)

            # 2. Shuffle P (positions)
            domain_chars = list(domain)
            P = list(range(T))
            random.shuffle(P)
        
            # Algorithm 1: Perform exactly 2 character replacements
            for _ in range(2):
                X_seq = [self.char2idx[c] for c in domain_chars]
                X_padded = X_seq + [self.char2idx["<PAD>"]] * (self.max_len - T)
                input_tensor = torch.tensor([X_padded], dtype=torch.long).to(self.device)
                
                # 3. Output of M with input X
                with torch.no_grad():
                    Y_probs = self.model(input_tensor)[0] # Shape: (max_len, 38)
                
                replaced = False
                pos_to_remove = None
                
                # Iterate through shuffled positions
                for p in P:
                    logits = Y_probs[p]
                    best_char_idx = int(torch.argmax(logits).item())
                    best_char = self.idx2char[best_char_idx]
                    
                    # Check if the highest probability character is different from the original
                    if best_char != domain_chars[p]:
                        domain_chars[p] = best_char
                        replaced = True
                        pos_to_remove = p
                        break
                
                # If all highest prob characters matched original, use the second-highest at P[0]
                if not replaced and len(P) > 0:
                    p = P[0]
                    logits = Y_probs[p]
                    top2_indices = torch.topk(logits, k=2).indices.cpu().numpy()
                    best_char_idx = top2_indices[1]
                    best_char = self.idx2char[best_char_idx]
                    
                    domain_chars[p] = best_char
                    pos_to_remove = p
                
                # Remove R from P to ensure 2 different positions are replaced
                if pos_to_remove in P:
                    P.remove(pos_to_remove)

            sld = "".join(domain_chars)
            # Check validity of the generated domain
            if self.max_len >= len(sld) >= self.min_len and sld[0] != '-' and sld[-1] != '-':
                return sld + "." + random.choice(self.tlds)
        
        logger.warning(f"[{self.name}] No valid domain after {max_attempts} attempts. Returning a random fallback.")  # SAFETY GUARD
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(8, 15))) + "." + random.choice(self.tlds)  # SAFETY GUARD: random SLD fallback