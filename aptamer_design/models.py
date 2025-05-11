from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from itertools import product
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Abstract base class for all aptamer models
# ------------------------------
class AptamerModel(ABC):
    @abstractmethod
    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Given a list of sequences, returns predicted binding values.
        Output: np.ndarray of shape (N,)
        """
        pass

# ------------------------------
# Utility: k-mer frequency vector encoder
# ------------------------------
def compute_kmer_vector(seq: str, k: int = 6, alphabet: str = "ACGT") -> np.ndarray:
    """
    Convert a sequence into a normalized k-mer frequency vector.
    
    Args:
        seq (str): Aptamer sequence (should already be normalized to DNA).
        k (int): Size of k-mers.
        alphabet (str): Alphabet used, e.g., 'ACGT'.

    Returns:
        np.ndarray: Frequency vector of size (|alphabet|^k,)
    """
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]

    all_possible = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_possible)}

    vec = np.zeros(len(all_possible))
    counts = Counter(kmers)
    for kmer, count in counts.items():
        if kmer in kmer_to_index:
            vec[kmer_to_index[kmer]] = count

    # Normalize to relative frequency
    return vec / vec.sum() if vec.sum() > 0 else vec

# ------------------------------
# K-mer-based Multi-Layer Perceptron model
# ------------------------------
class KMerMLPModel(AptamerModel):
    def __init__(self, k: int = 6, hidden_dim: int = 128, weights_path: Optional[str] = None):
        """
        Initializes the k-mer MLP model.

        Args:
            k (int): Size of k-mers to use for input encoding.
            hidden_dim (int): Size of the hidden layer.
            weights_path (str): Optional path to a .pth file with pre-trained weights.
        """
        self.k = k
        self.alphabet = "ACGT"  # Assumes sequences are normalized to DNA
        self.input_dim = len(self.alphabet) ** k

        # Define simple 2-layer MLP
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        else:
            # Initialize with random weights (for PoC/demo)
            self.model.apply(self._init_weights)

        self.model.eval()  # Disable dropout/batchnorm during inference

    def _init_weights(self, m):
        """
        Applies Xavier initialization to MLP layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Predicts binding scores for a list of sequences.

        Args:
            sequences (List[str]): List of input aptamer sequences.

        Returns:
            np.ndarray: Array of predicted binding values of shape (N,)
        """
        # Encode each sequence as a k-mer frequency vector
        features = np.stack([
            compute_kmer_vector(seq, k=self.k, alphabet=self.alphabet)
            for seq in sequences
        ])

        # Run through the MLP model
        with torch.no_grad():
            inputs = torch.from_numpy(features).float()
            outputs = self.model(inputs).squeeze()
            return outputs.numpy()

class CNNModel(AptamerModel):
    def __init__(self, max_len: int = 100, weights_path: Optional[str] = None):
        """
        Initializes the 1D CNN model.

        Args:
            max_len (int): Maximum sequence length. Sequences longer than this are truncated.
            weights_path (str): Optional path to load model weights (.pth)
        """
        self.alphabet = "ACGT"
        self.base_to_index = {base: i for i, base in enumerate(self.alphabet)}
        self.max_len = max_len

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(32, 1)
        )

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.apply(self._init_weights)

        self.model.eval()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _one_hot_encode(self, seq: str) -> np.ndarray:
        """
        One-hot encodes a DNA sequence of length max_len.
        Output: shape (max_len, 4)
        """
        mat = np.zeros((self.max_len, 4), dtype=np.float32)
        for i, base in enumerate(seq.upper()):
            if i >= self.max_len:
                break
            if base in self.base_to_index:
                mat[i, self.base_to_index[base]] = 1.0
        return mat

    def predict(self, sequences: List[str]) -> np.ndarray:
        """
        Predicts binding scores for a list of sequences.
        """
        encoded = np.stack([self._one_hot_encode(seq) for seq in sequences])  # (B, L, 4)
        inputs = torch.from_numpy(encoded).float().permute(0, 2, 1)  # → (B, 4, L)

        with torch.no_grad():
            outputs = self.model(inputs).squeeze()
            return outputs.numpy()


def load_model(model_type: str, weights_path: Optional[str] = None, **kwargs) -> AptamerModel:
    """
    Factory to load a specific model type.

    Args:
        model_type (str): One of "mlp" or "cnn".
        weights_path (str): Optional path to pretrained .pth file.
        kwargs: Additional args like `k`, `hidden_dim`, `max_len`.

    Returns:
        AptamerModel: an instance of the chosen model.
    """
    model_type = model_type.lower()
    if model_type == "mlp":
        return KMerMLPModel(weights_path=weights_path, **kwargs)
    elif model_type == "cnn":
        return CNNModel(weights_path=weights_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
