import pytest
import numpy as np
from aptamer_design.models import compute_kmer_vector, KMerMLPModel, CNNModel, load_model

def test_kmer_vector_basic():
    vec = compute_kmer_vector("ACGTAC", k=2)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (16,)  # 4^2
    assert np.isclose(vec.sum(), 1.0)

def test_kmer_mlp_predict():
    model = KMerMLPModel(k=2, hidden_dim=16)  # small dummy MLP
    seqs = ["ACGTAC", "TTTTTT", "GGGGGG"]
    preds = model.predict(seqs)
    assert preds.shape == (3,)
    assert np.issubdtype(preds.dtype, np.floating)

def test_cnn_predict():
    model = CNNModel(max_len=50)
    seqs = ["ACGTACGTACGT", "TGCATGCA", "GATTACA"]
    preds = model.predict(seqs)
    assert preds.shape == (3,)
    assert np.issubdtype(preds.dtype, np.floating)

def test_model_loader():
    mlp = load_model("mlp", k=2, hidden_dim=16)
    cnn = load_model("cnn", max_len=50)
    assert isinstance(mlp, KMerMLPModel)
    assert isinstance(cnn, CNNModel)
