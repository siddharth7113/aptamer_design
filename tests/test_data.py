import pytest
from aptamer_design.data import AptamerSample, AptamerDataset, is_valid_sequence, normalize_sequence

def test_sample_and_dataset_basic():
    samples = [
        AptamerSample(sequence="ACGT", binding=1.2, id="apt1", molecule_type="DNA"),
        AptamerSample(sequence="UGCA", binding=None, id="apt2", molecule_type="RNA")
    ]
    ds = AptamerDataset(samples)
    assert len(ds) == 2
    assert ds[0].sequence == "ACGT"
    assert ds[1].molecule_type == "RNA"
    assert ds.has_labels is False

def test_is_valid_sequence_dna():
    assert is_valid_sequence("ACGT", "DNA")
    assert not is_valid_sequence("ACGU", "DNA")

def test_is_valid_sequence_rna():
    assert is_valid_sequence("ACGU", "RNA")
    assert not is_valid_sequence("ACGT", "RNA")

def test_is_valid_sequence_peptide():
    assert is_valid_sequence("ACDEFGHIK", "Peptide")
    assert not is_valid_sequence("XYZ", "Peptide")

def test_normalize_sequence_dna():
    assert normalize_sequence("5'-acgu-3'", "DNA") == "ACGT"

def test_normalize_sequence_rna():
    assert normalize_sequence("ACGT", "RNA") == "ACGU"
    assert normalize_sequence("a c-g*u", "RNA") == "ACGU"

# Optional: test loading from CSV or DataFrame (mock with tempfile if needed)
