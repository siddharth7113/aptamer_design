from dataclasses import dataclass
from typing import Optional, List
import logging
import pandas as pd
import os

# Define alphabets for different molecule types
MOLECULE_ALPHABETS = {
    "DNA": {"A", "C", "G", "T"},
    "RNA": {"A", "C", "G", "U"},
    "PEPTIDE": set("ACDEFGHIKLMNPQRSTVWY")
}

@dataclass
class AptamerSample:
    sequence: str
    binding: Optional[float] = None
    id: Optional[str] = None
    molecule_type: str = "DNA"  # Can be DNA, RNA, PEPTIDE


class AptamerDataset:
    def __init__(self, samples: List[AptamerSample]):
        self.samples = samples
        self.has_labels = all(s.binding is not None for s in samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AptamerSample:
        return self.samples[idx]

    def get_max_length(self) -> int:
        return max(len(s.sequence) for s in self.samples)

    @classmethod
    def from_table(
        cls,
        path: str,
        filetype: Optional[str] = None,
        seq_col: str = "sequence",
        binding_col: str = "binding",
        id_col: Optional[str] = "id",
        type_col: Optional[str] = None,
        default_type: str = "DNA"
    ) -> "AptamerDataset":
        if not filetype:
            _, ext = os.path.splitext(path)
            filetype = ext.lower().replace('.', '')

        # Load table
        if filetype == "csv":
            df = pd.read_csv(path)
        elif filetype in ["tsv", "txt"]:
            df = pd.read_csv(path, sep="\t")
        elif filetype in ["xlsx", "xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type: {filetype}")

        if seq_col not in df.columns:
            raise ValueError(f"Missing required column: '{seq_col}'")

        samples = []
        for idx, row in df.iterrows():
            raw_seq = str(row[seq_col]).strip()

            # Determine molecule type
            mol_type = str(row[type_col]).upper() if type_col and type_col in df.columns else default_type.upper()

            # Normalize and validate sequence
            clean_seq = normalize_sequence(raw_seq, mol_type)
            if not is_valid_sequence(clean_seq, mol_type):
                continue

            # Optional binding
            binding = None
            if binding_col in df.columns:
                val = row[binding_col]
                if pd.notnull(val):
                    try:
                        binding = float(val)
                    except ValueError:
                        continue

            # Optional ID
            apt_id = str(row[id_col]) if id_col and id_col in df.columns else None

            samples.append(AptamerSample(sequence=clean_seq, binding=binding, id=apt_id, molecule_type=mol_type))

        logging.info(f"Loaded {len(samples)} samples from {path} ({filetype})")
        return cls(samples)

    @classmethod
    def from_public_db(cls, db_name: str, **kwargs) -> "AptamerDataset":
        """
        Future: Load data from a supported public database like AptaDB or UTexas.
        For now, this function is not implemented.
        """
        raise NotImplementedError(f"Support for public DB '{db_name}' not implemented yet.")

def is_valid_sequence(seq: str, molecule_type: str = "DNA") -> bool:
    """
    Checks if the sequence is valid for the given molecule type.
    """
    allowed = MOLECULE_ALPHABETS.get(molecule_type.upper(), set())
    return all(base in allowed for base in seq.upper())


def normalize_sequence(seq: str, molecule_type: str = "DNA") -> str:
    """
    Cleans and standardizes an aptamer sequence based on molecule type.
    """
    seq = seq.upper().replace(" ", "").replace("-", "")
    if molecule_type.upper() == "DNA":
        seq = seq.replace("U", "T")
    elif molecule_type.upper() == "RNA":
        seq = seq.replace("T", "U")
    return "".join(base for base in seq if base in MOLECULE_ALPHABETS.get(molecule_type.upper(), set()))


