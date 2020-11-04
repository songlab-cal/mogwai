from typing import Sequence, List


class _FastaVocab:

    def __init__(self):
        self.ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
        self.A2N = {a: n for n, a in enumerate(self.ALPHABET)}
        self.A2N["X"] = 20

        self.IUPAC_CODES = {
            "Ala": "A",
            "Arg": "R",
            "Asn": "N",
            "Asp": "D",
            "Cys": "C",
            "Gln": "Q",
            "Glu": "E",
            "Gly": "G",
            "His": "H",
            "Ile": "I",
            "Leu": "L",
            "Lys": "K",
            "Met": "M",
            "Phe": "F",
            "Pro": "P",
            "Ser": "S",
            "Thr": "T",
            "Trp": "W",
            "Val": "V",
            "Tyr": "Y",
            "Asx": "B",
            "Sec": "U",
            "Xaa": "X",
            "Glx": "Z",
        }

        self.THREE_LETTER = {aa: name for name, aa in self.IUPAC_CODES.items()}

    def convert_indices_to_tokens(self, indices: Sequence[int]) -> List[str]:
        return [self.ALPHABET[i] for i in indices]

    def convert_tokens_to_indices(self, tokens: Sequence[str]) -> List[int]:
        return [self.A2N.get(token, 20) for token in tokens]

    def tokenize(self, sequence: str) -> List[int]:
        return self.convert_tokens_to_indices(list(sequence))

    def __len__(self) -> int:
        return 20

    @property
    def pad_idx(self) -> int:
        return 20


FastaVocab = _FastaVocab()
