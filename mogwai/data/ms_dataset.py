from argparse import ArgumentParser
from typing import Union, NamedTuple, List
from pathlib import Path
import torch

from ..parsing import parse_fasta
from ..vocab import FastaVocab
from ..utils import collate_tensors

MSStats = NamedTuple(
    "MSStats", [("num_seqs", int), ("reference", torch.Tensor)]
)


class MSDataset(torch.utils.data.Dataset):
    """MSDataset: Loads a multiple sequences directly from a fasta or a3m file.

    Args:
        data (PathLike): Path to fasta or a3m file.
    """

    def __init__(self, data: Union[str, Path]):
        super().__init__()
        _, sequences = parse_fasta(data, remove_gaps=True)
        self.data = [
            torch.tensor(FastaVocab.tokenize(seq), dtype=torch.long)
            for seq in sequences
        ]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @property
    def num_seqs(self) -> int:
        return len(self)

    @property
    def reference(self) -> torch.Tensor:
        return self.data[0]

    def get_stats(self) -> MSStats:
        return MSStats(self.num_seqs, self.reference)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser

    def collater(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        return collate_tensors(sequences, FastaVocab.pad_idx)
