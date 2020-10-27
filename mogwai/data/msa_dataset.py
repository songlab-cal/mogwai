from argparse import ArgumentParser
from typing import Union, NamedTuple, List
from pathlib import Path
import torch

from ..parsing import parse_fasta
from ..vocab import FastaVocab

MSAStats = NamedTuple(
    "MSAStats", [("num_seqs", int), ("msa_length", int), ("msa_counts", torch.Tensor)]
)


class MSADataset(torch.utils.data.TensorDataset):
    """MSADataset: Loads a multiple sequence alignment directly from an A3M file.

    Args:
        data (PathLike): Path to a3m file.
    """

    def __init__(self, data: Union[str, Path]):
        _, sequences = parse_fasta(data, remove_insertions=True)
        indices = [FastaVocab.tokenize(seq) for seq in sequences]
        super().__init__(torch.tensor(indices, dtype=torch.long))

    @property
    def num_seqs(self) -> int:
        return self.tensors[0].size(0)

    @property
    def msa_length(self) -> int:
        return self.tensors[0].size(1)

    @property
    def msa_counts(self) -> torch.Tensor:
        if not hasattr(self, "_msa_counts"):
            self._msa_counts = torch.eye(len(FastaVocab) + 1, len(FastaVocab))[
                self.tensors[0]
            ].sum(0)
        return self._msa_counts

    def get_stats(self) -> MSAStats:
        return MSAStats(self.num_seqs, self.msa_length, self.msa_counts)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser

    def collater(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(sequences, 0)
