from typing import Union, NamedTuple, List

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import torch

from ..vocab import FastaVocab

MSAStats = NamedTuple(
    "MSAStats", [("num_seqs", int), ("msa_length", int), ("msa_counts", torch.Tensor)]
)


class TRRosetta_MSADataset(torch.utils.data.TensorDataset):
    """TRRosetta Dataset: Loads a multiple sequence alignment directly from a TRRosetta npz file.

    Args:
        data (PathLike): Path to npz file.
    """

    def __init__(self, data: Union[str, Path]):
        fam_data = np.load(data)
        msa = fam_data["msa"]
        super().__init__(torch.tensor(msa, dtype=torch.long))

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
