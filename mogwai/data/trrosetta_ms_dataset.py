from typing import Union, NamedTuple, List

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import torch

from ..vocab import FastaVocab

MSAStats = NamedTuple(
    "MSAStats", [("num_seqs", int), ("msa_length", int), ("msa_counts", torch.Tensor)]
)

PAD_IDX = FastaVocab.pad_idx


def construct_unalignment(msa):
    padded_ms = torch.zeros_like(msa)
    ms = [seq[seq != PAD_IDX] for seq in msa]
    for i, m in enumerate(ms):
        padded_ms[i, : len(m)] = m
    return padded_ms


class TRRosetta_MSDataset(torch.utils.data.TensorDataset):
    """TRRosetta Dataset: Loads a multiple sequence alignment directly from a TRRosetta npz file.

    Args:
        data (PathLike): Path to npz file.
    """

    def __init__(self, data: Union[str, Path]):
        fam_data = np.load(data)
        msa = fam_data["msa"]
        msa = torch.from_numpy(msa)
        ms = construct_unalignment(msa)
        super().__init__(ms.long())

    @property
    def num_seqs(self) -> int:
        return self.tensors[0].size(0)

    @property
    def msa_length(self) -> int:
        return self.tensors[0].size(1)

    @property
    def reference(self) -> torch.Tensor:
        return self.tensors[0]

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
