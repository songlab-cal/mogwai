from typing import Optional, Union

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from .parsing import a2n, one_hot, load_a3m_msa


class A3M_MSADataModule(pl.LightningDataModule):
    """Creates dataset from A3M file of an MSA.

    Args:
        a3m_file (Union[str, Path]): Path to a3m file.
        batch_size (int, optional): Batch size for DataLoader.
    """

    def __init__(
        self,
        a3m_file: Union[str, Path],
        batch_size: int = 64,
    ):
        super().__init__()
        a3m_file = Path(a3m_file)
        if not a3m_file.exists():
            raise FileNotFoundError(a3m_file)

        self.a3m_file = a3m_file
        self.batch_size = batch_size

    def setup(self):
        msa, _, _, reference = load_a3m_msa(self.a3m_file)
        msa = torch.FloatTensor(msa)
        self.msa_dataset = TensorDataset(msa)
        ref_int = np.array([a2n[aa] for aa in reference])
        self.reference = torch.tensor(one_hot(ref_int))
        self.dims = msa.shape
        self.msa_counts = msa.sum(0)

    @property
    def num_seqs(self) -> int:
        return self.dims[0]

    @property
    def msa_length(self) -> int:
        return self.dims[1]

    @property
    def vocab_size(self) -> int:
        return self.dims[2]

    def train_dataloader(self):
        return DataLoader(
            self.msa_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
