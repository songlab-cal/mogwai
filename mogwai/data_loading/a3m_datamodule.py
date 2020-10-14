from typing import Union

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from .parsing import a2n, one_hot, load_a3m_msa


class A3M_MSA_DataModule(pl.LightningDataModule):
    def __init__(self, a3m_file: Union[str, Path], batch_size: int = 64):
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

    def train_dataloader(self):
        return DataLoader(
            self.msa_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
