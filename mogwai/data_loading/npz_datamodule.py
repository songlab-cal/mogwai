from typing import Union

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from .parsing import load_npz_data


class NPZ_MSA_DataModule(pl.LightningDataModule):
    def __init__(
        self, npz_file: Union[str, Path], c_beta_cutoff: int = 8, batch_size: int = 64
    ):
        super().__init__()
        npz_file = Path(npz_file)
        if not npz_file.exists():
            raise FileNotFoundError(npz_file)
        self.npz_file = npz_file
        self.c_beta_cutoff = c_beta_cutoff
        self.batch_size = batch_size

    def setup(self):
        family_data = load_npz_data(self.npz_file, self.c_beta_cutoff)
        msa = torch.FloatTensor(family_data.msa)
        self.msa_dataset = TensorDataset(msa)
        self.true_contacts = torch.FloatTensor(family_data.contacts)
        self.dims = msa.shape

    def train_dataloader(self):
        return DataLoader(
            self.msa_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
