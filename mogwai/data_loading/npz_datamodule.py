from typing import Union

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from .parsing import load_npz_data


class NPZ_MSADataModule(pl.LightningDataModule):
    """Creates dataset from NPZ file of an MSA.

    Args:
        npz_file (Union[str, Path]): Path to npz file.
        batch_size (int, optional): Batch size for DataLoader.
    """

    def __init__(self, npz_file: Union[str, Path], batch_size: int = 64):
        super().__init__()
        npz_file = Path(npz_file)
        if not npz_file.exists():
            raise FileNotFoundError(npz_file)
        self.npz_file = npz_file
        self.batch_size = batch_size

    def setup(self):
        family_data = load_npz_data(self.npz_file)
        msa = torch.FloatTensor(family_data.msa)
        self.msa_dataset = TensorDataset(msa)
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
