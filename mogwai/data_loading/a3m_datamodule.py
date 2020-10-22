from argparse import ArgumentParser, Namespace

from typing import Union
from pathlib import Path
import torch
import pytorch_lightning as pl
from ..data.msa_dataset import MSADataset
from ..data.repeat_dataset import RepeatDataset
from ..data.pseudolikelihood_dataset import PseudolikelihoodDataset


class A3M_MSADataModule(pl.LightningDataModule):
    """Creates dataset from A3M file of an MSA.

    Args:
        a3m_file (Union[str, Path]): Path to a3m file.
        batch_size (int, optional): Batch size for DataLoader.
    """

    def __init__(self, data: Union[str, Path], batch_size: int = 128, num_repeats: int = 1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_repeats = num_repeats

    def setup(self):
        msa_dataset = MSADataset(self.data)
        dataset = RepeatDataset(msa_dataset, self.num_repeats)
        dataset = PseudolikelihoodDataset(dataset)
        self.dataset = dataset
        self.msa_dataset = msa_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            collate_fn=self.dataset.collater,
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "A3M_MSADataModule":
        return cls(args.data, args.batch_size, args.num_repeats)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        MSADataset.add_args(parser)
        RepeatDataset.add_args(parser)
        PseudolikelihoodDataset.add_args(parser)
        parser.add_argument(
            "--batch_size", type=int, default=128, help="Batch size for training."
        )
        parser.add_argument("data", type=str, help="Data file to load from.")
        return parser
