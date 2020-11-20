from argparse import ArgumentParser, Namespace

from typing import Union
from pathlib import Path
import torch
import pytorch_lightning as pl
from ..data.ms_dataset import MSDataset, MSStats
from ..data.repeat_dataset import RepeatDataset
from ..data.trrosetta_ms_dataset import TRRosetta_MSDataset
from ..data.pseudolikelihood_dataset import PseudolikelihoodDataset
from ..data.maskedlm_dataset import MaskedLMDataset
from ..vocab import FastaVocab


class MSDataModule(pl.LightningDataModule):
    """Creates dataset from A3M, Fasta file or TRRosetta npz file.

    Args:
        data (Union[str, Path]): Path to fasta, npz, or a3m file to load sequences.
        batch_size (int, optional): Batch size for DataLoader. Default 128.
        num_repeats (int, optional): Number of times to repeat dataset (can speed up
            training for small datasets). Default 1.
        task (str, optional): Which task to train with.
            Choices: ['pseudolikelihood', 'masked_lm']. Default: 'pseudolikelihhod'.
        mask_prob (float, optional): Probability of masking a token when using
            'masked_lm' task. Default: 0.15.
        mask_rnd_prob (float, optional): Probability of using a random token when using
            'masked_lm' task. Default: 0.1.
        mask_leave_prob (float, optional): Probability of leaving original token when
            using 'masked_lm' task. Default: 0.1.
    """

    def __init__(
        self,
        data: Union[str, Path],
        batch_size: int = 128,
        num_repeats: int = 100,
        task: str = "pseudolikelihood",
        mask_prob: float = 0.15,
        mask_rnd_prob: float = 0.1,
        mask_leave_prob: float = 0.1,
    ):
        super().__init__()
        self.data = Path(data)
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.task = task
        self.mask_prob = mask_prob
        self.mask_rnd_prob = mask_rnd_prob
        self.mask_leave_prob = mask_leave_prob

    def setup(self):
        if self.data.suffix == ".npz":
            ms_dataset = TRRosetta_MSDataset(self.data)
        else:
            ms_dataset = MSDataset(self.data)
        dataset = RepeatDataset(ms_dataset, self.num_repeats)
        if self.task == "pseudolikelihood":
            dataset = PseudolikelihoodDataset(dataset)
        elif self.task == "masked_lm":
            dataset = MaskedLMDataset(
                dataset,
                FastaVocab.pad_idx,
                FastaVocab.pad_idx,
                len(FastaVocab),
                self.mask_prob,
                self.mask_rnd_prob,
                self.mask_leave_prob,
            )
        elif self.task != "none":  # allow none to load raw sequences
            raise ValueError(f"Invalid task {self.task}")
        self.dataset = dataset
        self.ms_dataset = ms_dataset
        self.dims = (ms_dataset.num_seqs, len(ms_dataset.reference))

    def get_stats(self) -> MSStats:
        try:
            return self.ms_dataset.get_stats()
        except AttributeError:
            raise RuntimeError(
                "Trying to get MSA stats before calling setup on module."
            )

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
    def from_args(cls, args: Namespace) -> "MSDataModule":
        return cls(
            args.data,
            args.batch_size,
            args.num_repeats,
            args.task,
            args.mask_prob,
            args.mask_rnd_prob,
            args.mask_leave_prob,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        MSDataset.add_args(parser)
        RepeatDataset.add_args(parser)
        PseudolikelihoodDataset.add_args(parser)
        MaskedLMDataset.add_args(parser)
        parser.add_argument("--data", type=str, help="Data file to load from.")
        parser.add_argument(
            "--batch_size", type=int, default=128, help="Batch size for training."
        )
        parser.add_argument(
            "--task",
            choices=["pseudolikelihood", "masked_lm"],
            default="masked_lm",
            help="Whether to use Pseudolikelihood or Masked LM for training",
        )
        return parser
