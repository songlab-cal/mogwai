from argparse import ArgumentParser, Namespace

from typing import Union, Optional
from pathlib import Path
import torch
import pytorch_lightning as pl
from ..data.msa_dataset import MSADataset, MSAStats
from ..data.trrosetta_dataset import TRRosetta_MSADataset
from ..data.repeat_dataset import RepeatDataset
from ..data.max_steps_dataset import MaxStepsDataset
from ..data.pseudolikelihood_dataset import PseudolikelihoodDataset
from ..data.maskedlm_dataset import MaskedLMDataset
from ..vocab import FastaVocab


class MSADataModule(pl.LightningDataModule):
    """Creates dataset from an MSA file.

    Args:
        data (Union[str, Path]): Path to a3m or npz file to load MSA.
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
        mask_rnd_prob: float = 0.0,
        mask_leave_prob: float = 0.0,
        max_steps: int = -1,
    ):
        super().__init__()
        self.data = Path(data)
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.task = task
        self.mask_prob = mask_prob
        self.mask_rnd_prob = mask_rnd_prob
        self.mask_leave_prob = mask_leave_prob
        self.max_steps = max_steps

    def setup(self, stage: Optional[str] = None):
        if self.data.suffix == ".a3m":
            msa_dataset = MSADataset(self.data)
        elif self.data.suffix == ".npz":
            msa_dataset = TRRosetta_MSADataset(self.data)
        else:
            raise ValueError(
                f"Cannot read file of type {self.data.suffix}, must be one of (.a3m,"
                " .npz)."
            )
        if self.max_steps > 0:
            dataset = MaxStepsDataset(msa_dataset, self.max_steps, self.batch_size)
        else:
            dataset = RepeatDataset(msa_dataset, self.num_repeats)
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
        self.dataset = dataset
        self.msa_dataset = msa_dataset
        self.dims = (msa_dataset.num_seqs, msa_dataset.msa_length)

    def get_stats(self) -> MSAStats:
        try:
            return self.msa_dataset.get_stats()
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
            num_workers=4,
            collate_fn=self.dataset.collater,
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "MSADataModule":
        return cls(
            args.data,
            args.batch_size,
            args.num_repeats,
            args.task,
            args.mask_prob,
            args.mask_rnd_prob,
            args.mask_leave_prob,
            max_steps=getattr(args, "max_steps", -1),
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        MSADataset.add_args(parser)
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
            default="pseudolikelihood",
            help="Whether to use Pseudolikelihood or Masked LM for training",
        )
        return parser
