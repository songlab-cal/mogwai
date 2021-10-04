from argparse import ArgumentParser, Namespace
from typing import List, Optional

from abc import abstractmethod, abstractclassmethod

import torch
import pytorch_lightning as pl

from ..utils import apc
from ..metrics import contact_auc, precision_at_cutoff


class BaseModel(pl.LightningModule):
    """Base model containing shared init and functionality for all single-MSA models.

    Args:
        num_seqs (int): Number of sequences in MSA.
        msa_length (int): Length of MSA.
        learning_rate (float): Learning rate for training model.
        vocab_size (int, optional): Alphabet size of MSA.
        true_contacts (tensor, optional): True contacts for family. Used to compute metrics while training.
    """

    def __init__(
        self,
        num_seqs: int,
        msa_length: int,
        learning_rate: float,
        vocab_size: int = 20,
        true_contacts: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_seqs = num_seqs
        self.msa_length = msa_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        if true_contacts is not None:
            self.register_buffer("_true_contacts", true_contacts, persistent=False)
            self.has_true_contacts = True
        else:
            self.has_true_contacts = False

        self.register_buffer("_max_auc", torch.tensor(0.0), persistent=False)

    def training_step(self, batch, batch_nb):
        if isinstance(batch, tuple):
            loss, *_ = self.forward(*batch)
        elif isinstance(batch, dict):
            loss, *_ = self.forward(**batch)
        else:
            loss, *_ = self.forward(batch)

        if self.has_true_contacts:
            auc = self.get_auc(do_apc=False)
            auc_apc = self.get_auc(do_apc=True)

            self._max_auc.masked_fill_(self._max_auc < auc, auc)

            self.log("auc", auc, on_step=True, on_epoch=False, prog_bar=False)
            self.log("auc_apc", auc_apc, on_step=True, on_epoch=False, prog_bar=True)
            self.log(
                "max_auc", self._max_auc, on_step=True, on_epoch=False, prog_bar=True
            )
            self.log(
                "delta_auc",
                self._max_auc - auc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            p_at_l = self.get_precision(do_apc=False)
            p_at_l_5 = self.get_precision(do_apc=False, cutoff=5)
            self.log("pr_at_L", p_at_l, on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                "pr_at_L_5", p_at_l_5, on_step=True, on_epoch=False, prog_bar=False
            )

            p_at_l_apc = self.get_precision(do_apc=True)
            p_at_l_5_apc = self.get_precision(do_apc=True, cutoff=5)
            self.log(
                "pr_at_L_apc", p_at_l_apc, on_step=True, on_epoch=False, prog_bar=True
            )
            self.log(
                "pr_at_L_5_apc",
                p_at_l_5_apc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        return {
            "loss": loss,
        }

    @abstractmethod
    def get_contacts(self):
        raise NotImplementedError

    @torch.no_grad()
    def get_precision(
        self,
        do_apc: bool = True,
        thresh: float = 0.01,
        superdiag: int = 6,
        cutoff: int = 1,
    ):
        if not self.has_true_contacts:
            raise ValueError(
                "Model not provided with ground truth contacts, precision can't be computed."
            )
        contacts = self.get_contacts()
        if do_apc:
            contacts = apc(contacts)
        return precision_at_cutoff(
            contacts, self._true_contacts, thresh, superdiag, cutoff  # type: ignore
        )

    @torch.no_grad()
    def get_auc(
        self,
        do_apc: bool = True,
        thresh: float = 0.01,
        superdiag: int = 6,
        cutoff_range: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ):
        if not self.has_true_contacts:
            raise ValueError(
                "Model not provided with ground truth contacts, precision can't be computed."
            )
        contacts = self.get_contacts()
        if do_apc:
            contacts = apc(contacts)
        return contact_auc(
            contacts, self._true_contacts, thresh, superdiag, cutoff_range  # type: ignore
        )

    @abstractclassmethod
    def from_args(cls, args: Namespace, *unused, **unusedkw) -> "BaseModel":
        return NotImplemented

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser
