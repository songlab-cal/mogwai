from typing import List, Optional

from abc import abstractmethod

import torch
import pytorch_lightning as pl

from ..utils import apc
from ..metrics import contact_auc, precision_at_cutoff


class BaseModel(pl.LightningModule):
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
            self._true_contacts = true_contacts
            self.has_true_contacts = True
        else:
            self.has_true_contacts = False

        self.register_buffer("_max_auc", torch.tensor(0.0))

    def training_step(self, batch, batch_nb):
        loss, *_ = self.forward(*batch)
        metrics = {}
        compute_auc = self.global_step & 10 == 0
        if compute_auc or self.trainer.fast_dev_run:
            auc = self.get_auc(do_apc=False)
            auc_apc = self.get_auc(do_apc=True)

            self._max_auc.masked_fill_(self._max_auc < auc, auc)

            self.log("auc", auc, on_step=True, on_epoch=False, prog_bar=True)
            self.log("auc_apc", auc_apc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "max_auc", self._max_auc, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "delta_auc",
                self._max_auc - auc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
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
        cutoff: int = 1,
        thresh: float = 0.01,
        superdiag: int = 6,
    ):
        if not self.has_true_contacts:
            raise ValueError(
                "Model not provided with ground truth contacts, precision can't be computed."
            )
        contacts = self.get_contacts()
        if do_apc:
            contacts = apc(contacts)
        return precision_at_cutoff(
            contacts, self._true_contacts, cutoff, thresh, superdiag
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
            contacts, self._true_contacts, thresh, superdiag, cutoff_range
        )
