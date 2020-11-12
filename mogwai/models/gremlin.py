from argparse import ArgumentParser, Namespace
import math
from typing import Optional

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..optim import GremlinAdam
from ..utils import symmetrize_potts_
from ..utils.init import (
    init_potts_bias,
    init_potts_weight,
    init_pseudolik_mask,
    gremlin_weight_decay_coeffs,
)


class Gremlin(BaseModel):
    def __init__(
        self,
        num_seqs: int,
        msa_length: int,
        batch_size: int,
        msa_counts: Optional[torch.Tensor] = None,
        optimizer: str = "gremlin_adam",
        learning_rate: float = 0.5,
        vocab_size: int = 20,
        true_contacts: Optional[torch.Tensor] = None,
        l2_coeff: float = 1e-2,
        use_bias: bool = True,
        pad_idx: int = 20,
    ):
        super().__init__(num_seqs, msa_length, learning_rate, vocab_size, true_contacts)
        self.l2_coeff = l2_coeff
        self.use_bias = use_bias
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.optimizer = optimizer

        weight = init_potts_weight(msa_length, vocab_size)
        self.weight = nn.Parameter(weight, True)

        mask = init_pseudolik_mask(msa_length)
        self.register_buffer("diag_mask", mask)

        if self.use_bias:
            if msa_counts is not None:
                bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            else:
                bias = torch.zeros(msa_length, vocab_size)
            self.bias = nn.Parameter(bias, True)

        self.register_buffer("one_hot", torch.eye(vocab_size + 1, vocab_size))
        self._weight_reg_coeff, self._bias_reg_coeff = gremlin_weight_decay_coeffs(
            batch_size, msa_length, l2_coeff, vocab_size
        )

    @torch.no_grad()
    def apply_constraints(self):
        # Symmetrize and mask diagonal
        self.weight.data = symmetrize_potts_(self.weight.data)
        self.weight.data.mul_(self.diag_mask[:, None, :, None])

    def maybe_onehot_inputs(self, src_tokens):
        """Onehots src_tokens if necessary otherwise uses original tokens"""
        if src_tokens.dtype == torch.long:
            return self.one_hot[src_tokens]
        else:
            return src_tokens

    def forward(self, src_tokens, targets=None, src_lengths=None):
        self.apply_constraints()
        inputs = self.maybe_onehot_inputs(src_tokens)
        logits = torch.tensordot(inputs, self.weight, 2)
        if self.use_bias:
            logits = logits + self.bias

        outputs = (logits,)
        if targets is not None:
            loss = self.loss(logits, targets)
            outputs = (loss,) + outputs

        return outputs

    def loss(self, logits, targets):
        """Compute GREMLIN loss w/ L2 Regularization"""
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction="sum")(
            logits.view(-1, self.vocab_size), targets.view(-1)
        )
        loss *= self.num_seqs / self.batch_size
        loss += self.compute_regularization(targets)
        return loss

    def compute_regularization(self, targets):
        """Compute regularization weights based on the number of targets."""
        sample_size = (targets != self.pad_idx).sum()
        reg = self._weight_reg_coeff * self.weight.pow(2).sum()
        if self.use_bias:
            reg += self._bias_reg_coeff * self.bias.pow(2).sum()

        return reg * sample_size

    def configure_optimizers(self):
        if self.optimizer == "gremlin_adam":
            optimizer = GremlinAdam(
                self.parameters(), lr=self.learning_rate, weight_decay=0.0
            )
        elif self.optimizer == "adam":

            self.learning_rate *= math.log(self.num_seqs) / self.msa_length
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=0.0
            )
        return [optimizer]

    @torch.no_grad()
    def get_contacts(self):
        """Extracts contacts by taking Frobenius norm of each interaction matrix."""
        self.apply_constraints()
        contacts = self.weight.data.norm(p=2, dim=(1, 3))
        return contacts

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        num_seqs: int,
        msa_length: int,
        batch_size: int,
        msa_counts: Optional[torch.Tensor] = None,
        vocab_size: int = 20,
        pad_idx: int = 20,
        true_contacts: Optional[torch.Tensor] = None,
    ) -> "Gremlin":
        return cls(
            num_seqs=num_seqs,
            msa_length=msa_length,
            msa_counts=msa_counts,
            batch_size=batch_size,
            learning_rate=args.learning_rate,
            vocab_size=vocab_size,
            true_contacts=true_contacts,
            l2_coeff=args.l2_coeff,
            use_bias=args.use_bias,
            pad_idx=pad_idx,
            optimizer=args.optimizer,
        )

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.5,
            help="Learning rate for training.",
        )
        parser.add_argument(
            "--l2_coeff",
            type=float,
            default=1e-2,
            help="L2 Regularization Coefficient.",
        )
        parser.add_argument(
            "--use_bias", action="store_true", help="Use a bias when training GREMLIN."
        )
        parser.add_argument(
            "--no_bias",
            action="store_false",
            help="Use a bias when training GREMLIN.",
            dest="use_bias",
        )
        parser.add_argument(
            "--optimizer",
            choices=["adam", "gremlin_adam"],
            default="gremlin_adam",
            help="Which optimizer to use.",
        )
        return parser
