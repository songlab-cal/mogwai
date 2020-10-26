from argparse import ArgumentParser, Namespace
from typing import Optional

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..optim import GremlinAdam
from ..utils import symmetrize_potts_
from ..utils.init import init_potts_bias, init_potts_weight, init_pseudolik_mask


class Gremlin(BaseModel):
    def __init__(
        self,
        num_seqs: int,
        msa_length: int,
        msa_counts: torch.Tensor,
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

        weight = init_potts_weight(msa_length, vocab_size)
        weight = nn.Parameter(weight, True)
        self.register_parameter("weight", weight)

        mask = init_pseudolik_mask(msa_length)
        self.register_buffer("diag_mask", mask)

        if self.use_bias:
            bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            bias = nn.Parameter(bias, True)
            self.register_parameter("bias", bias)

        self.register_buffer("one_hot", torch.eye(vocab_size + 1, vocab_size))

    @torch.no_grad()
    def apply_constraints(self):
        # Symmetrize and mask diagonal
        self.weight.data = symmetrize_potts_(self.weight.data)
        self.weight.data.mul_(self.diag_mask[:, None, :, None])

    def forward(self, src_tokens, targets=None):
        self.apply_constraints()
        inputs = self.one_hot[src_tokens]
        logits = torch.tensordot(inputs, self.weight, 2)
        if self.use_bias:
            logits = logits + self.bias

        loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction="sum")(
            logits.view(-1, self.vocab_size), targets.view(-1)
        )
        loss = loss / inputs.size(0)

        return loss, logits

    def configure_optimizers(self):
        weight_decay = (
            self.l2_coeff * (self.msa_length - 1) * self.vocab_size / self.num_seqs
        )
        bias_weight_decay = self.l2_coeff * 2 / self.num_seqs

        optimizer_grouped_parameters = [
            {"params": [self.weight], "weight_decay": weight_decay}
        ]
        if self.use_bias:
            optimizer_grouped_parameters.append(
                {"params": [self.bias], "weight_decay": bias_weight_decay}
            )

        optimizer = GremlinAdam(
            optimizer_grouped_parameters, lr=self.learning_rate, weight_decay=0.0
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
        msa_counts: torch.Tensor,
        vocab_size: int = 20,
        pad_idx: int = 20,
        true_contacts: Optional[torch.Tensor] = None,
    ) -> "Gremlin":
        return cls(
            num_seqs=num_seqs,
            msa_length=msa_length,
            msa_counts=msa_counts,
            learning_rate=args.learning_rate,
            vocab_size=vocab_size,
            true_contacts=true_contacts,
            l2_coeff=args.l2_coeff,
            use_bias=args.use_bias,
            pad_idx=pad_idx,
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
        return parser
