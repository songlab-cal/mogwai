from math import log

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..optim import GremlinAdam


class GremlinPseudolikelihood(BaseModel):
    def __init__(
        self,
        num_seqs: int,
        msa_length: int,
        msa_counts: torch.Tensor,
        learning_rate: float = 0.5,
        vocab_size: int = 20,
        true_contacts=None,
        l2_coeff: float = 1e-2,
        use_bias: bool = True,
    ):
        super().__init__(num_seqs, msa_length, learning_rate, vocab_size, true_contacts)
        self.l2_coeff = l2_coeff
        self.use_bias = use_bias

        self.init_potts_weight(msa_length, vocab_size)
        self.init_pseudolik_mask(msa_length)

        if self.use_bias:
            self.init_potts_bias(msa_counts, l2_coeff, num_seqs)

    def init_potts_weight(self, msa_length: int, vocab_size: int):
        # Create [L,A,L,A] coupling tensor from zero.
        weight = torch.zeros(msa_length, vocab_size, msa_length, vocab_size)
        weight = nn.Parameter(weight, True)
        self.register_parameter("weight", weight)

    def init_potts_bias(self, msa_counts: torch.Tensor, l2_coeff: float, num_seqs: int):
        # Create [L, A] bias term from PSSM
        bias = (msa_counts + l2_coeff * log(num_seqs)).log()
        bias.add_(-bias.mean(-1, keepdims=True))
        bias = nn.Parameter(bias, True)
        self.register_parameter("bias", bias)

    def init_pseudolik_mask(self, msa_length: int):
        # Create diagonal mask for pseudolikelihood
        diag_mask = torch.ones(msa_length, msa_length).float()
        diag_mask.masked_fill_(torch.eye(msa_length).bool(), 0.0)
        self.register_buffer("diag_mask", diag_mask)

    @torch.no_grad()
    def apply_constraints(self):
        # Symmetrize and mask diagonal
        self.weight.data = 0.5 * (
            self.weight.data + self.weight.data.permute(2, 3, 0, 1)
        )
        self.weight.data.mul_(self.diag_mask[:, None, :, None])

    def forward(self, inputs):
        self.apply_constraints()
        logits = torch.tensordot(inputs, self.weight, 2)
        if self.use_bias:
            logits.add_(self.bias)

        targets = inputs.argmax(-1)
        targets.masked_fill_(inputs.sum(-1) == 0, -1)
        loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")(
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
        contacts = self.weight.data.norm(p=2, dim=(1, 3))
        return contacts
