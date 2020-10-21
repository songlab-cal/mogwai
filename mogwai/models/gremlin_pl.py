from typing import Optional

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..optim import GremlinAdam
from ..utils import symmetrize_potts_
from ..utils.init import init_potts_bias, init_potts_weight, init_pseudolik_mask


class GremlinPseudolikelihood(BaseModel):
    """GREMLIN, a Potts model trained with pseudolikelihood.

    Args:
        num_seqs (int): Number of sequences in MSA.
        msa_length (int): Length of MSA.
        msa_counts (tensor): Counts of each amino acid in each position of MSA. Used for initialization.
        learning_rate (float): Learning rate for training model.
        vocab_size (int, optional): Alphabet size of MSA.
        true_contacts (tensor, optional): True contacts for family. Used to compute metrics while training.
        l2_coeff (int, optional): Coefficient of L2 regularization for all weights.
        use_bias (bool, optional): Whether to include single-site potentials in the model.
    """

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
    ):
        super().__init__(num_seqs, msa_length, learning_rate, vocab_size, true_contacts)
        self.l2_coeff = l2_coeff
        self.use_bias = use_bias

        weight = init_potts_weight(msa_length, vocab_size)
        weight = nn.Parameter(weight, True)
        self.register_parameter("weight", weight)

        mask = init_pseudolik_mask(msa_length)
        self.register_buffer("diag_mask", mask)

        if self.use_bias:
            bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            bias = nn.Parameter(bias, True)
            self.register_parameter("bias", bias)

    @torch.no_grad()
    def apply_constraints(self):
        # Symmetrize and mask diagonal
        self.weight.data = symmetrize_potts_(self.weight.data)
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
        """Extracts contacts by taking Frobenius norm of each interaction matrix."""
        contacts = self.weight.data.norm(p=2, dim=(1, 3))
        return contacts
