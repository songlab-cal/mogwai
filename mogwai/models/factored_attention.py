from argparse import ArgumentParser, Namespace
from typing import Optional
import math

import torch
import torch.nn as nn
from apex.optimizers import FusedLAMB

from .base_model import BaseModel
from ..utils import symmetrize_matrix_, symmetrize_potts
from ..utils.init import init_potts_bias


class FactoredAttention(BaseModel):
    """FactoredAttention Layer.

    Args:
        num_seqs (int): Number of sequences in MSA.
        msa_length (int): Length of MSA.
        msa_counts (tensor): Counts of each amino acid in each position of MSA. Used
            for initialization.
        learning_rate (float): Learning rate for training model.
        vocab_size (int, optional): Alphabet size of MSA.
        true_contacts (tensor, optional): True contacts for family. Used to compute
            metrics while training.
        l2_coeff (int, optional): Coefficient of L2 regularization for all weights.
        use_bias (bool, optional): Whether to include single-site potentials.
    """

    def __init__(
        self,
        num_seqs: int,
        msa_length: int,
        msa_counts: Optional[torch.Tensor] = None,
        attention_head_size: int = 16,
        num_attention_heads: int = 32,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
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
        self.num_seqs = num_seqs
        self.msa_length = msa_length
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.optimizer = optimizer
        self.vocab_size = vocab_size

        hidden_size = attention_head_size * num_attention_heads

        self.query = nn.Parameter(
            0.01 * torch.randn(msa_length, num_attention_heads, attention_head_size)
        )
        self.key = nn.Parameter(
            0.01 * torch.randn(msa_length, num_attention_heads, attention_head_size)
        )
        self.value = nn.Linear(vocab_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        if self.use_bias:
            if msa_counts is not None:
                bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            else:
                bias = torch.zeros(msa_length, vocab_size)

            self.bias = nn.Parameter(bias, True)

        self.register_buffer("diag_mask", torch.eye(msa_length) * -10000)
        self.register_buffer("one_hot", torch.eye(vocab_size + 1, vocab_size))

        self._weight_reg_coeff = (
            l2_coeff * (msa_length - 1) * (vocab_size - 1) / num_seqs
        )
        self._bias_reg_coeff = l2_coeff / num_seqs

    def maybe_onehot_inputs(self, src_tokens):
        """Onehots src_tokens if necessary otherwise uses original tokens"""
        if src_tokens.dtype == torch.long:
            return self.one_hot[src_tokens]
        else:
            return src_tokens

    def forward(self, src_tokens, targets=None):
        inputs = self.maybe_onehot_inputs(src_tokens)
        # batch_size, seqlen = src_tokens.size()
        # values = self.value(inputs).view(
            # batch_size, seqlen, self.num_attention_heads, self.attention_head_size
        # )
        # attention = torch.einsum("ihd,jhd->hij", self.query, self.key)
        # attention = attention / math.sqrt(self.attention_head_size)
        # attention = attention + self.diag_mask
        # attention = attention.softmax(-1)
        # context = torch.einsum("hij,njhd->nihd", attention, values)
        # context = context.reshape(
            # batch_size, seqlen, self.num_attention_heads * self.attention_head_size
        # )
        # logits = self.output(context)
        gremlin_w = self.compute_gremlin_w()
        logits = torch.tensordot(inputs, gremlin_w, 2)

        if self.use_bias:
            logits = logits + self.bias

        outputs = (logits, gremlin_w.norm(dim=(1, 3)))
        if targets is not None:
            loss = self.loss(logits, targets)
            outputs = (loss,) + outputs
        return outputs

    def compute_regularization(self, targets, gremlin_w: Optional[torch.Tensor] = None):
        """Compute regularization weights based on the number of targets."""
        if gremlin_w is None:
            gremlin_w = self.compute_gremlin_w()
        sample_size = (targets != self.pad_idx).sum()
        reg = self._weight_reg_coeff * gremlin_w.norm()
        if self.use_bias:
            reg += self._bias_reg_coeff * self.bias.norm()

        return reg * sample_size

    def loss(self, logits, targets, gremlin_w: Optional[torch.Tensor] = None):
        """Compute GREMLIN loss w/ L2 Regularization"""
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction="sum")(
            logits.view(-1, self.vocab_size), targets.view(-1)
        )
        loss += self.compute_regularization(targets, gremlin_w)
        loss = loss / logits.size(0)
        return loss

    def compute_gremlin_w(self):
        attention = torch.einsum("ihd,jhd->hij", self.query, self.key)
        attention = attention / math.sqrt(self.attention_head_size)
        attention = attention + self.diag_mask
        attention = attention.softmax(-1)  # H x L x L

        value = self.value.weight
        value = value.view(
            self.vocab_size, self.num_attention_heads, self.attention_head_size
        )

        output = self.output.weight
        output = output.view(
            self.vocab_size, self.num_attention_heads, self.attention_head_size
        )

        embed = torch.einsum("ahd,bhd->hab", value, output)  # H x A x A

        W = torch.einsum("hij,hab->iajb", attention, embed)
        return W

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=0.0
            )
        elif self.optimizer == "lamb":
            optimizer = FusedLAMB(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Unrecognized optimizer {self.optimizer}")
        return [optimizer]

    @torch.no_grad()
    def get_contacts(self, gremlin_w: Optional[torch.Tensor] = None):
        """Extracts contacts by getting the attentions."""
        if gremlin_w is None:
            gremlin_w = self.compute_gremlin_w()
        gremlin_w = symmetrize_potts(gremlin_w)
        return gremlin_w.norm(dim=(1, 3))

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
    ) -> "FactoredAttention":
        return cls(
            num_seqs=num_seqs,
            msa_length=msa_length,
            msa_counts=msa_counts,
            attention_head_size=args.attention_head_size,
            num_attention_heads=args.num_attention_heads,
            optimizer=args.optimizer,
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
            default=1e-3,
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
            "--num_attention_heads",
            type=int,
            default=32,
            help="Number of attention heads.",
        )
        parser.add_argument(
            "--attention_head_size",
            type=int,
            default=16,
            help="Dims in each attention head.",
        )
        parser.add_argument(
            "--optimizer",
            choices=["adam", "lamb"],
            default="adam",
            help="Which optimizer to use.",
        )
        return parser
