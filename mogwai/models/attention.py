from argparse import ArgumentParser, Namespace
from typing import Optional
import math

import torch
import torch.nn as nn
from apex.optimizers import FusedLAMB

from .base_model import BaseModel
from ..utils import symmetrize_matrix_, symmetrize_potts
from ..utils.init import init_potts_bias, gremlin_weight_decay_coeffs
from .. import lr_schedulers


class Attention(BaseModel):
    """Attention Layer.

    Args:
        num_seqs (int): Number of sequences in MSA.
        msa_length (int): Length of MSA.
        msa_counts (tensor, optional): Counts of each amino acid in each position of MSA. Used
            for initialization.
        attention_head_size (int, optional): Dimension of queries and keys for a single head.
        num_attention_heads (int, optional): Number of attention heads.
        optimizer (str, optional): Choice of optimizer from ["adam", "lamb"]
        learning_rate (float, optional): Learning rate for training model.
        vocab_size (int, optional): Alphabet size of MSA.
        true_contacts (tensor, optional): True contacts for family. Used to compute
            metrics while training.
        l2_coeff (int, optional): Coefficient of L2 regularization for all weights.
        use_bias (bool, optional): Whether to include single-site potentials.
        pad_idx (int, optional): Integer for padded positions.
        lr_scheduler (str, optional): Learning schedule to use. Choose from ["constant", "warmup_constant"].
        warmup_steps (int, optional): Number of warmup steps for learning rate schedule.
        max_steps (int, optional): Maximum number of training batches before termination.
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
        lr_scheduler: str = "warmup_constant",
        warmup_steps: int = 0,
        max_steps: int = 10000,
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
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        hidden_size = attention_head_size * num_attention_heads

        self.query = nn.Linear(msa_length + vocab_size, hidden_size, bias=False)
        self.key = nn.Linear(msa_length + vocab_size, hidden_size, bias=False)
        self.value = nn.Linear(msa_length + vocab_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

        if self.use_bias:
            if msa_counts is not None:
                bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            else:
                bias = torch.zeros(msa_length, vocab_size)
            self.bias = nn.Parameter(bias, requires_grad=True)

        self.register_buffer("posembed", torch.eye(msa_length).unsqueeze(0))
        self.register_buffer("one_hot", torch.eye(vocab_size + 1, vocab_size))
        self.register_buffer("diag_mask", torch.eye(msa_length) * -10000)

        self._weight_reg_coeff, self._bias_reg_coeff = gremlin_weight_decay_coeffs(
            num_seqs, msa_length, l2_coeff, vocab_size
        )
        # self.save_hyperparameters()

    def maybe_onehot_inputs(self, src_tokens):
        """Onehots src_tokens if necessary otherwise uses original tokens"""
        if src_tokens.dtype == torch.long:
            return self.one_hot[src_tokens]
        else:
            return src_tokens

    def forward(self, src_tokens, targets=None, src_lengths=None):
        batch_size, seqlen = src_tokens.size()[:2]
        aa_inputs = self.maybe_onehot_inputs(src_tokens)
        posembed = self.posembed.repeat(batch_size, 1, 1)
        inputs = torch.cat((aa_inputs, posembed), -1)

        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)

        queries = queries.view(
            batch_size, seqlen, self.num_attention_heads, self.attention_head_size
        )
        keys = keys.view(
            batch_size, seqlen, self.num_attention_heads, self.attention_head_size
        )
        values = values.view(
            batch_size, seqlen, self.num_attention_heads, self.attention_head_size
        )
        attention = torch.einsum("nihd,njhd->nhij", queries, keys)
        attention = attention / math.sqrt(self.attention_head_size)
        attention = attention + self.diag_mask
        attention = attention.softmax(-1)

        context = torch.einsum("nhij,njhd->nihd", attention, values)
        context = context.reshape(
            batch_size, seqlen, self.num_attention_heads * self.attention_head_size
        )
        logits = self.output(context).contiguous()

        if self.use_bias:
            logits = logits + self.bias

        outputs = (logits, attention)
        if targets is not None:
            mrf_weight = self.compute_mrf_weight(attention)
            loss = self.loss(logits, targets, mrf_weight)
            outputs = (loss,) + outputs

        return outputs

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

        lr_scheduler = lr_schedulers.get(self.lr_scheduler)(
            optimizer, self.warmup_steps, self.trainer.max_steps
        )
        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_dict]

    def compute_regularization(self, targets, mrf_weight: torch.Tensor):
        """Compute regularization weights based on the number of targets."""
        sample_size = (targets != self.pad_idx).sum()

        batch_size = mrf_weight.size()[0]
        reg = self._weight_reg_coeff * mrf_weight.norm()
        if self.use_bias:
            reg += self._bias_reg_coeff * self.bias.norm()

        return reg * sample_size

    def loss(self, logits, targets, mrf_weight: torch.Tensor):
        """Compute GREMLIN loss w/ L2 Regularization"""
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction="sum")(
            logits.view(-1, self.vocab_size), targets.view(-1)
        )
        loss += self.compute_regularization(targets, mrf_weight)
        loss = loss / logits.size(0)
        return loss

    def compute_mrf_weight(self, attention):
        # Note that attention gives a mapping x -> MRF Weights(x),
        # so it is more general than a simple Pairwise MRF.
        value = self.value.weight.view(
            self.vocab_size + self.msa_length,
            self.num_attention_heads,
            self.attention_head_size,
        )
        output = self.output.weight
        output = output.view(
            self.vocab_size, self.num_attention_heads, self.attention_head_size
        )
        embed = torch.einsum("ahd,bhd->hab", value, output)  # H x (A + L) x A
        mrf_weight = torch.einsum("hij,hab->iajb", attention.sum(0), embed)
        return mrf_weight

    @torch.no_grad()
    def get_contacts(self):
        """Extracts contacts by getting the attentions."""
        inputs = torch.full(
            [1, self.msa_length],
            self.pad_idx,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        *_, attention = self.forward(inputs)
        mrf_weight = self.compute_mrf_weight(attention).squeeze()
        return mrf_weight.norm(dim=(1, 3))

    @classmethod
    def from_args(
        cls,
        args: Namespace,
        num_seqs: int,
        msa_length: int,
        msa_counts: Optional[torch.Tensor] = None,
        vocab_size: int = 20,
        pad_idx: int = 20,
        true_contacts: Optional[torch.Tensor] = None,
    ) -> "Attention":
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
        parser.add_argument(
            "--lr_scheduler",
            choices=lr_schedulers.LR_SCHEDULERS.keys(),
            default="warmup_constant",
            help="Learning rate scheduler to use.",
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=0,
            help="How many warmup steps to use when using a warmup schedule.",
        )
        return parser
