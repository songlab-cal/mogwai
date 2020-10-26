from argparse import ArgumentParser, Namespace
from typing import Optional
import math

import torch
import torch.nn as nn
from apex.optimizers import FusedLAMB

from .base_model import BaseModel
from ..utils import symmetrize_matrix_
from ..utils.init import init_potts_bias


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_attention_heads: int,
        attention_head_size: int,
        output_size: int,
    ):
        super().__init__()
        hidden_size = num_attention_heads * attention_head_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.qkv = nn.Linear(input_size, hidden_size * 3, bias=False)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        batch_size, seqlen, _ = inputs.size()
        queries, keys, values = self.qkv(inputs).chunk(dim=-1, chunks=3)
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
        attention = attention.softmax(-1)
        context = torch.einsum("nhij,njhd->nihd", attention, values)
        context = context.reshape(
            batch_size, seqlen, self.num_attention_heads * self.attention_head_size
        )
        return self.output(context), attention


class MultilayerAttention(BaseModel):
    """Attention Layer.

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
        msa_counts: torch.Tensor,
        attention_head_size: int = 16,
        num_attention_heads: int = 32,
        num_layers: int = 1,
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

        hidden_size = attention_head_size * num_attention_heads

        layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = hidden_size if layer > 0 else msa_length + vocab_size
            output_size = hidden_size if layer + 1 < num_layers else vocab_size
            layers.append(
                MultiheadAttention(
                    input_size, num_attention_heads, attention_head_size, output_size
                )
            )
        self.layers = layers

        if self.use_bias:
            bias = init_potts_bias(msa_counts, l2_coeff, num_seqs)
            bias = nn.Parameter(bias, True)
            self.register_parameter("bias", bias)

        self.register_buffer("posembed", torch.eye(msa_length).unsqueeze(0))
        self.register_buffer("one_hot", torch.eye(vocab_size + 1, vocab_size))
        self.register_buffer("diag_mask", torch.eye(msa_length) * -10000)

    def forward(self, src_tokens, targets=None):
        batch_size, seqlen = src_tokens.size()
        inputs = self.one_hot[src_tokens]
        posembed = self.posembed.repeat(batch_size, 1, 1)
        inputs = torch.cat((inputs, posembed), -1)

        for layer in self.layers:
            outputs, attention = layer(inputs)
            if outputs.size() == inputs.size():
                outputs = outputs + inputs
            inputs = outputs

        logits = inputs
        if self.use_bias:
            logits = logits + self.bias

        outputs = (logits, attention)
        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction="sum")(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )
            loss = loss / inputs.size(0)
            outputs = (loss,) + outputs
        return outputs

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.l2_coeff
            )
        elif self.optimizer == "lamb":
            optimizer = FusedLAMB(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_coeff,
            )
        else:
            raise ValueError(f"Unrecognized optimizer {self.optimizer}")
        return [optimizer]

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
        attention = attention.mean((0, 1))
        attention = symmetrize_matrix_(attention)
        return attention

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
    ) -> "Attention":
        return cls(
            num_seqs=num_seqs,
            msa_length=msa_length,
            msa_counts=msa_counts,
            attention_head_size=args.attention_head_size,
            num_attention_heads=args.num_attention_heads,
            num_layers=args.num_layers,
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
            "--num_layers",
            type=int,
            default=1,
            help="Number of attention layers.")
        parser.add_argument(
            "--optimizer",
            choices=["adam", "lamb"],
            default="adam",
            help="Which optimizer to use.",
        )
        return parser
