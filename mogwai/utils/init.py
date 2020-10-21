"""Contains initializers shared by models."""

from math import log
import torch

from .functional import zero_diag_


def init_potts_bias(
    msa_counts: torch.Tensor, l2_coeff: float, num_seqs: int
) -> torch.Tensor:
    """Initialize single-site log-potential as regularized PSSM.

    Args:
        msa_counts (tensor): Counts of amino acids per-position of MSA.
        l2_coeff (float): L2 regularization weight.
        num_seqs (int): Number of sequences in MSA.
    """
    bias = (msa_counts + l2_coeff * log(num_seqs)).log()
    bias.add_(-bias.mean(-1, keepdims=True))  # type: ignore
    return bias


def init_potts_weight(msa_length: int, vocab_size: int) -> torch.Tensor:
    """Initializes Potts coupling matrices of all zeros.

    Args:
        msa_length (int): Length of MSA.
        vocab_size (int): Number of characters in MSA alphabet.
    """
    weight = torch.zeros(msa_length, vocab_size, msa_length, vocab_size)
    return weight


def init_pseudolik_mask(msa_length: int) -> torch.Tensor:
    """Creates mask for efficient pseudolikelihood calculation.

    Args:
        msa_length (int): Length of MSA.
    """
    diag_mask = torch.ones(msa_length, msa_length, dtype=torch.float)
    diag_mask = zero_diag_(diag_mask)
    return diag_mask