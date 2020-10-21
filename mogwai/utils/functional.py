"""Contains transformations shared by models."""

import torch


def symmetrize_matrix(inp: torch.Tensor) -> torch.Tensor:
    """Symmetrize matrix in additive fashion.

    Symmetrizes A with (A + A.T)/2.

    Args:
        inp (tensor): Matrix to symmetrize.
    """
    return 0.5 * (inp + inp.transpose(-1, -2))


def symmetrize_matrix_(inp: torch.Tensor) -> torch.Tensor:
    """Inplace version of symmetrize_matrix.

    Args:
        inp (tensor): Matrix to symmetrize.
    """
    inp.add_(inp.transpose(-1, -2).clone())
    inp.mul_(0.5)
    return inp


def symmetrize_potts(weight: torch.Tensor) -> torch.Tensor:
    """Symmetrize 4D Potts coupling tensor.

    Enforces the constraint that W(i,j) = W(j,i) for coupling matrices.

    Args:
        weight (tensor): 4D tensor of shape (length, vocab, length, vocab) to symmetrize.
    """
    return 0.5 * (weight + weight.permute(2, 3, 0, 1))


def symmetrize_potts_(weight: torch.Tensor) -> torch.Tensor:
    """Inplace version of symmetrize_potts.

    Args:
        weight (tensor): 4D tensor of shape (length, vocab, length, vocab) to symmetrize.
    """
    weight.add_(weight.permute(2, 3, 0, 1).clone())
    weight.mul_(0.5)
    return weight


def zero_diag(inp: torch.Tensor) -> torch.Tensor:
    """Zeros all elements of diagonal.

    Computes diagonal along last two axes of input tensor.

    Args:
        inp (tensor): Tensor to zero out.
    """

    diag_mask = torch.eye(
        inp.size(-2),
        inp.size(-1),
        dtype=torch.bool,
        device=inp.device,
    )
    return inp.masked_fill(diag_mask, 0.0)


def zero_diag_(inp: torch.Tensor) -> torch.Tensor:
    """Inplace version of zero_diag.

    Args:
        inp (tensor): Tensor to zero out.
    """
    diag_mask = torch.eye(
        inp.size(-2),
        inp.size(-1),
        dtype=torch.bool,
        device=inp.device,
    )
    return inp.masked_fill_(diag_mask, 0.0)


def apc(inp: torch.Tensor, remove_diag: bool = False) -> torch.Tensor:
    """Compute Average Product Correction (APC) of tensor.

    Applies correction along last two axes.

    Args:
        inp (tensor): Tensor to correct.
        remove_diag (bool, optional): Whether to zero out diagonal before correcting.
    """

    if remove_diag:
        inp = zero_diag(inp)

    a1 = inp.sum(-1, keepdims=True)  # type: ignore
    a2 = inp.sum(-2, keepdims=True)  # type: ignore
    corr = inp - (a1 * a2) / inp.sum((-1, -2), keepdims=True)  # type: ignore
    corr = zero_diag(corr)
    return corr


def apc_(inp: torch.Tensor, remove_diag: bool = False) -> torch.Tensor:
    """Inplace version of apc.

    Args:
        inp (tensor): Tensor to correct.
        remove_diag (bool, optional): Whether to zero out diagonal before correcting.
    """
    if remove_diag:
        zero_diag_(inp)

    a1 = inp.sum(-1, keepdims=True)  # type: ignore
    a2 = inp.sum(-2, keepdims=True)  # type: ignore
    a12 = inp.sum((-1, -2), keepdims=True)  # type: ignore
    corr = a1 * a2
    corr.div_(a12)

    inp.sub_(corr)
    zero_diag_(inp)
    return inp
