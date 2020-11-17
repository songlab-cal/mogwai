from typing import List, Optional
from typing import List

import numpy as np
import torch
from scipy.stats import entropy


def precision_at_cutoff(
    pred: torch.Tensor,
    meas: torch.Tensor,
    thresh: float = 0.01,
    superdiag: int = 6,
    cutoff: int = 1,
):
    """Computes precision for top L/k contacts.

    Args:
        pred (tensor): Predicted contact scores or probabilities.
        meas (tensor): Binary matrix of true contacts.
        thresh (float, optional): Threshold at which to call a predicted contact.
        superdiag (int, optional): Ignore all true and predicted contacts from diag to superdiag.
        cutoff (int, optional): Only compute precision of top L/cutoff predictions.
    """

    # Subset everything above superdiag
    eval_idx = np.triu_indices_from(meas, superdiag)
    pred_, meas_ = pred[eval_idx], meas[eval_idx]

    # Sort by model confidence
    sort_idx = pred_.argsort(descending=True)

    # Extract top predictions and calculate
    len_cutoff = (len(meas) / torch.tensor(cutoff)).int()
    preds = meas_[sort_idx][:len_cutoff]

    num_positives = (meas_[sort_idx][:len_cutoff] > thresh).sum().float()
    num_preds = len(preds)
    precision = num_positives / num_preds

    return precision


def precision_at_cutoff(
    pred: torch.Tensor,
    meas: torch.Tensor,
    thresh: float = 0.01,
    superdiag: int = 6,
    cutoff: int = 1,
):
    """Computes precision for top L/k contacts.

    Args:
        pred (tensor): Predicted contact scores or probabilities.
        meas (tensor): Binary matrix of true contacts.
        thresh (float, optional): Threshold at which to call a predicted contact.
        superdiag (int, optional): Ignore all true and predicted contacts from diag to superdiag.
        cutoff (int, optional): Only compute precision of top L/cutoff predictions.
    """

    # Subset everything above superdiag
    eval_idx = np.triu_indices_from(meas, superdiag)
    pred_, meas_ = pred[eval_idx], meas[eval_idx]

    # Sort by model confidence
    sort_idx = pred_.argsort(descending=True)

    # Extract top predictions and calculate
    len_cutoff = (len(meas) / torch.tensor(cutoff)).int()
    preds = meas_[sort_idx][:len_cutoff]

    num_positives = (meas_[sort_idx][:len_cutoff] > thresh).sum().float()
    num_preds = len(preds)
    precision = num_positives / num_preds

    return precision


# https://github.com/rmrao/explore-protein-attentcion/blob/main/metrics.py
def precisions_in_range(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
):
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full(
            [batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    indices = predictions_upper.topk(
        dim=-1, k=seqlen, sorted=True, largest=True
    ).indices

    topk_targets = targets_upper[torch.arange(batch_size), indices] > 0.01

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0)
        * src_lengths.unsqueeze(1)
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl10 = binned_precisions[:, 0]
    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {
        "auc": auc,
        "pr_at_l": pl,
        "pr_at_l_2": pl2,
        "pr_at_l_5": pl5,
        "pr_at_l_10": pl10
    }


def contact_auc(
    pred: torch.Tensor,
    meas: torch.Tensor,
    thresh: float = 0.01,
    superdiag: int = 6,
    cutoff_range: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
):
    """Compute modified Area Under PR Curve.

    Args:
        pred (tensor): Predicted contact scores or probabilities.
        meas (tensor): Binary matrix of true contacts.
        thresh (float, optional): Threshold at which to call a predicted contact.
        superdiag (int, optional): Ignore all true and predicted contacts from diag to superdiag.
        cutoff_range (List[int], optional): Range of precision cutoffs to use for averaging.
    """

    # Nick: This does not agree with normal AUPR, but instead computes
    # precision for top L/10, L/9, ... then averages them together.
    # Compared to normal precision this puts more weight on a small number
    # of high confidence true positives.

    # True aupr could be computed via
    #
    # from sklearn.metrics import average_precision_score
    #
    # eval_idx = np.triu_indices_from(meas, superdiag)
    # pred_, meas_ = pred[eval_idx], meas[eval_idx]
    # aupr = average_precision_score(meas_ > thresh, pred_)

    binned_precisions = [
        precision_at_cutoff(pred, meas, thresh, superdiag, c) for c in cutoff_range
    ]
    return torch.stack(binned_precisions, 0).mean()


def get_len_stdev(msa):
    lengths = []
    for seq in msa:
        num_gaps = len(np.where(seq == 20)[0])
        len_unaligned = len(seq) - num_gaps
        lengths.append(len_unaligned)
    return np.std(lengths)


def get_len_entropy(msa):
    lengths = []
    for seq in msa:
        num_gaps = len(np.where(seq == 20)[0])
        len_unaligned = len(seq) - num_gaps
        lengths.append(len_unaligned)
    return entropy(lengths)
