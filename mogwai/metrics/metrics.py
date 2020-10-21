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
