import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_precision_vs_length(
    pred: torch.Tensor,
    meas: torch.Tensor,
    thresh: float = 1e-4,
    superdiag: int = 6,
):
    """Plot precision versus length for various length cutoffs.

    Analogous to a precision-recall curve.

    Args:
        pred (tensor): Predicted contact scores or probabilities.
        meas (tensor): Binary matrix of true contacts.
        thresh (float, optional): Threshold at which to call a predicted contact.
        superdiag (int, optional): Ignore all true and predicted contacts from diag to superdiag.
    """
    # Ignore nearby contacts
    eval_idx = np.triu_indices_from(meas, superdiag)
    pred_, meas_ = pred[eval_idx], meas[eval_idx]

    # Sort by model confidence
    sort_idx = pred_.argsort(descending=True)

    # want to separate correct from incorrect indices
    true_pos = list()
    false_pos = list()
    length = meas.shape[0]
    precision = list()
    optimal_precision = list()

    num_contacts = len(np.nonzero(meas_))
    # Only consider top 2L predictions
    for i, idx in enumerate(sort_idx[: (2 * length)]):
        # idx is in the flattened array of upper triang. values
        # recover the position in the matrix

        # Update optimal precision based on number of true contacts
        if i <= num_contacts:
            optimal_precision.append(1.0)
        else:
            num_false = i - num_contacts
            optimal_precision.append(num_contacts / (num_contacts + num_false))

        # Update model precision based on predictions
        xy = (eval_idx[0][idx], eval_idx[1][idx])
        if meas_[idx] >= thresh:
            true_pos.append(xy)
        else:
            false_pos.append(xy)

        precision.append(len(true_pos) / (len(true_pos) + len(false_pos)))

    plt.plot(precision, color="b")
    plt.plot(optimal_precision, color="k")