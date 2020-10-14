import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_colored_preds_on_trues(
    pred: torch.Tensor,
    meas: torch.Tensor,
    cutoff: int = 1,
    thresh: float = 1e-4,
    superdiag: int = 6,
    point_size: int = 1,
):
    # Ignore nearby contacts
    eval_idx = np.triu_indices_from(meas, superdiag)
    pred_, meas_ = pred[eval_idx], meas[eval_idx]

    # Sort by model confidence
    sort_idx = pred_.argsort(descending=True)

    # want to plot the indexes that are right in blue and
    # the ones that are wrong in red
    # just consider the top L/cutoff contacts
    true_pos = list()
    false_pos = list()
    length = meas.shape[0]
    len_cutoff = int(length / cutoff)

    for idx in sort_idx[:len_cutoff]:
        # idx is in the flattened array of upper triang. values
        # recover the position in the matrix
        xy = (eval_idx[0][idx], eval_idx[1][idx])
        if meas_[idx] >= thresh:
            true_pos.append(xy)
        else:
            false_pos.append(xy)

    # there should only be len_cutoff total level contacts
    assert len(true_pos) + len(false_pos) == len_cutoff

    true_contacts_ij = list()
    for i, j in zip(eval_idx[0], eval_idx[1]):
        if meas[i, j] >= thresh:
            true_contacts_ij.append((i, j))

    plt.imshow(meas, cmap="gray_r", alpha=0.1, label="measured contacts")
    if len(true_contacts_ij) > 1:
        x, y = zip(*true_contacts_ij)
        plt.scatter(x, y, c="grey", s=point_size, alpha=0.3)
        # plot symmetric values
        plt.scatter(y, x, c="grey", s=point_size, alpha=0.3)
    if len(true_pos) > 1:
        x, y = zip(*true_pos)
        plt.scatter(x, y, c="b", s=point_size, alpha=0.4, label="true positives")
        plt.scatter(y, x, c="b", s=point_size, alpha=0.4, label="true positives")
    if len(false_pos) > 1:
        x, y = zip(*false_pos)
        plt.scatter(x, y, c="r", s=point_size, alpha=0.4, label="false positives")
        plt.scatter(y, x, c="r", s=point_size, alpha=0.4, label="false positives")
