import torch


def apc(contacts, remove_diagonal: bool = True):
    if remove_diagonal:
        contacts.fill_diagonal_(0.0)

    a1 = contacts.sum(0, keepdims=True)
    a2 = contacts.sum(1, keepdims=True)
    normalized = contacts - (a1 * a2) / contacts.sum()
    normalized.fill_diagonal_(0.0)
    return normalized
