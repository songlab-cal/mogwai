"""Contains common tensor operations."""

from typing import Sequence, Union
import torch
import numpy as np


def collate_tensors(
    tensors: Sequence[torch.Tensor], pad_value: Union[int, float, bool, str] = 0
):
    dtype = tensors[0].dtype
    device = tensors[0].device
    batch_size = len(tensors)
    shape = (batch_size,) + tuple(np.max([tensor.size() for tensor in tensors], 0))

    padded = torch.full(shape, pad_value, dtype=dtype, device=device)
    for position, tensor in zip(padded, tensors):
        tensorslice = tuple(slice(dim) for dim in tensor.shape)
        position[tensorslice] = tensor

    return padded
