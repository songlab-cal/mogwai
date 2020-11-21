from argparse import ArgumentParser
import torch
from .base_wrapper_dataset import BaseWrapperDataset


class MaxStepsDataset(BaseWrapperDataset):
    """MaxSteps repeats the same dataset in order to have a length of exactly max_steps.
    This can help when working with relatively small datasets, since PyTorch dataloading
    operations reset at the end of each epoch. It will also provide accurate timing
    estimates for pytorch lightning.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to wrap
        max_steps (int): Total number of training steps.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, max_steps: int, batch_size: int):
        super().__init__(dataset)
        self._max_steps = max_steps
        self._batch_size = batch_size

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(
                f"index {idx} out of bounds for dataset of size {len(self)}"
            )
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return self._max_steps * self._batch_size
