from argparse import ArgumentParser
import torch
from .base_wrapper_dataset import BaseWrapperDataset


class RepeatDataset(BaseWrapperDataset):
    """RepeatDataset repeats the same dataset multiple times. This can help when
    working with relatively small datasets, since PyTorch dataloading operations
    reset at the end of each epoch.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to wrap
        n (int): Number of times to repeat the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, n: int):
        super().__init__(dataset)
        self._n = n

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError(
                f"index {idx} out of bounds for dataset of size {len(self)}"
            )
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return self._n * len(self.dataset)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--num_repeats",
            type=int,
            default=1,
            help="Number of times to repeat the input dataset.",
        )
        return parser
