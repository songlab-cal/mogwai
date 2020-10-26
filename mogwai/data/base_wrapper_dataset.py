from argparse import ArgumentParser
import torch
from typing import List, Any


class BaseWrapperDataset(torch.utils.data.Dataset):
    """BaseWrapperDataset. Wraps an existing dataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset to wrap.
    """

    def __init__(self, dataset: torch.utils.data.dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser

    def collater(self, batch: List[Any]) -> Any:
        return self.dataset.collater(batch)
