from typing import List, Dict
from argparse import ArgumentParser
import torch
from .base_wrapper_dataset import BaseWrapperDataset
from ..utils import collate_tensors
from ..vocab import FastaVocab


class PseudolikelihoodDataset(BaseWrapperDataset):
    """PseudolikelihoodDataset implements a mostly-dummy dataset, which simply wraps an
    existing token dataset. It is designed to act as a drop-in replacement of the
    MaskedLMDataset.

    Args:
        dataset (torch.utils.data.dataset): Dataset of tensors to wrap.
    """

    def __init__(self, dataset: torch.utils.data.dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) == 1:
            item = item[0]
        return {"src_tokens": item, "targets": item.clone()}

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser

    def collater(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        concat = {
            "src_tokens": collate_tensors(
                [element["src_tokens"] for element in batch], FastaVocab.pad_idx
            ),
            "targets": collate_tensors(
                [element["targets"] for element in batch], FastaVocab.pad_idx
            ),
        }
        return concat
