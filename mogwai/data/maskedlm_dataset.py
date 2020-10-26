from typing import List, Dict
from argparse import ArgumentParser
import torch
from .base_wrapper_dataset import BaseWrapperDataset
from ..utils import collate_tensors
from ..vocab import FastaVocab


class MaskedLMDataset(BaseWrapperDataset):
    """MaskedLMDataset implements masking tokens with a specified mask index.

    Args:
        dataset (torch.utils.data.Dataset): Dataset of tensors to wrap.
        mask_idx (int): Index of mask token.
        vocab_size (int): Vocab Size.
        mask_prob (float): Probability of masking a token.
        mask_rnd_prob (float): Probability of replacing token with a random token.
        mask_leave_prob (float): Probability of leaving token as-is.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        mask_idx: int,
        pad_idx: int,
        vocab_size: int,
        mask_prob: float = 0.15,
        mask_rnd_prob: float = 0.0,
        mask_leave_prob: float = 0.0,
    ):
        super().__init__(dataset)
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.mask_rnd_prob = mask_rnd_prob
        self.mask_leave_prob = mask_leave_prob

        assert 0 < mask_prob < 1
        assert 0 <= mask_rnd_prob < 1
        assert 0 <= mask_leave_prob < 1
        assert mask_leave_prob + mask_rnd_prob < 1

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) == 1:
            item = item[0]

        mask = torch.rand_like(item, dtype=torch.float) < self.mask_prob
        targets = item.masked_fill(~mask, self.pad_idx)

        token_type_probs = torch.rand_like(item, dtype=torch.float)
        is_leave = token_type_probs < self.mask_leave_prob
        is_rnd = (
            token_type_probs < (self.mask_leave_prob + self.mask_rnd_prob)
        ) & ~is_leave
        is_mask = ~(is_leave | is_rnd)

        # Do not make this in-place
        item = item.masked_fill(mask & is_mask, self.mask_idx)
        item[is_rnd & mask] = torch.randint_like(item, self.vocab_size)[is_rnd & mask]

        return {"src_tokens": item, "targets": targets}

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--mask_prob",
            type=float,
            default=0.15,
            help="Probability of masking tokens.",
        )
        parser.add_argument(
            "--mask_rnd_prob",
            type=float,
            default=0.0,
            help="Probability of replacing masked token with a random token.",
        )
        parser.add_argument(
            "--mask_leave_prob",
            type=float,
            default=0.0,
            help="Probability of keeping correct token as mask token.",
        )
        return parser

    def collater(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        concat = {
            "src_tokens": collate_tensors(
                [element["src_tokens"] for element in batch], FastaVocab.pad_idx
            ),
            "targets": collate_tensors(
                [element["targets"] for element in batch], FastaVocab.pad_idx
            ),
            "src_lengths": torch.tensor(
                [len(element["src_tokens"]) for element in batch], dtype=torch.long
            ),
        }
        return concat
