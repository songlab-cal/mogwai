from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from mogwai.data_loading import MSADataModule
from mogwai.parsing import read_contacts
from mogwai import models
from mogwai.utils.functional import apc
from mogwai.metrics import contact_auc
from mogwai.vocab import FastaVocab


def train():
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        default="gremlin",
        choices=models.MODELS.keys(),
        help="Which model to train.",
    )
    model_name = parser.parse_known_args()[0].model
    parser.add_argument(
        "--structure_file",
        type=str,
        default=None,
        help=(
            "Optional pdb or cf file containing protein structure. "
            "Used for evaluation."
        ),
    )
    parser = MSADataModule.add_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=1,
        min_steps=50,
        max_steps=1000,
    )
    model_type = models.get(model_name)
    model_type.add_args(parser)
    args = parser.parse_args()

    # Load msa
    msa_dm = MSADataModule.from_args(args)
    msa_dm.setup()

    # Load contacts
    true_contacts = (
        torch.from_numpy(read_contacts(args.structure_file))
        if args.structure_file is not None
        else None
    )

    # Initialize model
    num_seqs, msa_length, msa_counts = msa_dm.get_stats()
    model = model_type.from_args(
        args,
        num_seqs=num_seqs,
        msa_length=msa_length,
        msa_counts=msa_counts,
        vocab_size=len(FastaVocab),
        pad_idx=FastaVocab.pad_idx,
        true_contacts=true_contacts,
    )

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, msa_dm)

    if true_contacts is not None:
        contacts = model.get_contacts()
        auc = contact_auc(contacts, true_contacts).item()
        contacts = apc(contacts)
        auc_apc = contact_auc(contacts, true_contacts).item()
        print(f"AUC: {auc:0.3f}, AUC_APC: {auc_apc:0.3f}")


if __name__ == "__main__":
    train()
