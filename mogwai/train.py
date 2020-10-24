from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from mogwai.data_loading import MSADataModule
from mogwai.parsing import read_contacts
from mogwai.models import Gremlin
from mogwai.utils.functional import apc
from mogwai.metrics import contact_auc
from mogwai.vocab import FastaVocab

# Initialize parser
parser = ArgumentParser()
parser.add_argument(
    "--structure_file",
    type=str,
    default=None,
    help="Optional pdb or cf file containing protein structure. Used for evaluation.",
)
parser = MSADataModule.add_args(parser)
parser = Gremlin.add_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(
    gpus=1,
    min_steps=50,
    max_steps=1000,
)
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
model = Gremlin(
    msa_dm.msa_dataset.num_seqs,
    msa_dm.msa_dataset.msa_length,
    msa_dm.msa_dataset.msa_counts,
    true_contacts=true_contacts,
    pad_idx=FastaVocab.pad_idx,
)

# Initialize Trainer
trainer = pl.Trainer.from_argparse_args(args)

trainer.fit(model, msa_dm)
contacts = model.get_contacts()
contacts = apc(contacts)

if true_contacts is not None:
    auc = contact_auc(contacts, true_contacts)
    print(auc)
