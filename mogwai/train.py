from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

from mogwai.data_loading import A3M_MSADataModule
from mogwai.parsing import parse_cf
from mogwai.models import GremlinPseudolikelihood
from mogwai.utils.functional import apc
from mogwai.metrics import contact_auc
from mogwai.vocab import FastaVocab

# Initialize parser
parser = ArgumentParser()
parser = A3M_MSADataModule.add_args(parser)
parser = GremlinPseudolikelihood.add_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(
    gpus=1,
    min_steps=50,
    max_steps=1000,
)
args = parser.parse_args()

# Load msa
msa_dm = A3M_MSADataModule.from_args(args)
msa_dm.setup()

# Load contacts
true_contacts = parse_cf("data/test/4rb6Y/4rb6Y.cf")

# Initialize model
model = GremlinPseudolikelihood(
    msa_dm.msa_dataset.num_seqs,
    msa_dm.msa_dataset.msa_length,
    msa_dm.msa_dataset.msa_counts,
    true_contacts=torch.from_numpy(true_contacts),
    pad_idx=FastaVocab.pad_idx,
)

# Initialize Trainer
trainer = pl.Trainer.from_argparse_args(args)

trainer.fit(model, msa_dm)
contacts = model.get_contacts()
contacts = apc(contacts)

auc = contact_auc(contacts, torch.from_numpy(true_contacts))
print(auc)
