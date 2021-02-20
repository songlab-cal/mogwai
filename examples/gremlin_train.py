import ipdb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from mogwai.data_loading import MSADataModule
from mogwai.parsing import contacts_from_cf
from mogwai.models import Gremlin
from mogwai.plotting import plot_colored_preds_on_trues
from mogwai.utils import apc


# Load msa
msa_dm = MSADataModule("../data/test/4rb6Y/4rb6Y.i90c75.a3m", batch_size=4096)
msa_dm.setup()

# Load contacts
true_contacts = contacts_from_cf("../data/test/4rb6Y/4rb6Y.cf")

# Initialize model
num_seqs, msa_length, msa_counts = msa_dm.get_stats()
model = Gremlin(num_seqs, msa_length, msa_counts, true_contacts=torch.tensor(true_contacts))

# Initialize Trainer
trainer = pl.Trainer(min_steps=50, max_steps=50, gpus=1)
trainer.fit(model, msa_dm)

# real test pass is currently running out of memory for some reason...
# trainer.test(test_dataloaders=msa_dm.train_dataloader())


# this is on CPU so it is much slower
energies = []
for batch_n, batch in tqdm(enumerate(msa_dm.train_dataloader())):
    if batch_n > 10:
        break
    batch_energies = model.hamiltonian(batch['src_tokens'])
    energies.append(batch_energies)
# sequence = batch['src_tokens'][0]

ipdb.set_trace()
# batch_energies = model.hamiltonian(batch['src_tokens'])

