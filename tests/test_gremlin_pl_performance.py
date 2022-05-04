import torch
import unittest

import pytorch_lightning as pl

from mogwai.data_loading import MSADataModule
from mogwai.parsing import contacts_from_cf
from mogwai.models import Gremlin


class TestGremlinPLPerformance(unittest.TestCase):
    def setUp(self):
        npz_path = "data/test/4rb6Y/4rb6Y.i90c75.a3m"
        self.dm = MSADataModule(npz_path, batch_size=4096)
        self.dm.setup()

        true_contacts = contacts_from_cf("data/test/4rb6Y/4rb6Y.cf")

        n, l, msa_counts = self.dm.get_stats()
        self.model = Gremlin(
            n, l, msa_counts, true_contacts=torch.tensor(true_contacts)
        )
        self.trainer = pl.Trainer(
            min_steps=50,
            max_steps=50,
            gpus=1,
        )

    def test_training_performance(self):
        self.trainer.fit(self.model, self.dm)
        final_auc_apc = self.model.get_auc(do_apc=True)
        self.assertGreaterEqual(final_auc_apc, 0.91)


if __name__ == "__main__":
    unittest.main()
