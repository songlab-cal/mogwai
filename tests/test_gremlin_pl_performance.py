import torch
import unittest

import pytorch_lightning as pl

from mogwai.data_loading import NPZ_MSADataModule
from mogwai.models import GremlinPseudolikelihood


class TestGremlinPLPerformance(unittest.TestCase):
    def setUp(self):
        npz_path = "data/test/3er7_1_A.npz"
        self.dm = NPZ_MSADataModule(npz_path, batch_size=4096)
        self.dm.setup()

        n, l, _ = self.dm.dims
        self.model = GremlinPseudolikelihood(
            n, l, self.dm.msa_counts, true_contacts=self.dm.true_contacts
        )
        self.trainer = pl.Trainer(
            min_steps=200,
            max_steps=200,
            gpus=1,
        )

    def test_training_performance(self):
        self.trainer.fit(self.model, self.dm)
        final_auc_apc = self.model.get_auc(do_apc=True)
        self.assertGreaterEqual(final_auc_apc, 0.74)


if __name__ == "__main__":
    unittest.main()
