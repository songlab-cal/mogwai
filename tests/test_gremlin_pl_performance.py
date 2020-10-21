import torch
import unittest

import pytorch_lightning as pl

from mogwai.data_loading import A3M_MSADataModule, parse_cf
from mogwai.models import GremlinPseudolikelihood


class TestGremlinPLPerformance(unittest.TestCase):
    def setUp(self):
        npz_path = "data/test/4rb6Y/4rb6Y.i90c75.a3m"
        self.dm = A3M_MSADataModule(npz_path, batch_size=4096)
        self.dm.setup()

        true_contacts = parse_cf("data/test/4rb6Y/4rb6Y.cf")

        n, l, _ = self.dm.dims
        self.model = GremlinPseudolikelihood(
            n, l, self.dm.msa_counts, true_contacts=torch.tensor(true_contacts)
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
