import unittest

from mogwai.data_loading import A3M_MSADataModule, NPZ_MSADataModule


class TestA3MDataModule(unittest.TestCase):
    def setUp(self):
        a3m_path = "data/test/4rb6Y/4rb6Y.i90c75.a3m"
        self.dm = A3M_MSADataModule(a3m_path)
        self.dm.setup()

    def test_datamodule_shape(self):
        self.assertTupleEqual(self.dm.dims, (7569, 107, 20))

    def test_batch_shape(self):
        batch = next(iter(self.dm.train_dataloader()))[0]
        self.assertTupleEqual(batch.shape, (64, 107, 20))

    def test_pssm_shape(self):
        self.assertTupleEqual(self.dm.msa_counts.shape, (107, 20))


class TestNPZDataModule(unittest.TestCase):
    def setUp(self):
        npz_path = "data/test/3er7_1_A.npz"
        self.dm = NPZ_MSADataModule(npz_path)
        self.dm.setup()

    def test_datamodule_shape(self):
        self.assertTupleEqual(self.dm.dims, (33672, 118, 20))

    def test_batch_shape(self):
        batch = next(iter(self.dm.train_dataloader()))[0]
        self.assertTupleEqual(batch.shape, (64, 118, 20))

    def test_contact_shape(self):
        self.assertTupleEqual(self.dm.true_contacts.shape, (118, 118))

    def test_pssm_shape(self):
        self.assertTupleEqual(self.dm.msa_counts.shape, (118, 20))


if __name__ == "__main__":
    unittest.main()
