import unittest

from mogwai.data_loading import MSADataModule


class TestA3MDataModule(unittest.TestCase):
    def setUp(self):
        a3m_path = "data/test/4rb6Y/4rb6Y.i90c75.a3m"
        self.dm = MSADataModule(a3m_path, batch_size=64)
        self.dm.setup()

    def test_datamodule_stats(self):
        num_seqs, msa_length, msa_counts = self.dm.get_stats()
        self.assertEqual(num_seqs, 7569)
        self.assertEqual(msa_length, 107)
        self.assertTupleEqual(msa_counts.shape, (107, 20))

    def test_batch_shape(self):
        batch = next(iter(self.dm.train_dataloader()))
        self.assertTupleEqual(batch['src_tokens'].shape, (64, 107))


class TestNPZDataModule(unittest.TestCase):
    def setUp(self):
        npz_path = "data/test/3er7_1_A.npz"
        self.dm = MSADataModule(npz_path, batch_size=64)
        self.dm.setup()

    def test_datamodule_stats(self):
        num_seqs, msa_length, msa_counts = self.dm.get_stats()
        self.assertEqual(num_seqs, 33672)
        self.assertEqual(msa_length, 118)
        self.assertTupleEqual(msa_counts.shape, (118, 20))

    def test_batch_shape(self):
        batch = next(iter(self.dm.train_dataloader()))
        self.assertTupleEqual(batch['src_tokens'].shape, (64, 118))


if __name__ == "__main__":
    unittest.main()
