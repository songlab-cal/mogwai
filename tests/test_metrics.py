import torch
import unittest

from mogwai.metrics import contact_auc, precision_at_cutoff


class TestPrecision(unittest.TestCase):
    def setUp(self):
        self.pred = torch.FloatTensor(
            [
                [1e-3, 1e-2, 0.8],
                [1e-2, 1e-4, 0.3],
                [0.8, 0.3, 1e-10],
            ]
        )
        self.meas = torch.IntTensor([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    def test_precision_cutoffs(self):
        p_at_1 = precision_at_cutoff(self.pred, self.meas, cutoff=1, superdiag=0)
        p_at_2 = precision_at_cutoff(self.pred, self.meas, cutoff=2, superdiag=0)
        p_at_3 = precision_at_cutoff(self.pred, self.meas, cutoff=3, superdiag=0)

        self.assertEqual(p_at_1, 2.0 / 3)
        self.assertEqual(p_at_2, 1.0)
        self.assertEqual(p_at_3, 1.0)

    def test_superdiag(self):
        superdiag_0 = precision_at_cutoff(self.pred, self.meas, cutoff=1, superdiag=0)
        superdiag_1 = precision_at_cutoff(self.pred, self.meas, cutoff=1, superdiag=1)
        superdiag_2 = precision_at_cutoff(self.pred, self.meas, cutoff=1, superdiag=2)
        superdiag_3 = precision_at_cutoff(self.pred, self.meas, cutoff=1, superdiag=3)
        self.assertEqual(superdiag_0, 2.0 / 3)
        self.assertEqual(superdiag_1, 2.0 / 3)
        self.assertEqual(superdiag_2, 1.0)
        self.assertTrue(superdiag_3.isnan())


class TestAUC(unittest.TestCase):
    def setUp(self):
        self.pred = torch.FloatTensor(
            [
                [1e-3, 1e-2, 0.8],
                [1e-2, 1e-4, 0.3],
                [0.8, 0.3, 1e-10],
            ]
        )
        self.meas = torch.IntTensor([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    def test_range(self):
        auc = contact_auc(self.pred, self.meas, superdiag=0, cutoff_range=[1, 2, 3])
        self.assertEqual(auc, 8.0 / 9)

    def test_superdiag_range(self):
        auc_superdiag_1 = contact_auc(
            self.pred, self.meas, superdiag=1, cutoff_range=[1, 2, 3]
        )
        auc_superdiag_2 = contact_auc(
            self.pred, self.meas, superdiag=2, cutoff_range=[1, 2, 3]
        )
        self.assertEqual(auc_superdiag_1, 8.0 / 9)
        self.assertEqual(auc_superdiag_2, 1.0)


if __name__ == "__main__":
    unittest.main()