from pathlib import Path
import numpy as np
import unittest

from mogwai.data_loading import one_hot, load_a3m_msa, parse_cf


class TestOneHot(unittest.TestCase):
    def setUp(self):
        self.msa = np.array(
            [
                [2, 5, 4],
                [3, 2, 19],
            ]
        )
        self.oh = one_hot(self.msa)

    def test_shape(self):
        self.assertTupleEqual(self.oh.shape, (2, 3, 20))

    def test_argmax(self):
        idx = np.argmax(self.oh, -1)
        np.testing.assert_array_equal(idx, self.msa)

    def test_pad(self):
        padded_msa = np.array([[12, 13, -1, -1], [-1, 9, -1, 8]])
        padded_idx = np.array([[False, False, True, True], [True, False, True, False]])

        oh = one_hot(padded_msa)
        test_padded_idx = np.sum(oh, -1) == 0

        np.testing.assert_array_equal(padded_idx, test_padded_idx)


class TestA3MLoading(unittest.TestCase):
    def setUp(self):
        self.path = Path("data/test/4rb6Y") / "4rb6Y.i90c75.a3m"
        if not self.path.exists():
            raise FileNotFoundError(
                "Please download 4rb6Y using scripts/download_example.sh to run tests."
            )
        self.msa, self.ms, _, self.ref = load_a3m_msa(self.path)

    def test_msa_shape(self):
        self.assertTupleEqual(self.msa.shape, (7569, 107, 20))

    def test_ms_shape(self):
        self.assertTupleEqual(self.ms.shape, (7569, 162, 20))

    def test_msa_indel(self):
        # Test for gaps at start of second seq in msa.
        seq = self.msa[1].argmax(-1)
        self.assertEqual(seq[0], 0)
        self.assertEqual(seq[1], 0)

        # Test for no gaps at end
        self.assertEqual(seq[-3], 5)
        self.assertEqual(seq[-4], 6)

    def test_ms_indel(self):
        # Test no gaps at start of second seq in ms.
        seq = self.ms[1].argmax(-1)
        self.assertEqual(seq[0], 11)
        self.assertEqual(seq[1], 16)

        # Test for gaps at end
        self.assertEqual(seq[-3], 0)
        self.assertEqual(seq[-4], 0)

    def test_reference(self):
        reference = "MRVKMHVKKGDTVLVASGKYKGRVGKVKEVLPKKYAVIVEGVNIVKKAVRVSPKYPQGGFIEKEAPLHASKVRPICPACGKPTRVRKKFLENGKKIRVCAKCGGALD"
        self.assertEqual(self.ref, reference)


class TestCfLoading(unittest.TestCase):
    def setUp(self):
        self.path = Path("data/test/4rb6Y") / "4rb6Y.cf"
        if not self.path.exists():
            raise FileNotFoundError(
                "Please download 4rb6Y using scripts/download_example.sh to run tests."
            )
        self.contacts = parse_cf(self.path)

    def test_contact_shape(self):
        shape = self.contacts.shape
        self.assertTupleEqual(shape, (107, 107))

    def test_zero_contacts(self):
        num_zero_contacts = np.sum(self.contacts == 0)
        self.assertEqual(num_zero_contacts, 10169)

    def test_contact_mass(self):
        contact_mass = np.sum(self.contacts)
        self.assertAlmostEqual(contact_mass, 1158.90602)


if __name__ == "__main__":
    unittest.main()
