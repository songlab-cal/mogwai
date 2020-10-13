from pathlib import Path
import numpy as np
import unittest

from mogwai.utils.data_loading import load_a3m_msa, parse_cf


class TestA3MLoading(unittest.TestCase):
    def setUp(self):
        self.path = Path("data/test/4rb6Y") / "4rb6Y.i90c75.a3m"
        if not self.path.exists():
            raise FileNotFoundError(
                "Please download 4rb6Y using scripts/download_example.sh to run tests."
            )
        self.msa, self.ms, _, self.ref = load_a3m_msa(self.path)

    def test_msa_shape(self):
        N, L, A = self.msa.shape
        self.assertEqual(N, 7569)
        self.assertEqual(L, 107)
        self.assertEqual(A, 20)

    def test_ms_shape(self):
        N, L, A = self.ms.shape
        self.assertEqual(N, 7569)
        self.assertEqual(L, 162)
        self.assertEqual(A, 20)

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
        self.assertEqual(shape, (107, 107))

    def test_zero_contacts(self):
        num_zero_contacts = np.sum(self.contacts == 0)
        self.assertEqual(num_zero_contacts, 10169)

    def test_contact_mass(self):
        contact_mass = np.sum(self.contacts)
        self.assertAlmostEqual(contact_mass, 1158.90602)


if __name__ == "__main__":
    unittest.main()
