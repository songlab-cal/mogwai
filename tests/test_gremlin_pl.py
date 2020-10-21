import itertools
import numpy as np
import torch
import unittest

from mogwai.data_loading import one_hot
from mogwai.models import GremlinPseudolikelihood


class TestGremlinPL(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        N = 100
        L = 20
        A = 8
        msa = torch.randint(0, A, [N, L])
        msa = torch.FloatTensor(one_hot(msa.numpy()))
        msa_counts = msa.sum(0)

        self.msa = msa
        self.model = GremlinPseudolikelihood(N, L, msa_counts, vocab_size=A)

        # Need nonzero weights but don't want to take a grad for this test
        wt = self.model.weight.data
        self.model.weight.data = torch.randn_like(wt)

        # Used for data leakage test.
        self.A = A

    def test_parameter_shapes(self):
        self.assertTupleEqual(self.model.weight.shape, (20, 8, 20, 8))
        self.assertTupleEqual(self.model.bias.shape, (20, 8))

    def test_forward_shape(self):
        batch = self.msa[:64]
        loss, logits = self.model(batch)
        self.assertTupleEqual(logits.shape, (64, 20, 8))

    def onehot_vector(self, idx: int):
        oh = torch.zeros(self.A)
        oh[idx] = 1.0
        return oh

    @torch.no_grad()
    def test_data_leakage(self):
        # Confirm that logits for position 0 do not change
        # when sequence at position 0 is exhaustively changed.
        logits_list = []
        example = self.msa[0]

        seq_pos = 0
        for i in range(self.A):
            example[seq_pos] = self.onehot_vector(i)
            _, logits = self.model(example.unsqueeze(0))
            logits_list.append(logits[0, seq_pos])
        all_pairs = itertools.combinations(logits_list, 2)
        for x, y in all_pairs:
            np.testing.assert_array_almost_equal(x.numpy(), y.numpy())


class TestGremlinPLGrad(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        N = 100
        L = 20
        A = 8
        msa = torch.randint(0, A, [N, L])
        msa = torch.FloatTensor(one_hot(msa.numpy()))
        msa_counts = msa.sum(0)

        self.msa = msa
        self.model = GremlinPseudolikelihood(N, L, msa_counts, vocab_size=A)

    def test_gradient(self):
        # Tests that backward runs.
        batch = self.msa[:64]
        loss, _ = self.model(batch)
        loss.backward()
        # TODO: Presumably there's a less stupid approach
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
