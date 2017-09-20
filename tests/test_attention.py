import unittest

import torch
from torch.autograd import Variable

from models import attention


class TestDotAttention(unittest.TestCase):

    def test_compute_attention_weights(self):
        att = attention.DotAttention(input_dim=3, hidden_dim=4,
                                     input_feeding=True)
        att_queries = Variable(torch.randn(3, 2, 4))
        enc_states = Variable(torch.randn(5, 2, 4))
        enc_lengths = torch.LongTensor([4, 2])
        att_weights = att.compute_attention_weights(
            attention_queries=att_queries,
            encoder_states=enc_states,
            encoder_lengths=enc_lengths)
        self.assertTrue((att_weights[0, :, 4] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 2] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 3] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 4] < 1e-5).data.all())

    def test_compute_contexts(self):
        att = attention.DotAttention(input_dim=3, hidden_dim=4,
                                     input_feeding=True)
        att_queries = Variable(torch.randn(3, 2, 4))
        enc_states = Variable(torch.randn(5, 2, 4))
        enc_lengths = torch.LongTensor([4, 2])
        att_weights = att.compute_attention_weights(
            attention_queries=att_queries,
            encoder_states=enc_states,
            encoder_lengths=enc_lengths)
        contexts = att.compute_contexts(attention_weights=att_weights,
                                        encoder_states=enc_states)
        self.assertTupleEqual(tuple(contexts.size()), (3, 2, 4))
