import unittest

import torch
from torch.autograd import Variable

from models import attention


class TestDotAttention(unittest.TestCase):

    def test_score(self):
        att = attention.DotAttention(hidden_dim=4, dropout_prob=0.1)
        att_queries = Variable(torch.randn(2, 3, 4))
        enc_states = Variable(torch.randn(2, 5, 4))
        enc_length = torch.LongTensor([4, 2])
        att_weights = att.score(
            queries=att_queries,
            annotations=enc_states, annotations_length=enc_length)
        self.assertTrue((att_weights[0, :, 4] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 2] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 3] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 4] < 1e-5).data.all())

    def test_forward(self):
        att = attention.DotAttention(hidden_dim=4, dropout_prob=0.1)
        att_queries = Variable(torch.randn(3, 2, 4))
        enc_states = Variable(torch.randn(5, 2, 4))
        enc_length = torch.LongTensor([4, 2])
        att_states, scores = att.forward(
            queries=att_queries,
            annotations=enc_states, annotations_length=enc_length)
        self.assertTupleEqual(att_states.size(), (3, 2, 4))


class TestMLPAttention(unittest.TestCase):

    def test_score(self):
        att = attention.MLPAttention(hidden_dim=4, annotation_dim=8,
                                     dropout_prob=0.1)
        att_queries = Variable(torch.randn(2, 3, 4))
        enc_states = Variable(torch.randn(2, 5, 8))
        enc_length = torch.LongTensor([4, 2])
        att_weights = att.score(
            queries=att_queries,
            annotations=enc_states, annotations_length=enc_length)
        self.assertTrue((att_weights[0, :, 4] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 2] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 3] < 1e-5).data.all())
        self.assertTrue((att_weights[1, :, 4] < 1e-5).data.all())

    def test_forward(self):
        att = attention.MLPAttention(hidden_dim=4, annotation_dim=8,
                                     dropout_prob=0.1)
        att_queries = Variable(torch.randn(3, 2, 4))
        enc_states = Variable(torch.randn(5, 2, 8))
        enc_length = torch.LongTensor([4, 2])
        att_states, scores = att.forward(
            queries=att_queries,
            annotations=enc_states, annotations_length=enc_length)
        self.assertTupleEqual(att_states.size(), (3, 2, 4))
