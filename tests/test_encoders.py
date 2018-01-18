import unittest

import torch
from torch.autograd import Variable

from models import encoders


class TestEncoders(unittest.TestCase):

    def test_gru_recurrent_encoder_simple(self):
        enc = encoders.RecurrentEncoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=3,
            bidirectional=False, num_layers=1, dropout_prob=0.1)
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        length = torch.LongTensor([4, 3, 1])
        enc_states, enc_last_state = enc.forward(words=words, length=length)
        self.assertTupleEqual(tuple(enc_states.size()), (4, 3, 3))
        self.assertTupleEqual(tuple(enc_last_state.size()), (1, 3, 3))

    def test_gru_recurrent_encoder_complex(self):
        enc = encoders.RecurrentEncoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=3,
            bidirectional=True, num_layers=3, dropout_prob=0.1)
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        length = torch.LongTensor([4, 3, 1])
        enc_states, enc_last_state = enc.forward(words=words, length=length)
        self.assertTupleEqual(tuple(enc_states.size()), (4, 3, 6))
        self.assertTupleEqual(tuple(enc_last_state.size()), (6, 3, 3))

    def test_lstm_recurrent_encoder_simple(self):
        enc = encoders.RecurrentEncoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=3,
            bidirectional=False, num_layers=1, dropout_prob=0.1)
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        length = torch.LongTensor([4, 3, 1])
        enc_states, enc_last_state = enc.forward(words=words, length=length)
        enc_last_h, enc_last_c = enc_last_state
        self.assertTupleEqual(tuple(enc_states.size()), (4, 3, 3))
        self.assertTupleEqual(tuple(enc_last_h.size()), (1, 3, 3))
        self.assertTupleEqual(tuple(enc_last_c.size()), (1, 3, 3))

    def test_lstm_recurrent_encoder_complex(self):
        enc = encoders.RecurrentEncoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=3,
            bidirectional=True, num_layers=3, dropout_prob=0.1)
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        length = torch.LongTensor([4, 3, 1])
        enc_states, enc_last_state = enc.forward(words=words, length=length)
        enc_last_h, enc_last_c = enc_last_state
        self.assertTupleEqual(tuple(enc_states.size()), (4, 3, 6))
        self.assertTupleEqual(tuple(enc_last_h.size()), (6, 3, 3))
        self.assertTupleEqual(tuple(enc_last_c.size()), (6, 3, 3))
