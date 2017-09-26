import unittest

import torch
from torch.autograd import Variable

from models import decoders


class TestDecoders(unittest.TestCase):

    def test_gru_decoder_train_simple(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            encoder_states=enc_states, encoder_lengths=enc_lengths,
            prev_state=enc_last_state, input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTupleEqual(tuple(decoder_state.size()), (1, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_gru_decoder_train_complex(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            encoder_states=enc_states, encoder_lengths=enc_lengths,
            prev_state=enc_last_state, input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTupleEqual(tuple(decoder_state.size()), (3, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_lstm_decoder_train_simple(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            encoder_states=enc_states, encoder_lengths=enc_lengths,
            prev_state=enc_last_state, input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTupleEqual(tuple(decoder_state[0].size()), (1, 3, 6))
        self.assertTupleEqual(tuple(decoder_state[1].size()), (1, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_lstm_decoder_train_complex(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            encoder_states=enc_states, encoder_lengths=enc_lengths,
            prev_state=enc_last_state, input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTupleEqual(tuple(decoder_state[0].size()), (3, 3, 6))
        self.assertTupleEqual(tuple(decoder_state[1].size()), (3, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))
