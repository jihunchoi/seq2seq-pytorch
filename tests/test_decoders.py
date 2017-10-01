import unittest

import torch
from torch.autograd import Variable

from models import decoders
from models.decoders import DecoderState


class TestDecoders(unittest.TestCase):

    def test_gru_decoder_train_simple(self):
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_length = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        prev_state = DecoderState(
            rnn_state=enc_last_state, input_feeding=False)
        logits, decoder_state, attention_weights = dec.forward(
            context=enc_states, src_length=enc_length,
            prev_state=prev_state, words=words)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertFalse('attention' in decoder_state)
        self.assertTupleEqual(tuple(decoder_state['rnn'].size()), (1, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_gru_decoder_train_complex(self):
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_length = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        prev_state = DecoderState(
            rnn_state=enc_last_state, input_feeding=True)
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            context=enc_states, src_length=enc_length,
            prev_state=prev_state, words=words)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTrue('attention' in decoder_state)
        self.assertTupleEqual(tuple(decoder_state['attention'].size()), (3, 6))
        self.assertTupleEqual(tuple(decoder_state['rnn'].size()), (3, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_lstm_decoder_train_simple(self):
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_length = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        prev_state = DecoderState(
            rnn_state=enc_last_state, input_feeding=False)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            context=enc_states, src_length=enc_length,
            prev_state=prev_state, words=words)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertFalse('attention' in decoder_state)
        self.assertTupleEqual(tuple(decoder_state['rnn'][0].size()), (1, 3, 6))
        self.assertTupleEqual(tuple(decoder_state['rnn'][1].size()), (1, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))

    def test_lstm_decoder_train_complex(self):
        words = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_length = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        prev_state = DecoderState(
            rnn_state=enc_last_state, input_feeding=True)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2, dropout_prob=0.1)
        logits, decoder_state, attention_weights = dec.forward(
            context=enc_states, src_length=enc_length,
            prev_state=prev_state, words=words)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))
        self.assertTrue('attention' in decoder_state)
        self.assertTupleEqual(tuple(decoder_state['attention'].size()), (3, 6))
        self.assertTupleEqual(tuple(decoder_state['rnn'][0].size()), (3, 3, 6))
        self.assertTupleEqual(tuple(decoder_state['rnn'][1].size()), (3, 3, 6))
        self.assertTupleEqual(tuple(attention_weights.size()), (3, 4, 5))
