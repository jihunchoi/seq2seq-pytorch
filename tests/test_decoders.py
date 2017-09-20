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
            pad_id=0, bos_id=1, eos_id=2)
        logits = dec.forward(encoder_states=enc_states,
                             encoder_lengths=enc_lengths,
                             encoder_last_state=enc_last_state,
                             input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))

    def test_gru_decoder_train_complex(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2)
        logits = dec.forward(encoder_states=enc_states,
                             encoder_lengths=enc_lengths,
                             encoder_last_state=enc_last_state,
                             input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))

    def test_gru_decoder_generate(self):
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        dec = decoders.RecurrentDecoder(
            rnn_type='gru', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=21)  # avoid generating EOS
        dec.eval()
        generated = dec.forward(encoder_states=enc_states,
                                encoder_lengths=enc_lengths,
                                encoder_last_state=enc_last_state,
                                max_length=8, beam_size=1)
        self.assertTupleEqual(tuple(generated.size()), (8, 3))

    def test_lstm_decoder_train_simple(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=2)
        logits = dec.forward(encoder_states=enc_states,
                             encoder_lengths=enc_lengths,
                             encoder_last_state=enc_last_state,
                             input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))

    def test_lstm_decoder_train_complex(self):
        input_ = Variable(torch.arange(0, 12).view(4, 3).long())
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(3, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=3, attention_type='dot', input_feeding=True,
            pad_id=0, bos_id=1, eos_id=2)
        logits = dec.forward(encoder_states=enc_states,
                             encoder_lengths=enc_lengths,
                             encoder_last_state=enc_last_state,
                             input=input_)
        self.assertTupleEqual(tuple(logits.size()), (4, 3, 20))

    def test_lstm_decoder_generate(self):
        enc_states = Variable(torch.randn(5, 3, 6))
        enc_lengths = torch.LongTensor([5, 4, 2])
        enc_last_state = Variable(torch.randn(1, 3, 6))
        enc_last_state = (enc_last_state, enc_last_state)
        dec = decoders.RecurrentDecoder(
            rnn_type='lstm', num_words=20, word_dim=2, hidden_dim=6,
            num_layers=1, attention_type='dot', input_feeding=False,
            pad_id=0, bos_id=1, eos_id=21)  # avoid generating EOS
        dec.eval()
        generated = dec.forward(encoder_states=enc_states,
                                encoder_lengths=enc_lengths,
                                encoder_last_state=enc_last_state,
                                max_length=8, beam_size=1)
        self.assertTupleEqual(tuple(generated.size()), (8, 3))