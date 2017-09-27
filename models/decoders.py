import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from . import attention, basic


class RecurrentDecoder(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 num_layers, attention_type, input_feeding, dropout_prob,
                 pad_id, bos_id, eos_id):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_feeding = input_feeding
        self.dropout_prob = dropout_prob
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim,
                                           padding_idx=pad_id)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=word_dim, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=word_dim, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        else:
            raise ValueError('Unknown RNN type!')
        if attention_type == 'dot':
            self.attention = attention.DotAttention(
                input_dim=word_dim, hidden_dim=hidden_dim,
                input_feeding=input_feeding, dropout_prob=dropout_prob)
        else:
            raise ValueError('Unknown attention type!')
        self.output_linear = nn.Linear(in_features=hidden_dim,
                                       out_features=num_words)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.word_embedding.weight.data[self.pad_id].fill_(0)
        for i in range(self.num_layers):
            weight_ih = getattr(self.rnn, f'weight_ih_l{i}')
            weight_hh = getattr(self.rnn, f'weight_hh_l{i}')
            bias_ih = getattr(self.rnn, f'bias_ih_l{i}')
            bias_hh = getattr(self.rnn, f'bias_hh_l{i}')
            init.orthogonal(weight_hh.data)
            init.kaiming_normal(weight_ih.data)
            init.constant(bias_ih.data, val=0)
            init.constant(bias_hh.data, val=0)
            if self.rnn_type == 'lstm':  # Set initial forget bias to 1
                bias_ih.data.chunk(4)[1].fill_(1)
        init.kaiming_normal(self.output_linear.weight.data)
        init.constant(self.output_linear.bias.data, val=0)

    def forward(self, encoder_hidden_states, encoder_length,
                prev_state, words):
        words_emb = self.word_embedding(words)
        words_emb = self.dropout(words_emb)
        rnn_outputs, rnn_state = self.rnn(
            input=words_emb, hx=prev_state)
        attentional_states, attention_weights = self.attention(
            encoder_states=encoder_hidden_states,
            encoder_lengths=encoder_length,
            decoder_states=rnn_outputs,
            input=words_emb)
        logits = basic.apply_nd(fn=self.output_linear,
                                input=attentional_states)
        return logits, rnn_state, attention_weights
