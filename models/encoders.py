from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentEncoder(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 dropout_prob, bidirectional, num_layers):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=word_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers,
                dropout=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=word_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers,
                dropout=dropout_prob)
        else:
            raise ValueError('Unknown RNN type!')
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        for i in range(self.num_layers):
            suffixes = ['']
            if self.bidirectional:
                suffixes.append('_reverse')
            for suffix in suffixes:
                weight_ih = getattr(self.rnn, f'weight_ih_l{i}{suffix}')
                weight_hh = getattr(self.rnn, f'weight_hh_l{i}{suffix}')
                bias_ih = getattr(self.rnn, f'bias_ih_l{i}{suffix}')
                bias_hh = getattr(self.rnn, f'bias_hh_l{i}{suffix}')
                init.orthogonal(weight_hh.data)
                init.kaiming_normal(weight_ih.data)
                init.constant(bias_ih.data, val=0)
                init.constant(bias_hh.data, val=0)
                if self.rnn_type == 'lstm':  # Set initial forget bias to 1
                    bias_ih.data.chunk(4)[1].fill_(1)

    def forward(self, words, length):
        words_emb = self.word_embedding(words)
        words_emb = self.dropout(words_emb)
        words_emb_packed = pack_padded_sequence(
            input=words_emb, lengths=length.tolist())
        rnn_outputs_packed, rnn_state = self.rnn(words_emb_packed)
        encoder_hidden_states, _ = pad_packed_sequence(rnn_outputs_packed)
        # For LSTM, encoder_states does not contain cell states.
        # Thus it is necessary to explicitly return the last state
        encoder_state = rnn_state
        return encoder_hidden_states, encoder_state
