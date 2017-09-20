from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentEncoder(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 bidirectional, num_layers, pad_id):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.pad_id = pad_id

        self.embedding = nn.Embedding(num_embeddings=num_words,
                                      embedding_dim=word_dim,
                                      padding_idx=pad_id)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=word_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=word_dim, hidden_size=hidden_dim,
                bidirectional=bidirectional, num_layers=num_layers)
        else:
            raise ValueError('Unknown RNN type!')
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.embedding.weight.data, mean=0, std=0.01)
        self.embedding.weight.data[self.pad_id].fill_(0)
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

    def forward(self, input, lengths):
        input_emb = self.embedding(input)
        input_emb_packed = pack_padded_sequence(
            input=input_emb, lengths=lengths.tolist())
        rnn_output_packed, rnn_last_state = self.rnn(input_emb_packed)
        rnn_output, _ = pad_packed_sequence(rnn_output_packed)
        encoder_states = rnn_output
        # For LSTM, encoder_states does not contain cell states.
        # Thus it is necessary to explicitly return the last state
        encoder_last_state = rnn_last_state
        return encoder_states, encoder_last_state
