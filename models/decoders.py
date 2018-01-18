import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from . import attention


class RecurrentDecoder(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 annotation_dim, num_layers, attention_type, input_feeding,
                 dropout_prob):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.annotation_dim = annotation_dim
        self.num_layers = num_layers
        self.input_feeding = input_feeding
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        rnn_input_size = word_dim
        if input_feeding:
            rnn_input_size += hidden_dim
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        else:
            raise ValueError('Unknown RNN type!')
        if attention_type == 'dot':
            assert hidden_dim == annotation_dim, (
                'hidden_dim and annotation_dim must be same when using'
                ' dot attention.')
            self.attention = attention.DotAttention(
                hidden_dim=hidden_dim, dropout_prob=dropout_prob)
        elif attention_type == 'mlp':
            self.attention = attention.MLPAttention(
                hidden_dim=hidden_dim, annotation_dim=annotation_dim,
                dropout_prob=dropout_prob)
        else:
            raise ValueError('Unknown attention type!')
        self.output_linear = nn.Linear(in_features=hidden_dim,
                                       out_features=num_words)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
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
        self.attention.reset_parameters()
        init.normal(self.output_linear.weight.data, mean=0, std=0.01)
        init.constant(self.output_linear.bias.data, val=0)

    def forward(self, annotations, annotations_length, words, state):
        """
        Args:
            annotations (Variable): A float variable of size
                (max_src_length, batch_size, context_dim).
            annotations_length (Tensor): A long 1D tensor of
                source lengths.
            words (Variable): A long variable of size
                (length, batch_size) that contains indices of words.
            state (DecoderState): The current state of the decoder.

        Returns:
            logits (Variable): A float variable containing unnormalized
                log probabilities. It has the same size as words.
            state (DecoderState): The updated state of the decoder.
            attention_weights (Variable): A float variable of size
                (length, batch_size, max_src_length), which contains
                the attention weight for each time step of the context.
        """

        state = copy.copy(state)
        words_emb = self.word_embedding(words)
        words_emb = self.dropout(words_emb)
        if not self.input_feeding:
            rnn_outputs, rnn_state = self.rnn(input=words_emb, hx=state.rnn)
            attentional_states, attention_weights = self.attention.forward(
                queries=rnn_outputs, annotations=annotations,
                annotations_length=annotations_length)
            state.update(rnn_state=rnn_state)
        else:
            words_length, batch_size = words.size()
            attentional_states = []
            attention_weights = []
            if state.attention is None:
                zero_attentional_state = (
                    words_emb.data.new(batch_size, self.hidden_dim).zero_())
                zero_attentional_state = Variable(zero_attentional_state)
                state.update_attentional_state(zero_attentional_state)
            for t in range(words_length):
                words_emb_t = words_emb[t]
                decoder_input_t = torch.cat(
                    [words_emb_t, state.attention], dim=1)
                decoder_input_t = decoder_input_t.unsqueeze(0)
                rnn_output_t, rnn_state_t = self.rnn(
                    input=decoder_input_t, hx=state.rnn)
                attentional_state_t, attention_weights_t = self.attention(
                    queries=rnn_output_t, annotations=annotations,
                    annotations_length=annotations_length)
                attentional_state_t = attentional_state_t.squeeze(0)
                attentional_states.append(attentional_state_t)
                attention_weights.append(attention_weights_t)
                state.update_state(rnn_state=rnn_state_t,
                                   attentional_state=attentional_state_t)
            attentional_states = torch.stack(attentional_states, dim=0)
            attention_weights = torch.cat(attention_weights, dim=0)
        logits = self.output_linear(attentional_states)
        return logits, state, attention_weights


class DecoderState(dict):

    def __init__(self, rnn_state, attentional_state=None,
                 input_feeding=False):
        super().__init__()
        self['input_feeding'] = input_feeding
        self['rnn'] = rnn_state
        assert input_feeding or attentional_state is None
        self['attention'] = None
        if input_feeding:
            self['attention'] = attentional_state

    @staticmethod
    def apply_to_rnn_state(fn, rnn_state):
        if isinstance(rnn_state, tuple):  # LSTM
            return tuple(fn(s) for s in rnn_state)
        else:
            return fn(rnn_state)

    @property
    def input_feeding(self):
        return self['input_feeding']

    @property
    def rnn(self):
        return self['rnn']

    @property
    def attention(self):
        return self['attention']

    def update_state(self, rnn_state=None, attentional_state=None):
        self.update_rnn_state(rnn_state)
        self.update_attentional_state(attentional_state)

    def update_rnn_state(self, rnn_state):
        self['rnn'] = rnn_state

    def update_attentional_state(self, attentional_state):
        if self.input_feeding and attentional_state is not None:
            self['attention'] = attentional_state

    def repeat(self, beam_size):
        new_state = copy.copy(self)
        new_state['rnn'] = self.apply_to_rnn_state(
            fn=lambda s: s.repeat(1, beam_size, 1),
            rnn_state=new_state['rnn'])
        if self.input_feeding and new_state['attention'] is not None:
            new_state['attention'] = (
                new_state['attention'].repeat(beam_size, 1))
        return new_state

    def beam_update(self, batch_index, beam_indices, beam_size):
        def beam_update_fn(v):
            # The shape is (..., beam_size * batch_size, state_dim).
            orig_size = v.size()
            new_size = (
                orig_size[:-2]
                + (beam_size, orig_size[-2] // beam_size, orig_size[-1]))
            # beam_of_batch: (..., beam_size, state_dim)
            beam_of_batch = v.view(*new_size).select(-2, batch_index)
            beam_of_batch.data.copy_(
                beam_of_batch.data.index_select(-2, beam_indices))

        self.apply_to_rnn_state(fn=beam_update_fn, rnn_state=self['rnn'])
        if self.input_feeding and self['attention'] is not None:
            beam_update_fn(self['attention'])
