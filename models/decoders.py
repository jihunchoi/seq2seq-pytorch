import copy

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
            self.attention = attention.DotAttention(
                hidden_dim=hidden_dim, dropout_prob=dropout_prob)
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
        self.attention.reset_parameters()
        init.kaiming_normal(self.output_linear.weight.data)
        init.constant(self.output_linear.bias.data, val=0)

    def forward(self, context, src_length, words, prev_state=None):
        """
        Args:
            context (Variable): A float variable of size
                (max_src_length, batch_size, context_dim).
            src_length (Tensor): A long 1D tensor of source lengths.
            words (Variable): A long variable of size
                (length, batch_size) that contains indices of words.
            prev_state (DecoderState): The previous state of the decoder.

        Returns:
            logits (Variable): A float variable containing unnormalized
                log probabilities. It has the same size as words.
            state (DecoderState): The current state of the decoder.
            attention_weights (Variable): A float variable of size
                (batch_size, length, max_src_length), which contains
                the attention weight for each timestep of the context.
        """

        words_emb = self.word_embedding(words)
        words_emb = self.dropout(words_emb)
        if not self.input_feeding:
            rnn_outputs, rnn_state = self.rnn(
                input=words_emb, hx=prev_state['rnn'])
            attentional_states, attention_weights = self.attention(
                context=context, src_length=src_length,
                decoder_hidden_states=rnn_outputs)
            state = DecoderState(rnn_state=rnn_state)
        else:
            words_length, batch_size = words.size()
            attentional_states = []
            attention_weights = []
            state = prev_state
            if ('attention' not in prev_state or
                        prev_state['attention'] is None):
                zero_attentional_state = (
                    words_emb.data.new(batch_size, self.hidden_dim).zero_())
                zero_attentional_state = Variable(zero_attentional_state)
                prev_state['attention'] = zero_attentional_state
            for t in range(words_length):
                words_emb_t = words_emb[t]
                decoder_input_t = torch.cat(
                    [words_emb_t, prev_state['attention']], dim=1)
                decoder_input_t = decoder_input_t.unsqueeze(0)
                rnn_output_t, rnn_state_t = self.rnn(
                    input=decoder_input_t, hx=prev_state['rnn'])
                attentional_state_t, attention_weights_t = self.attention(
                    context=context, src_length=src_length,
                    decoder_hidden_states=rnn_output_t)
                attentional_state_t = attentional_state_t.squeeze(0)
                attentional_states.append(attentional_state_t)
                attention_weights.append(attention_weights_t)
                state = DecoderState(
                    rnn_state=rnn_state_t,
                    attentional_state=attentional_state_t, input_feeding=True)
                prev_state = state
            attentional_states = torch.stack(attentional_states, dim=0)
            attention_weights = torch.cat(attention_weights, dim=1)
        logits = basic.apply_nd(fn=self.output_linear,
                                input=attentional_states)
        return logits, state, attention_weights


class DecoderState(dict):

    def __init__(self, rnn_state, attentional_state=None,
                 input_feeding=False):
        super().__init__()
        self.input_feeding = input_feeding
        self['rnn'] = rnn_state
        assert input_feeding or attentional_state is None
        if input_feeding:
            self['attention'] = attentional_state

    @staticmethod
    def apply_to_rnn_state(fn, rnn_state):
        if isinstance(rnn_state, tuple):  # LSTM
            return tuple(fn(s) for s in rnn_state)
        else:
            return fn(rnn_state)

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
        def update(v):
            # The shape is (..., beam_size * batch_size, state_dim).
            orig_size = v.size()
            new_size = (
                orig_size[:-2]
                + (beam_size, orig_size[-2] // beam_size, orig_size[-1]))
            # beam_of_batch: (..., beam_size, state_dim)
            beam_of_batch = v.view(*new_size).select(-2, batch_index)
            beam_of_batch.data.copy_(
                beam_of_batch.data.index_select(-2, beam_indices))

        self.apply_to_rnn_state(fn=update, rnn_state=self['rnn'])
        if self.input_feeding and self['attention'] is not None:
            update(self['attention'])
