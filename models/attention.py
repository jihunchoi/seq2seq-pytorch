import torch
from torch import nn
from torch.nn import init, functional

from . import basic


class DotAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, input_feeding, dropout_prob):
        super().__init__()
        self.input_feeding = input_feeding

        num_attention_features = 2 * hidden_dim
        if input_feeding:
            num_attention_features += input_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_linear = nn.Linear(in_features=num_attention_features,
                                          out_features=hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.attention_linear.weight.data)
        init.constant(self.attention_linear.bias.data, val=0)

    def compute_attention_weights(self, attention_queries,
                                  encoder_states, encoder_lengths):
        encoder_states_bm = encoder_states.transpose(0, 1)
        encoder_mask = basic.sequence_mask(lengths=encoder_lengths,
                                           max_length=encoder_states.size(0))
        encoder_mask_bm = encoder_mask.transpose(0, 1)
        attention_queries_bm = attention_queries.transpose(0, 1)
        attention_weights = torch.bmm(
            attention_queries_bm, encoder_states_bm.transpose(1, 2))
        attention_weights.data.masked_fill_(
            ~encoder_mask_bm.unsqueeze(1), -float('inf'))
        attention_weights = basic.apply_nd(
            fn=functional.softmax, input=attention_weights)
        return attention_weights

    def compute_contexts(self, attention_weights, encoder_states):
        encoder_states_bm = encoder_states.transpose(0, 1)
        contexts = torch.bmm(attention_weights, encoder_states_bm)
        contexts = contexts.transpose(0, 1)
        return contexts

    def forward(self, rnn,
                encoder_states, encoder_lengths, initial_state,
                input):
        decoder_states, decoder_last_state = rnn(
            input=input, hx=initial_state)
        attention_weights = self.compute_attention_weights(
            attention_queries=decoder_states,
            encoder_states=encoder_states, encoder_lengths=encoder_lengths)
        contexts = self.compute_contexts(
            attention_weights=attention_weights, encoder_states=encoder_states)
        attention_features = [contexts, decoder_states]
        if self.input_feeding:
            attention_features.append(input)
        attention_input = torch.cat(attention_features, dim=2)
        attention_input = self.dropout(attention_input)
        attentional_states = basic.apply_nd(
            fn=self.attention_linear, input=attention_input)
        return functional.tanh(attentional_states), decoder_last_state
