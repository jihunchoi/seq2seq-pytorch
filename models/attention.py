import torch
from torch import nn
from torch.nn import init, functional

from . import basic


class DotAttention(nn.Module):

    def __init__(self, hidden_dim, dropout_prob):
        super().__init__()

        num_attention_features = 2 * hidden_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.attention_linear = nn.Linear(in_features=num_attention_features,
                                          out_features=hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.attention_linear.weight.data)
        init.constant(self.attention_linear.bias.data, val=0)

    def compute_attention_weights(self, attention_queries,
                                  context, src_length):
        encoder_states_bm = context.transpose(0, 1)
        encoder_mask = basic.sequence_mask(length=src_length,
                                           max_length=context.size(0))
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

    def forward(self, context, src_length, decoder_hidden_states):
        attention_weights = self.compute_attention_weights(
            attention_queries=decoder_hidden_states,
            context=context, src_length=src_length)
        contexts = self.compute_contexts(
            attention_weights=attention_weights, encoder_states=context)
        attention_features = [contexts, decoder_hidden_states]
        attention_input = torch.cat(attention_features, dim=2)
        attention_input = self.dropout(attention_input)
        attentional_states = basic.apply_nd(
            fn=self.attention_linear, input=attention_input)
        return functional.tanh(attentional_states), attention_weights
