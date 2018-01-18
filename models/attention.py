import torch
from torch import nn
from torch.nn import init, functional

from . import basic


class DotAttention(nn.Module):

    def __init__(self, hidden_dim, dropout_prob):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.attend = nn.Sequential(nn.Linear(in_features=2 * hidden_dim,
                                              out_features=hidden_dim),
                                    nn.Tanh())
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.attend[0].weight.data)
        init.constant(self.attend[0].bias.data, val=0)

    def score(self, queries, annotations, annotations_length):
        """
        Args:
            queries (Variable): Query vectors to annotations,
                (batch_size, tgt_len, hidden_dim).
            annotations (Variable): Encoded source vectors,
                (batch_size, src_len, hidden_dim).
            annotations_length (Tensor): Lengths of annotations,
                (batch_size,).

        Returns:
            scores (Variable): Alignment vectors,
                (batch_size, tgt_len, src_len).
        """

        mask = basic.sequence_mask(length=annotations_length,
                                   max_length=annotations.size(1))
        scores = torch.bmm(queries, annotations.transpose(1, 2))
        scores.data.masked_fill_(mask=~mask.unsqueeze(1), value=-float('inf'))
        scores = functional.softmax(scores, dim=2)
        return scores

    def forward(self, queries, annotations, annotations_length):
        """
        Args:
            queries (Variable): Query vectors to annotations,
                (tgt_len, batch_size, hidden_dim).
            annotations (Variable): Encoded source vectors,
                (src_len, batch_size, hidden_dim).
            annotations_length (Tensor): Lengths of annotations,
                (batch_size,).

        Returns:
            attentional_states (Variable): Hidden states after
                being attended, (tgt_len, batch_size, hidden_dim).
            scores (Variable): Alignment vectors,
                (tgt_len, batch_size, src_len).
        """

        # Make queries and annotations batch-major.
        queries = queries.t()
        annotations = annotations.t()
        scores = self.score(queries=queries, annotations=annotations,
                            annotations_length=annotations_length)
        contexts = torch.bmm(scores, annotations)
        attention_features = [queries, contexts]
        attention_input = torch.cat(attention_features, dim=2)
        attention_input = self.dropout(attention_input)
        attentional_states = self.attend(attention_input)

        # Convert back to time-major.
        attentional_states = attentional_states.t()
        scores = scores.t()
        return attentional_states, scores


class MLPAttention(nn.Module):

    def __init__(self, hidden_dim, annotation_dim, dropout_prob):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.query_linear = nn.Linear(in_features=hidden_dim,
                                      out_features=hidden_dim)
        self.annotation_linear = nn.Linear(in_features=annotation_dim,
                                           out_features=hidden_dim,
                                           bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(in_features=hidden_dim, out_features=1,
                           bias=False)
        self.attend = nn.Sequential(
            nn.Linear(in_features=hidden_dim + annotation_dim,
                      out_features=hidden_dim),
            nn.Tanh())
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.query_linear.weight.data, mean=0, std=0.001)
        init.constant(self.query_linear.bias.data, val=0)
        init.normal(self.annotation_linear.weight.data, mean=0, std=0.001)
        init.constant(self.v.weight.data, val=0)
        init.kaiming_normal(self.attend[0].weight.data)
        init.constant(self.attend[0].bias.data, val=0)

    def score(self, queries, annotations, annotations_length):
        """
        Args:
            queries (Variable): Query vectors to annotations,
                (batch_size, tgt_len, hidden_dim).
            annotations (Variable): Encoded source vectors,
                (batch_size, src_len, annotation_dim).
            annotations_length (Tensor): Lengths of annotations,
                (batch_size,).

        Returns:
            scores (Variable): Alignment vectors,
                (batch_size, tgt_len, src_len).
        """

        mask = basic.sequence_mask(length=annotations_length,
                                   max_length=annotations.size(1))
        # queries_proj: (batch_size, tgt_len, hidden_dim)
        queries_proj = self.query_linear(queries)
        # annotations_proj: (batch_size, src_len, hidden_dim)
        annotations_proj = self.annotation_linear(annotations)
        pre_tanh = queries_proj.unsqueeze(2) + annotations_proj.unsqueeze(1)
        scores = self.v(self.tanh(pre_tanh)).squeeze(3)
        scores.data.masked_fill_(mask=~mask.unsqueeze(1), value=-float('inf'))
        scores = functional.softmax(scores, dim=2)
        return scores

    def forward(self, queries, annotations, annotations_length):
        """
        Args:
            queries (Variable): Query vectors to annotations,
                (tgt_len, batch_size, hidden_dim).
            annotations (Variable): Encoded source vectors,
                (src_len, batch_size, hidden_dim).
            annotations_length (Tensor): Lengths of annotations,
                (batch_size,).

        Returns:
            attentional_states (Variable): Hidden states after
                being attended, (tgt_len, batch_size, hidden_dim).
            scores (Variable): Alignment vectors,
                (tgt_len, batch_size, src_len).
        """

        # Make queries and annotations batch-major.
        queries = queries.t()
        annotations = annotations.t()
        scores = self.score(queries=queries, annotations=annotations,
                            annotations_length=annotations_length)
        contexts = torch.bmm(scores, annotations)
        attention_features = [queries, contexts]
        attention_input = torch.cat(attention_features, dim=2)
        attention_input = self.dropout(attention_input)
        attentional_states = self.attend(attention_input)

        # Convert back to time-major.
        attentional_states = attentional_states.t()
        scores = scores.t()
        return attentional_states, scores
