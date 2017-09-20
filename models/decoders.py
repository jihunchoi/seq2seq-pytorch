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
        self.embedding = nn.Embedding(num_embeddings=num_words,
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
        init.normal(self.embedding.weight.data, mean=0, std=0.01)
        self.embedding.weight.data[self.pad_id].fill_(0)
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

    def forward(self, encoder_states, encoder_lengths, encoder_last_state,
                input=None, max_length=None, beam_size=None):
        if self.training:
            input_emb = self.embedding(input)
            input_emb = self.dropout(input_emb)
            attentional_states, _ = self.attention(
                rnn=self.rnn,
                encoder_states=encoder_states,
                encoder_lengths=encoder_lengths,
                initial_state=encoder_last_state,
                input=input_emb)
            logits = basic.apply_nd(fn=self.output_linear,
                                    input=attentional_states)
            return logits
        else:
            if beam_size == 1:
                return self.greedy_search(
                    encoder_states=encoder_states,
                    encoder_lengths=encoder_lengths,
                    encoder_last_state=encoder_last_state,
                    max_length=max_length)
            else:
                raise NotImplementedError

    def make_bos(self, batch_size):
        bos = Variable(torch.LongTensor([[self.bos_id] * batch_size]))
        return bos

    def update_done(self, done, output):
        done.masked_fill_(mask=torch.eq(output.data, self.eos_id).squeeze(0),
                          value=1)

    def greedy_search(self, encoder_states, encoder_lengths,
                      encoder_last_state, max_length):
        batch_size = encoder_states.size(1)
        done = torch.zeros(batch_size).byte()
        prev_input = self.make_bos(batch_size)
        prev_state = encoder_last_state
        outputs = []
        if encoder_states.is_cuda:
            device = encoder_states.get_device()
            done = done.cuda(device)
            prev_input = prev_input.cuda(device)
        for time in range(max_length):
            prev_input_emb = self.embedding(prev_input)
            attentional_state, cur_state = self.attention(
                rnn=self.rnn,
                encoder_states=encoder_states,
                encoder_lengths=encoder_lengths,
                initial_state=prev_state,
                input=prev_input_emb)
            attentional_state = self.dropout(attentional_state)
            cur_logits = basic.apply_nd(fn=self.output_linear,
                                        input=attentional_state)
            cur_output = cur_logits.max(2)[1]
            outputs.append(cur_output)
            self.update_done(done=done, output=cur_output)
            if done.all():
                break
            prev_input = cur_output
            prev_state = cur_state
        outputs = torch.cat(outputs, dim=0)
        return outputs
