from torch import nn

from . import encoders, decoders


class RecurrentSeq2Seq(nn.Module):

    def __init__(self, num_src_words, num_tgt_words, word_dim, hidden_dim,
                 dropout_prob, rnn_type, num_layers, attention_type,
                 input_feeding,
                 src_pad_id, tgt_pad_id, tgt_bos_id, tgt_eos_id):
        super().__init__()
        self.num_src_words = num_src_words
        self.num_tgt_words = num_tgt_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.attention_type = dropout_prob
        self.input_feeding = input_feeding
        self.src_pad_id = src_pad_id
        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id

        self.encoder = encoders.RecurrentEncoder(
            rnn_type=rnn_type, num_words=num_src_words, word_dim=word_dim,
            hidden_dim=hidden_dim, dropout_prob=dropout_prob,
            bidirectional=False, num_layers=num_layers, pad_id=src_pad_id)
        self.decoder = decoders.RecurrentDecoder(
            rnn_type=rnn_type, num_words=num_tgt_words, word_dim=word_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            attention_type=attention_type, input_feeding=input_feeding,
            dropout_prob=dropout_prob, pad_id=tgt_pad_id, bos_id=tgt_bos_id,
            eos_id=tgt_eos_id)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, src_words, src_length, tgt_words):
        encoder_hidden_states, encoder_state = self.encoder(
            words=src_words, length=src_length)
        logits, _, _ = self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            encoder_length=src_length,
            prev_state=encoder_state, words=tgt_words)
        return logits
