from torch import nn

from . import encoders, decoders


class RecurrentSeq2Seq(nn.Module):

    def __init__(self, num_src_words, num_tgt_words, word_dim, hidden_dim,
                 dropout_prob, rnn_type, bidirectional, num_layers,
                 attention_type, input_feeding):
        super().__init__()
        self.num_src_words = num_src_words
        self.num_tgt_words = num_tgt_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.attention_type = dropout_prob
        self.input_feeding = input_feeding

        self.encoder = encoders.RecurrentEncoder(
            rnn_type=rnn_type, num_words=num_src_words, word_dim=word_dim,
            hidden_dim=hidden_dim, dropout_prob=dropout_prob,
            bidirectional=bidirectional, num_layers=num_layers)
        annotation_dim = hidden_dim
        if bidirectional:
            annotation_dim = hidden_dim * 2
        self.decoder = decoders.RecurrentDecoder(
            rnn_type=rnn_type, num_words=num_tgt_words, word_dim=word_dim,
            hidden_dim=hidden_dim, annotation_dim=annotation_dim,
            num_layers=num_layers, attention_type=attention_type,
            input_feeding=input_feeding, dropout_prob=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, src_words, src_length, tgt_words):
        encoder_hidden_states, encoder_state = self.encoder(
            words=src_words, length=src_length)
        decoder_state = decoders.DecoderState(
            rnn_state=encoder_state, input_feeding=self.input_feeding)
        logits, _, _ = self.decoder(
            annotations=encoder_hidden_states, annotations_length=src_length,
            words=tgt_words, state=decoder_state)
        return logits
