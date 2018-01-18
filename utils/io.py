"""Some lines are copied from OpenNMT-py."""

import codecs
from collections import defaultdict
from itertools import chain

import torchtext.data
import torchtext.vocab


PAD_WORD = '<pad>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'


def __getstate__(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def __setstate__(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = __getstate__
torchtext.vocab.Vocab.__setstate__ = __setstate__


def join_dicts(*args):
    """
    args: dictionaries with disjoint keys
    returns: a single dictionary that has the union of these keys
    """
    return dict(chain(*[d.items() for d in args]))


class NMTDataset(torchtext.data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, src_path, tgt_path, fields):
        """
        Create a TranslationDataset given paths and fields.

        src_path: location of source-side data
        tgt_path: location of target-side data or None. If it exists, it
                  source and target data must be the same length.
        fields: tuple of (src_field, trg_field).
        """

        src_data = self._read_corpus_file(src_path, 0)
        src_examples = self._construct_examples(src_data, "src")

        tgt_data = self._read_corpus_file(tgt_path, 0)
        assert len(src_data) == len(tgt_data), \
            "Len src and tgt do not match"
        tgt_examples = self._construct_examples(tgt_data, "tgt")

        examples = [join_dicts(src, tgt)
                    for src, tgt in zip(src_examples, tgt_examples)]

        keys = examples[0].keys()
        examples = [torchtext.data.Example.fromlist([ex[k] for k in keys],
                                                    fields)
                    for ex in examples]

        super(NMTDataset, self).__init__(examples, fields)

    @staticmethod
    def _read_corpus_file(path, truncate):
        """
        path: location of a src or tgt file
        truncate: maximum sequence length (0 for unlimited)

        returns: (word, features, nfeat) triples for each line
        """
        with codecs.open(path, "r", "utf-8") as corpus_file:
            lines = [line.split() for line in corpus_file]
            if truncate:
                lines = (line[:truncate] for line in lines)
            return lines

    @staticmethod
    def _construct_examples(lines, side):
        assert side in ["src", "tgt"]
        examples = []
        for line in lines:
            words = line
            example_dict = {side: words}
            examples.append(example_dict)
        return examples

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __reduce_ex__(self, proto):
        """This is a hack. Something is broken with torch pickle."""
        return super(NMTDataset, self).__reduce_ex__()


class SrcField(torchtext.data.Field):

    def __init__(self, lower=True):
        super().__init__(lower=lower, pad_token=PAD_WORD,
                         eos_token=EOS_WORD, include_lengths=True)


class TgtField(torchtext.data.Field):

    def __init__(self, lower=True):
        super().__init__(lower=lower, pad_token=PAD_WORD,
                         init_token=BOS_WORD, eos_token=EOS_WORD,
                         include_lengths=True)
