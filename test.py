import os

import configargparse as argparse
import dill
import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn import functional
from torchtext import data
from tqdm import tqdm

from models.seq2seq import RecurrentSeq2Seq
from utils.beam import Beam


def test(args):
    test_path = os.path.join(args.data_dir, 'test_dataset.pkl')
    fields_path = os.path.join(args.data_dir, 'fields.pkl')
    test_dataset = torch.load(test_path, pickle_module=dill)
    fields = torch.load(fields_path, pickle_module=dill)
    test_dataset.fields = fields
    test_iter = data.Iterator(
        dataset=test_dataset, batch_size=args.batch_size, device=args.gpu,
        train=False, sort=False)

    src_field = test_dataset.fields['src']
    tgt_field = test_dataset.fields['tgt']
    model = RecurrentSeq2Seq(
        num_src_words=len(src_field.vocab),
        num_tgt_words=len(tgt_field.vocab),
        word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout_prob, rnn_type=args.rnn_type,
        num_layers=args.num_layers, attention_type='dot',
        input_feeding=args.input_feeding,
        src_pad_id=src_field.vocab.stoi[src_field.pad_token],
        tgt_pad_id=tgt_field.vocab.stoi[tgt_field.pad_token],
        tgt_bos_id=tgt_field.vocab.stoi[tgt_field.init_token],
        tgt_eos_id=tgt_field.vocab.stoi[tgt_field.eos_token])
    model.load_state_dict(
        torch.load(args.model_path, map_location=lambda storage, loc: storage))
    if args.gpu > -1:
        model.cuda()

    def apply_state(fn, state):
        if isinstance(state, tuple):  # LSTM
            state = [fn(s) for s in state]
            return tuple(state)
        else:
            return fn(state)

    def generate(batch):
        vocab = tgt_field.vocab
        pad_id = vocab.stoi[tgt_field.pad_token]
        bos_id = vocab.stoi[tgt_field.init_token]
        eos_id = vocab.stoi[tgt_field.eos_token]

        src_words, src_lengths = batch.src
        src_max_length, batch_size = src_words.size()
        src_lengths_sorted, sort_indices = src_lengths.sort(0, descending=True)
        orig_indices = sort_indices.sort()[1]
        src_words_sorted = src_words[:, sort_indices]

        beam = [Beam(size=args.beam_size, n_best=1,
                     pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
                     device=args.gpu, vocab=vocab, global_scorer=None)
                for _ in range(batch_size)]
        context, prev_state = model.encoder(
            input=src_words_sorted, lengths=src_lengths_sorted)
        context = context[:, orig_indices, :]
        prev_state = apply_state(
            fn=lambda s: s[:, orig_indices, :], state=prev_state)

        context = context.repeat(1, args.beam_size, 1)
        src_lengths = src_lengths.repeat(args.beam_size)
        prev_state = apply_state(
            fn=lambda s: s.repeat(1, args.beam_size, 1), state=prev_state)

        for t in range(src_max_length * 2):
            if all((b.done() for b in beam)):
                break
            decoder_input = torch.stack(
                [b.get_current_state() for b in beam], dim=1)
            # decoder_input: (1, beam_size * batch_size)
            decoder_input = decoder_input.view(1, -1)
            decoder_input = Variable(decoder_input, volatile=True)
            logits, prev_state, attn_weights = model.decoder(
                encoder_states=context, encoder_lengths=src_lengths,
                prev_state=prev_state, input=decoder_input)
            log_probs = functional.log_softmax(logits.squeeze(0))
            # log_prob: (beam_size, batch_size, num_words)
            log_probs = log_probs.view(args.beam_size, batch_size, -1)
            # attn_weights: (beam_size, batch_size, source_length)
            attn_weights = attn_weights.view(args.beam_size, batch_size, -1)
            # prev_state: (Tuple of) (beam_size, batch_size, hidden_dim)
            prev_state = apply_state(
                fn=lambda s: s.view(args.beam_size, batch_size, -1),
                state=prev_state)
            for j, b in enumerate(beam):
                b.advance(word_lk=log_probs[:, j].data,
                          attn_out=attn_weights[:, j].data)
                # Update prev_state to point correct parents
                current_origin = b.get_current_origin()
                apply_state(
                    fn=lambda s: s[:, j].data.copy_(
                        s[:, j].data.index_select(0, current_origin)),
                    state=prev_state)
            prev_state = apply_state(
                fn=lambda s: s.view(1, args.beam_size * batch_size, -1),
                state=prev_state)

        hyps = []
        for i, b in enumerate(beam):
            scores, ks = b.sort_finished(minimum=1)
            hyp, att = b.get_hyp(timestep=ks[0][0], k=ks[0][1])
            hyps.append(hyp[:src_lengths[i] * 2])
        return hyps

    def ids_to_words(ids, vocab, eos_id, remove_eos=False):
        words = []
        for id_ in ids:
            words.append(vocab.itos[id_])
            if id_ == eos_id:
                if remove_eos:
                    words = words[:-1]
                break
        return words

    save_path = args.save_path
    if not save_path:
        save_path = (f'{os.path.splitext(args.model_path)[0]}'
                     f'_pred_b{args.beam_size}.txt')
    save_file = open(save_path, 'w')

    model.eval()
    references = []
    hypotheses = []
    for test_batch in tqdm(test_iter):
        hyps_batch = generate(test_batch)
        for i, hyp in enumerate(hyps_batch):
            ref_words = ids_to_words(
                ids=test_batch.tgt[0][1:, i].data,
                vocab=tgt_field.vocab,
                eos_id=tgt_field.vocab.stoi[tgt_field.pad_token],
                remove_eos=True)
            hyp_words = ids_to_words(
                ids=hyp,
                vocab=tgt_field.vocab,
                eos_id=tgt_field.vocab.stoi[tgt_field.eos_token],
                remove_eos=True)
            save_file.write(' '.join(hyp_words))
            save_file.write('\n')
            references.append([ref_words])
            hypotheses.append(hyp_words)
    bleu = corpus_bleu(list_of_references=references,
                       hypotheses=hypotheses,
                       emulate_multibleu=True)
    print(f'BLEU score: {bleu * 100:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--rnn-type', default='lstm')
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--input-feeding', default=False, action='store_true')
    parser.add_argument('--word-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout-prob', type=float, default=0.2)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=-1)
    args, _ = parser.parse_known_args()

    test(args)


if __name__ == '__main__':
    main()
