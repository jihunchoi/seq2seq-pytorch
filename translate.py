import argparse
import os

import dill
import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn import functional
from torchtext import data
from tqdm import tqdm

from models.seq2seq import RecurrentSeq2Seq
from models.decoders import DecoderState
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

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    model = RecurrentSeq2Seq(
        num_src_words=len(src_field.vocab),
        num_tgt_words=len(tgt_field.vocab),
        **config['model'])
    model.load_state_dict(
        torch.load(args.model_path, map_location=lambda storage, loc: storage))
    if args.gpu > -1:
        model.cuda()

    def generate(batch):
        vocab = tgt_field.vocab
        pad_id = vocab.stoi[tgt_field.pad_token]
        bos_id = vocab.stoi[tgt_field.init_token]
        eos_id = vocab.stoi[tgt_field.eos_token]

        src_words, src_length = batch.src
        src_max_length, batch_size = src_words.size()
        src_length_sorted, sort_indices = src_length.sort(0, descending=True)
        orig_indices = sort_indices.sort()[1]
        src_words_sorted = src_words[:, sort_indices]

        beam = [Beam(size=args.beam_size, n_best=1,
                     pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
                     device=args.gpu, vocab=vocab, global_scorer=None)
                for _ in range(batch_size)]
        context, encoder_state = model.encoder(
            words=src_words_sorted, length=src_length_sorted)
        context = context[:, orig_indices, :]
        encoder_state = DecoderState.apply_to_rnn_state(
            lambda s: s[:, orig_indices], rnn_state=encoder_state)

        prev_state = DecoderState(
            rnn_state=encoder_state, input_feeding=model.input_feeding)

        context = context.repeat(1, args.beam_size, 1)
        src_length = src_length.repeat(args.beam_size)
        prev_state = prev_state.repeat(args.beam_size)

        for t in range(src_max_length * 2):
            if all((b.done() for b in beam)):
                break
            decoder_input = torch.stack(
                [b.get_current_state() for b in beam], dim=1)
            # decoder_input: (1, beam_size * batch_size)
            decoder_input = decoder_input.view(1, -1)
            decoder_input = Variable(decoder_input, volatile=True)
            logits, prev_state, attn_weights = model.decoder(
                annotations=context, annotations_length=src_length,
                state=prev_state, words=decoder_input)
            log_probs = functional.log_softmax(logits.squeeze(0), dim=1)
            # log_probs: (beam_size, batch_size, num_words)
            log_probs = log_probs.view(args.beam_size, batch_size, -1)
            # attn_weights: (beam_size, batch_size, source_length)
            attn_weights = attn_weights.view(args.beam_size, batch_size, -1)
            for j, b in enumerate(beam):
                b.advance(word_lk=log_probs[:, j].data,
                          attn_out=attn_weights[:, j].data)
                # Update prev_state to point correct parents
                current_origin = b.get_current_origin()
                prev_state.beam_update(
                    batch_index=j, beam_indices=current_origin,
                    beam_size=args.beam_size)

        hyps = []
        for i, b in enumerate(beam):
            scores, ks = b.sort_finished(minimum=1)
            hyp, att = b.get_hyp(timestep=ks[0][0], k=ks[0][1])
            hyps.append(hyp[:src_length[i] * 2])
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
            hyp_words = ids_to_words(
                ids=hyp,
                vocab=tgt_field.vocab,
                eos_id=tgt_field.vocab.stoi[tgt_field.eos_token],
                remove_eos=True)
            save_file.write(' '.join(hyp_words))
            save_file.write('\n')
            hypotheses.append(hyp_words)
            ref_words = test_dataset[len(hypotheses) - 1].tgt
            references.append([ref_words])
    bleu = corpus_bleu(list_of_references=references,
                       hypotheses=hypotheses,
                       emulate_multibleu=True)
    print(f'BLEU score: {bleu * 100:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    args, _ = parser.parse_known_args()

    test(args)


if __name__ == '__main__':
    main()
