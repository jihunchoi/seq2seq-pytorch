import argparse
import os
import random

import dill
import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torchtext import data

from models import basic
from models.seq2seq import RecurrentSeq2Seq


def train(args):
    train_path = os.path.join(args.data_dir, 'train_dataset.pkl')
    valid_path = os.path.join(args.data_dir, 'valid_dataset.pkl')
    fields_path = os.path.join(args.data_dir, 'fields.pkl')
    train_dataset = torch.load(train_path, pickle_module=dill)
    valid_dataset = torch.load(valid_path, pickle_module=dill)
    fields = torch.load(fields_path, pickle_module=dill)
    train_dataset.fields = fields
    valid_dataset.fields = fields
    train_iter, valid_iter = data.BucketIterator.splits(
        datasets=(train_dataset, valid_dataset), batch_size=args.batch_size,
        device=args.gpu)
    train_iter.sort_within_batch = True

    src_field = train_dataset.fields['src']
    tgt_field = train_dataset.fields['tgt']
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
    if args.gpu > -1:
        model.cuda(args.gpu)
    optimizer = optim.Adam(model.parameters())

    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'log'))

    def add_scalar_summary(tag, value, step):
        summary_writer.add_scalar(tag=tag, scalar_value=value,
                                  global_step=step)

    def run_train_iter(batch):
        src_words, src_length = batch.src
        tgt_words, tgt_length = batch.tgt
        logits = model(src_words=src_words, src_length=src_length,
                       tgt_words=tgt_words[:-1])
        loss = basic.sequence_cross_entropy(
            logits=logits, targets=tgt_words[1:], length=tgt_length - 1)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        return loss

    def generate(batch):
        # Greedy search
        vocab = tgt_field.vocab
        bos_id = vocab.stoi[tgt_field.init_token]
        batch_size = batch.src[0].size(1)
        max_length = batch.src[0].size(0) * 2

        src_words, src_length = batch.src
        context, encoder_rnn_state = model.encoder(
            words=src_words, length=src_length)

        prev_pred = Variable(
            src_length.new(1, batch_size).fill_(bos_id))
        done = torch.zeros(batch_size).byte()
        hyps = []
        prev_state = {'rnn': encoder_rnn_state}
        for t in range(max_length):
            if done.all():
                break
            decoder_input = prev_pred
            logits, prev_state, attn_weights = model.decoder(
                context=context, src_length=src_length,
                prev_state=prev_state, words=decoder_input)
            pred = logits.max(2)[1]
            prev_pred = pred
            hyps.append(pred.data)
        hyps = torch.cat(hyps, dim=0).transpose(0, 1).tolist()
        hyps = [hyps[i][:2 * length]
                for i, length in enumerate(src_length)]
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

    def print_samples():
        model.eval()
        num_samples = 2
        random_indices = [random.randrange(len(valid_dataset))
                          for _ in range(num_samples)]
        sample_data = [valid_dataset[i] for i in random_indices]
        sample_data.sort(key=lambda s: len(s.src), reverse=True)
        sample_batch = data.Batch(
            data=sample_data, dataset=valid_dataset,
            device=args.gpu, train=False)
        generated_sample = generate(sample_batch)
        for i in range(num_samples):
            print(f'  - Sample #{i}')
            src_sentence = ' '.join(ids_to_words(
                ids=sample_batch.src[0][:, i].data,
                vocab=src_field.vocab,
                eos_id=src_field.vocab.stoi[src_field.pad_token],
                remove_eos=True))
            tgt_sentence = ' '.join(ids_to_words(
                ids=sample_batch.tgt[0][:, i].data,
                vocab=tgt_field.vocab,
                eos_id=tgt_field.vocab.stoi[tgt_field.eos_token]))
            output_sentence = ' '.join(ids_to_words(
                ids=generated_sample[i],
                vocab=tgt_field.vocab,
                eos_id=tgt_field.vocab.stoi[tgt_field.eos_token]))
            print(f'    Source: {src_sentence}')
            print(f'    Target: {tgt_sentence}')
            print(f'    Output: {output_sentence}')

    def validate():
        model.eval()
        refs = []
        hyps = []
        for valid_batch in valid_iter:
            hyps_batch = generate(valid_batch)
            for i, hyp in enumerate(hyps_batch):
                ref_words = ids_to_words(
                    ids=valid_batch.tgt[0][1:, i].data,
                    vocab=tgt_field.vocab,
                    eos_id=tgt_field.vocab.stoi[tgt_field.pad_token],
                    remove_eos=True)
                hyp_words = ids_to_words(
                    ids=hyp,
                    vocab=tgt_field.vocab,
                    eos_id=tgt_field.vocab.stoi[tgt_field.eos_token],
                    remove_eos=True)
                refs.append([ref_words])
                hyps.append(hyp_words)
        return corpus_bleu(list_of_references=refs, hypotheses=hyps,
                           emulate_multibleu=True)

    best_valid_bleu = -10000

    for train_batch in train_iter:
        if not model.training:
            model.train()
        train_loss = run_train_iter(train_batch)
        iter_count = train_iter.iterations
        add_scalar_summary(tag='train_loss', value=train_loss.data[0],
                           step=iter_count)

        if iter_count % args.print_every == 0:
            print(f'* Epoch {train_iter.epoch:.3f} '
                  f'(Iter #{iter_count}): Sample')
            print_samples()
            print()

        if iter_count % args.validate_every == 0:
            print(f'* Epoch {train_iter.epoch:.3f} '
                  f'(Iter #{iter_count}): Validation')
            valid_bleu_score = validate()
            add_scalar_summary(tag='valid_bleu', value=valid_bleu_score,
                               step=iter_count)
            print(f'  - Valid BLEU: {valid_bleu_score:.4f}')
            if valid_bleu_score > best_valid_bleu:
                best_valid_bleu = valid_bleu_score
                model_filename = (f'model-{train_iter.epoch:.3f}'
                                  f'-{valid_bleu_score:.4f}.pkl')
                model_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(f'  - Saved the model to: {model_path}')
            print()

        if train_iter.epoch > args.max_epoch:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--rnn-type', default='lstm')
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--input-feeding', default=False, action='store_true')
    parser.add_argument('--word-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout-prob', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--validate-every', type=int, default=1000)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    config = {'model': {'rnn_type': args.rnn_type,
                        'num_layers': args.num_layers,
                        'input_feeding': args.input_feeding,
                        'word_dim': args.word_dim,
                        'hidden_dim': args.hidden_dim,
                        'dropout_prob': args.dropout_prob},
              'train': {'batch_size': args.batch_size}}
    config_path = os.path.join(args.save_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    train(args)


if __name__ == '__main__':
    main()
