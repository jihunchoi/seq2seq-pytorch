import argparse
import os

import dill
import torch

from utils import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--src-vocab-size', type=int, default=30000)
    parser.add_argument('--tgt-vocab-size', type=int, default=30000)
    args = parser.parse_args()

    src_field = io.SrcField(lower=True)
    tgt_field = io.TgtField(lower=True)
    fields = [('src', src_field), ('tgt', tgt_field)]
    train_prefix = f'{args.data_path}/train.{args.src}-{args.tgt}'
    valid_prefix = f'{args.data_path}/valid.{args.src}-{args.tgt}'
    test_prefix = f'{args.data_path}/test.{args.src}-{args.tgt}'
    train_dataset = io.NMTDataset(
        src_path=f'{train_prefix}.{args.src}',
        tgt_path=f'{train_prefix}.{args.tgt}',
        fields=fields)
    valid_dataset = io.NMTDataset(
        src_path=f'{valid_prefix}.{args.src}',
        tgt_path=f'{valid_prefix}.{args.tgt}',
        fields=fields)
    test_dataset = io.NMTDataset(
        src_path=f'{test_prefix}.{args.src}',
        tgt_path=f'{test_prefix}.{args.tgt}',
        fields=fields)

    src_field.build_vocab(train_dataset.src, max_size=args.src_vocab_size)
    tgt_field.build_vocab(train_dataset.tgt, max_size=args.tgt_vocab_size)

    # Let's save fields and datasets separately, to reduce file sizes.
    fields = train_dataset.fields
    train_dataset.fields = None
    valid_dataset.fields = None
    test_dataset.fields = None

    filename_objs = [('train_dataset.pkl', train_dataset),
                     ('valid_dataset.pkl', valid_dataset),
                     ('test_dataset.pkl', test_dataset),
                     ('fields.pkl', fields)]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for filename, obj in filename_objs:
        file_path = os.path.join(args.save_dir, filename)
        print(f'Saving {file_path}...')
        with open(file_path, 'wb') as f:
            torch.save(obj, f, pickle_module=dill)


if __name__ == '__main__':
    main()
