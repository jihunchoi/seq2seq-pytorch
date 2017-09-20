import argparse
import os

import dill
import torch

from utils import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-src', required=True)
    parser.add_argument('--train-tgt', required=True)
    parser.add_argument('--valid-src', required=True)
    parser.add_argument('--valid-tgt', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--src-vocab-size', type=int, default=50000)
    parser.add_argument('--tgt-vocab-size', type=int, default=50000)
    args = parser.parse_args()

    src_field = io.SrcField(lower=True)
    tgt_field = io.TgtField(lower=True)
    fields = {'src': src_field, 'tgt': tgt_field}
    train_dataset = io.NMTDataset(
        src_path=args.train_src, tgt_path=args.train_tgt,
        fields=fields)
    valid_dataset = io.NMTDataset(
        src_path=args.valid_src, tgt_path=args.valid_tgt,
        fields=fields)

    src_field.build_vocab(train_dataset.src, max_size=args.src_vocab_size)
    tgt_field.build_vocab(train_dataset.tgt, max_size=args.tgt_vocab_size)

    filename_objs = [('train_dataset.pkl', train_dataset),
                     ('valid_dataset.pkl', valid_dataset)]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for filename, obj in filename_objs:
        file_path = os.path.join(args.save_dir, filename)
        print(f'Saving {file_path}...')
        with open(file_path, 'wb') as f:
            torch.save(obj, f, pickle_module=dill)


if __name__ == '__main__':
    main()
