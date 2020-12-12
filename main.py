import os
import argparse
import torch
import torch.optim as optim
from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO: use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    model = Transformer(src_vocab_sz=src_vocab_size, tgt_vocab_sz=tgt_vocab_size,
                        pad_idx=pad_idx, enc_stack=6, dec_stack=6, max_len=max_length,
                        model_dim=512, ff_dim=2048, num_head=8).to(device)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        # TODO: train
        for epoch in range(args.epochs):
            model.train()
            for src_batch, tgt_batch in train_loader:
                optimizer.zero_grad()
                src_batch, tgt_batch = torch.tensor(src_batch), torch.tensor(tgt_batch)
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                pred = model(src_batch, tgt_batch)
                loss = criterion(pred, tgt_batch)
                loss.backward()
                optimizer.step()

            # TODO: validation
            model.eval()
            with torch.no_grad():
                for src_batch, tgt_batch in valid_loader:
                    src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

                    pred = model(src_batch, tgt_batch)
                    loss = criterion(pred, tgt_batch)
                    print(f'val loss: {loss:.3f}')

    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in test_loader:
            # TODO: predict pred_batch from src_batch with your model.
            pred_batch = tgt_batch

            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
