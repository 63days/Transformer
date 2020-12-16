import os
import argparse
import torch
import torch.optim as optim
from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from model import Transformer
from tqdm import tqdm

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


    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        train_losses = []
        val_losses = []
        best_loss = float('inf')

        # TODO: train
        for epoch in range(args.epochs):
            model.train()
            train_loss = []
            pbar = tqdm(train_loader)
            for src_batch, tgt_batch in pbar:
                loss = model.train_batch(src_batch, tgt_batch)

                pbar.set_description(f'E: {epoch+1:3d} | L: {loss:.3f}')
                train_loss.append(loss)

            train_loss = sum(train_loss) / len(train_loss)
            train_losses.append(train_loss)


            # TODO: validation
            model.eval()
            with torch.no_grad():
                val_loss = []
                for src_batch, tgt_batch in valid_loader:
                    model.validation_batch(src_batch, tgt_batch)

                    val_loss.append(loss)

                val_loss = sum(val_loss) / len(val_loss)
                val_losses.append(val_loss)

                print(f'-----VL: {val_loss:.3f}-----')

                if best_loss > val_loss:
                    best_loss = val_loss
                    model.save(epoch, best_loss, train_losses, val_losses)



    else:
        # test
        load_state = torch.load('./ckpt/transformer.ckpt', map_location=device)
        model.load_state_dict(load_state['model_state_dict'])
        model.eval()
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in test_loader:
            # TODO: predict pred_batch from src_batch with your model.
            pred_batch = model.inference(src_batch, tgt_batch)
            pred_batch = pred_batch.tolist()
            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            p = seq2sen(pred_batch, tgt_vocab)
            print(p)
            pred += p

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
