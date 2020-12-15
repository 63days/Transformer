import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from masking import *
from layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):

    def __init__(self, len_seq, model_dim):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.register_buffer('PE', self.get_pos_table(len_seq, model_dim))

    def get_pos_table(self, len_seq, dim):
        N = np.arange(len_seq)
        D = np.arange(dim)

        def get_angles(pos):
            angles = pos / np.power(10000, 2 * (D // 2) / self.model_dim)
            return angles

        pos_table = np.array([get_angles(pos) for pos in N])
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])

        return torch.FloatTensor(pos_table)  # unsqueeze(0) ?

    def forward(self, x):
        return x + self.PE[:x.size(-2), :].clone().detach()


class Transformer(nn.Module):

    def __init__(self, src_vocab_sz, tgt_vocab_sz, pad_idx, enc_stack, dec_stack,
                 max_len, model_dim, ff_dim, num_head):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.model_dim = model_dim

        self.encoder = Encoder(src_vocab_sz, pad_idx, enc_stack, max_len, model_dim,
                               ff_dim, num_head)
        self.decoder = Decoder(tgt_vocab_sz, pad_idx, dec_stack, max_len, model_dim, ff_dim, num_head)
        self.fc = nn.Linear(model_dim, tgt_vocab_sz)

        self.optimizer = optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.step = 1
        self.warmup_steps = 4000

    def forward(self, src_seq, tgt_seq):
        src_seq_mask = get_pad_mask(src_seq, self.pad_idx) # [B, 1, len_src]
        tgt_seq_mask = get_pad_mask(tgt_seq, self.pad_idx) | get_subsequent_mask(tgt_seq) # [B, len_tgt, len_tgt]

        enc_output = self.encoder(src_seq, src_seq_mask)
        dec_output = self.decoder(enc_output, tgt_seq, tgt_seq_mask, src_seq_mask)

        pred_tgt_seq = self.fc(dec_output)

        return pred_tgt_seq

    def train_batch(self, src_batch, tgt_batch):
        src_batch = torch.LongTensor(src_batch).to(device)
        tgt_batch = torch.LongTensor(tgt_batch).to(device)

        pred = self.forward(src_batch, tgt_batch[:, :-1])
        loss = self.criterion(pred.reshape(-1, pred.size(-1)), tgt_batch[:, 1:].reshape(-1))

        ####### lr scheduler ########
        lr = (self.model_dim ** -0.5) * min(self.step ** -0.5, self.step * self.warmup_steps ** -1.5)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        #############################

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1

        return loss.item()

    def validation_batch(self, src_batch, tgt_batch):
        src_batch = torch.LongTensor(src_batch).to(device)
        tgt_batch = torch.LongTensor(tgt_batch).to(device)

        pred = self.forward(src_batch, tgt_batch[:, :-1])
        loss = self.criterion(pred.reshape(-1, pred.size(-1)), tgt_batch[:, 1:].reshape(-1))

        return loss.item()

    def save(self, epoch, best_loss, train_losses, val_losses):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': best_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, './ckpt/transformer.ckpt')


class Encoder(nn.Module):

    def __init__(self, src_vocab_sz, pad_idx, num_stack, max_len, model_dim, ff_dim, num_head):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(num_head, model_dim, ff_dim)
                                          for _ in range(num_stack)])
        self.pos_table = PositionalEncoding(max_len, model_dim)
        self.emb_layer = nn.Embedding(src_vocab_sz, model_dim, padding_idx=pad_idx)

    def forward(self, src_seq, src_mask=None):
        output = self.emb_layer(src_seq)
        output = self.pos_table(output)

        for layer in self.layer_stack:
            output = layer(output, src_mask)

        return output


class Decoder(nn.Module):

    def __init__(self, tgt_vocab_sz, pad_idx, num_stack, max_len, model_dim, ff_dim, num_head):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList([DecoderLayer(num_head, model_dim, ff_dim)
                                          for _ in range(num_stack)])
        self.pos_table = PositionalEncoding(max_len, model_dim)
        self.emb_layer = nn.Embedding(tgt_vocab_sz, model_dim, padding_idx=pad_idx)

    def forward(self, enc_output, tgt_seq, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = self.emb_layer(tgt_seq)

        for layer in self.layer_stack:
            dec_output = layer(dec_output, enc_output, slf_attn_mask, dec_enc_attn_mask)

        return dec_output
