import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class Transformer(nn.Module):

    def __init__(self, src_vocab_sz, tgt_vocab_sz, pad_idx, enc_stack, dec_stack,
                 max_len, model_dim, ff_dim, num_head):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_sz, pad_idx, enc_stack, max_len, model_dim,
                               ff_dim, num_head)
        self.decoder = Decoder(tgt_vocab_sz, pad_idx, dec_stack, max_len, model_dim, ff_dim, num_head)
        self.fc = nn.Linear(model_dim, tgt_vocab_sz)

    def forward(self, src_seq, tgt_seq):
        enc_output = self.encoder(src_seq)
        dec_output = self.decoder(enc_output, tgt_seq)

        pred_tgt_seq = self.fc(dec_output)
        pred_tgt_seq = F.softmax(pred_tgt_seq, dim=-1)

        return pred_tgt_seq

class Encoder(nn.Module):

    def __init__(self, src_vocab_sz, pad_idx, num_stack, max_len, model_dim, ff_dim, num_head):
        super(Encoder, self).__init__()
        self.layer_stack = nn.ModuleList([EncoderLayer(num_head, model_dim, ff_dim)
                                          for _ in range(num_stack)])
        self.pos_table = PositionalEncoding(max_len, model_dim)
        self.emb_layer = nn.Embedding(src_vocab_sz, model_dim, padding_idx=pad_idx)

    def forward(self, src_seq):
        output = self.emb_layer(src_seq)
        output = self.pos_table(output)

        for layer in self.layer_stack:
            output = layer(output)

        return output


class Decoder(nn.Module):

    def __init__(self, tgt_vocab_sz, pad_idx, num_stack, max_len, model_dim, ff_dim, num_head):
        super(Decoder, self).__init__()
        self.layer_stack = nn.ModuleList([DecoderLayer(num_head, model_dim, ff_dim)
                                          for _ in range(num_stack)])
        self.pos_table = PositionalEncoding(max_len, model_dim)
        self.emb_layer = nn.Embedding(tgt_vocab_sz, model_dim, padding_idx=pad_idx)

    def forward(self, enc_output, tgt_seq):
        dec_output = self.emb_layer(tgt_seq)

        for layer in self.layer_stack:
            dec_output = layer(dec_output, enc_output)

        return dec_output
