import torch
import torch.nn as nn
from sublayers import *
import numpy as np


class EncoderLayer(nn.Module):

    def __init__(self, num_head, model_dim, ff_dim):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(num_head, model_dim)
        self.ff_layer = PositionWiseFeedForward(model_dim, ff_dim)

    def forward(self, enc_input, src_mask=None):
        output = self.attn_layer(enc_input, enc_input, enc_input, src_mask)
        output = self.ff_layer(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, num_head, model_dim, ff_dim):
        super(DecoderLayer, self).__init__()
        self.attn_layer1 = MultiHeadAttention(num_head, model_dim)
        self.attn_layer2 = MultiHeadAttention(num_head, model_dim)
        self.ff_layer = PositionWiseFeedForward(model_dim, ff_dim)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        output = self.attn_layer1(dec_input, dec_input, dec_input, slf_attn_mask)
        output = self.attn_layer2(enc_output, enc_output, output, dec_enc_attn_mask)
        output = self.ff_layer(output)

        return output
