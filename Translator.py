import torch
import torch.nn as nn
from model import Transformer
from masking import *
from utils import seq2sen

class Translator(nn.Module):

    def __init__(self, model, tgt_vocab, sos_idx=0, eos_idx=1, pad_idx=2, max_len=50):
        super(Translator, self).__init__()
        assert type(model) == Transformer
        self.model = model
        self.model.eval()
        self.load_state = self.model.load('./ckpt/transformer.ckpt')

        self.tgt_vocab = tgt_vocab

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_len = max_len

    def inference(self, src_batch):
        B = src_batch.size(0)
        src_mask = get_pad_mask(src_batch)

        tgt_batch = torch.full([B, self.max_len], self.pad_idx, dtype=torch.long, device=device)
        tgt_batch[:, 0] = self.sos_idx

        for i in range(1, self.max_len):
            pred = self.model.forward(src_batch, tgt_batch[:, :i])
            tgt_batch[:, i] = pred[:, i-1]


