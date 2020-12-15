import torch
import torch.nn as nn
from masking import *

class Translator(nn.Module):

    def __init__(self, model, sos_idx=0, eos_idx=1, pad_idx=2, max_len=50):
        super(Translator, self).__init__()
        self.model = model

    def inference(self, src_batch):
        B = src_batch.size(0)
        src_mask = get_pad_mask(src_batch)

        tgt_batch = torch.full([B, 50], pad_idx, dtype=torch.long, device=device)

