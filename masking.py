import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2) #[B, 1, N]

def get_subsequent_mask(seq):
    len_seq = seq.size(-1)
    mask = torch.triu(torch.ones((len_seq, len_seq), device=device), diagonal=1).bool()

    return mask.unsqueeze(0) #[1, N, N]
