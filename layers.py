import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPAttention(nn.Module):  # Scaled Dot-Product Attention
    def __init__(self):
        super(SDPAttention, self).__init__()

    def forward(self, q, k, v):  # [B, h, len, d_k]
        k_dim = k.size(-1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(k_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, h, len_q, len_v]

        attn = torch.matmul(attn_weights, v)  # [B, h, len_q, d_k]

        return attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, model_dim=512):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.model_dim = model_dim
        self.k_dim = model_dim // num_head

        self.w_qs = nn.Linear(model_dim, num_head * self.k_dim, bias=False)
        self.w_ks = nn.Linear(model_dim, num_head * self.k_dim, bias=False)
        self.w_vs = nn.Linear(model_dim, num_head * self.k_dim, bias=False)
        self.w_os = nn.Linear(num_head * self.k_dim, model_dim, bias=False)

        self.attention = SDPAttention()
        self.layernorm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v):
        B, N = q.size(0), q.size(1)
        qs = self.w_qs(q).view(B, N, self.num_head, self.k_dim).transpose(1, 2)  # [B, h, N, d_k]
        ks = self.w_ks(k).view(B, N, self.num_head, self.k_dim).transpose(1, 2)
        vs = self.w_vs(v).view(B, N, self.num_head, self.k_dim).transpose(1, 2)

        attn = self.attention(qs, ks, vs).transpose(1, 2)  # [B, N, h, d_k]
        attn = attn.view(B, N, -1) #[B, N, h * d_k]
        attn = self.w_os(attn) #[B, N, model_dim]

        attn = attn + q
        attn = self.layernorm(attn)

        return attn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.layernorm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x += residual
        x = self.layernorm(x)

        return x

