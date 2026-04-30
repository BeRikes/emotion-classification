import torch
import torch.nn as nn
import math


# 嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx, device):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=padding_idx, device=device)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout, device, n=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, requires_grad=False, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.pow(n, (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 多头自注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, nh, d_model, device):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nh == 0
        self.d_k = d_model // nh
        self.nh = nh     # 头的数量
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False, device=device) for _ in range(4)])
        self.attn = None

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1. 线性变换并分头, 这里的nh头求qkv不通过nh次wx+b，而是只进行一次，全连接层是512x512的，而不是64x64的
        query, key, value = [l(x).view(nbatches, -1, self.nh, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2. 计算注意力
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3. 拼接多头输出并线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.nh * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-1)      # shape: [b_size, nh, seq_len, seq_len]
        return torch.matmul(p_attn, value), p_attn

    @property
    def get_attension(self):
        return self.attn


# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, device='cuda:0'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False, device=device)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.act(self.dropout(self.w_1(x))))


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout, device):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, device)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, device)
        self.norm0 = nn.LayerNorm(d_model, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, mask):
        x_ = self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm0(x + x_)
        x_ = self.dropout2(self.feed_forward(x))
        return self.norm1(x + x_)


class mini_bert(nn.Module):
    def __init__(self, num_class, vocab_size, seq_len, pad, h=4, d_model=256, d_ff=1024, dropout=0.1, N=1, device='cuda:0'):
        super(mini_bert, self).__init__()
        self.pad = pad
        self.embed = nn.Sequential(Embeddings(d_model, vocab_size, pad, device), PositionalEncoding(d_model, seq_len, dropout, device))
        self.transformer_blocks = nn.Sequential()
        for i in range(N):
            self.transformer_blocks.add_module(f'encoder_{i}', EncoderLayer(h, d_model, d_ff, dropout, device))
        self.proj = nn.Linear(d_model, num_class, device=device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')

    def forward(self, x):   # (b, seq_len)
        pad_mask = self.create_padding_mask(x, self.pad)   # 对于填充的位置不计算注意力
        x = self.embed(x)    # (b, seq_len, d_model)
        for blk in self.transformer_blocks:
            x = blk(x, pad_mask)
        return self.proj(torch.mean(x, dim=1))      # 获取句子的平均表示

    # 填充掩码
    def create_padding_mask(self, seq, pad=0):
        padding_mask = (seq == pad).unsqueeze(1).unsqueeze(2)
        return padding_mask