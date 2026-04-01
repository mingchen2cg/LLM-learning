import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, kv_cache=None):
        batch_size, seq_len = query.shape[:2]

        Q = self.w_q(query).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, -1, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)

            kv_cache["k"] = K
            kv_cache["v"] = V

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask[:, :, -seq_len:, :] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        output = (
            torch.matmul(attn_weights, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        return self.w_o(output)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None, kv_cache=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask, kv_cache))
        x = self.norm2(x + self.ffn(x))

        return x


class TransformerDecoderWithKVCache(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=4096):
        super().__init__()
        self.n_layer = n_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, kv_cache=None):
        batch_size, seq_len = input_ids.shape

        # 增量模式
        if kv_cache is not None and kv_cache[0]:
            x = (
                self.embedding(input_ids[:, -1:])
                + self.pos_embedding[:, seq_len - 1 : seq_len, :]
            )

            mask = torch.ones(1, 1, 1, seq_len, device=input_ids.device)

        else:
            x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=input_ids.device)
            ).view(1, 1, seq_len, seq_len)

            if kv_cache is None:
                kv_cache = [{} for _ in range(self.n_layer)]

        for i, layer in enumerate(self.layers):
            x = layer(x, mask, kv_cache[i])

        return self.output_proj(self.layer_norm(x)), kv_cache
