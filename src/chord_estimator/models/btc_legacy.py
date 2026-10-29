"""BTC-ISMIR19 25-class architecture compatibility adapter.

Derived from jayg996/BTC-ISMIR19 (MIT, Copyright 2019 Jonggwon Park),
commit 2682317be668032e6e4b269ded36adaa2ad57df0. The original software
license is reproduced in docs/BTC_UPSTREAM_LICENSE.txt.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


def _gen_bias_mask(max_length: int) -> torch.Tensor:
    return torch.from_numpy(np.triu(np.full((max_length, max_length), -np.inf), 1)).float().unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length: int, channels: int) -> torch.Tensor:
    position = np.arange(length)
    count = channels // 2
    increment = math.log(1.0e4) / (count - 1)
    inv = np.exp(np.arange(count).astype(float) * -increment)
    scaled = np.expand_dims(position, 1) * np.expand_dims(inv, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, ((0, 0), (0, channels % 2)))
    return torch.from_numpy(signal.reshape(1, length, channels)).float()


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        return self.gamma * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.beta


class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super().__init__()
        padding = (kernel_size - 1, 0) if pad_type == "left" else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        return self.conv(self.pad(inputs.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth, layer_config="cc", padding="both", dropout=0.0):
        super().__init__()
        sizes = [(input_depth, filter_size)] + [(filter_size, filter_size)] * (len(layer_config) - 2) + [(filter_size, output_depth)]
        self.layers = nn.ModuleList([Conv(*size, kernel_size=3, pad_type=padding) if kind == "c" else nn.Linear(*size) for kind, size in zip(layer_config, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads, bias_mask, dropout=0.0, attention_map=True):
        super().__init__()
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.attention_map = attention_map
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values):
        queries = self._split_heads(self.query_linear(queries)) * self.query_scale
        keys = self._split_heads(self.key_linear(keys))
        values = self._split_heads(self.value_linear(values))
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        outputs = self.output_linear(self._merge_heads(torch.matmul(weights, values)))
        return (outputs, weights) if self.attention_map else outputs


class self_attention_block(nn.Module):
    def __init__(self, hidden_size, bias_mask, dropout=0.2):
        super().__init__()
        self.attention_map = True
        self.multi_head_attention = MultiHeadAttention(hidden_size, 128, 128, hidden_size, 4, bias_mask, dropout, True)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, 128, hidden_size, "cc", "both", dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs
        y, weights = self.multi_head_attention(*(self.layer_norm_mha(x),) * 3)
        x = self.dropout(x + y)
        y = self.positionwise_convolution(self.layer_norm_ffn(x))
        return self.dropout(x + y), weights


class bi_directional_self_attention(nn.Module):
    def __init__(self, max_length=108):
        super().__init__()
        mask = _gen_bias_mask(max_length)
        self.weights_list = []
        self.attn_block = self_attention_block(128, mask)
        self.backward_attn_block = self_attention_block(128, mask.transpose(2, 3))
        self.linear = nn.Linear(256, 128)

    def forward(self, inputs):
        x, weights_list = inputs
        forward, weights = self.attn_block(x)
        backward, reverse_weights = self.backward_attn_block(x)
        output = self.linear(torch.cat((forward, backward), dim=2))
        weights_list.append(weights)
        weights_list.append(reverse_weights)
        return output, weights_list


class bi_directional_self_attention_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.timing_signal = _gen_timing_signal(108, 128)
        self.embedding_proj = nn.Linear(144, 128, bias=False)
        self.self_attn_layers = nn.Sequential(*[bi_directional_self_attention(108) for _ in range(8)])
        self.layer_norm = LayerNorm(128)
        self.input_dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        x = self.embedding_proj(self.input_dropout(inputs))
        x = x + self.timing_signal[:, :inputs.shape[1], :].type_as(inputs)
        y, weights = self.self_attn_layers((x, []))
        return self.layer_norm(y), weights


class SoftmaxOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 25
        self.output_projection = nn.Linear(128, 25)
        self.probs_out = False
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.hidden_size = 128


class LegacyBTCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.timestep = 108
        self.probs_out = False
        self.self_attn_layers = bi_directional_self_attention_layers()
        self.output_layer = SoftmaxOutputLayer()

    def forward(self, features, valid_mask=None):
        if features.shape[1] != 108 or features.shape[2] != 144:
            raise ValueError("BTC-ISMIR19 requires [batch, 108, 144]")
        hidden, _ = self.self_attn_layers(features)
        return {"chord_logits": self.output_layer.output_projection(hidden)}

