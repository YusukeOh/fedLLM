"""
Core layers for Time-LLM, adapted from Jin et al. (2024).

Contains PatchEmbedding, ReprogrammingLayer, FlattenHead, and Normalize.
These are structurally identical to the original implementation so that
weights / behaviours remain comparable.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
from torch import Tensor


class Normalize(nn.Module):
    """Reversible instance normalization (RevIN-style)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"Unknown mode: {mode}")

    def _get_statistics(self, x: Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class ReplicationPad1d(nn.Module):
    def __init__(self, padding: tuple[int, int]):
        super().__init__()
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        replicate = x[:, :, -1:].repeat(1, 1, self.padding[-1])
        return torch.cat([x, replicate], dim=-1)


class PatchEmbedding(nn.Module):
    """Split a univariate time series into patches and project to d_model."""

    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_layer = ReplicationPad1d((0, stride))
        self.value_embedding = nn.Sequential(
            nn.Conv1d(patch_len, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        # x: (B, N, T) → patches: (B*N, n_patches, patch_len)
        n_vars = x.shape[1]
        x = self.padding_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class ReprogrammingLayer(nn.Module):
    """Cross-attention that maps time-series patches into word-embedding space."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_llm: int | None = None,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        target_embedding: Tensor,
        source_embedding: Tensor,
        value_embedding: Tensor,
    ) -> Tensor:
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target = self.query_projection(target_embedding).view(B, L, H, -1)
        source = self.key_projection(source_embedding).view(S, H, -1)
        value = self.value_projection(value_embedding).view(S, H, -1)

        scale = 1.0 / sqrt(target.shape[-1])
        scores = torch.einsum("blhe,she->bhls", target, source)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", A, value)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


class FlattenHead(nn.Module):
    """Flatten patch dimension and project to prediction horizon."""

    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.linear(self.flatten(x)))
