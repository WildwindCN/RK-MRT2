"""Transformer 组件 —— PyTorch 参考实现

ONNX 兼容设计:
- 所有 einsum 用 MatMul + Reshape + Transpose 替代
- 无动态 shape 操作 (固定 max_len 预分配)
- 无 CumSum / STFT 等 RKNN 不支持算子

参考: magenta_rt/jax/transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS 归一化 (等同于 T5 的 RMSNorm)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)


class AttentionSink(nn.Module):
    """Attention Sink: 可学习的固定注意锚点

    在 KV cache 前插入 1 个可学习的 sink token,
    防止滑动窗口注意力在长序列生成中退化。
    """

    def __init__(self, num_heads: int, dim_per_head: int, num_sinks: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_sinks = num_sinks
        self.sink_k = nn.Parameter(torch.randn(num_sinks, num_heads, dim_per_head) * 0.02)
        self.sink_v = nn.Parameter(torch.randn(num_sinks, num_heads, dim_per_head) * 0.02)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """在 KV 序列前添加 sink tokens"""
        B, H, T, D = k.shape
        sk = self.sink_k.unsqueeze(0).expand(B, -1, -1, -1)  # [B, S, H, D]
        sv = self.sink_v.unsqueeze(0).expand(B, -1, -1, -1)
        sk = sk.transpose(1, 2)  # [B, H, S, D]
        sv = sv.transpose(1, 2)
        return torch.cat([sk, k], dim=2), torch.cat([sv, v], dim=2)


class SlidingWindowAttention(nn.Module):
    """滑动窗口因果自注意力 (NoPE + Attention Sink)

    ONNX 兼容: 使用 MatMul 替代 einsum
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_per_head: int,
        max_past_horizon: int = 41,
        dropout: float = 0.0,
        num_sink_embeddings: int = 1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.max_past_horizon = max_past_horizon
        self.num_sink_embeddings = num_sink_embeddings
        self.use_rope = use_rope
        self.inner_dim = num_heads * dim_per_head

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if num_sink_embeddings > 0:
            self.attention_sink = AttentionSink(num_heads, dim_per_head, num_sink_embeddings)
        else:
            self.attention_sink = None

        self.scale = 1.0 / math.sqrt(dim_per_head)

        if use_rope:
            raise NotImplementedError("RoPE 暂未实现，当前使用 NoPE")

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H*D] → [B, H, T, D]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.dim_per_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] 输入 (T=1 用于单帧推理)
            kv_cache: 可选的 (k_cache, v_cache) [B, H, L, D]
            attention_mask: [B, 1, T, L] 或 None

        Returns:
            output: [B, T, dim]
            new_kv_cache: (new_k, new_v)
        """
        B, T, _ = x.shape

        q = self._reshape_for_attention(self.q_proj(x))  # [B, H, T, D]
        k = self._reshape_for_attention(self.k_proj(x))
        v = self._reshape_for_attention(self.v_proj(x))

        # KV Cache 更新
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Attention Sink
        if self.attention_sink is not None:
            k, v = self.attention_sink(k, v)

        # 计算注意力分数 (MatMul, 不用 einsum)
        # q: [B, H, T, D], k: [B, H, T+L_sink, D]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, K]

        # 因果遮罩
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 滑动窗口: 屏蔽超出 max_past_horizon 的过去位置
        K = k.shape[2]
        if self.max_past_horizon > 0 and T == 1:
            # 单帧推理: 保留最近的 max_past_horizon 帧
            causal_mask = torch.ones(1, 1, 1, K, device=x.device, dtype=attn_weights.dtype)
            if K > self.max_past_horizon + self.num_sink_embeddings:
                # 屏蔽最老的帧, 保留 sink + 最近 max_past_horizon
                old_len = K - self.max_past_horizon - self.num_sink_embeddings
                causal_mask[:, :, :, self.num_sink_embeddings:self.num_sink_embeddings + old_len] = float("-inf")
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和 (MatMul, 不用 einsum)
        out = torch.matmul(attn_weights, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

        return self.out_proj(out), new_kv_cache


class CrossAttention(nn.Module):
    """流式交叉注意力 (MIDI conditioning → Temporal Body)

    ONNX 兼容: MatMul 替代 einsum
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_per_head: int,
        max_past_horizon: int = 41,
        dropout: float = 0.0,
        source_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.max_past_horizon = max_past_horizon
        self.inner_dim = num_heads * dim_per_head
        src_dim = source_dim if source_dim is not None else dim

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(src_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(src_dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(dim_per_head)
        self.attention_sink = None  # cross-attn sink supported in checkpoint but not in our arch

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.dim_per_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] 查询 (T=1 单帧)
            conditioning: [B, T_cond, dim] 条件信号
            kv_cache: 可选的 K/V cache

        Returns:
            output: [B, T, dim]
            new_kv_cache: (new_k, new_v)
        """
        B, T, _ = x.shape

        q = self._reshape(self.q_proj(x))

        k = self._reshape(self.k_proj(conditioning))
        v = self._reshape(self.v_proj(conditioning))

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

        return self.out_proj(out), new_kv_cache


class GatedFFN(nn.Module):
    """门控 FFN (Gated GELU)

    使用 2 个独立 Linear 替代 GatedUnit (ONNX 更友好)
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class FFN(nn.Module):
    """标准 FFN (非门控)"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class TransformerBlock(nn.Module):
    """单个 Transformer Block: Self-Attn + Cross-Attn + FFN

    使用 Pre-Norm (primer_hybrid 风格)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_per_head: int,
        ffn_dim: int,
        max_past_horizon: int = 41,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        num_sink_embeddings: int = 1,
        gated_ffn: bool = False,
        cross_attn_source_dim: int | None = None,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        self.pre_self_attn_norm = RMSNorm(dim)
        self.self_attn = SlidingWindowAttention(
            dim=dim,
            num_heads=num_heads,
            dim_per_head=dim_per_head,
            max_past_horizon=max_past_horizon,
            dropout=dropout,
            num_sink_embeddings=num_sink_embeddings,
        )

        if use_cross_attention:
            self.pre_cross_attn_norm = RMSNorm(dim)
            self.cross_attn = CrossAttention(
                dim=dim,
                num_heads=num_heads,
                dim_per_head=dim_per_head,
                max_past_horizon=max_past_horizon,
                dropout=dropout,
                source_dim=cross_attn_source_dim,
            )

        self.pre_ffn_norm = RMSNorm(dim)
        if gated_ffn:
            self.ffn = GatedFFN(dim, ffn_dim, dropout)
        else:
            self.ffn = FFN(dim, ffn_dim, dropout)

        self.post_self_attn_norm = RMSNorm(dim)
        self.post_cross_attn_norm = RMSNorm(dim) if use_cross_attention else None
        self.post_ffn_norm = RMSNorm(dim)

        self.self_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cross_attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        self_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple, tuple | None]:
        """
        Returns:
            output, new_self_kv_cache, new_cross_kv_cache
        """
        # Self-Attention (Pre-Norm + Residual)
        normed = self.pre_self_attn_norm(x)
        attn_out, new_self_kv = self.self_attn(normed, self_kv_cache, attention_mask)
        x = x + self.self_attn_dropout(attn_out)
        x = self.post_self_attn_norm(x)

        # Cross-Attention
        new_cross_kv = None
        if self.use_cross_attention and conditioning is not None:
            normed = self.pre_cross_attn_norm(x)
            cross_out, new_cross_kv = self.cross_attn(normed, conditioning, cross_kv_cache)
            x = x + self.cross_attn_dropout(cross_out)
            x = self.post_cross_attn_norm(x)

        # FFN (Pre-Norm + Residual)
        normed = self.pre_ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + self.ffn_dropout(ffn_out)
        x = self.post_ffn_norm(x)

        return x, new_self_kv, new_cross_kv
