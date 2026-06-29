"""DepthFormer: Multivariate Decoder-Only Transformer

MRT2 Small 规格:
- Temporal Body: 12 层, d=1024, 8 头, dim_per_head=128, 滑动窗口 41
- Depth Body:    2 层,  d=768,  6 头, dim_per_head=128
- NoPE + Attention Sink
- Cross-Attention to MIDI conditioning

ONNX 导出策略:
- TemporalBodyStateful: 单帧 stateful graph, KV cache 作为外部 I/O
- DepthBodyAR: 12 RVQ 层级联自回归, [B, Q, D] → [B, Q, vocab]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DepthFormerConfig
from .transformer import RMSNorm, TransformerBlock


class TokenEmbedding(nn.Module):
    """多码本 Token Embedding

    输入: [B, T, Q] int32 (Q 个 RVQ 层的 token indices)
    输出: [B, T, Q, D] float (每层独立嵌入后求和)

    MRT2 使用 unique codes: 不同 RVQ 层的 token offset 不同,
    codebook_size 个 token × num_codebooks 层 = 大词表。
    每层 token = base_token + rvq_index * codebook_size + num_reserved_tokens
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.num_reserved = config.num_reserved_tokens
        self.dim = config.temporal_spec.model_dims

        # 总词表 = 保留token + 所有层的码本
        total_embeddings = config.num_reserved_tokens + config.num_codebooks * config.codebook_size
        self.embed = nn.Embedding(total_embeddings, self.dim)
        self.scale = math.sqrt(self.dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T, Q] → embeddings: [B, T, Q, D]"""
        return self.embed(tokens) * self.scale


class TemporalBodyStateful(nn.Module):
    """单帧 Stateful Temporal Body

    设计用于 ONNX 导出为 stateful graph:
    - 输入: x [B, 1, D] + 24 个 KV cache tensor (12 层 × 2 种 attention)
    - 输出: out [B, 1, D] + 24 个更新的 KV cache tensor

    每帧推理时:
    1. 接收上一帧 token (mean over RVQ embedding)
    2. 接收 MIDI conditioning (encoder 输出, 一次计算)
    3. 更新 self-attn KV cache + cross-attn KV cache
    4. 输出当前帧的 temporal representation
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        spec = config.temporal_spec
        self.dim = spec.model_dims
        self.num_layers = spec.num_layers
        self.max_past_horizon = config.max_past_horizon
        encoder_dim = config.encoder_spec.model_dims

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=spec.model_dims,
                num_heads=spec.num_heads,
                dim_per_head=spec.dim_per_head,
                ffn_dim=spec.hidden_dims,
                max_past_horizon=config.max_past_horizon,
                dropout=spec.dropout_prob,
                use_cross_attention=True,
                num_sink_embeddings=config.num_attention_sink_embeddings,
                gated_ffn=spec.ffn_use_gated_activation,
                cross_attn_source_dim=encoder_dim,
            )
            for _ in range(spec.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        kv_caches: list[dict] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Args:
            x: [B, 1, D] 当前帧输入
            conditioning: [B, T_cond, D] MIDI encoder 输出
            kv_caches: 可选, 每层的 KV cache dict

        Returns:
            output: [B, 1, D]
            new_kv_caches: 更新后的 KV cache 列表
        """
        if kv_caches is None:
            kv_caches = [{} for _ in range(self.num_layers)]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i]
            x, new_self_kv, new_cross_kv = layer(
                x,
                conditioning=conditioning,
                self_kv_cache=cache.get("self_kv"),
                cross_kv_cache=cache.get("cross_kv"),
            )
            new_kv_caches.append({
                "self_kv": new_self_kv,
                "cross_kv": new_cross_kv,
            })

        return x, new_kv_caches


class TemporalBodyFull(nn.Module):
    """完整 Temporal Body (非 stateful, 用于训练/验证)

    输入整个序列 [B, T, D], 一次性前向传播
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        spec = config.temporal_spec
        self.dim = spec.model_dims
        self.num_layers = spec.num_layers
        encoder_dim = config.encoder_spec.model_dims

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=spec.model_dims,
                num_heads=spec.num_heads,
                dim_per_head=spec.dim_per_head,
                ffn_dim=spec.hidden_dims,
                max_past_horizon=config.max_past_horizon,
                dropout=spec.dropout_prob,
                use_cross_attention=True,
                num_sink_embeddings=config.num_attention_sink_embeddings,
                gated_ffn=spec.ffn_use_gated_activation,
                cross_attn_source_dim=encoder_dim,
            )
            for _ in range(spec.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 完整输入序列
            conditioning: [B, T_cond, D]

        Returns:
            output: [B, T, D]
        """
        for layer in self.layers:
            x, _, _ = layer(x, conditioning=conditioning)
        return x


class DepthBodyAR(nn.Module):
    """RVQ 维度自回归 Depth Body

    设计用于 ONNX 导出:
    - 输入: [B, Q, D] (Q=12 RVQ levels 的输入)
    - 输出: [B, Q, vocab_size] (每层一个 softmax logits)

    深度维度自注意力: 在 Q 轴上做因果自注意力,
    RVQ_i 只能看到 RVQ_0..i-1 (不包括自己)。

    推理时逐 RVQ 层级联:
    for rvq in range(12):
        logits_rvq = depth_body(x_input)  # x_input: [B, rvq+1, D]
        next_token = sample(logits_rvq[:, -1])
        x_input = append(x_input, embed(next_token))
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        spec = config.depth_spec
        self.dim = spec.model_dims
        self.temporal_dim = config.temporal_spec.model_dims
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size

        # 从 temporal_dim → depth_dim 的投影
        if self.temporal_dim != self.dim:
            self.input_adapter = nn.Linear(self.temporal_dim, self.dim, bias=False)
        else:
            self.input_adapter = nn.Identity()

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=spec.model_dims,
                num_heads=spec.num_heads,
                dim_per_head=spec.dim_per_head,
                ffn_dim=spec.hidden_dims,
                max_past_horizon=config.num_codebooks,  # 因果: 只能看前面的 RVQ
                dropout=spec.dropout_prob,
                use_cross_attention=False,
                num_sink_embeddings=0,  # Depth body 不用 attention sink
                gated_ffn=spec.ffn_use_gated_activation,
            )
            for _ in range(spec.num_layers)
        ])

        self.final_norm = RMSNorm(self.dim)
        self.to_logits = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.soft_cap = config.soft_cap_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Q, D] depth 维度输入

        Returns:
            logits: [B, Q, vocab_size]
        """
        x = self.input_adapter(x)

        for layer in self.layers:
            x, _, _ = layer(x, conditioning=None)

        x = self.final_norm(x)

        if self.soft_cap is not None:
            logits = self.soft_cap * torch.tanh(self.to_logits(x) / self.soft_cap)
        else:
            logits = self.to_logits(x)

        return logits


class DepthFormer(nn.Module):
    """完整 DepthFormer 模型

    输入: MIDI conditioning + RVQ tokens
    输出: 每步每 RVQ 层的 logits

    训练时使用完整序列前向传播。
    推理时拆分为 TemporalBodyStateful + DepthBodyAR 两个独立 graph。
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.sos_id = config.sos_id

        self.token_embedding = TokenEmbedding(config)
        self.temporal_body = TemporalBodyFull(config)
        self.depth_body = DepthBodyAR(config)

    def _pad_sos(self, tokens: torch.Tensor) -> torch.Tensor:
        """在序列开头插入 SOS token"""
        B, T, Q = tokens.shape
        sos = torch.full((B, 1, Q), self.sos_id, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([sos, tokens], dim=1)

    def forward(
        self,
        conditioning: torch.Tensor,
        tokens: torch.Tensor,
        return_loss: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            conditioning: [B, T_cond, D_enc] MIDI encoder 输出
            tokens: [B, T, Q] 目标 token 序列 (不含 SOS)
            return_loss: 是否返回 loss

        Returns:
            logits: [B, T, Q, vocab_size] 如果 return_loss=False
            (logits, loss): 如果 return_loss=True
        """
        B, T, Q = tokens.shape

        # 插入 SOS
        padded = self._pad_sos(tokens)  # [B, T+1, Q]

        # Token Embedding
        embedded = self.token_embedding(padded)  # [B, T+1, Q, D_temp]

        # Temporal: mean over RVQ dim
        temporal_input = embedded.mean(dim=-2)[:, :-1]  # [B, T, D_temp]

        # Temporal Body
        temporal_out = self.temporal_body(temporal_input, conditioning)  # [B, T, D_temp]

        # Depth Body 输入: concat(temporal_out, embedded[1:, :-1])
        # temporal_out: [B, T, D_temp] → [B, T, 1, D_temp]
        # embedded[1:, :-1]: [B, T, Q-1, D_temp]
        depth_temporal = temporal_out.unsqueeze(2)  # [B, T, 1, D_temp]
        depth_embedded = embedded[:, 1:, :-1]       # [B, T, Q-1, D_temp]
        depth_input = torch.cat([depth_temporal, depth_embedded], dim=2)  # [B, T, Q, D_temp]

        # Flatten B, T for depth body processing
        # depth_body expects [B, Q, D]
        depth_input_flat = depth_input.view(B * T, Q, -1)

        # Depth Body
        logits_flat = self.depth_body(depth_input_flat)  # [B*T, Q, vocab_size]
        logits = logits_flat.view(B, T, Q, -1)  # [B, T, Q, vocab_size]

        if return_loss:
            # Cross-entropy per RVQ level
            targets = tokens  # [B, T, Q]
            # Reshape for F.cross_entropy: [B*T*Q, vocab_size] vs [B*T*Q]
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        return logits


# ═══════════════════════════════════════════════════════
# 单帧推理辅助 (用于 ONNX 导出和 C++ Runtime)
# ═══════════════════════════════════════════════════════

class TemporalBodyStatefulExport(nn.Module):
    """Temporal Body 的单帧 stateful 版本 —— 专用于 ONNX 导出

    将 12 层 × 24 个 KV buffer (每层 self_k, self_v + cross_k, cross_v)
    作为显式输入/输出, 满足 RKNN 静态图要求。

    输入:
      x:              [1, 1, D_temp]      当前帧
      conditioning:   [1, T_cond, D_enc]  MIDI 条件 (预计算)
      self_k_i:       [1, H, L, D_h]     第 i 层 self-attn K cache (i=0..11)
      self_v_i:       [1, H, L, D_h]     第 i 层 self-attn V cache
      cross_k_i:      [1, H, L, D_h]     第 i 层 cross-attn K cache
      cross_v_i:      [1, H, L, D_h]     第 i 层 cross-attn V cache

    输出:
      output:         [1, 1, D_temp]
      self_k_i_out:   [1, H, L+1, D_h]
      self_v_i_out:   [1, H, L+1, D_h]
      cross_k_i_out:  [1, H, L, D_h]
      cross_v_i_out:  [1, H, L, D_h]

    共 1 + 48 个输入, 1 + 48 个输出。
    """

    def __init__(self, config: DepthFormerConfig):
        super().__init__()
        self.body = TemporalBodyStateful(config)
        self.num_layers = config.temporal_spec.num_layers

    def forward(self, x, conditioning, *kv_buffers):
        # kv_buffers: 每层 4 个 (self_k, self_v, cross_k, cross_v)
        kv_caches = []
        idx = 0
        for _ in range(self.num_layers):
            kv_caches.append({
                "self_kv": (kv_buffers[idx], kv_buffers[idx + 1]),
                "cross_kv": (kv_buffers[idx + 2], kv_buffers[idx + 3]),
            })
            idx += 4

        output, new_caches = self.body(x, conditioning, kv_caches)

        outputs = [output]
        for cache in new_caches:
            outputs.extend([
                cache["self_kv"][0],
                cache["self_kv"][1],
                cache["cross_kv"][0],
                cache["cross_kv"][1],
            ])

        return tuple(outputs)
