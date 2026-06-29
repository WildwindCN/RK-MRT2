"""MRT2 架构配置 —— PyTorch 参考实现

基于 magenta_rt/config.py 和 magenta_rt/jax/model.py 的规格，
针对 RK3588 NPU 部署做了 ONNX 兼容性适配。
"""

from dataclasses import dataclass


@dataclass
class SpectroStreamConfig:
    """SpectroStream 编解码器配置 (48kHz 立体声, 25Hz 帧率)"""

    # STFT 参数
    audio_sample_rate: float = 48000.0
    stft_frame_length: int = 960
    stft_frame_step: int = 480
    stft_fft_length: int = 960

    # 编解码器结构
    ratios: tuple = ((1, 2), (1, 2), (1, 3), (1, 2), (1, 2), (2, 2), (2, 1))
    mults: tuple = (2, 1, 2, 1, 1, 2, 1)
    is_resnet: bool = True
    causal: bool = True

    # 输入输出维度
    num_bins: int = 480          # STFT 频域 bins (fft_length // 2)
    num_channels: int = 4        # 立体声复数 → 4 通道实数
    num_features: int = 256      # 编码器瓶颈维度

    # 编码器参数
    encoder_base_conv_depth: int = 32
    encoder_base_conv_size: int = 7

    # 解码器参数
    decoder_base_conv_depth: int = 64
    decoder_base_conv_size: int = 7
    decoder_lookahead: int = 1

    # RVQ 量化器
    num_quantizers: int = 64
    codebook_size: int = 1024
    embedding_dim: int = 256
    rvq_truncation_level: int = 12   # 实际使用的 RVQ 层数

    # channel_splits
    channel_splits: int = 2
    channel_recombo_block: int = -2

    @property
    def total_time_stride(self) -> int:
        s = 1
        for rt, _ in self.ratios:
            s *= rt
        return s

    @property
    def total_freq_stride(self) -> int:
        s = 1
        for _, rf in self.ratios:
            s *= rf
        return s

    @property
    def input_bins(self) -> int:
        return self.num_bins // self.total_freq_stride

    @property
    def waveform_to_codes_ratio(self) -> int:
        return self.total_time_stride * self.stft_frame_step


@dataclass
class ModelSpec:
    """Transformer 层的架构规格"""
    num_layers: int = 12
    model_dims: int = 1024
    hidden_dims: int = 4096
    num_heads: int = 8
    dim_per_head: int = 128
    dropout_prob: float = 0.0
    use_mqa: bool = False
    ffn_use_gated_activation: bool = False


@dataclass
class DepthFormerConfig:
    """DepthFormer 完整模型配置 (MRT2 Small 规格)

    Temporal Body: 12 层, d=1024, 8 头, 滑动窗口 41 帧
    Depth Body:    2 层,  d=768,  6 头, 因果自注意力
    """

    # Encoder (MIDI conditioning)
    encoder_spec: ModelSpec = None

    # Temporal Body (逐帧自回归, 单帧 stateful)
    temporal_spec: ModelSpec = None

    # Depth Body (RVQ 维度自回归)
    depth_spec: ModelSpec = None

    # Token 配置
    num_reserved_tokens: int = 6
    codebook_size: int = 1024
    num_codebooks: int = 12       # rvq_truncation_level
    sos_id: int = 0

    # 注意力配置
    max_past_horizon: int = 41    # 滑动窗口大小
    num_attention_sink_embeddings: int = 1
    use_rope: bool = False        # NoPE
    soft_cap_logits: float = 30.0

    # 数据类型
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.encoder_spec is None:
            self.encoder_spec = ModelSpec(
                num_layers=6, model_dims=256, hidden_dims=1024,
                num_heads=8, dim_per_head=32,
                ffn_use_gated_activation=False,
            )
        if self.temporal_spec is None:
            self.temporal_spec = ModelSpec(
                num_layers=12, model_dims=1024, hidden_dims=4096,
                num_heads=8, dim_per_head=128,
                ffn_use_gated_activation=False,
            )
        if self.depth_spec is None:
            self.depth_spec = ModelSpec(
                num_layers=2, model_dims=768, hidden_dims=3072,
                num_heads=6, dim_per_head=128,
                ffn_use_gated_activation=False,
            )

    @property
    def vocab_size(self) -> int:
        """多分类输出: 每个 RVQ 层一个 softmax"""
        return self.codebook_size + self.num_reserved_tokens

    @property
    def total_vocab_size(self) -> int:
        """展平后的总词表大小"""
        return self.num_codebooks * self.codebook_size + self.num_reserved_tokens
