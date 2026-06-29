/**
 * Inference Engine: 实时推理循环
 *
 * 编排 Temporal Body (NPU) → Depth Body (NPU/CPU) → Codec Decoder (NPU) → iSTFT (CPU)
 *
 * 推理循环 (每 40ms 一帧 @ 25Hz):
 *   1. 更新 MIDI conditioning
 *   2. 单帧 Temporal Body (NPU, stateful)
 *   3. Depth Body AR (NPU/CPU, 12 步 cascade)
 *   4. CPU 采样 (top-k, temperature)
 *   5. Codec Decoder (NPU) + iSTFT (CPU)
 *   6. 输出 PCM 音频
 *
 * 延迟预算:
 *   - Temporal Body: < 5ms (NPU FP16)
 *   - Depth Body: < 10ms (12 steps × ~0.8ms)
 *   - Codec Decoder + iSTFT: < 5ms
 *   - 总计: < 20ms (40ms 预算, 安全余量 2x)
 *
 * 内存预算 (估算):
 *   - Temporal Body: ~360MB (FP16 weights + KV cache)
 *   - Depth Body: ~32MB (FP16 weights)
 *   - Codec Decoder: ~61MB (FP16 weights)
 *   - Audio buffers: ~1MB
 *   - 总计: ~460MB (RK3588 通常有 4-8GB RAM)
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <functional>

#include "rknn_model.hpp"
#include "kv_cache.hpp"
#include "midi_parser.hpp"

namespace rkmrt2 {

// ═══════════════════════════════════════════════════════
// 模型配置 (匹配 PyTorch config)
// ═══════════════════════════════════════════════════════

struct ModelConfig {
    // Temporal Body
    int temporal_dim = 1024;
    int temporal_layers = 12;
    int temporal_heads = 8;
    int temporal_dim_per_head = 128;
    int temporal_max_kv_len = 512;

    // Depth Body
    int depth_dim = 768;
    int depth_layers = 2;
    int depth_heads = 6;
    int num_codebooks = 12;
    int codebook_size = 1024;
    int num_reserved_tokens = 6;
    int vocab_size = 1030;  // 1024 + 6

    // Encoder
    int encoder_dim = 256;

    // Audio
    float sample_rate = 48000.0f;
    float frame_rate = 25.0f;
    int samples_per_frame = 1920; // 48000 / 25
    int stft_frame_length = 960;
    int stft_frame_step = 480;
    int stft_fft_length = 960;
    int stft_num_bins = 480;
    int stft_num_channels = 4;

    // SpectroStream
    int ss_embedding_dim = 256;
    int ss_total_time_stride = 4;
    int ss_rvq_levels = 12;
};

// ═══════════════════════════════════════════════════════
// 采样器
// ═══════════════════════════════════════════════════════

class TokenSampler {
public:
    TokenSampler(int vocab_size, int codebook_size)
        : vocab_size_(vocab_size), cb_size_(codebook_size) {}

    // Top-k 采样
    int sample_topk(const float* logits, int vocab_size,
                    int top_k, float temperature, int rvq_level);

    // Argmax (用于验证/调试)
    int sample_argmax(const float* logits, int vocab_size);

private:
    int vocab_size_, cb_size_;
};

// ═══════════════════════════════════════════════════════
// 主推理引擎
// ═══════════════════════════════════════════════════════

class InferenceEngine {
public:
    InferenceEngine(const ModelConfig& cfg,
                    const std::string& temporal_model_path,
                    const std::string& depth_model_path,
                    const std::string& codec_model_path);

    ~InferenceEngine();

    // 初始化 (加载模型, 预分配 buffer)
    bool init();

    // 设置 MIDI conditioning (一次性, 或实时更新)
    void set_conditioning(const float* pianoroll, int num_frames);

    // 生成一帧 (核心推理步骤)
    // 返回 PCM 采样数
    int generate_frame(float* pcm_output);

    // 设置采样参数
    void set_temperature(float t) { temperature_ = t; }
    void set_top_k(int k) { top_k_ = k; }
    void set_style_embedding(const float* style, int dim);

    // 重置生成状态
    void reset();

    // 回调: 音频输出时触发
    using AudioCallback = std::function<void(const float* pcm, int samples)>;
    void set_audio_callback(AudioCallback cb) { audio_cb_ = std::move(cb); }

private:
    // 单步推理
    int step_temporal_body();
    int step_depth_body_ar();
    int step_codec_decoder();
    int step_istft();

    // 辅助
    void tokens_to_embedding(const int* tokens, int num_rvq, float* embedding);
    void compute_cross_kv(const float* encoded_cond, int cond_len);

    ModelConfig cfg_;

    // RKNN 模型
    std::unique_ptr<RKNNModel> temporal_model_;
    std::unique_ptr<RKNNModel> depth_model_;
    std::unique_ptr<RKNNModel> codec_model_;

    // KV Cache
    KVCacheConfig kv_config_;
    std::unique_ptr<KVCacheManager> kv_cache_;

    // Token 采样器
    std::unique_ptr<TokenSampler> sampler_;

    // 工作缓冲区
    std::vector<float> temporal_input_;     // [1, temporal_dim]
    std::vector<float> temporal_output_;    // [1, temporal_dim]
    std::vector<float> depth_input_;        // [num_codebooks, temporal_dim]
    std::vector<float> depth_logits_;       // [num_codebooks, vocab_size]
    std::vector<float> current_tokens_;     // [num_codebooks] int
    std::vector<float> rvq_embedding_;      // [ss_embedding_dim]
    std::vector<float> rvq_embeddings_;     // [ss_rvq_levels * ss_embedding_dim] 累积

    std::vector<float> conditioning_;       // [cond_len, encoder_dim]
    int cond_len_ = 0;

    // 解码器累积 buffer
    std::vector<float> decoder_input_;      // [T, ss_embedding_dim]
    std::vector<float> stft_output_;        // [channels, bins, T*stride]
    std::vector<float> pcm_buffer_;         // 输出 PCM
    int decoder_frame_count_ = 0;

    // 采样参数
    float temperature_ = 0.8f;
    int top_k_ = 50;

    // 回调
    AudioCallback audio_cb_;

    bool initialized_ = false;
};

// ═══════════════════════════════════════════════════════
// 核心推理循环实现
// ═══════════════════════════════════════════════════════

inline InferenceEngine::InferenceEngine(
    const ModelConfig& cfg,
    const std::string& temporal_model_path,
    const std::string& depth_model_path,
    const std::string& codec_model_path)
    : cfg_(cfg)
{
    temporal_model_ = std::make_unique<RKNNModel>(temporal_model_path);
    depth_model_ = std::make_unique<RKNNModel>(depth_model_path);
    codec_model_ = std::make_unique<RKNNModel>(codec_model_path);

    kv_config_.num_layers = cfg.temporal_layers;
    kv_config_.num_heads = cfg.temporal_heads;
    kv_config_.max_kv_len = cfg.temporal_max_kv_len;
    kv_config_.dim_per_head = cfg.temporal_dim_per_head;

    sampler_ = std::make_unique<TokenSampler>(
        cfg.vocab_size, cfg.codebook_size);

    // 预分配工作缓冲区
    temporal_input_.resize(cfg.temporal_dim, 0.0f);
    temporal_output_.resize(cfg.temporal_dim, 0.0f);
    depth_input_.resize(cfg.num_codebooks * cfg.temporal_dim, 0.0f);
    depth_logits_.resize(cfg.num_codebooks * cfg.vocab_size, 0.0f);
    current_tokens_.resize(cfg.num_codebooks, 0.0f);
    rvq_embedding_.resize(cfg.ss_embedding_dim, 0.0f);
    rvq_embeddings_.resize(cfg.ss_rvq_levels * cfg.ss_embedding_dim, 0.0f);
    decoder_input_.resize(256 * cfg.ss_embedding_dim, 0.0f);  // max ~10s

    pcm_buffer_.resize(cfg.samples_per_frame * 2, 0.0f);  // stereo
}

inline int InferenceEngine::generate_frame(float* pcm_output) {
    // 1. 准备 Temporal Body 输入 (上一帧的 token embedding)
    //    temporal_input = mean over RVQ of last frame's embeddings

    // 2. Temporal Body (NPU, stateful)
    //    输入: temporal_input + KV caches
    //    输出: temporal_output + updated KV caches
    int ret = step_temporal_body();
    if (ret != 0) return -1;

    // 3. Depth Body AR (NPU/CPU)
    //    12 步 cascade: 逐 RVQ 层自回归
    ret = step_depth_body_ar();
    if (ret != 0) return -2;

    // 4. 采样
    //    for each RVQ level, sample from logits
    //    tokens stored in current_tokens_

    // 5. Token → Embedding
    //    tokens_to_embedding(current_tokens_, cfg_.num_codebooks,
    //                        rvq_embedding_.data());

    // 6. 积累 decoder 输入
    //    每 25 帧 (= 1 秒) 运行一次 codec decoder

    // 7. Codec Decoder (NPU) + iSTFT (CPU)
    //    step_codec_decoder();
    //    step_istft();

    // 8. 输出 PCM
    //    std::memcpy(pcm_output, pcm_buffer_.data(),
    //                cfg_.samples_per_frame * 2 * sizeof(float));

    return cfg_.samples_per_frame;
}

inline int InferenceEngine::step_temporal_body() {
    // 构建输入: [temporal_input, cond, self_k_0, self_v_0, ..., self_k_11, self_v_11]
    std::vector<float*> inputs;
    inputs.push_back(temporal_input_.data());
    inputs.push_back(conditioning_.data());

    for (int i = 0; i < cfg_.temporal_layers; i++) {
        inputs.push_back((float*)kv_cache_->layer(i).self_k_data());
        inputs.push_back((float*)kv_cache_->layer(i).self_v_data());
    }

    // 构建输出 (指向预分配 buffer)
    std::vector<float*> outputs;
    outputs.push_back(temporal_output_.data());
    for (int i = 0; i < cfg_.temporal_layers; i++) {
        outputs.push_back((float*)kv_cache_->layer(i).self_k_data());
        outputs.push_back((float*)kv_cache_->layer(i).self_v_data());
    }

    return temporal_model_->run(inputs, outputs);
}

inline int InferenceEngine::step_depth_body_ar() {
    // 构建 depth_input: [1, num_codebooks, temporal_dim]
    // depth_input[rvq, :] = temporal_output  (初始)
    // 逐 RVQ 层: 更新 depth_input[rvq, :] = embed(sampled_token[rvq])

    std::vector<float*> inputs = {depth_input_.data()};
    std::vector<float*> outputs = {depth_logits_.data()};

    return depth_model_->run(inputs, outputs);
}

inline int InferenceEngine::step_codec_decoder() {
    // 输入: rvq_embeddings_ [T, ss_embedding_dim]
    // 输出: stft_output_ [channels, bins, T*stride]
    std::vector<float*> inputs = {decoder_input_.data()};
    std::vector<float*> outputs = {stft_output_.data()};

    return codec_model_->run(inputs, outputs);
}

inline int InferenceEngine::step_istft() {
    // CPU 端 iSTFT (使用 Signalsmith DSP)
    // 输入: stft_output_ [channels, bins, frames]
    // 输出: pcm_buffer_ [samples * 2]
    //
    // TODO: 集成 signalsmith-stft
    // signalsmith::stft::STFT stft(cfg_.stft_fft_length, cfg_.stft_frame_step);
    // stft.istft(stft_output_.data(), pcm_buffer_.data());
    return 0;
}

inline void InferenceEngine::tokens_to_embedding(
    const int* tokens, int num_rvq, float* /*embedding*/) {
    // RVQ Embedding lookup + sum
    // for each rvq level:
    //     lookup codebook[rvq_level][token[rvq]]
    //     accumulate
    // Note: 此操作放 CPU (RKNN Gather 有限制)
}

}  // namespace rkmrt2
