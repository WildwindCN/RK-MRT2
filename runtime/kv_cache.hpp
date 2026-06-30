/**
 * KV Cache Ring Buffer + Attention Mask Manager
 *
 * 管理 Temporal Body 12 层的 Self-Attention KV Cache。
 * Ring buffer 维护完整历史 (capacity=512)，每帧提取最近 42 位置窗口送入 NPU。
 *
 * 内存模型:
 *   Ring buffer (CPU): 12 × 8 × 512 × 128 × 2 bytes ≈ 12.6 MB FP16 per K/V → 25 MB total
 *   NPU window:        12 × 8 × 42 × 128 × 2 bytes  ≈ 2.1 MB FP16 per K/V → 4.2 MB I/O
 *
 * Cross-attention 在 NPU 内部从 cond 重计算，CPU 不缓存。
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

namespace rkmrt2 {

struct KVCacheConfig {
    int num_layers = 12;
    int num_heads = 8;
    int dim_per_head = 128;
    int ring_capacity = 512;      // CPU ring buffer 总容量
    int window_size = 42;         // NPU 固定窗口 (41 window + 1 sink)
    int num_sinks = 1;            // Attention sink 数量
    int cond_len = 50;            // MIDI conditioning 序列长度
    int dtype_size = 2;           // FP16 = 2 bytes
};

/**
 * 单层 Self-Attention KV Ring Buffer
 */
class LayerKVCache {
public:
    LayerKVCache(const KVCacheConfig& cfg)
        : num_heads_(cfg.num_heads), capacity_(cfg.ring_capacity),
          dim_per_head_(cfg.dim_per_head), window_size_(cfg.window_size),
          byte_per_head_(cfg.dim_per_head * cfg.dtype_size)
    {
        size_t ring_size = num_heads_ * capacity_ * byte_per_head_;
        ring_k_.resize(ring_size, 0);
        ring_v_.resize(ring_size, 0);

        // NPU window buffer (contiguous, for RKNN I/O)
        size_t window_size_bytes = num_heads_ * window_size_ * byte_per_head_;
        window_k_.resize(window_size_bytes, 0);
        window_v_.resize(window_size_bytes, 0);

        write_pos_ = 0;
        cur_len_ = 0;
    }

    /** 写入当前帧的 K/V 到 ring buffer */
    void write_self_kv(const void* k, const void* v) {
        size_t offset = num_heads_ * write_pos_ * byte_per_head_;
        size_t size = num_heads_ * byte_per_head_;
        std::memcpy(ring_k_.data() + offset, k, size);
        std::memcpy(ring_v_.data() + offset, v, size);
        write_pos_ = (write_pos_ + 1) % capacity_;
        if (cur_len_ < capacity_) cur_len_++;
    }

    /**
     * 提取最近 window_size 个位置到 NPU 输入 buffer。
     * 如果 cur_len < window_size，前面用 0 填充。
     * 返回指向 window buffer 的指针。
     */
    void extract_window() {
        size_t head_bytes = num_heads_ * byte_per_head_;
        size_t window_bytes = num_heads_ * window_size_ * byte_per_head_;

        int valid = std::min(cur_len_, window_size_);
        int pad = window_size_ - valid;

        // 前置零填充 (早期帧)
        if (pad > 0) {
            std::memset(window_k_.data(), 0, pad * head_bytes);
            std::memset(window_v_.data(), 0, pad * head_bytes);
        }

        // 复制最近 valid 个位置 (从 ring buffer 按序提取)
        for (int i = 0; i < valid; i++) {
            int ring_idx = (write_pos_ - valid + i + capacity_) % capacity_;
            size_t ring_offset = ring_idx * head_bytes;
            size_t win_offset = (pad + i) * head_bytes;
            std::memcpy(window_k_.data() + win_offset, ring_k_.data() + ring_offset, head_bytes);
            std::memcpy(window_v_.data() + win_offset, ring_v_.data() + ring_offset, head_bytes);
        }
    }

    /** 从 NPU 输出合并: 提取最新的 K/V 位置写入 ring buffer */
    void merge_window(const void* updated_k, const void* updated_v) {
        // NPU 输出 window 的最后 window_size 个位置。最后一个即当前帧的 K/V。
        size_t head_bytes = num_heads_ * byte_per_head_;
        size_t last_offset = (window_size_ - 1) * head_bytes;

        const uint8_t* uk = static_cast<const uint8_t*>(updated_k);
        const uint8_t* uv = static_cast<const uint8_t*>(updated_v);

        write_self_kv(uk + last_offset, uv + last_offset);
    }

    const void* window_k() const { return window_k_.data(); }
    const void* window_v() const { return window_v_.data(); }
    size_t window_bytes() const { return window_k_.size(); }

    int current_length() const { return cur_len_; }
    void reset() { write_pos_ = 0; cur_len_ = 0;
        std::memset(ring_k_.data(), 0, ring_k_.size());
        std::memset(ring_v_.data(), 0, ring_v_.size()); }

private:
    int num_heads_, capacity_, dim_per_head_, window_size_, byte_per_head_;
    int write_pos_ = 0, cur_len_ = 0;
    std::vector<uint8_t> ring_k_, ring_v_;     // CPU ring buffer
    std::vector<uint8_t> window_k_, window_v_; // NPU I/O buffer
};

/**
 * Attention Mask 预计算
 *
 * Mask 形状: [1, 1, 1, total_k] where total_k = window_size + 1 + num_sinks = 44
 *
 * 掩码逻辑:
 *   - Position 0 (sink): 总是 0.0 (可见)
 *   - Positions 1..(total_k - 1 - valid_positions): -inf (padding)
 *   - 剩余位置 (valid_positions 个): 0.0 (可见)
 *   - 如果总有效位置 > window + sink (42): 最老的超出窗口的位置也被 mask
 */
class AttentionMask {
public:
    AttentionMask(int window_size, int num_sinks)
        : window_size_(window_size), num_sinks_(num_sinks),
          total_k_(window_size + 1 + num_sinks)  // 42 + 1 + 1 = 44
    {
        mask_.resize(total_k_, 0.0f);
    }

    /** 根据当前有效帧数更新 mask */
    void update(int frame_index) {
        int valid_cache = std::min(frame_index + 1, window_size_);
        int valid_total = num_sinks_ + valid_cache + 1;  // sink + cache + current

        std::fill(mask_.begin(), mask_.end(), 0.0f);

        // Padding: 尚未填充的 KV cache 位置
        int padded = window_size_ - valid_cache;
        if (padded > 0) {
            // Padded positions are right after the sink
            std::fill(mask_.begin() + num_sinks_,
                      mask_.begin() + num_sinks_ + padded,
                      -INFINITY);
        }

        // Sliding window: 如果总有效位置超出窗口, 屏蔽最老的
        int exceed = valid_total - (window_size_ + num_sinks_);
        if (exceed > 0) {
            // 最老的 exceed 个有效位置需要 mask (它们在 padding 之后)
            int start = num_sinks_ + padded;
            std::fill(mask_.begin() + start,
                      mask_.begin() + start + exceed,
                      -INFINITY);
        }
    }

    const float* data() const { return mask_.data(); }
    int size() const { return total_k_; }

private:
    int window_size_, num_sinks_, total_k_;
    std::vector<float> mask_;
};

/**
 * 完整 KV Cache Manager (12 层 + attention mask)
 */
class KVCacheManager {
public:
    KVCacheManager(const KVCacheConfig& cfg)
        : cfg_(cfg), attn_mask_(cfg.window_size, cfg.num_sinks)
    {
        for (int i = 0; i < cfg.num_layers; i++) {
            layers_.emplace_back(cfg);
        }
    }

    /** 准备当前帧的 NPU 输入: 提取 window + 更新 mask */
    void prepare_frame(int frame_index) {
        for (auto& layer : layers_) {
            layer.extract_window();
        }
        attn_mask_.update(frame_index);
    }

    /** NPU 推理后合并: 提取当前帧的 K/V 写回 ring buffer */
    void merge_frame(int layer, const void* updated_k, const void* updated_v) {
        layers_[layer].merge_window(updated_k, updated_v);
    }

    LayerKVCache& layer(int i) { return layers_[i]; }
    const LayerKVCache& layer(int i) const { return layers_[i]; }
    const AttentionMask& mask() const { return attn_mask_; }

    void reset() {
        for (auto& l : layers_) l.reset();
    }

    int num_layers() const { return cfg_.num_layers; }
    int window_size() const { return cfg_.window_size; }

private:
    KVCacheConfig cfg_;
    std::vector<LayerKVCache> layers_;
    AttentionMask attn_mask_;
};

}  // namespace rkmrt2
