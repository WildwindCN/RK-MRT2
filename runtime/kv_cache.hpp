/**
 * KV Cache Ring Buffer Manager
 *
 * 管理 Temporal Body 12 层 × 2 种 attention 的 KV Cache。
 * 使用 ring buffer 避免动态内存分配, 适配 RKNN 的静态图要求。
 *
 * 架构:
 *   每层维护:
 *     - self_attn_k: [num_heads, max_kv_len, dim_per_head] FP16
 *     - self_attn_v: [num_heads, max_kv_len, dim_per_head] FP16
 *     - cross_attn_k: [num_heads, T_cond, dim_per_head] FP16 (预计算, 不变)
 *     - cross_attn_v: [num_heads, T_cond, dim_per_head] FP16 (预计算, 不变)
 *
 * Self-attn KV 使用 ring buffer:
 *   每次写入 position = step % max_kv_len
 *   Attention 读取最近 max_kv_len 个位置 (滑动窗口)
 *
 * Cross-attn KV 从 MIDI conditioning 预计算, 不更新。
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <algorithm>

namespace rkmrt2 {

struct KVCacheConfig {
    int num_layers = 12;
    int num_heads = 8;
    int max_kv_len = 512;       // 滑动窗口大小 (41 帧 + 余量)
    int dim_per_head = 128;
    int cond_len = 50;          // MIDI conditioning 序列长度
    // 数据类型: 0=FP32, 1=FP16
    int dtype_size = 2;         // FP16 = 2 bytes
};

/**
 * 单层 KV Cache
 */
class LayerKVCache {
public:
    LayerKVCache(const KVCacheConfig& cfg)
        : num_heads_(cfg.num_heads)
        , max_len_(cfg.max_kv_len)
        , dim_per_head_(cfg.dim_per_head)
        , cond_len_(cfg.cond_len)
        , byte_per_head_(cfg.dim_per_head * cfg.dtype_size)
    {
        // Self-attn K/V ring buffer
        size_t self_buf_size = num_heads_ * max_len_ * byte_per_head_;
        self_k_.resize(self_buf_size, 0);
        self_v_.resize(self_buf_size, 0);

        // Cross-attn K/V (预计算, 固定)
        size_t cross_buf_size = num_heads_ * cond_len_ * byte_per_head_;
        cross_k_.resize(cross_buf_size, 0);
        cross_v_.resize(cross_buf_size, 0);

        write_pos_ = 0;
        cur_len_ = 0;
    }

    // 写入新的 self-attn K/V (在当前位置)
    void write_self_kv(const void* k, const void* v) {
        size_t offset = num_heads_ * write_pos_ * byte_per_head_;
        std::memcpy(self_k_.data() + offset, k,
                    num_heads_ * byte_per_head_);
        std::memcpy(self_v_.data() + offset, v,
                    num_heads_ * byte_per_head_);

        write_pos_ = (write_pos_ + 1) % max_len_;
        if (cur_len_ < max_len_) cur_len_++;
    }

    // 获取 self-attn K buffer (用于 ONNX 输入)
    const void* self_k_data() const { return self_k_.data(); }
    const void* self_v_data() const { return self_v_.data(); }

    // 设置 cross-attn K/V (从 MIDI conditioning 预计算)
    void set_cross_kv(const void* k, const void* v) {
        size_t size = num_heads_ * cond_len_ * byte_per_head_;
        std::memcpy(cross_k_.data(), k, size);
        std::memcpy(cross_v_.data(), v, size);
    }

    const void* cross_k_data() const { return cross_k_.data(); }
    const void* cross_v_data() const { return cross_v_.data(); }

    int current_length() const { return cur_len_; }
    int write_position() const { return write_pos_; }

private:
    int num_heads_, max_len_, dim_per_head_, cond_len_;
    int byte_per_head_;
    int write_pos_ = 0;
    int cur_len_ = 0;

    std::vector<uint8_t> self_k_;
    std::vector<uint8_t> self_v_;
    std::vector<uint8_t> cross_k_;
    std::vector<uint8_t> cross_v_;
};

/**
 * 完整 KV Cache Manager (12 层)
 */
class KVCacheManager {
public:
    KVCacheManager(const KVCacheConfig& cfg) : cfg_(cfg) {
        for (int i = 0; i < cfg.num_layers; i++) {
            layers_.emplace_back(cfg);
        }
    }

    // 写入当前帧的 self-attn K/V (更新 ring buffer)
    void update_self_kv(int layer, const void* k, const void* v) {
        layers_[layer].write_self_kv(k, v);
    }

    // 设置 cross-attn K/V (在每段生成开始时调用)
    void set_cross_kv(int layer, const void* k, const void* v) {
        layers_[layer].set_cross_kv(k, v);
    }

    // 获取各层 buffer 指针 (用于填充 RKNN 输入)
    LayerKVCache& layer(int i) { return layers_[i]; }
    const LayerKVCache& layer(int i) const { return layers_[i]; }

    // 重置所有 self-attn KV cache (新生成开始时)
    void reset_self() {
        for (auto& l : layers_) {
            l = LayerKVCache(cfg_);
        }
    }

    int num_layers() const { return cfg_.num_layers; }
    int max_len() const { return cfg_.max_kv_len; }

private:
    KVCacheConfig cfg_;
    std::vector<LayerKVCache> layers_;
};

}  // namespace rkmrt2
