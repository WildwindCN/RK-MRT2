/**
 * RKNN Model RAII Wrapper
 *
 * 封装 RKNPU2 Runtime API, 管理模型生命周期和推理。
 *
 * 依赖: librknnrt.so (RKNPU2 Runtime)
 * 头文件: rknn_api.h (from rknpu2 SDK)
 *
 * 参考: rknpu2/examples/common/rknn_utils.h
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

// RKNPU2 API (从 Rockchip SDK 获取)
// #include "rknn_api.h"

// 前向声明 RKNN 类型 (实际编译时使用 rknn_api.h)
typedef void* rknn_context;
typedef struct { int dtype; float scale; float zp; } rknn_tensor_attr;
typedef struct { float* buf; int size; } rknn_output;

namespace rkmrt2 {

// ═══════════════════════════════════════════════════════
// Tensor 描述
// ═══════════════════════════════════════════════════════

enum class DataType { FP32, FP16, INT8, INT32 };

struct TensorShape {
    int dims[4];  // N, C, H, W
    int rank;
    int size_bytes() const {
        int n = 1;
        for (int i = 0; i < rank; i++) n *= dims[i];
        return n * elem_size();
    }
    int elem_size() const;
};

// ═══════════════════════════════════════════════════════
// RAII Wrapper
// ═══════════════════════════════════════════════════════

class RKNNModel {
public:
    RKNNModel(const std::string& model_path, int max_batch = 1);
    ~RKNNModel();

    // 禁止拷贝
    RKNNModel(const RKNNModel&) = delete;
    RKNNModel& operator=(const RKNNModel&) = delete;

    // 移动
    RKNNModel(RKNNModel&& other) noexcept;
    RKNNModel& operator=(RKNNModel&& other) noexcept;

    // 查询
    int num_inputs() const { return input_attrs_.size(); }
    int num_outputs() const { return output_attrs_.size(); }
    const TensorShape& input_shape(int idx) const;
    const TensorShape& output_shape(int idx) const;

    // 推理 (所有输入/输出预分配)
    int run(const std::vector<float*>& inputs, std::vector<float*>& outputs);

    // 查询状态
    bool is_loaded() const { return initialized_; }

    // NPU 保活: 微小推理防止时钟降频
    void keep_alive();

    // 获取可写的输入/输出 buffer
    float* input_buffer(int idx);
    const float* output_buffer(int idx) const;

private:
    void init_context(const std::string& model_path);
    void query_io();

    rknn_context ctx_ = nullptr;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<float*> input_bufs_;
    std::vector<float*> output_bufs_;
    bool initialized_ = false;
};

// ═══════════════════════════════════════════════════════
// 实现 (简化版, 实际需链接 librknnrt.so)
// ═══════════════════════════════════════════════════════

inline RKNNModel::RKNNModel(const std::string& model_path, int /*max_batch*/) {
    // 实际实现:
    // rknn_init(&ctx_, model_data, model_size, 0, nullptr);
    // query_io();
    // 预分配输入输出 buffer
}

inline RKNNModel::~RKNNModel() {
    if (initialized_) {
        // rknn_destroy(ctx_);
    }
}

inline int RKNNModel::run(const std::vector<float*>& inputs,
                           std::vector<float*>& outputs) {
    // 实际实现:
    // 1. 拷贝输入到预分配 buffer
    // 2. rknn_run(ctx_, nullptr)
    // 3. 拷贝输出
    return 0;
}

inline void RKNNModel::keep_alive() {
    // NPU 保活: 跑微小推理防止时钟降频
    // 借鉴 MRT2 MLX engine: mx::array({0.0f}) + mx::array({0.0f})
    // RK3576 NPU 空闲超时后会降频, 导致延迟增加 ~5-10ms
    // 跑一个微小的推理操作保持 NPU 活跃
    if (initialized_ && !input_bufs_.empty()) {
        std::vector<float*> outputs(output_bufs_.size());
        for (size_t i = 0; i < output_bufs_.size(); ++i)
            outputs[i] = output_bufs_[i];
        run(input_bufs_, outputs);
    }
}

}  // namespace rkmrt2
