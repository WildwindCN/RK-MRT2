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
    // for (int i = 0; i < inputs.size(); i++)
    //     memcpy(input_bufs_[i], inputs[i], input_attrs_[i].size);
    //
    // 2. 运行推理
    // rknn_inputs_set(ctx_, num_inputs_, input_attrs_.data());
    // rknn_run(ctx_, nullptr);
    // rknn_outputs_get(ctx_, num_outputs_, output_attrs_.data(), outputs.data());
    //
    // 3. 拷贝输出
    // for (int i = 0; i < outputs.size(); i++)
    //     memcpy(outputs[i], output_bufs_[i], output_attrs_[i].size);
    return 0;
}

}  // namespace rkmrt2
