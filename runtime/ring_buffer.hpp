/**
 * Lock-free Single-Producer / Single-Consumer Ring Buffer
 *
 * 借鉴 MRT2 MLX C++ engine (Apache 2.0), 适配 RK3576 音频输出管线。
 *
 * 推理线程 (producer) 以 25Hz 固定写入 kFrameSamples 个样本;
 * 音频回调 (consumer) 以硬件速率读取变长块。
 *
 * Thread safety: 恰好 1 producer + 1 consumer. 多 producer/consumer 未定义行为.
 * reset() 必须在两线程都暂停时调用.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>

namespace rkmrt2 {

class RingBuffer {
public:
    /// 最大缓冲样本数 (~170ms @ 48kHz stereo, 4 帧余量). 必须是 2 的幂.
    static constexpr size_t kCapacity = 8192;

    RingBuffer() : buffer_{}, write_pos_(0), read_pos_(0), virtual_capacity_(4096) {}

    /// 调低有效容量以减少延迟, 调高以增加抗抖动能力.
    void set_virtual_capacity(size_t cap) {
        if (cap > kCapacity) cap = kCapacity;
        virtual_capacity_.store(cap, std::memory_order_relaxed);
    }

    size_t get_virtual_capacity() const {
        return virtual_capacity_.load(std::memory_order_relaxed);
    }

    /// Consumer 当前可读样本数.
    size_t available() const {
        return write_pos_.load(std::memory_order_acquire) -
               read_pos_.load(std::memory_order_relaxed);
    }

    /// Producer 可不阻塞写入的样本数.
    size_t free_space() const {
        size_t avail = available();
        size_t cap = get_virtual_capacity();
        return (cap > avail) ? (cap - avail) : 0;
    }

    /// Producer-only. 写入 count 个样本. 空间不足时返回 false (不部分写入).
    bool write(const float* data, size_t count) {
        if (free_space() < count) return false;
        size_t pos = write_pos_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < count; ++i) {
            buffer_[(pos + i) & (kCapacity - 1)] = data[i];
        }
        write_pos_.store(pos + count, std::memory_order_release);
        return true;
    }

    /// Consumer-only. 读取 count 个样本到 dest. 欠载部分补零.
    /// 返回 false 表示发生了欠载 (掉帧).
    bool read(float* dest, size_t count) {
        size_t avail = available();
        size_t to_read = (avail < count) ? avail : count;

        size_t pos = read_pos_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < to_read; ++i) {
            dest[i] = buffer_[(pos + i) & (kCapacity - 1)];
        }
        for (size_t i = to_read; i < count; ++i) {
            dest[i] = 0.0f;
        }
        read_pos_.store(pos + to_read, std::memory_order_release);

        return to_read == count;
    }

    /// Consumer-only. 跳过所有已缓冲样本.
    void drain() {
        read_pos_.store(write_pos_.load(std::memory_order_acquire),
                        std::memory_order_release);
    }

    /// 硬重置游标. 仅在两线程都暂停时调用.
    void reset() {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

private:
    float buffer_[kCapacity];
    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
    std::atomic<size_t> virtual_capacity_;
};

}  // namespace rkmrt2
