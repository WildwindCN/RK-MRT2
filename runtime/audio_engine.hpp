/**
 * Audio Engine: 双线程实时推理 → 音频输出
 *
 * 借鉴 MRT2 MLX RealtimeRunner, 适配 RK3576 NPU 管线.
 *
 * 架构:
 *   Inference Thread (25Hz producer):
 *     每 40ms: generate_frame() → write ring buffer (1920 samples × 2ch)
 *     NPU 保活: 等待 buffer 空间时跑微小推理防降频
 *
 *   Audio Callback (consumer, 硬件速率):
 *     任意块大小 (64/128/256/512 samples) → read ring buffer → 输出
 *
 * 预生成: start() 时先同步生成 3 帧 (120ms) 防止首帧欠载.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>

#include "inference_engine.hpp"
#include "ring_buffer.hpp"

namespace rkmrt2 {

class AudioEngine {
public:
    using AudioCallback = std::function<void(const float* pcm, int samples)>;

    AudioEngine(const ModelConfig& cfg,
                const std::string& temporal_path,
                const std::string& depth_path,
                const std::string& codec_path)
        : engine_(cfg, temporal_path, depth_path, codec_path)
        , ring_buffer_()
        , running_(false)
        , dropped_frames_(0)
    {}

    ~AudioEngine() { stop(); }

    /// 初始化模型, 加载 MIDI conditioning.
    bool init(const float* pianoroll = nullptr, int cond_frames = 0) {
        if (!engine_.init()) return false;
        if (pianoroll && cond_frames > 0) {
            engine_.set_conditioning(pianoroll, cond_frames);
        }
        return true;
    }

    /// 启动实时推理. 先预生成 3 帧, 再启动推理线程.
    void start() {
        if (running_.load(std::memory_order_acquire)) return;
        running_.store(true, std::memory_order_release);

        // 预生成 3 帧 (120ms) 防止首帧欠载
        for (int i = 0; i < 3; ++i) {
            std::vector<float> pcm(kSamplesPerFrame * 2);
            int n = engine_.generate_frame(pcm.data());
            ring_buffer_.write(pcm.data(), n * 2);
        }

        inference_thread_ = std::thread(&AudioEngine::inference_loop, this);
    }

    /// 停止推理线程.
    void stop() {
        running_.store(false, std::memory_order_release);
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

    /// Consumer: 音频回调读取变长块. 返回 false 表示发生欠载.
    bool read_audio(float* dest, size_t count) {
        bool ok = ring_buffer_.read(dest, count);
        if (!ok) {
            dropped_frames_++;
        }
        return ok;
    }

    /// 跳过所有已缓冲音频.
    void drain() { ring_buffer_.drain(); }

    /// 重置生成状态 (换 MIDI / 风格时).
    void reset() {
        stop();
        ring_buffer_.reset();
        dropped_frames_ = 0;
        engine_.reset();
    }

    void set_temperature(float t) { engine_.set_temperature(t); }
    void set_top_k(int k) { engine_.set_top_k(k); }
    void set_conditioning(const float* pr, int n) { engine_.set_conditioning(pr, n); }

    int dropped_frames() const { return dropped_frames_; }
    const RingBuffer& buffer() const { return ring_buffer_; }

    /// 设置延迟 (环形缓冲区 virtual_capacity, 越小延迟越低).
    void set_latency_ms(float ms) {
        size_t samples = static_cast<size_t>(ms / 1000.0f * kSampleRate * 2);
        if (samples < kSamplesPerFrame * 2 * 2) samples = kSamplesPerFrame * 2 * 2;
        ring_buffer_.set_virtual_capacity(samples);
    }

private:
    static constexpr int kSampleRate = 48000;
    static constexpr int kSamplesPerFrame = 1920;  // 40ms @ 48kHz
    static constexpr int kFrameIntervalUs = 40000; // 40ms

    void inference_loop() {
        std::vector<float> pcm(kSamplesPerFrame * 2);  // stereo

        while (running_.load(std::memory_order_acquire)) {
            auto frame_start = std::chrono::steady_clock::now();

            // 生成一帧
            int n = engine_.generate_frame(pcm.data());

            // 等待环形缓冲区有空间 (NPU 保活)
            while (running_.load(std::memory_order_acquire) &&
                   ring_buffer_.free_space() < static_cast<size_t>(n * 2)) {
                engine_.keep_alive();
            }

            if (!running_.load(std::memory_order_acquire)) break;

            // 写入环形缓冲区
            if (!ring_buffer_.write(pcm.data(), n * 2)) {
                dropped_frames_++;
            }

            // 帧率控制: 等待到下一帧时刻
            auto frame_end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                frame_end - frame_start).count();
            if (elapsed < kFrameIntervalUs) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(kFrameIntervalUs - elapsed));
            }
        }
    }

    InferenceEngine engine_;
    RingBuffer ring_buffer_;
    std::thread inference_thread_;
    std::atomic<bool> running_;
    int dropped_frames_;
};

}  // namespace rkmrt2
