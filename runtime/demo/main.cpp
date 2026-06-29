/**
 * RK-MRT2 推理 Demo
 *
 * 实时 MIDI → 音频生成主循环。
 *
 * 用法:
 *   ./rk_mrt2_demo \
 *     --temporal temporal_body.rknn \
 *     --depth depth_body.rknn \
 *     --codec codec_decoder.rknn \
 *     --midi input.mid \
 *     --output audio.wav \
 *     --temperature 0.8 \
 *     --duration 30
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

#include "inference_engine.hpp"
#include "midi_parser.hpp"

using namespace rkmrt2;

// 全局标志 (信号处理)
static volatile bool g_running = true;

void signal_handler(int) {
    g_running = false;
}

// WAV 文件写入 (简化)
class WAVWriter {
public:
    WAVWriter(const std::string& path, int sample_rate, int channels)
        : sample_rate_(sample_rate), channels_(channels) {
        file_.open(path, std::ios::binary);
        write_header_placeholder();
    }

    ~WAVWriter() {
        finalize_header();
        file_.close();
    }

    void write(const float* samples, int count) {
        for (int i = 0; i < count; i++) {
            // float → int16
            float s = samples[i];
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            int16_t v = static_cast<int16_t>(s * 32767.0f);
            file_.write(reinterpret_cast<const char*>(&v), sizeof(v));
            data_size_ += sizeof(v);
        }
    }

private:
    void write_header_placeholder() {
        // 占位 header (最后更新)
        char header[44] = {0};
        std::memcpy(header, "RIFF", 4);
        std::memcpy(header + 8, "WAVE", 4);
        std::memcpy(header + 12, "fmt ", 4);
        // fmt chunk size = 16
        *reinterpret_cast<int32_t*>(header + 16) = 16;
        // PCM format = 1
        *reinterpret_cast<int16_t*>(header + 20) = 1;
        *reinterpret_cast<int16_t*>(header + 22) = channels_;
        *reinterpret_cast<int32_t*>(header + 24) = sample_rate_;
        *reinterpret_cast<int32_t*>(header + 28) = sample_rate_ * channels_ * 2;
        *reinterpret_cast<int16_t*>(header + 32) = channels_ * 2;
        *reinterpret_cast<int16_t*>(header + 34) = 16;
        std::memcpy(header + 36, "data", 4);
        file_.write(header, 44);
    }

    void finalize_header() {
        file_.seekp(0, std::ios::end);
        int file_size = file_.tellp();
        // RIFF size
        file_.seekp(4);
        int32_t riff_size = file_size - 8;
        file_.write(reinterpret_cast<const char*>(&riff_size), 4);
        // data size
        file_.seekp(40);
        file_.write(reinterpret_cast<const char*>(&data_size_), 4);
    }

    std::ofstream file_;
    int sample_rate_, channels_;
    int data_size_ = 0;
};

int main(int argc, char* argv[]) {
    // 参数解析 (简化)
    std::string temporal_path = "temporal_body.rknn";
    std::string depth_path = "depth_body.rknn";
    std::string codec_path = "codec_decoder.rknn";
    std::string midi_path;
    std::string output_path = "output.wav";
    float temperature = 0.8f;
    int top_k = 50;
    float duration_sec = 30.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--temporal" && i + 1 < argc) temporal_path = argv[++i];
        else if (arg == "--depth" && i + 1 < argc) depth_path = argv[++i];
        else if (arg == "--codec" && i + 1 < argc) codec_path = argv[++i];
        else if (arg == "--midi" && i + 1 < argc) midi_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--top_k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (arg == "--duration" && i + 1 < argc) duration_sec = std::stof(argv[++i]);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║   RK-MRT2 Real-Time Music Generation      ║\n";
    std::cout << "║   Platform: RK3588 NPU (RKNN Runtime)     ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n\n";

    // 初始化模型
    ModelConfig cfg;
    cfg.temporal_max_kv_len = 512;

    std::cout << "[1/4] Loading models...\n";
    InferenceEngine engine(cfg, temporal_path, depth_path, codec_path);
    if (!engine.init()) {
        std::cerr << "Failed to initialize models\n";
        return 1;
    }
    engine.set_temperature(temperature);
    engine.set_top_k(top_k);

    // 加载 MIDI
    std::cout << "[2/4] Loading MIDI...\n";
    MIDIParser midi_parser;
    if (!midi_path.empty()) {
        midi_parser.load(midi_path.c_str());
    }
    // 设置 conditioning (一次性)
    if (midi_parser.num_frames() > 0) {
        engine.set_conditioning(midi_parser.data(), midi_parser.num_frames());
    }

    // 音频输出
    std::cout << "[3/4] Setting up audio output...\n";
    WAVWriter wav(output_path, static_cast<int>(cfg.sample_rate), 2);

    int total_frames_written = 0;
    engine.set_audio_callback([&](const float* pcm, int samples) {
        wav.write(pcm, samples);
        total_frames_written += samples;
    });

    // 推理循环
    std::cout << "[4/4] Starting generation...\n\n";

    int max_frames = static_cast<int>(duration_sec * cfg.frame_rate);
    std::vector<float> pcm_frame(cfg.samples_per_frame * 2);
    auto start_time = std::chrono::steady_clock::now();

    for (int frame = 0; frame < max_frames && g_running; frame++) {
        auto frame_start = std::chrono::steady_clock::now();

        // 生成一帧
        int samples = engine.generate_frame(pcm_frame.data());

        // 输出音频
        wav.write(pcm_frame.data(), samples);

        auto frame_end = std::chrono::steady_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end - frame_start).count();

        // 实时打印
        if (frame % 25 == 0) {  // 每秒一次
            float elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                frame_end - start_time).count();
            float rtf = frame_time / 40000.0f;  // 40ms budget
            std::cout << "\r  Frame " << frame
                      << " | RTF: " << rtf
                      << " | Elapsed: " << elapsed << "s"
                      << " | Audio: " << (total_frames_written / cfg.sample_rate)
                      << "s  " << std::flush;
        }

        // 检查延迟 (超过 40ms 预算时告警)
        if (frame_time > 40000) {
            std::cerr << "\n  [WARN] Frame " << frame
                      << " took " << (frame_time / 1000.0f) << "ms (>40ms budget)\n";
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    float total_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count() / 1000.0f;

    std::cout << "\n\nGeneration complete!\n";
    std::cout << "  Total time: " << total_sec << "s\n";
    std::cout << "  Audio duration: " << (total_frames_written / cfg.sample_rate) << "s\n";
    std::cout << "  Output: " << output_path << "\n";

    return 0;
}
