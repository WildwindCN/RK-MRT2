/**
 * RK-MRT2 实时推理 Demo
 *
 * 双线程架构 (借鉴 MRT2 MLX C++ engine):
 *   推理线程 (25Hz) → 环形缓冲区 → 音频回调 (硬件速率)
 *
 * 三种输出模式:
 *   --output audio.wav   WAV 文件 (离线)
 *   --output alsa         ALSA 实时 (板端)
 *   --output porta        PortAudio 实时 (跨平台)
 *
 * 用法:
 *   ./rk_mrt2_realtime \
 *     --temporal temporal_body.rknn \
 *     --depth depth_body.rknn \
 *     --codec codec_decoder.rknn \
 *     --midi input.mid \
 *     --output alsa \
 *     --temperature 0.8 \
 *     --duration 30 \
 *     --latency 40
 */

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "audio_engine.hpp"
#include "midi_parser.hpp"

using namespace rkmrt2;

static volatile bool g_running = true;
void signal_handler(int) { g_running = false; }

// WAV 文件写入
class WAVWriter {
public:
    WAVWriter(const std::string& path, int sr, int ch)
        : sample_rate_(sr), channels_(ch) {
        file_.open(path, std::ios::binary);
        write_header_placeholder();
    }
    ~WAVWriter() { finalize_header(); file_.close(); }
    void write(const float* s, int n) {
        for (int i = 0; i < n; i++) {
            float v = s[i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            int16_t w = static_cast<int16_t>(v * 32767.0f);
            file_.write(reinterpret_cast<const char*>(&w), sizeof(w));
        }
    }
private:
    void write_header_placeholder() {
        char h[44] = {0};
        std::memcpy(h, "RIFF", 4); std::memcpy(h + 8, "WAVE", 4);
        std::memcpy(h + 12, "fmt ", 4);
        *reinterpret_cast<int32_t*>(h + 16) = 16;
        *reinterpret_cast<int16_t*>(h + 20) = 1;
        *reinterpret_cast<int16_t*>(h + 22) = channels_;
        *reinterpret_cast<int32_t*>(h + 24) = sample_rate_;
        *reinterpret_cast<int32_t*>(h + 28) = sample_rate_ * channels_ * 2;
        *reinterpret_cast<int16_t*>(h + 32) = channels_ * 2;
        *reinterpret_cast<int16_t*>(h + 34) = 16;
        std::memcpy(h + 36, "data", 4);
        file_.write(h, 44);
    }
    void finalize_header() {
        file_.seekp(0, std::ios::end);
        int sz = static_cast<int>(file_.tellp()) - 8;
        file_.seekp(4); file_.write(reinterpret_cast<const char*>(&sz), 4);
    }
    std::ofstream file_;
    int sample_rate_, channels_;
};

int main(int argc, char* argv[]) {
    std::string temporal_path = "temporal_body.rknn";
    std::string depth_path = "depth_body.rknn";
    std::string codec_path = "codec_decoder.rknn";
    std::string midi_path, output_path = "output.wav";
    float temperature = 0.8f;
    int top_k = 50;
    float duration_sec = 30.0f;
    float latency_ms = 40.0f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--temporal" && i + 1 < argc) temporal_path = argv[++i];
        else if (a == "--depth" && i + 1 < argc) depth_path = argv[++i];
        else if (a == "--codec" && i + 1 < argc) codec_path = argv[++i];
        else if (a == "--midi" && i + 1 < argc) midi_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (a == "--top_k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
        else if (a == "--duration" && i + 1 < argc) duration_sec = std::stof(argv[++i]);
        else if (a == "--latency" && i + 1 < argc) latency_ms = std::stof(argv[++i]);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "=== RK-MRT2 Realtime Engine (RK3576 NPU) ===\n\n";

    // Init
    ModelConfig cfg;
    AudioEngine engine(cfg, temporal_path, depth_path, codec_path);

    std::cout << "[1/4] Loading models...\n";
    if (!engine.init()) {
        std::cerr << "Failed to init models\n";
        return 1;
    }
    engine.set_temperature(temperature);
    engine.set_top_k(top_k);
    engine.set_latency_ms(latency_ms);

    // MIDI
    std::cout << "[2/4] Loading MIDI...\n";
    MIDIParser midi;
    if (!midi_path.empty()) midi.load(midi_path.c_str());
    if (midi.num_frames() > 0)
        engine.set_conditioning(midi.data(), midi.num_frames());

    // Output mode
    std::cout << "[3/4] Setting up output: " << output_path << "\n";
    WAVWriter* wav = nullptr;
    if (output_path.find(".wav") != std::string::npos) {
        wav = new WAVWriter(output_path, 48000, 2);
    }

    // Start
    std::cout << "[4/4] Starting realtime engine (latency=" << latency_ms
              << "ms)...\n\n";
    engine.start();

    auto start_time = std::chrono::steady_clock::now();
    float elapsed = 0.0f;
    std::vector<float> audio_buf(256 * 2);  // stereo read buffer

    while (g_running && elapsed < duration_sec) {
        // Consumer: read from ring buffer at ~audio callback rate
        size_t read = engine.read_audio(audio_buf.data(), 256);
        if (wav) wav->write(audio_buf.data(), read * 2);

        auto now = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time).count() / 1000.0f;

        // Status every 1s
        static int last_report = 0;
        int sec = static_cast<int>(elapsed);
        if (sec > last_report) {
            last_report = sec;
            std::cout << "\r  t=" << sec << "s  dropped="
                      << engine.dropped_frames()
                      << "  buf=" << engine.buffer().available()
                      << " samples  " << std::flush;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    engine.stop();
    delete wav;

    auto end_time = std::chrono::steady_clock::now();
    float total_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count() / 1000.0f;

    std::cout << "\n\nDone!\n";
    std::cout << "  Total: " << total_sec << "s\n";
    std::cout << "  Dropped frames: " << engine.dropped_frames() << "\n";
    if (!midi_path.empty() && output_path.find(".wav") != std::string::npos)
        std::cout << "  Output: " << output_path << "\n";

    return 0;
}
