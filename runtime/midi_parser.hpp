/**
 * MIDI Parser: MIDI → 128-dim Pianoroll Conditioning
 *
 * 将 MIDI 事件转换为 25Hz 帧对齐的 multihot pianoroll,
 * 作为 DepthFormer Encoder 的输入。
 *
 * 格式: [T_cond, 128] float32, 每帧 128 个 MIDI 音高
 *   值 0.0 = 音符关闭, 1.0 = 音符打开
 *   onset masking: 仅音符开始时为 1.0, 持续帧为 0.0
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <map>

namespace rkmrt2 {

struct MIDIEvent {
    int frame;       // 帧索引 (25Hz)
    int pitch;       // MIDI 音高 (0-127)
    int velocity;    // 力度 (0-127, 0 = note off)
    bool is_note_on() const { return velocity > 0; }
};

class MIDIParser {
public:
    MIDIParser(float frame_rate = 25.0f, int max_frames = 512)
        : frame_rate_(frame_rate), max_frames_(max_frames)
    {
        pianoroll_.resize(max_frames * 128, 0.0f);
    }

    // 从 MIDI 文件路径加载
    bool load(const char* midi_path);

    // 从 MIDI 事件列表加载
    void load_events(const std::vector<MIDIEvent>& events);

    // 获取指定帧的 pianoroll
    const float* frame(int t) const {
        return pianoroll_.data() + t * 128;
    }

    // 获取完整 pianoroll buffer [max_frames, 128]
    const float* data() const { return pianoroll_.data(); }
    int num_frames() const { return max_frames_; }

    // 实时模式: 更新当前帧
    void set_note(int pitch, float velocity) {
        if (pitch >= 0 && pitch < 128) {
            int offset = current_frame_ * 128 + pitch;
            if (offset < (int)pianoroll_.size()) {
                pianoroll_[offset] = velocity;
            }
        }
    }

    void advance_frame() { current_frame_++; }

    int current_frame() const { return current_frame_; }
    void reset() {
        current_frame_ = 0;
        std::memset(pianoroll_.data(), 0,
                     pianoroll_.size() * sizeof(float));
    }

private:
    float frame_rate_;
    int max_frames_;
    int current_frame_ = 0;
    std::vector<float> pianoroll_;  // [max_frames, 128]
};

// 简化实现 (实际需集成 MIDI 解析库如 midifile)
inline bool MIDIParser::load(const char* /*midi_path*/) {
    // TODO: 集成 MIDI 文件解析
    // 使用 tiny MIDI parser 或 fluidsynth MIDI 解析
    return false;
}

inline void MIDIParser::load_events(
    const std::vector<MIDIEvent>& events) {
    reset();
    for (const auto& ev : events) {
        if (ev.frame < max_frames_ && ev.pitch < 128) {
            int offset = ev.frame * 128 + ev.pitch;
            pianoroll_[offset] = ev.velocity / 127.0f;
        }
    }
}

}  // namespace rkmrt2
