#pragma once

#include <cstdint>
#include <vector>

class AudioBuffer {
public:
    AudioBuffer();
    void enqueue(const float* samples, int n);
    void dequeue_up_to(int64_t absolute_sample);
    void read_range(int64_t abs_start, int64_t abs_end, std::vector<float>& out) const;
    int64_t dequeued_frames() const;
    int64_t total_frames() const;
    int size() const;
    const float* data() const;

private:
    std::vector<float> buffer_;
    int64_t dequeued_frames_;
};
