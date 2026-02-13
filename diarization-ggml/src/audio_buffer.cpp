#include "audio_buffer.h"

#include <algorithm>

AudioBuffer::AudioBuffer() : dequeued_frames_(0) {}

void AudioBuffer::enqueue(const float* samples, int n) {
    if (samples == nullptr || n <= 0) {
        return;
    }
    buffer_.insert(buffer_.end(), samples, samples + n);
}

void AudioBuffer::dequeue_up_to(int64_t absolute_sample) {
    if (absolute_sample <= dequeued_frames_) {
        return;
    }

    int64_t to_remove = absolute_sample - dequeued_frames_;
    if (to_remove > static_cast<int64_t>(buffer_.size())) {
        to_remove = static_cast<int64_t>(buffer_.size());
    }

    buffer_.erase(buffer_.begin(), buffer_.begin() + to_remove);
    dequeued_frames_ += to_remove;
}

void AudioBuffer::read_range(int64_t abs_start, int64_t abs_end, std::vector<float>& out) const {
    out.clear();
    if (abs_end <= abs_start) {
        return;
    }

    int64_t local_start = abs_start - dequeued_frames_;
    int64_t local_end = abs_end - dequeued_frames_;
    local_start = std::max<int64_t>(0, local_start);
    local_end = std::min<int64_t>(static_cast<int64_t>(buffer_.size()), local_end);

    if (local_end <= local_start) {
        return;
    }

    const size_t count = static_cast<size_t>(local_end - local_start);
    out.resize(count);
    std::copy(
        buffer_.begin() + local_start,
        buffer_.begin() + local_end,
        out.begin());
}

int64_t AudioBuffer::dequeued_frames() const {
    return dequeued_frames_;
}

int64_t AudioBuffer::total_frames() const {
    return dequeued_frames_ + static_cast<int64_t>(buffer_.size());
}

int AudioBuffer::size() const {
    return static_cast<int>(buffer_.size());
}

const float* AudioBuffer::data() const {
    return buffer_.data();
}
