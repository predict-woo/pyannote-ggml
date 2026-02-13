#include "../src/audio_buffer.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& name, const std::string& detail) {
    std::cerr << "FAIL: " << name << " - " << detail << std::endl;
    std::exit(1);
}

void pass(const std::string& name) {
    std::cout << "PASS: " << name << std::endl;
}

void expect_true(bool condition, const std::string& name, const std::string& detail) {
    if (!condition) {
        fail(name, detail);
    }
}

}

int main() {
    AudioBuffer buffer;

    std::vector<float> samples(48000);
    for (int i = 0; i < 48000; ++i) {
        samples[i] = static_cast<float>(i);
    }

    buffer.enqueue(samples.data(), static_cast<int>(samples.size()));
    expect_true(buffer.total_frames() == 48000, "Test 1", "total_frames != 48000");
    expect_true(buffer.dequeued_frames() == 0, "Test 1", "dequeued_frames != 0");
    pass("Test 1");

    buffer.dequeue_up_to(16000);
    expect_true(buffer.dequeued_frames() == 16000, "Test 2", "dequeued_frames != 16000");
    expect_true(buffer.size() == 32000, "Test 2", "size != 32000");
    pass("Test 2");

    std::vector<float> out;
    buffer.read_range(16000, 32000, out);
    expect_true(out.size() == 16000, "Test 3", "read size != 16000");
    for (int i = 0; i < 16000; ++i) {
        if (out[i] != static_cast<float>(16000 + i)) {
            fail("Test 3", "read_range sample mismatch at index " + std::to_string(i));
        }
    }
    pass("Test 3");

    std::vector<float> extra1(4000, 1.0f);
    buffer.enqueue(extra1.data(), static_cast<int>(extra1.size()));
    expect_true(buffer.total_frames() == 52000, "Test 4", "total_frames mismatch after enqueue");
    buffer.dequeue_up_to(20000);
    expect_true(buffer.dequeued_frames() == 20000, "Test 4", "dequeued_frames mismatch after dequeue");
    std::vector<float> extra2(1000, 2.0f);
    buffer.enqueue(extra2.data(), static_cast<int>(extra2.size()));
    expect_true(buffer.total_frames() == 53000, "Test 4", "total_frames mismatch after second enqueue");
    expect_true(buffer.size() == 33000, "Test 4", "size mismatch after cycles");
    pass("Test 4");

    buffer.dequeue_up_to(999999);
    expect_true(buffer.size() == 0, "Test 5", "size not clamped to zero");
    expect_true(buffer.dequeued_frames() == 53000, "Test 5", "dequeued_frames incorrect after clamp");
    expect_true(buffer.total_frames() == 53000, "Test 5", "total_frames incorrect after clamp");
    pass("Test 5");

    out.assign(4, 42.0f);
    buffer.read_range(53000, 54000, out);
    expect_true(out.empty(), "Test 6", "read_range on empty buffer should be empty");
    pass("Test 6");

    std::cout << "All tests passed." << std::endl;
    return 0;
}
