#pragma once

// Fixed-capacity ring buffer with an overwrite-oldest policy.
// Not thread-safe; the Collector serialises access.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llm_trace {

template <typename T>
class RingBuffer {
 public:
  explicit RingBuffer(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("ring buffer capacity must be > 0");
    slots_.reserve(capacity);
  }

  // Returns true if an older element was overwritten.
  bool push(T value) {
    if (slots_.size() < capacity_) {
      slots_.push_back(std::move(value));
      ++pushed_;
      return false;
    }
    slots_[head_] = std::move(value);
    head_ = (head_ + 1) % capacity_;
    ++pushed_;
    ++overwritten_;
    return true;
  }

  // Index 0 is the oldest retained element.
  const T& at(std::size_t index) const {
    if (index >= slots_.size()) throw std::out_of_range("ring buffer index");
    return slots_[(head_ + index) % slots_.size()];
  }

  std::size_t size() const { return slots_.size(); }
  std::size_t capacity() const { return capacity_; }
  bool empty() const { return slots_.empty(); }
  bool full() const { return slots_.size() == capacity_; }
  std::uint64_t pushed() const { return pushed_; }
  std::uint64_t overwritten() const { return overwritten_; }

  std::vector<T> snapshot() const {
    std::vector<T> out;
    out.reserve(slots_.size());
    for (std::size_t i = 0; i < slots_.size(); ++i) out.push_back(at(i));
    return out;
  }

  void clear() {
    slots_.clear();
    head_ = 0;
  }

 private:
  std::size_t capacity_;
  std::size_t head_ = 0;  // position of the oldest element once full
  std::vector<T> slots_;
  std::uint64_t pushed_ = 0;
  std::uint64_t overwritten_ = 0;
};

}  // namespace llm_trace
