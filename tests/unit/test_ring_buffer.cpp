#include <catch2/catch_test_macros.hpp>
#include <string>

#include "llm_trace/ring_buffer.hpp"

using llm_trace::RingBuffer;

TEST_CASE("ring buffer rejects zero capacity") {
  CHECK_THROWS_AS(RingBuffer<int>(0), std::invalid_argument);
}

TEST_CASE("empty ring buffer") {
  RingBuffer<int> buffer(3);
  CHECK(buffer.empty());
  CHECK(buffer.size() == 0);
  CHECK(buffer.snapshot().empty());
  CHECK_THROWS_AS(buffer.at(0), std::out_of_range);
}

TEST_CASE("ring buffer keeps insertion order before wraparound") {
  RingBuffer<int> buffer(3);
  CHECK_FALSE(buffer.push(1));
  CHECK_FALSE(buffer.push(2));
  CHECK(buffer.size() == 2);
  CHECK(buffer.at(0) == 1);
  CHECK(buffer.at(1) == 2);
  CHECK(buffer.overwritten() == 0);
}

TEST_CASE("ring buffer overwrites the oldest element when full") {
  RingBuffer<int> buffer(3);
  for (int i = 1; i <= 3; ++i) buffer.push(i);
  CHECK(buffer.full());
  CHECK(buffer.push(4));
  CHECK(buffer.push(5));
  CHECK(buffer.size() == 3);
  CHECK(buffer.snapshot() == std::vector<int>{3, 4, 5});
  CHECK(buffer.overwritten() == 2);
  CHECK(buffer.pushed() == 5);
}

TEST_CASE("ring buffer stays bounded across many wraparounds") {
  RingBuffer<std::string> buffer(7);
  for (int i = 0; i < 1000; ++i) buffer.push(std::to_string(i));
  CHECK(buffer.size() == 7);
  CHECK(buffer.capacity() == 7);
  CHECK(buffer.overwritten() == 993);
  for (std::size_t i = 0; i < buffer.size(); ++i)
    CHECK(buffer.at(i) == std::to_string(993 + static_cast<int>(i)));
}

TEST_CASE("ring buffer clear keeps counters") {
  RingBuffer<int> buffer(2);
  for (int i = 0; i < 5; ++i) buffer.push(i);
  buffer.clear();
  CHECK(buffer.empty());
  CHECK(buffer.overwritten() == 3);
  buffer.push(9);
  CHECK(buffer.at(0) == 9);
}
