// Minimal ONNX Runtime C++ sample that runs a model with profiling enabled and
// writes the profile file that `llm-trace --open` can inspect.
//
// It also measures the profiling overhead on the same workload: it times N runs
// in a session with profiling disabled and N runs in a session with profiling
// enabled, after warm-up, and prints both. Nothing is estimated.
//
// With --activations TRACE.jsonl, the model must come from
// scripts/expose_activations.py. Every run of the profiled session then reads the
// exposed intermediate tensors, summarizes their real values, and the profile plus
// those summaries are written as an llm-trace JSONL trace.
//
// Usage:
//   onnx_profile_runner <model.onnx> <profile_prefix> [--runs N] [--warmup N] [--seq L]
//                       [--threads T] [--activations TRACE.jsonl] [--epsilon E]

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "llm_trace/activation.hpp"
#include "llm_trace/onnx_profile_adapter.hpp"
#include "llm_trace/trace_io.hpp"

namespace {

constexpr const char* kExposedKey = "llm_trace.exposed_activations";

struct Options {
  std::string model;
  std::string prefix;
  int runs = 50;
  int warmup = 5;
  std::int64_t seq = 16;
  int threads = 1;
  std::string activations;  // output trace path; empty = off
  double epsilon = 1e-6;
};

struct Capture {
  std::map<std::string, std::string> node_of;  // exposed tensor -> producing node
  std::vector<llm_trace::ActivationRecord> records;
  std::size_t non_float = 0;
  double stats_ms = 0;
};

std::basic_string<ORTCHAR_T> to_ort_path(const std::string& path) {
#ifdef _WIN32
  return std::wstring(path.begin(), path.end());
#else
  return path;
#endif
}

struct Timing {
  double mean_ms = 0;
  double median_ms = 0;
  double min_ms = 0;
};

Timing summarize(std::vector<double> samples) {
  Timing t;
  if (samples.empty()) return t;
  std::sort(samples.begin(), samples.end());
  t.mean_ms =
      std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  t.median_ms = samples[samples.size() / 2];
  t.min_ms = samples.front();
  return t;
}

std::vector<double> run_session(Ort::Session& session, const Options& opt, std::int64_t hidden,
                                Capture* capture = nullptr) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> input(static_cast<std::size_t>(opt.seq * hidden));
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
  const std::int64_t shape[] = {1, opt.seq, hidden};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), shape, 3);
  const char* in_names[] = {"input"};
  // With --activations, both sessions fetch every graph output so their timings
  // describe the same work.
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names{"output"};
  if (!opt.activations.empty()) {
    names.clear();
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
      names.push_back(session.GetOutputNameAllocated(i, allocator).get());
  }
  std::vector<const char*> out_names;
  for (const auto& n : names) out_names.push_back(n.c_str());

  std::vector<double> samples;
  for (int i = 0; i < opt.warmup + opt.runs; ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names.data(),
                               out_names.size());
    const auto stop = std::chrono::steady_clock::now();
    if (i >= opt.warmup)
      samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    if (!capture) continue;
    const auto s0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < names.size(); ++k) {
      const auto node = capture->node_of.find(names[k]);
      if (node == capture->node_of.end()) continue;
      const auto info = outputs[k].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        ++capture->non_float;
        continue;
      }
      llm_trace::ActivationRecord r;
      r.run_index = static_cast<std::size_t>(i);
      r.node = node->second;
      r.tensor = names[k];
      r.summary = llm_trace::summarize_activation(outputs[k].GetTensorData<float>(),
                                                  info.GetElementCount(), opt.epsilon);
      capture->records.push_back(std::move(r));
    }
    capture->stats_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
  }
  return samples;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--runs")
      opt.runs = std::atoi(next());
    else if (arg == "--warmup")
      opt.warmup = std::atoi(next());
    else if (arg == "--seq")
      opt.seq = std::atoll(next());
    else if (arg == "--threads")
      opt.threads = std::atoi(next());
    else if (arg == "--activations")
      opt.activations = next();
    else if (arg == "--epsilon")
      opt.epsilon = std::atof(next());
    else
      positional.push_back(arg);
  }
  if (positional.size() != 2 || opt.runs <= 0 || opt.seq <= 0 || !(opt.epsilon >= 0)) {
    std::fprintf(stderr,
                 "usage: onnx_profile_runner <model.onnx> <profile_prefix> [--runs N] "
                 "[--warmup N] [--seq L] [--threads T] [--activations TRACE.jsonl] "
                 "[--epsilon E]\n");
    return 2;
  }
  opt.model = positional[0];
  opt.prefix = positional[1];

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "llm-trace-runner");
    const std::int64_t hidden = 32;  // matches scripts/make_tiny_model.py

    Ort::SessionOptions plain;
    plain.SetIntraOpNumThreads(opt.threads);
    plain.SetInterOpNumThreads(1);
    Ort::Session plain_session(env, to_ort_path(opt.model).c_str(), plain);
    const Timing off = summarize(run_session(plain_session, opt, hidden));

    Ort::SessionOptions profiled;
    profiled.SetIntraOpNumThreads(opt.threads);
    profiled.SetInterOpNumThreads(1);
    profiled.EnableProfiling(to_ort_path(opt.prefix).c_str());
    Ort::Session profiled_session(env, to_ort_path(opt.model).c_str(), profiled);
    Ort::AllocatorWithDefaultOptions allocator;
    Capture capture;
    if (!opt.activations.empty()) {
      auto value = profiled_session.GetModelMetadata().LookupCustomMetadataMapAllocated(kExposedKey,
                                                                                        allocator);
      if (!value) {
        std::fprintf(stderr,
                     "%s has no %s metadata; create it with scripts/expose_activations.py\n",
                     opt.model.c_str(), kExposedKey);
        return 2;
      }
      const auto map = nlohmann::json::parse(value.get(), nullptr, false);
      if (!map.is_object()) {
        std::fprintf(stderr, "%s metadata is not a JSON object\n", kExposedKey);
        return 2;
      }
      for (const auto& [tensor, node] : map.items())
        if (node.is_string()) capture.node_of[tensor] = node.get<std::string>();
    }
    const Timing on = summarize(
        run_session(profiled_session, opt, hidden, opt.activations.empty() ? nullptr : &capture));
    auto profile_path = profiled_session.EndProfilingAllocated(allocator);

    std::printf("onnxruntime %s\n", Ort::GetVersionString().c_str());
    std::printf("model %s  seq %lld  intra-op threads %d  warmup %d  runs %d\n", opt.model.c_str(),
                static_cast<long long>(opt.seq), opt.threads, opt.warmup, opt.runs);
    std::printf("profiling off: mean %.4f ms  median %.4f ms  min %.4f ms\n", off.mean_ms,
                off.median_ms, off.min_ms);
    std::printf("profiling on:  mean %.4f ms  median %.4f ms  min %.4f ms\n", on.mean_ms,
                on.median_ms, on.min_ms);
    if (off.median_ms > 0)
      std::printf("median overhead: %+.1f%%\n", (on.median_ms / off.median_ms - 1.0) * 100.0);
    std::printf("profile: %s\n", profile_path.get());
    if (!opt.activations.empty()) {
      llm_trace::LoadedTrace trace = llm_trace::OnnxProfileAdapter{}.load(profile_path.get());
      const auto result = llm_trace::attach_activations(trace, capture.records);
      trace.metadata.model = opt.model;
      trace.metadata.runtime_version = Ort::GetVersionString();
      for (const auto& w : trace.warnings) trace.metadata.notes.push_back("Warning: " + w);
      llm_trace::write_jsonl(opt.activations, trace.metadata, trace.events);
      const double per_run = capture.stats_ms / static_cast<double>(opt.warmup + opt.runs);
      std::printf(
          "activations: %zu exposed tensor(s), %zu summaries attached, %zu unmatched, "
          "%zu non-float skipped\n",
          capture.node_of.size(), result.attached, result.unmatched, capture.non_float);
      std::printf("activation statistics: %.4f ms per run (outside the timed region)\n", per_run);
      std::printf("trace: %s\n", opt.activations.c_str());
    }
  } catch (const Ort::Exception& e) {
    std::fprintf(stderr, "onnxruntime error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
