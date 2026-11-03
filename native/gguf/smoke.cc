#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::string Escape(const std::string &text) {
  std::string out;
  for (char c : text) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n') { out += "\\n"; continue; }
    out += c;
  }
  return out;
}

static double Since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char **argv) {
  try {
    if (argc < 3)
      throw std::runtime_error(
          "Usage: benchmark_gguf_smoke model.gguf sample.wav [threads] [warmups] [repeats] [gpu] [fa]");
    const int threads = argc > 3 ? std::max(1, std::atoi(argv[3])) : 4;
    const int warmups = argc > 4 ? std::max(0, std::atoi(argv[4])) : 0;
    const int repeats = argc > 5 ? std::max(1, std::atoi(argv[5])) : 2;
    const bool use_gpu = argc > 6 && std::string(argv[6]) == "gpu";
    const bool flash_attn = argc > 7 && std::string(argv[7]) == "fa";
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<unsigned char> wav((std::istreambuf_iterator<char>(input)), {});
    if (wav.size() < 44 || std::memcmp(wav.data(), "RIFF", 4) != 0)
      throw std::runtime_error("Expected PCM16 WAV");
    size_t pos = 12;
    std::vector<float> pcm;
    while (pos + 8 <= wav.size()) {
      uint32_t size = 0;
      std::memcpy(&size, wav.data() + pos + 4, 4);
      if (pos + 8 + size > wav.size()) throw std::runtime_error("Invalid WAV chunk");
      if (std::memcmp(wav.data() + pos, "data", 4) == 0) {
        pcm.resize(size / 2);
        for (size_t i = 0; i < pcm.size(); ++i) {
          int16_t value = 0;
          std::memcpy(&value, wav.data() + pos + 8 + i * 2, 2);
          pcm[i] = value / 32768.0f;
        }
        break;
      }
      pos += 8 + size + size % 2;
    }
    if (pcm.empty()) throw std::runtime_error("Missing WAV data");

    auto start = std::chrono::steady_clock::now();
    GgufEngine engine(argv[1], threads, use_gpu, flash_attn);
    const double load_ms = Since(start);
    start = std::chrono::steady_clock::now();
    std::string text = engine.recognize(pcm);
    const double first_ms = Since(start);
    for (int i = 0; i < warmups; ++i) engine.recognize(pcm);
    std::vector<double> samples;
    for (int i = 0; i < repeats; ++i) {
      start = std::chrono::steady_clock::now();
      text = engine.recognize(pcm);
      samples.push_back(Since(start));
    }
    std::cout << "RESULT: {\"loadMs\": " << load_ms << ", \"firstInferenceMs\": " << first_ms
              << ", \"samplesMs\": [";
    for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? ", " : "") << samples[i];
    std::cout << "], \"text\": \"" << Escape(text) << "\"}" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
