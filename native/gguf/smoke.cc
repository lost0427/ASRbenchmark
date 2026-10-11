#include "engine.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstring>

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: benchmark_gguf_smoke model.gguf sample.wav");
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
    GgufEngine engine(argv[1], 4);
    auto first = engine.recognize(pcm);
    auto second = engine.recognize(pcm);
    if (first.empty() || first != second) throw std::runtime_error("Empty or inconsistent recognition");
    std::cout << "TEXT: " << second << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
