#pragma once
#include "sense-voice.h"
#include "silero-vad.h"
#include <stdexcept>
#include <string>
#include <vector>

class GgufEngine {
 public:
  GgufEngine(const std::string &filename, int threads) : threads_(threads) {
    auto params = sense_voice_context_default_params();
    params.use_gpu = false;
    params.use_itn = false;
    params.flash_attn = false;
    context_ = sense_voice_small_init_from_file_with_params(filename.c_str(), params);
    if (!context_ || !context_->state) {
      release();
      throw std::runtime_error("SenseVoice.cpp could not load model");
    }
    context_->language_id = sense_voice_lang_id("en");
  }
  GgufEngine(const GgufEngine &) = delete;
  GgufEngine &operator=(const GgufEngine &) = delete;
  ~GgufEngine() { release(); }

  std::string recognize(const std::vector<float> &pcm) {
    // The upstream frontend expects PCM16 magnitudes, like its WAV reader.
    std::vector<double> samples(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) samples[i] = pcm[i] * 32768.0;
    auto &feature = context_->state->feature;
    // Upstream replaces these allocations each inference without releasing them.
    ggml_backend_buffer_free(feature.buffer);
    ggml_free(feature.ctx);
    feature.buffer = nullptr;
    feature.ctx = nullptr;
    feature.tensor = nullptr;
    auto params = sense_voice_full_default_params(SENSE_VOICE_SAMPLING_GREEDY);
    params.n_threads = threads_;
    params.language = "en";
    params.debug_mode = false;
    params.print_progress = false;
    params.print_timestamps = false;
    if (sense_voice_full_parallel(context_, params, samples, samples.size(), 1) != 0)
      throw std::runtime_error("SenseVoice.cpp inference failed");
    std::string text;
    const auto &ids = context_->state->ids;
    for (size_t i = 4; i < ids.size(); ++i) {
      if (ids[i] == 0 || (i > 0 && ids[i] == ids[i - 1])) continue;
      const auto &token = context_->vocab.id_to_token.at(ids[i]);
      if (token.rfind("<|", 0) != 0) text += token;
    }
    const std::string marker = "\xe2\x96\x81";
    size_t offset = 0;
    while ((offset = text.find(marker, offset)) != std::string::npos) {
      text.replace(offset, marker.size(), " ");
      ++offset;
    }
    return text;
  }

 private:
  sense_voice_context *context_ = nullptr;
  int threads_;
  void release() {
    if (!context_) return;
    sense_voice_free_state(context_->state);
    ggml_backend_buffer_free(context_->model.buffer);
    ggml_free(context_->model.ctx);
    if (context_->model.model) {
      delete context_->model.model->encoder;
      delete context_->model.model;
    }
    delete context_->vad_model.model;
    if (context_->backend) ggml_backend_free(context_->backend);
    delete context_;
    context_ = nullptr;
  }
};
