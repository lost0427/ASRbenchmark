#include "../common/jni_utils.h"
#include <algorithm>
#include "kaldi-native-fbank/csrc/online-feature.h"

extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_sensevoice_benchmark_FeatureNative_fbank(JNIEnv *env, jobject, jfloatArray input) {
  try {
    auto samples = Samples(env, input);
    for (auto &v : samples) v *= 32768.0f;
    knf::FbankOptions options;
    options.frame_opts.samp_freq = 16000;
    options.frame_opts.frame_length_ms = 25;
    options.frame_opts.frame_shift_ms = 10;
    options.frame_opts.dither = 0;
    options.frame_opts.snip_edges = true;
    options.frame_opts.window_type = "hamming";
    options.mel_opts.num_bins = 80;
    options.mel_opts.low_freq = 20;
    options.mel_opts.high_freq = 0;
    knf::OnlineFbank fbank(options);
    fbank.AcceptWaveform(16000, samples.data(), samples.size());
    fbank.InputFinished();
    std::vector<float> values(fbank.NumFramesReady() * 80);
    for (int i = 0; i < fbank.NumFramesReady(); ++i)
      std::copy(fbank.GetFrame(i), fbank.GetFrame(i) + 80, values.begin() + i * 80);
    auto out = env->NewFloatArray(values.size());
    env->SetFloatArrayRegion(out, 0, values.size(), values.data());
    return out;
  } catch (const std::exception &e) { Error(env, e); return nullptr; }
}
