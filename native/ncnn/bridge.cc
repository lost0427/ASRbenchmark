#include "../common/jni_utils.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"

extern "C" JNIEXPORT jlong JNICALL
Java_org_sensevoice_benchmark_NcnnNative_create(JNIEnv *env, jobject, jstring dir, jstring tokens, jint threads) {
  try {
    sherpa_ncnn::OfflineRecognizerConfig config;
    config.model_config.sense_voice.model_dir = String(env, dir);
    config.model_config.sense_voice.language = "en";
    config.model_config.sense_voice.use_itn = false;
    config.model_config.tokens = String(env, tokens);
    config.model_config.num_threads = threads;
    if (!config.Validate()) throw std::runtime_error("Invalid ncnn model configuration");
    return reinterpret_cast<jlong>(new sherpa_ncnn::OfflineRecognizer(config));
  } catch (const std::exception &e) { Error(env, e); return 0; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_sensevoice_benchmark_NcnnNative_recognize(JNIEnv *env, jobject, jlong handle, jfloatArray input) {
  try {
    auto samples = Samples(env, input);
    auto *r = reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(handle);
    auto s = r->CreateStream();
    s->AcceptWaveform(16000, samples.data(), samples.size());
    r->DecodeStream(s.get());
    return env->NewStringUTF(s->GetResult().text.c_str());
  } catch (const std::exception &e) { Error(env, e); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL
Java_org_sensevoice_benchmark_NcnnNative_destroy(JNIEnv *, jobject, jlong handle) {
  delete reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(handle);
}
