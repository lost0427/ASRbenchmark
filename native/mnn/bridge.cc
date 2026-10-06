#include "../common/jni_utils.h"
#include "sherpa-mnn/c-api/c-api.h"

extern "C" JNIEXPORT jlong JNICALL
Java_org_sensevoice_benchmark_MnnNative_create(JNIEnv *env, jobject, jstring model, jstring tokens, jint threads) {
  try {
    auto m = String(env, model), t = String(env, tokens);
    SherpaMnnOfflineRecognizerConfig config{};
    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;
    config.model_config.sense_voice.model = m.c_str();
    config.model_config.sense_voice.language = "en";
    config.model_config.sense_voice.use_itn = 0;
    config.model_config.tokens = t.c_str();
    config.model_config.num_threads = threads;
    config.model_config.provider = "cpu";
    config.decoding_method = "greedy_search";
    auto *p = SherpaMnnCreateOfflineRecognizer(&config);
    if (!p) throw std::runtime_error("MNN could not load model");
    return reinterpret_cast<jlong>(p);
  } catch (const std::exception &e) { Error(env, e); return 0; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_sensevoice_benchmark_MnnNative_recognize(JNIEnv *env, jobject, jlong handle, jfloatArray input) {
  try {
    auto samples = Samples(env, input);
    auto *r = reinterpret_cast<const SherpaMnnOfflineRecognizer *>(handle);
    auto *s = SherpaMnnCreateOfflineStream(r);
    if (!s) throw std::runtime_error("Could not create stream");
    SherpaMnnAcceptWaveformOffline(s, 16000, samples.data(), samples.size());
    SherpaMnnDecodeOfflineStream(r, s);
    auto *result = SherpaMnnGetOfflineStreamResult(s);
    std::string text = result && result->text ? result->text : "";
    if (result) SherpaMnnDestroyOfflineRecognizerResult(result);
    SherpaMnnDestroyOfflineStream(s);
    return env->NewStringUTF(text.c_str());
  } catch (const std::exception &e) { Error(env, e); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL
Java_org_sensevoice_benchmark_MnnNative_destroy(JNIEnv *, jobject, jlong handle) {
  SherpaMnnDestroyOfflineRecognizer(reinterpret_cast<const SherpaMnnOfflineRecognizer *>(handle));
}
