#include "../common/jni_utils.h"
#include "gpu.h"  // NOLINT
#include "sherpa-ncnn/csrc/offline-recognizer.h"

extern "C" JNIEXPORT jlong JNICALL
Java_org_sensevoice_benchmark_NcnnNative_create(JNIEnv *env, jobject, jstring dir, jstring tokens, jint threads,
                                                jboolean vulkan, jboolean fp16) {
  try {
    sherpa_ncnn::OfflineRecognizerConfig config;
    config.model_config.sense_voice.model_dir = String(env, dir);
    config.model_config.sense_voice.language = "en";
    config.model_config.sense_voice.use_itn = false;
    config.model_config.tokens = String(env, tokens);
    config.model_config.num_threads = threads;
    config.model_config.use_vulkan_compute = vulkan == JNI_TRUE;
    config.model_config.fp16 = fp16 == JNI_TRUE;
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
extern "C" JNIEXPORT jstring JNICALL
Java_org_sensevoice_benchmark_NcnnNative_deviceInfo(JNIEnv *env, jobject) {
  try {
    const int count = ncnn::get_gpu_count();
    if (count <= 0) return env->NewStringUTF("Vulkan unavailable");
    const char *name = ncnn::get_gpu_device(0)->get_device_name();
    return env->NewStringUTF((std::string("Vulkan 0: ") + (name ? name : "unknown")).c_str());
  } catch (...) {
    return env->NewStringUTF("Vulkan unavailable");
  }
}
extern "C" JNIEXPORT void JNICALL
Java_org_sensevoice_benchmark_NcnnNative_destroy(JNIEnv *, jobject, jlong handle) {
  delete reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(handle);
}
