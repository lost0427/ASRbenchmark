#include "../common/jni_utils.h"
#include "engine.h"

extern "C" JNIEXPORT jlong JNICALL
Java_org_sensevoice_benchmark_GgufNative_create(JNIEnv *env, jobject, jstring model, jint threads) {
  try {
    return reinterpret_cast<jlong>(new GgufEngine(String(env, model), threads));
  } catch (const std::exception &e) { Error(env, e); return 0; }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_sensevoice_benchmark_GgufNative_recognize(JNIEnv *env, jobject, jlong value, jfloatArray input) {
  try {
    auto text = reinterpret_cast<GgufEngine *>(value)->recognize(Samples(env, input));
    return env->NewStringUTF(text.c_str());
  } catch (const std::exception &e) { Error(env, e); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_org_sensevoice_benchmark_GgufNative_destroy(JNIEnv *, jobject, jlong value) {
  delete reinterpret_cast<GgufEngine *>(value);
}
