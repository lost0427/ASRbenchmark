#pragma once
#include <jni.h>
#include <string>
#include <vector>
#include <stdexcept>

inline std::string String(JNIEnv *env, jstring value) {
  const char *p = env->GetStringUTFChars(value, nullptr);
  if (!p) throw std::runtime_error("Cannot read string");
  std::string result(p);
  env->ReleaseStringUTFChars(value, p);
  return result;
}
inline std::vector<float> Samples(JNIEnv *env, jfloatArray input) {
  std::vector<float> result(env->GetArrayLength(input));
  env->GetFloatArrayRegion(input, 0, result.size(), result.data());
  return result;
}
inline void Error(JNIEnv *env, const std::exception &e) {
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}
