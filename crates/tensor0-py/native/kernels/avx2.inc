#pragma once

#include <cstdint>
#include <complex>
#include <type_traits>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TENSOR0_STRIDE_HAS_AVX2_TARGET 1

namespace tensor0::stride::simd {

inline bool CpuSupportsAvx2() {
  static const bool supported = __builtin_cpu_supports("avx2");
  return supported;
}

__attribute__((target("avx2")))
inline void Transpose8x8F32(__m256* values) {
  const auto pairs0 = _mm256_unpacklo_ps(values[0], values[1]);
  const auto pairs1 = _mm256_unpackhi_ps(values[0], values[1]);
  const auto pairs2 = _mm256_unpacklo_ps(values[2], values[3]);
  const auto pairs3 = _mm256_unpackhi_ps(values[2], values[3]);
  const auto pairs4 = _mm256_unpacklo_ps(values[4], values[5]);
  const auto pairs5 = _mm256_unpackhi_ps(values[4], values[5]);
  const auto pairs6 = _mm256_unpacklo_ps(values[6], values[7]);
  const auto pairs7 = _mm256_unpackhi_ps(values[6], values[7]);
  const auto quads0 = _mm256_shuffle_ps(pairs0, pairs2, 0x44);
  const auto quads1 = _mm256_shuffle_ps(pairs0, pairs2, 0xEE);
  const auto quads2 = _mm256_shuffle_ps(pairs1, pairs3, 0x44);
  const auto quads3 = _mm256_shuffle_ps(pairs1, pairs3, 0xEE);
  const auto quads4 = _mm256_shuffle_ps(pairs4, pairs6, 0x44);
  const auto quads5 = _mm256_shuffle_ps(pairs4, pairs6, 0xEE);
  const auto quads6 = _mm256_shuffle_ps(pairs5, pairs7, 0x44);
  const auto quads7 = _mm256_shuffle_ps(pairs5, pairs7, 0xEE);
  values[0] = _mm256_permute2f128_ps(quads0, quads4, 0x20);
  values[1] = _mm256_permute2f128_ps(quads1, quads5, 0x20);
  values[2] = _mm256_permute2f128_ps(quads2, quads6, 0x20);
  values[3] = _mm256_permute2f128_ps(quads3, quads7, 0x20);
  values[4] = _mm256_permute2f128_ps(quads0, quads4, 0x31);
  values[5] = _mm256_permute2f128_ps(quads1, quads5, 0x31);
  values[6] = _mm256_permute2f128_ps(quads2, quads6, 0x31);
  values[7] = _mm256_permute2f128_ps(quads3, quads7, 0x31);
}

inline bool CpuSupportsAvx2F16c() {
  static const bool supported =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
  return supported;
}

inline bool CpuSupportsAvx2Fma() {
  static const bool supported =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return supported;
}

__attribute__((target("avx2,fma")))
inline __m256 FullComplexMultiplyFourC64(
    __m256 value, float scale_real, float scale_imaginary) {
  const auto first_scale = _mm256_set_ps(
      scale_imaginary, scale_real, scale_imaginary, scale_real,
      scale_imaginary, scale_real, scale_imaginary, scale_real);
  const auto second_scale = _mm256_set_ps(
      scale_real, scale_imaginary, scale_real, scale_imaginary,
      scale_real, scale_imaginary, scale_real, scale_imaginary);
  const auto real = _mm256_moveldup_ps(value);
  const auto imaginary = _mm256_movehdup_ps(value);
  const auto negate_real = _mm256_castsi256_ps(
      _mm256_set1_epi64x(INT64_C(0x0000000080000000)));
  const auto base = _mm256_xor_ps(_mm256_mul_ps(imaginary, second_scale), negate_real);
  return _mm256_fmadd_ps(real, first_scale, base);
}

__attribute__((target("avx2,fma")))
inline void TransposeFourC64(__m256* values) {
  const auto row0 = _mm256_castps_pd(values[0]);
  const auto row1 = _mm256_castps_pd(values[1]);
  const auto row2 = _mm256_castps_pd(values[2]);
  const auto row3 = _mm256_castps_pd(values[3]);
  const auto low01 = _mm256_unpacklo_pd(row0, row1);
  const auto high01 = _mm256_unpackhi_pd(row0, row1);
  const auto low23 = _mm256_unpacklo_pd(row2, row3);
  const auto high23 = _mm256_unpackhi_pd(row2, row3);
  values[0] = _mm256_castpd_ps(_mm256_permute2f128_pd(low01, low23, 0x20));
  values[1] = _mm256_castpd_ps(_mm256_permute2f128_pd(high01, high23, 0x20));
  values[2] = _mm256_castpd_ps(_mm256_permute2f128_pd(low01, low23, 0x31));
  values[3] = _mm256_castpd_ps(_mm256_permute2f128_pd(high01, high23, 0x31));
}

__attribute__((target("avx2,fma")))
inline uint64_t ScaleC64Vectors(
    const std::complex<float>* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
  const uint64_t vector_count = element_count - element_count % 4;
  for (uint64_t element = 0; element < vector_count; element += 4) {
    const auto values = _mm256_loadu_ps(reinterpret_cast<const float*>(input + element));
    const auto mapped = FullComplexMultiplyFourC64(values, factor.real(), factor.imag());
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element), mapped);
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousC64Scale(
    const std::complex<float>* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
  if (element_count < 4 || !CpuSupportsAvx2Fma()) return 0;
  return ScaleC64Vectors(input, output, element_count, factor);
}

__attribute__((target("avx2,fma")))
inline uint64_t ScaleF32C64Vectors(
    const float* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
  const auto real_scale = _mm256_set1_ps(factor.real());
  const auto imaginary_scale = _mm256_set1_ps(factor.imag());
  const uint64_t vector_count = element_count - element_count % 8;
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const auto values = _mm256_loadu_ps(input + element);
    const auto real = _mm256_mul_ps(real_scale, values);
    const auto imaginary = _mm256_mul_ps(imaginary_scale, values);
    const auto low = _mm256_unpacklo_ps(real, imaginary);
    const auto high = _mm256_unpackhi_ps(real, imaginary);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element),
                    _mm256_permute2f128_ps(low, high, 0x20));
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element + 4),
                    _mm256_permute2f128_ps(low, high, 0x31));
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousF32C64Scale(
    const float* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
  if (element_count < 8 || !CpuSupportsAvx2Fma()) return 0;
  return ScaleF32C64Vectors(input, output, element_count, factor);
}

__attribute__((target("avx2,fma")))
inline uint64_t EmbedF32ProductVectors(
    const float* input, std::complex<float>* output,
    uint64_t element_count, float factor) {
  const auto scale = _mm256_set1_ps(factor);
  const auto zeros = _mm256_setzero_ps();
  const uint64_t vector_count = element_count - element_count % 8;
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const auto mapped = _mm256_mul_ps(_mm256_loadu_ps(input + element), scale);
    const auto low = _mm256_unpacklo_ps(mapped, zeros);
    const auto high = _mm256_unpackhi_ps(mapped, zeros);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element),
                    _mm256_permute2f128_ps(low, high, 0x20));
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element + 4),
                    _mm256_permute2f128_ps(low, high, 0x31));
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousF32ProductComplex(
    const float* input, std::complex<float>* output,
    uint64_t element_count, float factor) {
  if (element_count < 8 || !CpuSupportsAvx2Fma()) return 0;
  return EmbedF32ProductVectors(input, output, element_count, factor);
}

__attribute__((target("avx2,fma")))
inline uint64_t ProjectC64ProductVectors(
    const std::complex<float>* input, float* output,
    uint64_t element_count, std::complex<float> factor) {
  const auto real_scale = _mm256_set1_ps(factor.real());
  const auto imaginary_scale = _mm256_set1_ps(factor.imag());
  const uint64_t vector_count = element_count - element_count % 8;
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const auto first = _mm256_loadu_ps(reinterpret_cast<const float*>(input + element));
    const auto second = _mm256_loadu_ps(reinterpret_cast<const float*>(input + element + 4));
    const auto real = _mm256_castpd_ps(_mm256_permute4x64_pd(
        _mm256_castps_pd(_mm256_shuffle_ps(first, second, 0x88)), 0xD8));
    const auto imaginary = _mm256_castpd_ps(_mm256_permute4x64_pd(
        _mm256_castps_pd(_mm256_shuffle_ps(first, second, 0xDD)), 0xD8));
    const auto negative = _mm256_xor_ps(_mm256_mul_ps(imaginary_scale, imaginary),
        _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MIN)));
    _mm256_storeu_ps(output + element, _mm256_fmadd_ps(real_scale, real, negative));
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousC64ProductReal(
    const std::complex<float>* input, float* output,
    uint64_t element_count, std::complex<float> factor) {
  if (element_count < 8 || !CpuSupportsAvx2Fma()) return 0;
  return ProjectC64ProductVectors(input, output, element_count, factor);
}

__attribute__((target("avx2,fma")))
inline uint64_t ScaleC64RealVectors(
    const std::complex<float>* input, float* output,
    uint64_t element_count, float factor) {
  const auto scale = _mm_set1_ps(factor);
  const auto real_lanes = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  const uint64_t vector_count = element_count - element_count % 4;
  for (uint64_t element = 0; element < vector_count; element += 4) {
    const auto values = _mm256_loadu_ps(reinterpret_cast<const float*>(input + element));
    const auto real = _mm256_permutevar8x32_ps(values, real_lanes);
    _mm_storeu_ps(output + element, _mm_mul_ps(_mm256_castps256_ps128(real), scale));
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousC64RealScale(
    const std::complex<float>* input, float* output,
    uint64_t element_count, float factor) {
  if (element_count < 4 || !CpuSupportsAvx2Fma()) return 0;
  return ScaleC64RealVectors(input, output, element_count, factor);
}

__attribute__((target("avx2,f16c")))
inline void Transpose8x8F16(__m128i* values) {
  const __m128i pairs0 = _mm_unpacklo_epi16(values[0], values[1]);
  const __m128i pairs1 = _mm_unpackhi_epi16(values[0], values[1]);
  const __m128i pairs2 = _mm_unpacklo_epi16(values[2], values[3]);
  const __m128i pairs3 = _mm_unpackhi_epi16(values[2], values[3]);
  const __m128i pairs4 = _mm_unpacklo_epi16(values[4], values[5]);
  const __m128i pairs5 = _mm_unpackhi_epi16(values[4], values[5]);
  const __m128i pairs6 = _mm_unpacklo_epi16(values[6], values[7]);
  const __m128i pairs7 = _mm_unpackhi_epi16(values[6], values[7]);
  const __m128i quads0 = _mm_unpacklo_epi32(pairs0, pairs2);
  const __m128i quads1 = _mm_unpackhi_epi32(pairs0, pairs2);
  const __m128i quads2 = _mm_unpacklo_epi32(pairs1, pairs3);
  const __m128i quads3 = _mm_unpackhi_epi32(pairs1, pairs3);
  const __m128i quads4 = _mm_unpacklo_epi32(pairs4, pairs6);
  const __m128i quads5 = _mm_unpackhi_epi32(pairs4, pairs6);
  const __m128i quads6 = _mm_unpacklo_epi32(pairs5, pairs7);
  const __m128i quads7 = _mm_unpackhi_epi32(pairs5, pairs7);
  values[0] = _mm_unpacklo_epi64(quads0, quads4);
  values[1] = _mm_unpackhi_epi64(quads0, quads4);
  values[2] = _mm_unpacklo_epi64(quads1, quads5);
  values[3] = _mm_unpackhi_epi64(quads1, quads5);
  values[4] = _mm_unpacklo_epi64(quads2, quads6);
  values[5] = _mm_unpackhi_epi64(quads2, quads6);
  values[6] = _mm_unpacklo_epi64(quads3, quads7);
  values[7] = _mm_unpackhi_epi64(quads3, quads7);
}

__attribute__((target("avx2,f16c")))
inline uint64_t ConvertF16F32Vectors(
    const uint16_t* input, float* output, uint64_t element_count) {
  const uint64_t vector_count = element_count - element_count % 8;
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const __m128i half =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + element));
    _mm256_storeu_ps(output + element, _mm256_cvtph_ps(half));
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousF16F32(
    const uint16_t* input, float* output, uint64_t element_count) {
  if (element_count < 8 || !CpuSupportsAvx2F16c()) return 0;
  return ConvertF16F32Vectors(input, output, element_count);
}

template <typename Result>
__attribute__((target("avx2,f16c")))
inline uint64_t ScaleF16Vectors(
    const uint16_t* input, Result* output, uint64_t element_count, float factor) {
  const uint64_t vector_count = element_count - element_count % 8;
  const __m256 scale = _mm256_set1_ps(factor);
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const __m128i half =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + element));
    const __m256 values = _mm256_mul_ps(scale, _mm256_cvtph_ps(half));
    if constexpr (std::is_same_v<Result, uint16_t>) {
      const __m128i result = _mm256_cvtps_ph(
          values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output + element), result);
    } else {
      _mm256_storeu_ps(output + element, values);
    }
  }
  return vector_count;
}

template <typename Result>
inline uint64_t ExecuteContiguousF16Scale(
    const uint16_t* input, Result* output, uint64_t element_count, float factor) {
  if (element_count < 8 || !CpuSupportsAvx2F16c()) return 0;
  return ScaleF16Vectors(input, output, element_count, factor);
}

__attribute__((target("avx2,f16c")))
inline uint64_t NarrowF32ProductVectors(
    const float* input, uint16_t* output, uint64_t element_count, float factor) {
  const auto scale = _mm256_set1_ps(factor);
  const uint64_t vector_count = element_count - element_count % 8;
  for (uint64_t element = 0; element < vector_count; element += 8) {
    const auto mapped = _mm256_mul_ps(_mm256_loadu_ps(input + element), scale);
    const auto half = _mm256_cvtps_ph(
        mapped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + element), half);
  }
  return vector_count;
}

inline uint64_t ExecuteContiguousF32ProductF16(
    const float* input, uint16_t* output, uint64_t element_count, float factor) {
  if (element_count < 8 || !CpuSupportsAvx2F16c()) return 0;
  return NarrowF32ProductVectors(input, output, element_count, factor);
}

}
#else
namespace tensor0::stride::simd {

inline uint64_t ExecuteContiguousF32ProductF16(
    const float*, uint16_t*, uint64_t, float) {
  return 0;
}

inline uint64_t ExecuteContiguousF32ProductComplex(
    const float*, std::complex<float>*, uint64_t, float) {
  return 0;
}

inline uint64_t ExecuteContiguousC64ProductReal(
    const std::complex<float>*, float*, uint64_t, std::complex<float>) {
  return 0;
}

inline uint64_t ExecuteContiguousC64RealScale(
    const std::complex<float>*, float*, uint64_t, float) {
  return 0;
}

inline uint64_t ExecuteContiguousF32C64Scale(
    const float*, std::complex<float>*, uint64_t, std::complex<float>) {
  return 0;
}

inline uint64_t ExecuteContiguousC64Scale(
    const std::complex<float>*, std::complex<float>*, uint64_t, std::complex<float>) {
  return 0;
}

inline uint64_t ExecuteContiguousF16F32(const uint16_t*, float*, uint64_t) {
  return 0;
}

template <typename Result>
inline uint64_t ExecuteContiguousF16Scale(const uint16_t*, Result*, uint64_t, float) {
  return 0;
}

}
#endif
