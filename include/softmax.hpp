// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <float.h>

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);

#define HAVE_AVX512F
#ifdef HAVE_AVX512F
    inline __m512 cvt_bf16_to_fp32(const __m256i src) {
        __m512i y = _mm512_cvtepu16_epi32(src);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
    }

    inline __m512 mm512_uni_loadu_ps(const ov::bfloat16* a) {
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        return cvt_bf16_to_fp32(vec_bf16);
    }

    inline __m512 mm512_uni_loadu_ps(const float* a) {
        return _mm512_loadu_ps(a);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const float* a, size_t count) {
        __mmask16 mask = (1 << count) - 1;
        return _mm512_maskz_loadu_ps(mask, a);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const ov::bfloat16* a, size_t count) {
        auto mask = (1 << count) - 1;
        auto bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        return cvt_bf16_to_fp32(bf16_vec);
    }

    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
    inline void mm512_uni_storeu_ps(ov::bfloat16 *addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(addr), _mm512_cvtepi32_epi16(x));
    }
    inline void mm512_uni_mask_storeu_ps(ov::bfloat16 *addr, __mmask16 mask_addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
    }

#endif



#if defined(HAVE_AVX2)
inline void exp_ps_avx2(__m256& src) {
    static __m256 exp_ln_flt_min_f = _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));  // log(FLT_MIN)
    static __m256 exp_ln_flt_max_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));  // log(FLT_MAX)
    static __m256 exp_log2ef = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));        // log2(e)
    static __m256 half = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000));              // 0.5f
    static __m256 ln2f = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));              // ln(2)
    static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000));               // 1.0f
    static __m256i exponent_bias = _mm256_set1_epi32(0x0000007f);                         // 127
    static constexpr int n_mantissa_bits = 23;
    static __m256 exp_pol1 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb));  // p1 = 0.999999701f
    static __m256 exp_pol2 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3));  // p2 = 0.499991506f
    static __m256 exp_pol3 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40));  // p3 = 0.166676521f
    static __m256 exp_pol4 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d));  // p4 = 0.0418978221f
    static __m256 exp_pol5 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce));  // p5 = 0.00828929059f
    static __m256 two = _mm256_castsi256_ps(_mm256_set1_epi32(0x40000000));       // 2
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    auto zero_mask = _mm256_cmp_ps(src, exp_ln_flt_min_f, _CMP_LT_OS);

    // clip src
    src = _mm256_min_ps(src, exp_ln_flt_max_f);
    src = _mm256_max_ps(src, exp_ln_flt_min_f);

    // aux1 : r
    auto aux1 = src;

    // calculate exp(x)
    // fx = x * log2(e) + 0.5
    src = _mm256_mul_ps(src, exp_log2ef);
    src = _mm256_add_ps(src, half);

    // tmp = floorf(fx)
    src = _mm256_floor_ps(src);

    // aux1 = x - fx * ln2
    aux1 = _mm256_fnmadd_ps(src, ln2f, aux1);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    src = _mm256_sub_ps(src, one);
    auto aux2_i = _mm256_cvtps_epi32(src);
    aux2_i = _mm256_add_epi32(aux2_i, exponent_bias);
    aux2_i = _mm256_slli_epi32(aux2_i, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    auto zero = _mm256_setzero_ps();
    auto aux2 = _mm256_blendv_ps(_mm256_castsi256_ps(aux2_i), zero, zero_mask);

    // compute polynomial
    src = exp_pol5;
    src = _mm256_fmadd_ps(src, aux1, exp_pol4);
    src = _mm256_fmadd_ps(src, aux1, exp_pol3);
    src = _mm256_fmadd_ps(src, aux1, exp_pol2);
    src = _mm256_fmadd_ps(src, aux1, exp_pol1);
    src = _mm256_fmadd_ps(src, aux1, one);

    // y = y * 2^n
    src = _mm256_mul_ps(src, aux2);
    src = _mm256_mul_ps(src, two);
}
#endif

inline void scale_add_reduce_max(float* a, const float scale, const float* b, const size_t size, float& max) {
#if defined(HAVE_AVX512F)
    auto v_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm512_set1_ps(scale);
    auto v_a = v_max;
    auto v_b = v_max;
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_b = _mm512_loadu_ps(b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_max_ps(v_max, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    // process tails
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_b = _mm512_maskz_loadu_ps(mask, b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_mask_max_ps(v_max, mask, v_a, v_max);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }

    max = _mm512_reduce_max_ps(v_max);
#elif defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm256_set1_ps(scale);
    auto v_a = v_max;
    auto v_b = v_max;
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_b = _mm256_loadu_ps(b + i);
        v_a = _mm256_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    // process tails
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_b = _mm256_maskload_ps(b + i, mask);
        v_a = _mm256_fmadd_ps(v_a, v_scale, v_b);
        v_a = _mm256_blendv_ps(v_max, v_a, _mm256_castsi256_ps(mask));
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hmax(v_max);
    max = _mm256_cvtss_f32(v_max);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= scale;
        a[i] += b[i];
        max = a[i] > max ? a[i] : max;
    }
#endif
}

template <bool has_alibi, bool has_attn_mask, bool has_causal_mask, typename T>
static void __attribute__ ((noinline)) scale_add2_reduce_max(float* a,
                                  float scale,
                                  const float* alibi_lookup,
                                  const T* attn_mask,
                                  const uint8_t* causal_mask,
                                  bool select_nfltmax_at_0,  // true:  0 in mask set -FLT_MAX
                                  size_t size,
                                  float alibi_slope,
                                  float& max) {
#if defined(HAVE_AVX512F)
    auto v_max0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_max1 = v_max0;
    auto v_max2 = v_max0;
    auto v_max3 = v_max0;
    auto v_scale = _mm512_set1_ps(scale);
    size_t i = 0;
    auto v_zeroi32 = _mm512_setzero_epi32();
    auto v_nfltmax = _mm512_set1_ps(-FLT_MAX);
    auto kmask_xor = _cvtu32_mask16(select_nfltmax_at_0 ? 0xFFFF : 0);
    auto v_alibi_slope = _mm512_set1_ps(alibi_slope);
    __m512 v_a;
    //asm("int3");
    // process vector body
    for (; i + 4 * vec_len_f32_avx512 <= size; i += 4 * vec_len_f32_avx512) {
    #define C(n) \
        v_a = _mm512_loadu_ps(a + i + n * vec_len_f32_avx512);  \
        v_a = _mm512_mul_ps(v_a, v_scale);  \
        if (has_alibi) {    \
            auto v_lookup = _mm512_loadu_ps(alibi_lookup + i + n * vec_len_f32_avx512); \
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a); \
        } \
        if (has_attn_mask) {    \
            auto v_mask = mm512_uni_loadu_ps(attn_mask + i + n * vec_len_f32_avx512);   \
            v_a = _mm512_add_ps(v_a, v_mask);   \
        }   \
        if (has_causal_mask) {  \
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i + n * vec_len_f32_avx512));    \
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);    \
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);    \
            kmask = _kxor_mask16(kmask, kmask_xor);                                     \
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                          \
        }   \
        v_max##n = _mm512_max_ps(v_max##n, v_a);   \
        _mm512_storeu_ps(a + i + n * vec_len_f32_avx512, v_a);

        C(0);
        C(1);
        C(2);
        C(3);
    #undef C
    }
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm512_loadu_ps(alibi_lookup + i);
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm512_uni_loadu_ps(attn_mask + i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max0 = _mm512_max_ps(v_max0, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    // process tails
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm512_maskz_loadu_ps(mask, alibi_lookup + i);
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm512_uni_loadu_tail_ps(attn_mask + i, size - i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max0 = _mm512_mask_max_ps(v_max0, mask, v_a, v_max0);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }

    v_max0 = _mm512_max_ps(v_max0, v_max1);
    v_max2 = _mm512_max_ps(v_max2, v_max3);
    v_max0 = _mm512_max_ps(v_max0, v_max2);
    max = _mm512_reduce_max_ps(v_max0);
#elif defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm256_set1_ps(scale);
    auto v_a = v_max;
    auto v_zeroi32 = _mm256_setzero_si256();
    auto v_mask_xor = _mm256_set1_epi32(select_nfltmax_at_0 ? -1 : 0);
    auto v_nfltmax = _mm256_set1_ps(-FLT_MAX);
    auto v_alibi_slope = _mm256_set1_ps(alibi_slope);
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm256_loadu_ps(alibi_lookup + i);
            v_a = _mm256_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm256_uni_loadu_ps(attn_mask + i);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    // process tails
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm256_maskload_ps(alibi_lookup + i, mask);
            v_a = _mm256_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm256_uni_loadu_tail_ps(attn_mask + i, size - i);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_a = _mm256_blendv_ps(v_max, v_a, _mm256_castsi256_ps(mask));
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hmax(v_max);
    max = _mm256_cvtss_f32(v_max);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= scale;
        if (has_alibi) {
            a[i] += alibi_lookup[i] * alibi_slope;
        }

        if (has_attn_mask)
            a[i] += attn_mask[i];

        if (has_causal_mask) {
            if (select_nfltmax_at_0) {
                if (causal_mask[i] == 0)
                    a[i] = -FLT_MAX;
            } else {
                if (causal_mask[i] != 0)
                    a[i] = -FLT_MAX;
            }
        }

        max = a[i] > max ? a[i] : max;
    }
#endif
}

#if defined(HAVE_AVX512F)
static inline void exp_ps_avx512(__m512& src) {
    static const uint32_t c_min[] = {0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50,
        0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50, 0xc2aeac50};
    static const uint32_t c_max[] = {0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218,
        0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218, 0x42b17218};
    static const uint32_t c_e[] = {0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b,
        0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b, 0x3fb8aa3b};
    static const uint32_t c_half[] = {0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000,
        0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000};
    static const uint32_t c_ln2[] = {0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218,
        0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218, 0x3f317218};
    static const uint32_t c_1[] = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000};
    static const uint32_t c_bias[] = {0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f,
        0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f, 0x0000007f};
    static const uint32_t c_p1[] = {0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb,
        0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb, 0x3f7ffffb};
    static const uint32_t c_p2[] = {0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3,
        0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3, 0x3efffee3};
    static const uint32_t c_p3[] = {0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40,
        0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40, 0x3e2aad40};
    static const uint32_t c_p4[] = {0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d,
        0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d, 0x3d2b9d0d};
    static const uint32_t c_p5[] = {0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce,
        0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce, 0x3c07cfce};
    static const uint32_t c_2[] = {0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000,
        0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000, 0x40000000};
    static constexpr int n_mantissa_bits = 23;
    __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_load_epi32(c_min));  // log(FLT_MIN)
    __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_load_epi32(c_max));  // log(FLT_MAX)
    __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_load_epi32(c_e));        // log2(e)
    __m512 half = _mm512_castsi512_ps(_mm512_load_epi32(c_half));              // 0.5f
    __m512 ln2f = _mm512_castsi512_ps(_mm512_load_epi32(c_ln2));              // ln(2)
    __m512 one = _mm512_castsi512_ps(_mm512_load_epi32(c_1));               // 1.0f
    __m512i exponent_bias = _mm512_load_epi32(c_bias);                         // 127
    __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_load_epi32(c_p1));  // p1 = 0.999999701f
    __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_load_epi32(c_p2));  // p2 = 0.499991506f
    __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_load_epi32(c_p3));  // p3 = 0.166676521f
    __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_load_epi32(c_p4));  // p4 = 0.0418978221f
    __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_load_epi32(c_p5));  // p5 = 0.00828929059f
    __m512 two = _mm512_castsi512_ps(_mm512_load_epi32(c_2));       // 2
    // __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));  // log(FLT_MIN)
    // __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));  // log(FLT_MAX)
    // __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));        // log2(e)
    // __m512 half = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));              // 0.5f
    // __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));              // ln(2)
    // __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));               // 1.0f
    // __m512i exponent_bias = _mm512_set1_epi32(0x0000007f);                         // 127
    // __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7ffffb));  // p1 = 0.999999701f
    // __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3efffee3));  // p2 = 0.499991506f
    // __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3e2aad40));  // p3 = 0.166676521f
    // __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3d2b9d0d));  // p4 = 0.0418978221f
    // __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3c07cfce));  // p5 = 0.00828929059f
    // __m512 two = _mm512_castsi512_ps(_mm512_set1_epi32(0x40000000));       // 2

    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    auto zero_mask = _mm512_cmp_ps_mask(src, exp_ln_flt_min_f, _CMP_LT_OS);

    // clip src
    src = _mm512_min_ps(src, exp_ln_flt_max_f);
    src = _mm512_max_ps(src, exp_ln_flt_min_f);

    // aux1 : r
    auto aux1 = src;

    // calculate exp(x)
    // fx = x * log2(e) + 0.5
    src = _mm512_mul_ps(src, exp_log2ef);
    src = _mm512_add_ps(src, half);

    // tmp = floorf(fx)
    src = _mm512_floor_ps(src);

    // aux1 = x - fx * ln2
    aux1 = _mm512_fnmadd_ps(src, ln2f, aux1);
    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    src = _mm512_sub_ps(src, one);
    auto aux2_i = _mm512_cvtps_epi32(src);
    aux2_i = _mm512_add_epi32(aux2_i, exponent_bias);
    aux2_i = _mm512_slli_epi32(aux2_i, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    auto zero = _mm512_setzero_ps();
    auto aux2 = _mm512_mask_blend_ps(zero_mask, _mm512_castsi512_ps(aux2_i), zero);

    // compute polynomial
    src = exp_pol5;
    src = _mm512_fmadd_ps(src, aux1, exp_pol4);
    src = _mm512_fmadd_ps(src, aux1, exp_pol3);
    src = _mm512_fmadd_ps(src, aux1, exp_pol2);
    src = _mm512_fmadd_ps(src, aux1, exp_pol1);
    src = _mm512_fmadd_ps(src, aux1, one);

    // y = y * 2^n
    src = _mm512_mul_ps(src, aux2);
    src = _mm512_mul_ps(src, two);
}
#endif

inline void exp_reduce_sum(float* a, const float max, const size_t size, float& sum) {
#if defined(HAVE_AVX512F)
    size_t i = 0;
    __m512 v_a;
    auto v_max = _mm512_set1_ps(max);
    auto v_sum = _mm512_set1_ps(0.0f);
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_sub_ps(v_a, v_max);
        exp_ps_avx512(v_a);
        v_sum = _mm512_add_ps(v_sum, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_sub_ps(v_a, v_max);
        exp_ps_avx512(v_a);
        v_sum = _mm512_mask_add_ps(v_sum, mask, v_a, v_sum);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }
    sum = _mm512_reduce_add_ps(v_sum);
#elif defined(HAVE_AVX2)
    size_t i = 0;
    __m256 v_a;
    auto v_max = _mm256_set1_ps(max);
    auto v_sum = _mm256_set1_ps(0.0f);
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_sub_ps(v_a, v_max);
        exp_ps_avx2(v_a);
        v_sum = _mm256_add_ps(v_sum, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_sub_ps(v_a, v_max);
        exp_ps_avx2(v_a);
        v_a = _mm256_blendv_ps(_mm256_setzero_ps(), v_a, _mm256_castsi256_ps(mask));
        v_sum = _mm256_add_ps(v_a, v_sum);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hsum(v_sum);
    sum = _mm256_cvtss_f32(v_sum);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] = exp(a[i] - max);
        sum += a[i];
    }
#endif
}

inline void multiply_scalar(float* a, float* a_dst, const float val, const size_t size) {
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    size_t i = 0;
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_mask_storeu_ps(a_dst + i, mask, v_a);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(val);
    __m256 v_a = {0};
    size_t i = 0;
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);
        _mm256_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx2;
    }
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);
        _mm256_maskstore_ps(a_dst + i, mask, v_a);
    }
#else
    for (size_t i = 0; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

inline void multiply_scalar(float* a, ov::bfloat16* a_dst, const float val, const size_t size) {
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    size_t i = 0;
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_mask_storeu_ps(a_dst + i, mask, v_a);
    }
#else
    for (size_t i = 0; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

inline void attn_softmax_kernel(float* a,
                                void* a_dst,
                                float scale,
                                float* alibi,
                                void* attn_mask,
                                uint8_t* causal_mask,
                                bool select_nfltmax_at_0,
                                size_t len,
                                size_t total_size,
                                bool dst_precision_is_f32,
                                float alibi_slope = 0) {
    using func_bf16_type = void (*)(float*, float, const float*, const ov::bfloat16*, const uint8_t*, bool, size_t, float, float&);

    static constexpr func_bf16_type funcs_bf16[] = {
        scale_add2_reduce_max<false, false, false>,
        scale_add2_reduce_max<false, false, true>,
        scale_add2_reduce_max<false, true, false>,
        scale_add2_reduce_max<false, true, true>,
        scale_add2_reduce_max<true, false, false>,
        scale_add2_reduce_max<true, false, true>,
        scale_add2_reduce_max<true, true, false>,
        scale_add2_reduce_max<true, true, true>
    };
    int dispatch = (alibi ? 0b100 : 0) | (attn_mask ? 0b010 : 0) | (causal_mask ? 0b001 : 0);
    float max = std::numeric_limits<float>::lowest();
    funcs_bf16[dispatch](a, scale, alibi, static_cast<const ov::bfloat16*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);

    float sum = 0.0f;
    //asm("int3");
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    // divide sum
    float scalar = 1.0f / sum;
    if (dst_precision_is_f32) {
        multiply_scalar(a, static_cast<float*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<float*>(a_dst) + len, 0, sizeof(float) * (total_size - len));
    } else {
        //asm("int3");
        multiply_scalar(a, static_cast<ov::bfloat16*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<ov::bfloat16*>(a_dst) + len, 0, sizeof(ov::bfloat16) * (total_size - len));
    }
}
