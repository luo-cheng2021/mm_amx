#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>

#include "kernels_amx.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"

#include "thread_pool.hpp"
#include "dnnl_thread.hpp"
#include "fc_custom.hpp"
#include "matmul_custom.hpp"
#include "misc_custom.hpp"

using ov::bfloat16;
using namespace dnnl::impl;
using namespace dnnl::impl::utils;
////////////////////////////////////////////////////////////////////////
// fc
#if 1
struct FC::Impl {
    void init(size_t threads, FCType t);

    template<typename T, typename PP>
    void fc_s8s8(T* src, T* weight, size_t M, size_t N, size_t K, PP ppkernel);
    FCType _type;
    size_t _threads;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> opsFC_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, int8_t, float>>> opsFC_BF16xi8;
    std::vector<std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>>> opsFC_i8xi8;
};

void FC::Impl::init(size_t threads, FCType t) {
    _threads = threads;
    _type = t;
    if (t == FCType_S8) {
        opsFC_i8xi8.resize(_threads);
        for (size_t i = 0; i < _threads; i++) {
            opsFC_i8xi8[i] = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(true, true);
        }
    }
}

template<typename T, typename PP>
void FC::Impl::fc_s8s8(T* src, T* weight, size_t M, size_t N, size_t K, PP ppkernel) {
    tensor2D<T> matA(M, K, reinterpret_cast<T*>(src), K * sizeof(T));
    tensor2D<T> matB(N, K, reinterpret_cast<T*>(weight), K * sizeof(T));
    auto work_amount = rndup(N, 32) / 32;
    auto kernel = [&](int tid, int cnt) {
        size_t start, end;
        splitter(work_amount, cnt, tid, start, end);
        int n0 = start*32;
        int n1 = end*32;
        if (n1 > N) n1 = N;
        (*opsFC_i8xi8[tid].get())(matA, matB, n0, n1, ppkernel);
    };

    
    // for(size_t i = 0; i< _threads; i++) {
    //     kernel(i, _threads);
    // }
    parallel(_threads, kernel);    
}

FC::FC(): _impl(std::make_shared<Impl>()) {
}

void FC::init(size_t threads, FCType t) {
    _impl->init(threads, t);
}

void FC::fc_s8s8s8_dq_gelu_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* q) {
    tensor2D<int8_t> matC(M, N, reinterpret_cast<int8_t*>(dst), N * sizeof(int8_t));
    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_GELU_QUANT> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    ppkernel.set_q_scale(q);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8s8_dq_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* q) {
    tensor2D<int8_t> matC(M, N, reinterpret_cast<int8_t*>(dst), N * sizeof(int8_t));
    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_QUANT> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    ppkernel.set_q_scale(q);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8s8_dq_bias_gelu_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias, float* q) {
    tensor2D<int8_t> matC(M, N, reinterpret_cast<int8_t*>(dst), N * sizeof(int8_t));
    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_QUANT> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    ppkernel.set_q_scale(q);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8s8_dq_bias_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias, float* q) {
    tensor2D<int8_t> matC(M, N, reinterpret_cast<int8_t*>(dst), N * sizeof(int8_t));
    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_QUANT> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    ppkernel.set_q_scale(q);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8bf16_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq) {
    tensor2D<bfloat16> matC(M, N, reinterpret_cast<bfloat16*>(dst), N * sizeof(bfloat16));
    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8bf16_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq) {
    tensor2D<bfloat16> matC(M, N, reinterpret_cast<bfloat16*>(dst), N * sizeof(bfloat16));
    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8bf16_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias) {
    tensor2D<bfloat16> matC(M, N, reinterpret_cast<bfloat16*>(dst), N * sizeof(bfloat16));
    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8bf16_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias) {
    tensor2D<bfloat16> matC(M, N, reinterpret_cast<bfloat16*>(dst), N * sizeof(bfloat16));
    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8f32_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq) {
    tensor2D<float> matC(M, N, reinterpret_cast<float*>(dst), N * sizeof(float));
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8f32_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq) {
    tensor2D<float> matC(M, N, reinterpret_cast<float*>(dst), N * sizeof(float));
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT> ppkernel(matC);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8f32_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias) {
    tensor2D<float> matC(M, N, reinterpret_cast<float*>(dst), N * sizeof(float));
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}

void FC::fc_s8s8f32_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias) {
    tensor2D<float> matC(M, N, reinterpret_cast<float*>(dst), N * sizeof(float));
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(matC, bias);
    ppkernel.set_deq_scale(dq);
    _impl->fc_s8s8(src, weight, M, N, K, ppkernel);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// misc
#if 1
/// Convert Packed BF16 Data to Packed float Data.
///
/// \headerfile <x86intrin.h>
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \returns A 512-bit vector of [16 x float] come from convertion of __A
// static __inline__ __m512 _mm512_cvtpbh_ps(__m256bh __A) {
//   return _mm512_castsi512_ps((__m512i)_mm512_slli_epi32(
//       (__m512i)_mm512_cvtepi16_epi32((__m256i)__A), 16));
// }

// Store masks. The highest bit in each byte indicates the byte to store.
alignas(16) const unsigned char masks[16][16] =
{
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00 },
    { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 }
};

inline void store_n(__m128i mm, unsigned int n, void* storage)
{
    _mm_maskmoveu_si128(mm, reinterpret_cast< const __m128i& >(masks[n]), static_cast< char* >(storage));
}

void add3(int8_t* _a, int8_t *_b, int8_t *_c, int8_t *_dst, size_t ele_num) {
    bfloat16* a = reinterpret_cast<bfloat16*>(_a);
    bfloat16* b = reinterpret_cast<bfloat16*>(_b);
    bfloat16* c = reinterpret_cast<bfloat16*>(_c);
    bfloat16* dst = reinterpret_cast<bfloat16*>(_dst);
    size_t i = 0;
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(a);
        auto b0 = _mm256_loadu_epi16(b);
        auto c0 = _mm256_loadu_epi16(c);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto b0_f = _mm512_cvtpbh_ps((__m256bh)b0);
        auto c0_f = _mm512_cvtpbh_ps((__m256bh)c0);
        auto d_f = _mm512_add_ps(a0_f, b0_f);
        d_f = _mm512_add_ps(d_f, c0_f);
        auto regOut = _mm512_cvtne2ps_pbh(d_f, d_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm512_extracti64x4_epi64(regOut, 0));
        a += 16;
        b += 16;
        c += 16;
        dst += 16;
    }
    if (i != ele_num) {
        // https://stackoverflow.com/questions/40391708/convert-16-bit-mask-mmask16-to-m128i-control-byte-mask-on-knl-xeon-phi-72
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, a);
        auto b0 = _mm256_maskz_loadu_epi16(msk, b);
        auto c0 = _mm256_maskz_loadu_epi16(msk, c);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto b0_f = _mm512_cvtpbh_ps((__m256bh)b0);
        auto c0_f = _mm512_cvtpbh_ps((__m256bh)c0);
        auto d_f = _mm512_add_ps(a0_f, b0_f);
        d_f = _mm512_add_ps(d_f, c0_f);
        auto regOut = _mm512_cvtne2ps_pbh(d_f, d_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_mask_storeu_epi16(dst, msk, _mm512_extracti64x4_epi64(regOut, 0));
    }
}

static float sum(bfloat16* src, size_t ele_num) {
    size_t i = 0;
    auto one = _mm512_set1_epi32(0x3f803f80);
    __m512 s;
    s = _mm512_xor_ps(s, s);
    for (; i < ele_num / 32 * 32; i += 32) {
        auto a0 = _mm512_loadu_epi16(src);
        s = _mm512_dpbf16_ps(s, (__m512bh)a0, (__m512bh)one);
        src += 32;
    }
    if (i != ele_num) {
        __mmask32 msk = _cvtu32_mask32(0xFFFFFFFFu >> (32 - (ele_num % 32)));
        auto a0 = _mm512_maskz_loadu_epi16(msk, src);
        s = _mm512_dpbf16_ps(s, (__m512bh)a0, (__m512bh)one);
    }
    // https://stackoverflow.com/questions/26896432/horizontal-add-with-m512-avx512
    return _mm512_reduce_add_ps(s);
}

static float sum_power2(bfloat16* src, float mean, size_t ele_num) {
    size_t i = 0;
    __m512 s;
    s = _mm512_xor_ps(s, s);
    auto m = _mm512_set1_ps(mean);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        s = _mm512_fmadd_ps(a0_f, a0_f, s);
        src += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_maskz_sub_ps(msk, a0_f, m);
        s = _mm512_fmadd_ps(a0_f, a0_f, s);
    }
    return _mm512_reduce_add_ps(s);
}

static void mvn(bfloat16* src, float mean, float var, size_t ele_num, bfloat16* dst) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm512_extracti64x4_epi64(regOut, 0));

        src += 16;
        dst += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_mask_storeu_epi16(dst, msk, _mm512_extracti64x4_epi64(regOut, 0));
    }
}

static void mvn_i8(bfloat16* src, float mean, float var, size_t ele_num, int8_t* dst, float* quant) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto q = _mm512_loadu_ps(quant);
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), a0_i8);

        src += 16;
        dst += 16;
        quant += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto q = _mm512_maskz_loadu_ps(msk, quant);
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        store_n(a0_i8, ele_num % 16, dst);
    }
}

void mvn_line(int8_t* _src, size_t ele_num, float eps, bool inside_sqrt, int8_t *_dst) {
    bfloat16* src = reinterpret_cast<bfloat16*>(_src);
    bfloat16* dst = reinterpret_cast<bfloat16*>(_dst);
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn(src, mean, var, ele_num, dst);
}

void mvn_line(int8_t* _src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst, float* quant) {
    bfloat16* src = reinterpret_cast<bfloat16*>(_src);
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn_i8(src, mean, var, ele_num, dst, quant);
}

void quant_i8(void* dst, void* src, size_t ele_num, float scale) {
    size_t i = 0;
    bfloat16* a = reinterpret_cast<bfloat16*>(src);
    int8_t* d = reinterpret_cast<int8_t*>(dst);
    auto s = _mm512_set1_ps(scale);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d), d_i8);
        a += 16;
        d += 16;
    }
    if (i != ele_num) {
        // https://stackoverflow.com/questions/40391708/convert-16-bit-mask-mmask16-to-m128i-control-byte-mask-on-knl-xeon-phi-72
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        store_n(d_i8, ele_num % 16, d);
    }
}

// NOTE: did not handle tail because there should be enough room
inline void cvt_i32_f32(float* dst, int32_t* src, size_t ele_num) {
    for (int i = 0; i < (ele_num + 15) / 16 * 16; i += 16) {
        auto a0 = _mm512_load_epi32(src);
        auto a_f = _mm512_cvtepi32_ps(a0);
        _mm512_storeu_ps(dst, a_f);
        src += 16;
        dst += 16;
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////
// matmul
struct Matmul::Impl {
    Impl(Type t, bool transpose);

    template<typename T, typename PP>
    void matmul_s8s8(T* src, T* weight, size_t M, size_t N, size_t K, PP ppkernel);
    Type _type;
    bool _transpose;

    void matmul_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void matmul_u8s8f32(uint8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void gemAvB_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t M, size_t K);

    void matmul_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void gemAvB_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);

    std::shared_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>> gemAvB_bf16xbf16;
    std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>> bf16xbf16;

    std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>> i8xi8;
    std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>> u8xi8;
    std::shared_ptr<amx_kernel::MatmulVector<int8_t, int8_t>> gemAvB_i8xi8;
};

Matmul::Impl::Impl(Type t, bool transpose): _type(t), _transpose(transpose) {
    switch (t) {
        case Type_S8:
            i8xi8 = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(false, transpose);
            break;
        case Type_S8_v:
            gemAvB_i8xi8 = std::make_shared<amx_kernel::MatmulVector<int8_t, int8_t>>();
            break;
        case Type_U8:
            u8xi8 = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(false, transpose);
            break;
        case Type_BF16:
            bf16xbf16 = std::make_shared<amx_kernel::Matmul<bfloat16, bfloat16>>(false, transpose);
            break;
        case Type_BF16_v:
            gemAvB_bf16xbf16 = std::make_shared<amx_kernel::MatmulVector<bfloat16, bfloat16>>();
            break;
    }
}

void Matmul::Impl::matmul_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {
    tensor2D<int8_t> a(M, K, A, lda);
    auto kb = _transpose ? N : K;
    auto nb = _transpose ? K : N;
    tensor2D<int8_t> b(kb, nb, B, ldb);
    tensor2D<float> c(M, N, C, ldc);
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
    (*i8xi8)(a, b, 0, N, pp);
}

void Matmul::Impl::matmul_u8s8f32(uint8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {
    tensor2D<uint8_t> a(M, K, A, lda);
    auto kb = _transpose ? N : K;
    auto nb = _transpose ? K : N;
    tensor2D<int8_t> b(kb, nb, B, ldb);
    tensor2D<float> c(M, N, C, ldc);
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
    (*u8xi8)(a, b, 0, N, pp);
}

void Matmul::Impl::gemAvB_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t M, size_t K) {
    tensor2D<int8_t> a(M, K, A, lda);
    (*gemAvB_i8xi8)(a, B, reinterpret_cast<int32_t*>(C));
    cvt_i32_f32(C, reinterpret_cast<int32_t*>(C), M);
}

Matmul::Matmul(Type t, bool transpose): _impl(std::make_shared<Impl>(t, transpose)) {
}

void Matmul::matmul_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {
    _impl->matmul_s8s8f32(A, B, C, lda, ldb, ldc, M, N, K);
}

void Matmul::matmul_u8s8f32(uint8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {
    _impl->matmul_u8s8f32(A, B, C, lda, ldb, ldc, M, N, K);
}

void Matmul::gemAvB_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t M, size_t K) {
    _impl->gemAvB_s8s8f32(A, B, C, lda, M, K);
}

void Matmul::matmul_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {
}

void Matmul::gemAvB_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K) {

}
