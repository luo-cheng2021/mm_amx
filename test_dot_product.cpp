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
#include <cmath>

#include "misc.hpp"
#include "softmax.hpp"
#include "tensor2D.hpp"
#include "timeit.hpp"
#include <omp.h>
#include "test_bw.hpp"
// https://raw.githubusercontent.com/intel/perfmon/main/SPR/events/sapphirerapids_core.json
timeit benchmark
(
    {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);

template<typename T>
static void dot_product_block4(T* a, uint8_t* b, float* c, size_t a_stride, size_t c_stride, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t j = 0;
            // asm("int3");
    for (; j < block_size; j++) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();

        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = _mm512_set1_ps(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va0 = mm512_uni_loadu_ps(a + i);
            auto va1 = mm512_uni_loadu_ps(a + i + a_stride);
            auto va2 = mm512_uni_loadu_ps(a + i + a_stride * 2);
            auto va3 = mm512_uni_loadu_ps(a + i + a_stride * 3);
            auto vb = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp);
            vsum0 = _mm512_fmadd_ps(va0, vb, vsum0);
            vsum1 = _mm512_fmadd_ps(va1, vb, vsum1);
            vsum2 = _mm512_fmadd_ps(va2, vb, vsum2);
            vsum3 = _mm512_fmadd_ps(va3, vb, vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);

        b += n;
        *c = sum0 * b0[0];
        *(c + c_stride) = sum1 * b0[0];
        *(c + c_stride * 2) = sum2 * b0[0];
        *(c + c_stride * 3) = sum3 * b0[0];
        c++;
    }
}

template<typename TA, typename TB>
void cvt_copy(TA* dst, TB* src, size_t n) {
    size_t i = 0;
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(src + i);
        mm512_uni_storeu_ps(dst + i, vb);
    }
}

template<typename T>
static void dot_product_block(T* a, uint8_t* b, float* c, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto b1 = reinterpret_cast<float*>(b + n + 8);
        auto b2 = reinterpret_cast<float*>(b + (n + 8) * 2);
        auto b3 = reinterpret_cast<float*>(b + (n + 8) * 3);
        auto v_zp0 = _mm512_set1_ps(b0[1]);
        auto v_zp1 = _mm512_set1_ps(b1[1]);
        auto v_zp2 = _mm512_set1_ps(b2[1]);
        auto v_zp3 = _mm512_set1_ps(b3[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp0);
            auto vb1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + n + 8)))), v_zp1);
            auto vb2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 2 * (n + 8))))), v_zp2);
            auto vb3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 3 * (n + 8))))), v_zp3);

            vsum0 = _mm512_fmadd_ps(va, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va, vb1, vsum1);
            vsum2 = _mm512_fmadd_ps(va, vb2, vsum2);
            vsum3 = _mm512_fmadd_ps(va, vb3, vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        // for (; i < n; i++) {
        //     sum0 += a[i] * (b[i] - b0[1]);
        //     sum1 += a[i] * (b[i + n + 8] - b1[1]);
        //     sum2 += a[i] * (b[i + 2 * (n + 8)] - b2[1]);
        //     sum3 += a[i] * (b[i + 3 * (n + 8)] - b3[1]);
        // }
        c[0] = sum0 * b0[0];
        c[1] = sum1 * b1[0];
        c[2] = sum2 * b2[0];
        c[3] = sum3 * b3[0];
        c += 4;
        b +=  4 * (n + 8) - 8;
    }
    // for (; j < block_size; j++) {
    //     auto vsum = _mm512_setzero_ps();
    //     auto b0 = reinterpret_cast<float*>(b);
    //     auto v_zp = _mm512_set1_ps(b0[1]);
    //     size_t i = 0;
    //     b += 8;
    //     for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
    //         auto va = mm512_uni_loadu_ps(a + i);
    //         auto vb = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp);
    //         vsum = _mm512_fmadd_ps(va, vb, vsum);
    //     }
    //     float sum = _mm512_reduce_add_ps(vsum);
    //     for (; i < n; i++) {
    //         sum += a[i] * (b[i] - b0[1]);
    //     }
    //     b += n;
    //     *c++ = sum * b0[0];
    // }
}

// vfmadd132ps ymm(8 floats)  Throughput (CPI)=0.5
const double vfmaddOpsPerCycle = 16;


int OMP_NT = omp_thread_count();

int test_dot_product() {
    tensor2D<ov::bfloat16> A;
    tensor2D<uint8_t> B;
    tensor2D<float> C;
    tensor2D<float> C1;
    tensor2D<float> A1;
    int N = 128;

    auto ref = [&](ov::bfloat16* a, uint8_t* b, float* c, size_t n, size_t block_size) {
        for (size_t j = 0; j < block_size; j++) {
            float sum = 0;
            auto b0 = reinterpret_cast<float*>(b);
            b += 8;
            for (size_t i = 0; i < n; i++) {
                sum += a[i] * (b[i] - b0[1]);
            }
            b += n;
            *c++ = sum * b0[0];
        }
    };
    int errors = 0;
    {
        A.resize(1, N);
        B.resize(32, N + 8);
        C.resize(1, 32);
        C1.resize(1, 32);
        A.fill_rnd();
        B.fill_rnd();
        float* p = (float*)&B[0];
        for (int i = 0; i < B.dims[0]; i++) {
            p[0] = 1.0f;
            p[1] = 0.0f;
            p += B.stride;
        }
        ref(&A[0], &B[0], &C1[0], A.dims[1], B.dims[0]);
        dot_product_block(&A[0], &B[0], &C[0], A.dims[1], B.dims[0]);
        for(int i=0;i<B.dims[0];i++) {
            if (abs((C[i] - C1[i])/C[i]) > 0.01f) {
                errors ++;
                std::cout << "#" << i << "/" << N << ":  " <<C[i] << " vs " << C1[i] << " diff " << (C[i] - C1[i]) << std::endl;
            }
        }
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }
    {
        int M = 100;
        A.resize(4, N);
        A1.resize(4, N);
        B.resize(32*M, N + 8);
        B = 0;
        A1 = 0;
        float* p = (float*)&B[0];
        for (int i = 0; i < B.dims[0]; i++) {
            p[0] = 1.0f;
            p[1] = 0.0f;
            p += B.stride;
        }
        C.resize(4, 32);
        C = 0;

        benchmark.tag(__func__, N, "dot")(1000, [&](){
            for (int j = 0; j < 4; j++)
            for (int i = 0; i < M * 32; i += 32)
                dot_product_block(&A(j, 0), &B(i, 0), &C(j, 0), A.dims[1], C.dims[1]);
        });
        benchmark.tag(__func__, N, "dot4")(1000, [&](){
            for (int i = 0; i < M * 32; i += 32) {
                for (int j = 0; j < 4; j++)
                    cvt_copy(&A1(j, 0), &A(j, 0), N);
                dot_product_block4(&A1[0], &B(i, 0), &C[0], A1.stride, C.stride, A1.dims[1], C.dims[1]);
            }
        });
    }
    return 0;
}

int main(int argc, const char *argv[]) {
    benchmark.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    test_dot_product();

    return 0;
}
