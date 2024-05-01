#include "jit.hpp"
#include <vector>

#include "dnnl_kernels.hpp"

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>

#include "bf16.hpp"
// #include "kernels_avx512.hpp"
#include "kernels_amx.hpp"
#include "tensor2D.hpp"

static tensor2D<ov::bfloat16> repack_weights(tensor2D<ov::bfloat16>& Bt) {
    int N = Bt.dims[0];
    int K = Bt.dims[1];
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    for (int n = 0, i = 0; n < N; n += 32) {
        for (int k = 0; k < K; k += 32) {
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
            i++;
        }
    }
    return BPacked;
}

EnvVar NOSWPF("NOSWPF", 0);
EnvVar SWPFA("SWPFA", 0); // prefetch A is slower, so disable it

class Linear32x32_AMX_mkernel : public jit_generator {
public:
    TileConfig m_tile_cfg;

    int64_t m_ktiles;

    int m_BM_hint; // blockM: L2-cache kernel

    bool m_do_accumulation;

    int m_prefetch_Blines;

    // both A & B data will be prefetched from memory for next kernel invokation
    // and the prefetches are evenly distributed into each kernel.
    //
    // we first tackle the prefetching of B, because each time
    // we will call it with a new B, and run() will have to prefetch new B
    // for next round, so next B is also of size (KxN) elements
    //    distributes into (BM/32)*(BN/32) kernels:
    //    each kernel has (BK/32) iterations, thus each kernel iteration
    //    need to prefetch (BKxBN)/(BMxBNxBK/32768) = 32768/BM bfloat16-elements
    //    which is 1024/BM cache lines, this has to be determined at
    //    code-generation time. with BM=256, this is only 4.
    //
    // prefetch A can be done in unit of 32xBK elements, which must be evenly distributed
    // into (BN/32)*(BK/32) kernel iterations, each iteration prefetch/copy 32xBK/(BN*BK/1024) = 32768/BN bfloat16-elements
    // or 1024/BN cache lines. with BM=256, this is only 4 too.
    //
    // prefetch or copy?
    //   prefetch strided sub-matrix of A is tricky, consider each 32x32 AMX jit kernel has [BK/32] iterations
    //   and it's called (BN/32) times, each kernel must prefetch 32*BK/(BN/32) = (1024/BN)*BK elements
    //   since each kernel has [BK/32] loop iterations, each iteration fetch (1024/BN)*BK/(BK/32) = 1024*32/BN
    //   bytes.
    //
    //   when 1024 is not divisible by BN, it's fine, just prefetch more
    //
    // copy data from A to a ping-pong buffer has advantage:
    //    - read can be done in continous way most suitable for HW prefetcher
    //    - write to ping-pong buffer is within L2 cache, which should be fast
    //    - data transfer rate is small comparing to L2-bandwidth, shouldn't be a big issue for interleaved write to L2.
    //    - read from ping-pong buffer is much faster and free of odd-multiple-cache-line restriction.
    // so we prefer distribute the repacking of A sub-matrix into ping-pong buffer into kernel.
    // for BN=256, each kernel read 4*BK elements into ping-pong, each iteration read 4*BK*sizeof(bfloat16)/(BK/32)=256bytes = 4-512bits zmm registers
    //
    //
    Linear32x32_AMX_mkernel() = default;

    Linear32x32_AMX_mkernel(int M_hint, bool do_accumulation) { setup(M_hint, do_accumulation); }

    void setup(int M_hint = 0, //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
               bool do_accumulation = false) {
        m_do_accumulation = do_accumulation;
        m_BM_hint = M_hint;

        if (m_BM_hint == 0 || NOSWPF) {
            m_prefetch_Blines = 0;
        } else {
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / m_BM_hint;
        }

        create_kernel("Linear32x32_AMX_mkernel");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    // row data is in layout [N, K], maybe smaller than [32, 16]
    template <typename T>
    void repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K) {
        assert(K <= 32);
        assert(N <= 16);
        int k = 0;
        ov::bfloat16 bf16zero(0.0f);
        for (; k < 32; k += 2) {
            int n = 0;
            bool is_k0_valid = (k) < K;
            bool is_k1_valid = (k + 1) < K;
            auto* psrc = src + k;
            for (; n < 16 && n < N; n++, psrc += N_stride) {
                *dst++ = is_k0_valid ? ov::bfloat16(psrc[0]) : bf16zero;
                *dst++ = is_k1_valid ? ov::bfloat16(psrc[1]) : bf16zero;
            }
            for (; n < 16; n++) {
                *dst++ = 0;
                *dst++ = 0;
            }
        }
    }

    // weight is supposed to be of shape[N, K], stride in unit of bytes
    // N should be m_BN
    // K should be m_BK
    template <typename T>
    tensor2D<ov::bfloat16> prepareB(T* p_weight, int stride, int N, int K) {
        tensor2D<ov::bfloat16> ret;
        ASSERT((N % 32) == 0);
        ASSERT((K % 32) == 0);
        // weight matrix is in unit of [N/32, Kx32]
        ret.resize(N / 32, K * 32, true);

        auto N_stride = stride / sizeof(T);
        for (int n = 0, blkn = 0; n < N; n += 32, blkn++) {
            for (int k = 0, blkk = 0; k < K; k += 32, blkk++) {
                // two adjacent 32x16 (512) block of weight: dst0 & dst1
                auto* dst0 = &ret(blkn, blkk * 1024);
                auto* dst1 = dst0 + 16 * 32;
                auto valid_k = (K - k) < 32 ? (K - k) : 32;

                auto* src0 = p_weight + n * N_stride + k;
                auto valid_n0 = (N - n) < 16 ? (N - n) : 16;
                repackB<T>(dst0, src0, N_stride, valid_n0, valid_k);

                auto* src1 = p_weight + (n + 16) * N_stride + k;
                auto valid_n1 = (N - (n + 16)) < 16 ? (N - (n + 16)) : 16;
                repackB<T>(dst1, src1, N_stride, valid_n1, valid_k);
            }
        }
        return ret;
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    void generate() {
        Xbyak::Reg64 reg_A_addr = abi_param1;
        Xbyak::Reg64 reg_A_stride = abi_param2;
        Xbyak::Reg64 reg_B_addr = abi_param3;
        Xbyak::Reg64 reg_C_addr = abi_param4;
        Xbyak::Reg64 reg_C_stride = abi_param5;
        Xbyak::Reg64 reg_prefetch = abi_param6; // prefetch B

        Xbyak::Reg64 reg_ktiles = rax;
        Xbyak::Reg64 reg_prefetch_A = r9;
        Xbyak::Reg64 reg_prefetch_A1 = r12;
        Xbyak::Reg64 reg_B_stride = r10;
        Xbyak::Reg64 reg_A1_addr = r11;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC10 = tmm1;
        Xbyak::Tmm tmmC01 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        auto num_PFB = m_prefetch_Blines;
        int cur_PFB = 0;
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;

        if (m_do_accumulation) {
            auto reg_C1_addr = reg_A1_addr; // reuse reg_A1_addr
#if 0
            mov(reg_B_stride, 64);
            tileloadd(tmmC00, ptr[reg_C_addr + reg_B_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_B_stride + 1024]);
            tileloadd(tmmC10, ptr[reg_C_addr + reg_B_stride + 1024 * 2]);
            tileloadd(tmmC11, ptr[reg_C_addr + reg_B_stride + 1024 * 3]);
#else
            tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
            lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
            lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
            tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
            tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
#endif
        } else {
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);
        }
        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&m_ktiles));
        mov(reg_ktiles, ptr[reg_B_stride + 0]);

        if (SWPFA) {
            push(reg_prefetch_A1);
            mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_next_A_addr));
            mov(reg_prefetch_A, ptr[reg_B_stride + 0]);
            mov(reg_prefetch_A1, ptr[reg_prefetch_A + 2 * reg_A_stride]);
        }

        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A]);

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A + reg_A_stride]);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [num_Ktiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A1]);

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        // prefetch next sub-block B matrix
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A1 + reg_A_stride]);

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);

        if (SWPFA)
            lea(reg_prefetch_A, ptr[reg_prefetch_A + 64]);
        if (SWPFA)
            lea(reg_prefetch_A1, ptr[reg_prefetch_A1 + 64]);

        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

#if 0
        tilestored(ptr[reg_C_addr + reg_B_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024], tmmC01);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 2], tmmC10);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 3], tmmC11);
#else
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);
#endif
        if (SWPFA)
            pop(reg_prefetch_A1);
        ret();
    }

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(int M,                                 // actual M
             uint8_t* pA, int strideA,              // A [M, K]
             tensor2D<ov::bfloat16>& repacked_B,    // B [N/32, K*32]
             uint8_t* pC, int strideC,              // C [M, N]
             uint8_t* prefetch_B                    // prefetch B
    ) {
        // number of blocks in N dimension (in unit of 32 columns)
        auto num_blkN = repacked_B.dims[0];
        auto K = repacked_B.dims[1]/32;
        auto* pB = reinterpret_cast<uint8_t*>(&repacked_B[0]);
        auto strideB = repacked_B.stride;
        m_ktiles = K / 32;

        assert((K % 32) == 0);

        auto prefetch_step = m_prefetch_Blines * 64 * m_ktiles;

        // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB1 = pB;
            // prefetch_next_A_addr = pA + 32 * strideA;
            // if (m + 32 >= BM)
            //     prefetch_next_A_addr = pA;
            for (int ni = 0; ni < num_blkN; ni ++, pB1 += strideB, prefetch_B += prefetch_step) {
                (*this)(pA, strideA, pB1, pC + ni * 32 * sizeof(float), strideC, prefetch_B);
                // prefetch_next_A_addr += 4 * strideA;
            }
        }
    }
};

// multi-threaded kernel, decoupled from weight
void clr_cache() {
    thread_local std::vector<uint8_t> big_buffer(1024 * 1024 * 2, 0);
    memset(&big_buffer[0], 1, big_buffer.size());
    load_prefetch_L2(&big_buffer[0], big_buffer.size());
};

void test(int BM, int BK, int BN, const int num_AB_pairs = 43) {
    tensor2D<ov::bfloat16> A(BM, BK, true);
    tensor2D<ov::bfloat16> B(BK, BN, true);
    tensor2D<float> C0(BM, BN, true); // reference result
    tensor2D<float> C1(BM, BN, true); // reference result

    Linear32x32_AMX_mkernel jit_amx(BM, false);

    auto Bt = B.Tr();
    // auto B1 = repack_weights(Bt);
    auto B1 = jit_amx.prepareB(&Bt[0], Bt.stride, BN, BK);

    C0 = 0;
    matmul(A, B, C0);

    auto strideB = (BK / 32) * 2048;
    {
        TileConfigScope tcfg(jit_amx.m_tile_cfg);
        jit_amx.run(BM,
                    reinterpret_cast<uint8_t*>(&A[0]), A.stride,   //
                    B1,                                            //
                    reinterpret_cast<uint8_t*>(&C1[0]), C1.stride, //
                    reinterpret_cast<uint8_t*>(&B1[0]));
    }

    std::string acc;
    const char* acc_color = nullptr;
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        if (std::getenv("SHOW_ERR")) {
            std::cout << "============= A ================ " << std::endl;
            std::cout << A << std::endl;
            std::cout << "============= B ================ " << std::endl;
            std::cout << B << std::endl;
            logger() << C0 << std::endl;
            logger() << C1 << std::endl;
        }
        acc = "[FAIL]";
        acc_color = "1;31";
    }

    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
    });
    plog.reserve(512);
    plog.tag("cache-COLD", BM, BK, BN, acc);
    plog.color(acc_color);

    std::vector<tensor2D<ov::bfloat16>> A1s;
    tensor2D<ov::bfloat16> Abig(BM, num_AB_pairs * BK, true);
    for (int i = 0; i < num_AB_pairs; i++) {
        A1s.emplace_back(BM, BK, &Abig(0, i * BK), Abig.stride);
        // A1s.emplace_back(A.clone());
    }

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();

        tensor2D<float> C2(BM, BN, true);
        std::vector<tensor2D<ov::bfloat16>> B1s;
        for (int i = 0; i < num_AB_pairs; i++) {
            B1s.emplace_back(B1.clone());
        }
        Linear32x32_AMX_mkernel jit_amx0(BM, false);
        Linear32x32_AMX_mkernel jit_amx1(BM, true);
        TileConfigScope tcfg(jit_amx0.m_tile_cfg);

#if 0
#pragma omp barrier
        {
            // plog.tag("cache-HOT", M, K, N, acc);
            // plog.color(acc_color);
            for (int r = 0; r < 10; r++) {
                tensor2D<ov::bfloat16>& blockB = B1s[0];
                tensor2D<ov::bfloat16>& blockB1 = B1s[0];
                plog(
                    [&]() {
                        jit_amx.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride,     //
                                    reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                    reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                    reinterpret_cast<uint8_t*>(&blockB1[0]));
                    },
                    2.0 * M * N * K // OPS per call per core
                );
            };
        }
#endif
        clr_cache();
        clr_cache();
        clr_cache();

#pragma omp barrier
        plog(
            [&]() {
                Linear32x32_AMX_mkernel* pkernel = &jit_amx0;
                for (int r = 0; r < num_AB_pairs; r++) {
                    tensor2D<ov::bfloat16>& blockA = A1s[r];
                    tensor2D<ov::bfloat16>& blockB = B1s[r];
                    auto r1 = r + 1;
                    tensor2D<ov::bfloat16>& blockB1 = B1s[r1 < B1s.size() ? r1 : r];

                    pkernel->run(BM, reinterpret_cast<uint8_t*>(&blockA[0]), blockA.stride, blockB, reinterpret_cast<uint8_t*>(&C2[0]), C2.stride, reinterpret_cast<uint8_t*>(&blockB1[0]));
                    pkernel = &jit_amx1;
                }
            },
            (num_AB_pairs) * 2.0 * BM * BN * BK // OPS per call per core
        );

        if (ithr == 0)
            plog(); // add separator

#pragma omp barrier
        plog(
            [&]() {
                Linear32x32_AMX_mkernel* pkernel = &jit_amx0;
                for (int r = 0; r < num_AB_pairs; r++) {
                    tensor2D<ov::bfloat16>& blockA = A1s[r];
                    tensor2D<ov::bfloat16>& blockB = B1s[r];
                    auto r1 = r + 1;
                    tensor2D<ov::bfloat16>& blockB1 = B1s[r1 < B1s.size() ? r1 : r];

                    pkernel->run(BM, reinterpret_cast<uint8_t*>(&blockA[0]), blockA.stride, blockB, reinterpret_cast<uint8_t*>(&C2[0]), C2.stride, reinterpret_cast<uint8_t*>(&blockB1[0]));
                    pkernel = &jit_amx1;
                }
            },
            (num_AB_pairs) * 2.0 * BM * BN * BK // OPS per call per core
        );
    }
}

#if 0
class LinearAMX {
public:
    Linear32x32_AMX_mkernel* p_jit_amx0;
    Linear32x32_AMX_mkernel* p_jit_amx1;

    std::vector<tensor2D<ov::bfloat16>> weights;
    int num_blk_N;
    int num_blk_K;
    std::vector<tensor2D<float>> Cs; // per-thread result

    LinearAMX() {}

    // weight [N, K]

    template <typename T, int BN = 256>
    void setup(T* p_weight, int stride, int N, int K) {
        static Linear32x32_AMX_mkernel jit_amx0(256, 256, BN, false);
        static Linear32x32_AMX_mkernel jit_amx1(256, 256, BN, true);
        p_jit_amx0 = &jit_amx0;
        p_jit_amx1 = &jit_amx1;

        // prepare weights, N split among threads
        auto nthr = get_nthr();
        num_blk_N = N / BN;
        num_blk_K = K / 256;
        ASSERT(num_blk_N < nthr);
        weights.resize(num_blk_N * num_blk_K);
        Cs.resize(num_blk_N);
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            if (ithr < num_blk_N) {
                auto wid = num_blk_K * ithr;
                auto* pw = p_weight + (ithr * BN) * stride / sizeof(T);
                for (int k = 0; k < K; k += 256, wid++) {
                    weights[wid] = jit_amx0.prepareB(pw + k, stride, BN, 256);
                }
                Cs[ithr].resize(256, BN);
            }
        }
    }

    // A bfloat16 [256,  num_blk_K * 256]
    void run(uint8_t* pA, int strideA) {
#pragma omp parallel
        {
            int ni = omp_get_thread_num();
            if (ni < num_blk_N) {
                TileConfigScope tcfg(p_jit_amx0->m_tile_cfg);
                tensor2D<float>& C = Cs[ni];
                auto offset = ni * num_blk_K;
                Linear32x32_AMX_mkernel* pkernel = p_jit_amx0;
                for (int ki = 0; ki < num_blk_K; ki++) {
                    auto wi = offset + ki;
                    tensor2D<ov::bfloat16>& blockB = weights[wi];
                    tensor2D<ov::bfloat16>& blockB1 = weights[(ki + 1) < num_blk_K ? (wi + 1) : wi];

                    pkernel->run(pA + ki * 256 * sizeof(ov::bfloat16), strideA, blockB, reinterpret_cast<uint8_t*>(&C[0]), C.stride, reinterpret_cast<uint8_t*>(&blockB1[0]), 256);
                    pkernel = p_jit_amx1;
                }
            }
        }
    }
};

void test2(int K = 4096, int N = 11008) {
    tensor2D<ov::bfloat16> A(256, K, true);
    tensor2D<ov::bfloat16> B(K, N, true);
    tensor2D<float> C0(256, N, true); // reference result
    tensor2D<float> C1(256, N, true); // reference result

    LinearAMX linear;
    LinearAMX linear2;

    auto Bt = B.Tr();

    linear.setup(&Bt[0], Bt.stride, N, K);
    linear2.setup(&Bt[0], Bt.stride, N, K);

    C0 = 0;
    matmul(A, B, C0);

    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
    });

    plog([&]() { linear.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride); }, 256.0 * N * K * 2);

    // concat all Cs into C1
    auto concat_Cs = [&]() {
        for (int i = 0; i < linear.Cs.size(); i++) {
            auto& blkC = linear.Cs[i];
            for (int r = 0; r < 256; r++) {
                memcpy(&C1(r, i * 256), &blkC(r, 0), 256 * sizeof(float));
            }
        }
    };
    concat_Cs();

    std::string acc;
    const char* acc_color = nullptr;
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        if (std::getenv("SHOW_ERR")) {
            std::cout << "============= A ================ " << std::endl;
            std::cout << A << std::endl;
            std::cout << "============= B ================ " << std::endl;
            std::cout << B << std::endl;
            logger() << C0 << std::endl;
            logger() << C1 << std::endl;
        }
        acc = "[FAIL]";
        acc_color = "1;31";
    }
    std::cout << acc << std::endl;

    for (int i = 0; i < 5; i++) {
        plog([&]() { linear.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride); }, 256.0 * N * K * 2);
        concat_Cs();
    }
}
#endif

int main() {
    MSRConfig _msr;
    bool initAMX = initXTILE();
    test(256, 256, 256);
    //test2();
    return 0;
}
