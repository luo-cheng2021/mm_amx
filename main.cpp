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

#include "kernels_amxbf16.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"
#include "test_bw.hpp"

#include "thread_pool.hpp"
#include <omp.h>
timeit timer;

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();

static amx_bf16::Matmul::WeightPrecision precision = amx_bf16::Matmul::Weight_BF16;

//================================================================================
int amx_unit_test_perf() {
    int M = 32;
    // K=12*32, A+B fits L1D, gives 100% HW usage
    // K=80*32  A+B fits L2, gives 70% HW usage
    // K=512*32 A+B fits L2, gives 30% HW usage
    int K = 80*32;
    int N = 32;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> BT(N, K);
    tensor2D<bfloat16> C(M, N);
    tileconfig_t tfg(1, 0, 8, 16, 64);

    std::cout << "A & B in L1D (should give theoratical peak Gflops)\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        auto * pA0 = &A(0,0);
        auto * pA1 = &A(16,0);
        auto * pB0 = &BT(0,0);
        auto * pB1 = &BT(16,0);
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for(int k = 0; k < K; k+=32) {
            _tile_loadd(A0, pA0, 64);
            _tile_loadd(B0, pB0, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA1, 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB1, 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "TileGemmKernel32x32:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(A0, &A(0, k), A.stride);
            _tile_loadd(B0, &BT(0, k), BT.stride);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, &BT(16, k), BT.stride);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "B is transposed and blocked:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(A0, &A(0, k), A.stride);
            _tile_stream_loadd(B0, pB, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, pB + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            pB += 32*32;
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "B is transposed and blocked; A is blocked:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pA = &A(0, 0);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(B0, pB + k*(32), 64);
            _tile_loadd(A0, pA + k*(32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );


    // now we go through real memory area, but assume the layout has been
    // optimized for performance.
    std::cout << "A & B are blocked and sequentially loaded:\n\t";
    tensor2D<bfloat16> AB(M*K + K*N, 32);
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *ptr = &AB(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(B0, ptr, 64);
            _tile_stream_loadd(A0, ptr + (16*32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, ptr + (2*16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, ptr + (3*16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            ptr += (4*16*32);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << C(0,0) << std::endl;
    return 0;
}

void amx_FC_acc(int M, int K, int N) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul fc(true, false, precision);
    fc(A, B, C);

    C0=0;
    matmul(A, B, C0);
    std::cout << __func__ << " [" << M << "," << K << "," << N << "," << precision << "] ";
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
}

void amx_FC_perf(int M, int K, int N, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    amx_bf16::Matmul mm(true, false, precision);

    timer.tag(__func__, M, K, N, precision)(times, [&](){
        mm(A, B, C);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

void amx_Matmul_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul mm(false, transB);

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0);
    mm(A, transB?BT:B, C);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
    std::cout << C0 << std::endl;
    std::cout << C << std::endl;
    timer(times, [&](){
        mm(A, transB?BT:B, C);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

void amx_unit_test_gemAvB(int M, int K, int times = -1000) {
    int N = 1;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(1, K);
    tensor2D<bfloat16> B0(K, 1);
    tensor2D<bfloat16> C0(M, 1);
    tensor2D<bfloat16> C(1, M);
    amx_bf16::GemAvB gemAvB;

    // same B, different layout
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    B0 = B;
    C0 = 0;
    matmul(A, B0, C0);
    auto C0Tr = C0.Tr();
    gemAvB(A, &B(0,0), &C(0,0));
    if (C0Tr == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << C0Tr << std::endl;
        std::cout << C << std::endl;
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        return;
    }

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    timer(times, [&](){
        gemAvB(A, &B(0,0), &C(0,0));
    },
    double(M * N) * K * 2,
    256 * 3e9);
}

void test_blk_loops() {
    int max = 9999;
    BlockIterator loc;
    BlockIterator::blkloop bloops[] = {
        {10,32,0},{max,0,32},{max,320,0}
    };
    //loc.reset(bloops, 896,10240);
    //do {
    //    std::cout << "[" << loc.seq << "]   " << loc.m << "," << loc.n
    //              << "  idx =  " << loc.idx[0] << "," << loc.idx[1] << "," << loc.idx[2] << std::endl;
    //}while(loc.next());

    loc.reset(bloops, 3, 896, 10240);
    do {
    }while(loc.next());
    std::cout << loc.seq << std::endl;
    
    std::cout << __func__;
    timer(-1000, [&](){
        loc.reset(bloops, 3, 10240, 10240);
        do {
        }while(loc.next());
    });
}

#if 0

// ThreadPool has much lower performance than OMP

ThreadPool thp;

// multi-threaded matmul
struct MatmulMT {
    amx_bf16::Matmul::WeightPrecision rt_precision;
    std::vector<std::shared_ptr<amx_bf16::Matmul>> ops;
    bool transposeB;
    MatmulMT(bool constB = false,
             bool transposeB = false,
             amx_bf16::Matmul::WeightPrecision precision=amx_bf16::Matmul::Weight_BF16) : transposeB(transposeB), rt_precision(precision) {
        for(int i = 0; i < thp.num_threads; i++)
            ops.push_back(std::make_shared<amx_bf16::Matmul>(constB, transposeB));
    }

    template<typename P>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    P ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0:1];
        // split along N dimension
        int work_amount = rndup(N, 32)/32;

        auto kernel = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount, cnt, tid, start, end);
            int n0 = start*32;
            int n1 = end*32;
            if (n1 > N) n1 = N;
            tensor2D<bfloat16> copyA = matA.clone();
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(copyA, matB, n0, n1, ppkernel);
        };
        thp.Paralell_NT(kernel);
    }
};
#endif

int OMP_NT = omp_thread_count();

struct MatmulMTOMP {
    amx_bf16::Matmul::WeightPrecision rt_precision;
    std::vector<std::shared_ptr<amx_bf16::Matmul>> ops;
    bool transposeB;
    MatmulMTOMP(bool constB = false,
                bool transposeB = false,
                amx_bf16::Matmul::WeightPrecision precision=amx_bf16::Matmul::Weight_BF16) : transposeB(transposeB), rt_precision(precision) {
        for(int i = 0; i < OMP_NT; i++)
            ops.push_back(std::make_shared<amx_bf16::Matmul>(constB, transposeB, rt_precision));
    }

    template<typename P>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    P ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0:1];
        // split along N dimension
        int work_amount = rndup(N, 32)/32;

        auto kernel = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount, cnt, tid, start, end);
            int n0 = start*32;
            int n1 = end*32;
            if (n1 > N) n1 = N;
            //tensor2D<bfloat16> copyA = matA.clone();
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(matA, matB, n0, n1, ppkernel);
        };

        #pragma omp parallel for
        for(int i = 0; i<OMP_NT; i++) {
            kernel(i, OMP_NT);
        }
    }
};

void amx_MatmulMT_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul mm(true, transB, precision);
    MatmulMTOMP      mmMT(true, transB, precision);
    //amx_bf16::PP::Store2bf16 pp0(C0);
    //amx_bf16::PP::Store2bf16 pp(C);
    tensor2D<float> Bias0(1, N);
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp0(C0, &Bias0(0,0));
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp(C, &Bias0(0,0));

    timer.tag(__func__, "ST", M, K, N, precision)(times, [&](){
        mm(A, transB?BT:B, pp0);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);

    // only test perf on MultiThread case when
    // N is big enough to be divided into OMP_NT cores
    if (N >= 32*OMP_NT) {
        timer.tag(__func__, "MT", M, K, N, precision)(times, [&](){
            mmMT(A, transB?BT:B, pp);
        },
        double(M * N) * K * 2,
        AMXBf16PeakGops2PerCore * 1e9);

        if (C0 == C) {
            std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
            //std::cout << C << std::endl;
        } else {
            std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
            std::cout << C0 << std::endl;
            std::cout << C << std::endl;
            return;
        }
    }
}

void amx_MatmulMT_BiasGelu_acc(int M, int K, int N, bool transB) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias(1, N);
    amx_bf16::Matmul mm(true, transB);
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp0(C, &Bias(0,0));

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    C0 = 0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return x*0.5*(1 + std::erf(x/std::sqrt(2)));
    });
    {
        mm(A, transB?BT:B, pp0);
    }

    if (C0.compare(C, 0.001f)) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
    }
}

void amx_MatmulMT_BiasGelu_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias(1, N);
    amx_bf16::Matmul mm(true, transB);
    MatmulMTOMP      mmMT(true, transB);
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp0(C0, &Bias(0,0));
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp(C, &Bias(0,0));

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    timer(times, [&](){
        mm(A, transB?BT:B, pp0);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);

    timer(times, [&](){
        mmMT(A, transB?BT:B, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
        return;
    }
}



void amx_Matmul_perf_float(int M, int K, int N, int times = -1000) {
    tensor2D<float> A(M, K);
    tensor2D<float> B(K, N);
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);
    tensor2D<float> Bias(1, N);
    avx512::Matmul mm;
    avx512::PP::AddbiasRelu pp(&Bias(0,0));
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return std::max(x, 0.0f);
    });
    mm(A, B, C, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }

    timer(times, [&](){
        mm(A, B, C, pp);
    },
    double(M * N) * K * 2,
    FP32PeakGopsPerCore * 1e9);
}

void test_bf16() {
    for(int i=0;i<65536;i++) {
        auto bf16i = bfloat16(i);
        auto bf16i2i = static_cast<int>(static_cast<float>(bf16i));
        if (bf16i2i != i) {
            std::cout << "bfloat16 cannot represent int " << i << std::endl;
            break;
        }
    }
    {
        auto a = bfloat16(std::nan("1"));
        auto b = bfloat16(0.0f);
        auto c = a*b;
        std::cout << c << std::endl;
    }
    {
        tensor2D<bfloat16> A(16, 32);
        tensor2D<bfloat16> BT(16, 32);
        tensor2D<bfloat16> C(16, 16);
        tileconfig_t tfg(1, 0, 8, 16, 64);
        const int tileA = 0;
        const int tileB = 1;
        const int tileC = 2;
        A = bfloat16(std::nan("0"));
        BT = bfloat16(0.0f);
        _tile_loadd(tileA, &A(0,0), 64);
        _tile_loadd(tileB, &BT(0,0), 64);
        _tile_dpbf16ps(tileC, tileA, tileB);
        tshow<float, 2>();
    }

}

/*
repeate following topology 
   [M,K][K,N] => [M,N]   A1*B1=>A2
   [M,N][N,K] => [M,K]   A2*B2=>A1
   ...

*/
void amx_FC_MTML_perf(int M, int K, int N, int repeates, int times = -1000) {
    tensor2D<bfloat16> A1(M, K);
    tensor2D<bfloat16> A2(M, N);

    std::vector<tensor2D<bfloat16>> B1s;
    std::vector<tensor2D<bfloat16>> B2s;
    std::vector<tensor2D<float>> biasA1;
    std::vector<tensor2D<float>> biasA2;
    std::vector<MatmulMTOMP> FC1;
    std::vector<MatmulMTOMP> FC2;
    //MatmulMTOMP               mmMT(true, false, precision);

    for(int i = 0; i<repeates; i++) {
        B1s.emplace_back(K, N);
        B2s.emplace_back(N, K);
        biasA1.emplace_back(1, K);
        biasA2.emplace_back(1, N);
        // MatmulMTOMP internally will cache B matrix, so we need
        // multiple instances, one for each FC layer.
        FC1.emplace_back(true, false, precision);
        FC2.emplace_back(true, false, precision);
    }

    double elesize = (precision == amx_bf16::Matmul::Weight_BF16)? sizeof(bfloat16) : sizeof(int8_t);

    timer.tag(__func__, M, K, N, precision, repeates)(times, [&](){
        for(int i = 0; i<repeates; i++) {
            amx_bf16::PP::Addbias_Gelu_Store2bf16 ppToA2(A2, &biasA2[i](0,0));
            amx_bf16::PP::Addbias_Gelu_Store2bf16 ppToA1(A1, &biasA1[i](0,0));
            //amx_bf16::PP::Store2bf16 ppToA2(A2);
            //amx_bf16::PP::Store2bf16 ppToA1(A1);
            FC1[i](A1, B1s[i], ppToA2);
            FC2[i](A2, B2s[i], ppToA1);
        }
    },
    (double(N) * K * elesize) * 2 * repeates,
    1e12,
    "Byte/s");
}

void test_acc() {
    auto do_test_acc = [&](){
        amx_FC_acc(32*22, 10*32, 256);
        amx_FC_acc(32*22 + 1, 10*32, 256 + 1);
        amx_FC_acc(32*22 + 16, 10*32, 256 + 17);
        amx_FC_acc(32*22 + 31, 10*32, 256 + 15);
        amx_FC_acc(32*22 + 31, 10*32 + 1, 256 + 15);
        amx_FC_acc(32*22 + 31, 10*32 + 17, 256 + 15);
        amx_FC_acc(2, 10*32, 256);
    };
    precision = amx_bf16::Matmul::Weight_BF16;
    do_test_acc();
    precision = amx_bf16::Matmul::Weight_INT8;
    do_test_acc();
}

void test_perf() {
    auto do_test_perf = [&](){
        amx_FC_perf(32*28, 32*80, 10240);
        amx_FC_perf(32*28 + 1, 32*80, 10240);
        amx_FC_perf(32*28 + 16, 32*80, 10240);
        amx_FC_perf(32*28 + 17, 32*80, 10240);
        amx_FC_perf(32*28 + 31, 32*80, 10240);
        amx_FC_perf(32*28, 32*80, 10240);
        amx_FC_perf(32*28 + 1, 32*80, 10240);
        amx_FC_perf(32*28, 32*80 + 1, 10240);
        amx_FC_perf(32*28, 32*80, 10240 + 1);
        amx_FC_perf(32*28 + 1, 32*80 + 1, 10240 + 1);
        amx_FC_perf(32*28 + 32, 32*80 + 32, 10240 + 32);
        amx_FC_perf(896, 256, 1024, 10000);
        amx_FC_perf(896, 256, 1024, 10000);
    };
    precision = amx_bf16::Matmul::Weight_BF16;
    do_test_perf();
    precision = amx_bf16::Matmul::Weight_INT8;
    do_test_perf();
}

/*
 B matrix is 50MB, 56-cores took 2.8GB, so it can use almost full HBM bandwidth 600GB+
    test_parallel_FC_2_2560_10240_bf16 Avg latency:
        4420.87 us x 221  HW Usage : 66% (664.125 GByte/s /1000 GByte/s)
*/
void test_parallel_FC(int L, int M, int K, int N, int times = -5000) {
    tensor2D<bfloat16> A0(M, K);
    tensor2D<bfloat16> B0(K, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias0(1, N);

    struct mm_single_layer {
        tensor2D<bfloat16> A;
        tensor2D<bfloat16> B;
        tensor2D<bfloat16> C;
        tensor2D<float> Bias;
        int _N;
        std::shared_ptr<amx_bf16::Matmul> mm;
        void create(tensor2D<bfloat16> & Atemplate,
                    tensor2D<bfloat16> & Btemplate,
                    tensor2D<bfloat16> & Ctemplate,
                    tensor2D<float> & BiasTemplate) {
            A = Atemplate.clone();
            B = Btemplate.clone();
            C = Ctemplate.clone();
            Bias = BiasTemplate.clone();
            _N = B.dims[1];
            mm.reset(new amx_bf16::Matmul(true, false, precision));
        }
        void run() {
            // post-ops do nothing
            //amx_bf16::PP::Dummy ppkernel(C);
            amx_bf16::PP::Addbias_Gelu_Store2bf16 ppkernel(C, &Bias(0,0));
            (*mm.get())(A, B, 0, _N, ppkernel);
        }
    };

    struct mm_multi_layer {
        std::vector<mm_single_layer> mms;
        void create(int layers,
                    tensor2D<bfloat16> & Atemplate,
                    tensor2D<bfloat16> & Btemplate,
                    tensor2D<bfloat16> & Ctemplate,
                    tensor2D<float> & BiasTemplate) {
            mms.resize(layers);
            for(int i = 0; i < layers; i++) {
                mms[i].create(Atemplate, Btemplate, Ctemplate, BiasTemplate);
            }
        }
        void run() {
            for(auto & layer : mms) {
                layer.run();
            }
        }
    };

    std::vector<mm_multi_layer> mms(OMP_NT);

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        mms[i].create(L, A0, B0, C0, Bias0);
    }

    double elesize = (precision == amx_bf16::Matmul::Weight_BF16)? sizeof(bfloat16) : sizeof(int8_t);
    timer.tag(__func__, L, M, K, N, precision)(times, [&](){
        #pragma omp parallel
        {
            int i = omp_get_thread_num();
            mms[i].run();
        }
    },
    (double(N) * K * elesize * L) * OMP_NT,
    1e12,
    "Byte/s");
}

void test_parallel_FC() {
    precision = amx_bf16::Matmul::Weight_BF16;
    // K*N is same, but K is bigger, bandwidth usage is high & more stable
    while(1) {
        std::cout << "=========================\n";
        test_parallel_FC(1, 2, 25600, 1024);
        test_parallel_FC(1, 2, 25600, 1024);
        test_parallel_FC(1, 2, 25600, 1024);
        std::cout << "=========================\n";
        test_parallel_FC(1, 2, 2560, 10240);
        test_parallel_FC(1, 2, 2560, 10240);
        test_parallel_FC(1, 2, 2560, 10240);
        std::cout << "=========================\n";
        // multi-layer, bandwidth usage is very unstable
        test_parallel_FC(40, 2, 2560, 256);
        test_parallel_FC(40, 2, 2560, 256);
        test_parallel_FC(40, 2, 2560, 256);
    }
}






//=====================================================================================================

void unittest_base(tensor2D<bfloat16> & A,
                    tensor2D<bfloat16> & B,
                    tensor2D<float> & C) {
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pB0 = &B(0,0);
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
    // A0 A1 only load once
    _tile_loadd(A0, pA0, 64); pA0 += 16*32;
    _tile_loadd(A1, pA1, 64); pA1 += 16*32;
    for(int k = 0; k < K; k+=32) {
        _tile_loadd(B0, pB0, 64); pB0 += 16*32;
        _tile_loadd(B1, pB0, 64);  pB0 += 16*32;
        _tile_dpbf16ps(C00, A0, B0);
        _tile_dpbf16ps(C10, A1, B0);
        _tile_dpbf16ps(C01, A0, B1);
        _tile_dpbf16ps(C11, A1, B1);
    }
    _tile_stored(C00, &C(0,0), C.stride);
    _tile_stored(C01, &C(0,16), C.stride);
    _tile_stored(C10, &C(16,0), C.stride);
    _tile_stored(C11, &C(16,16), C.stride);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}


void unittest_halfB(tensor2D<bfloat16> & A,
                    tensor2D<bfloat16> & B,
                    tensor2D<float> & C) {
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pB0 = &B(0,0);
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
    _tile_loadd(A0, pA0, 64); pA0 += 16*32;
    _tile_loadd(A1, pA1, 64); pA1 += 16*32;
    for(int k = 0; k < K; k+=32) {
        _tile_loadd(B0, pB0, 64); pB0 += 16*32;
        _tile_dpbf16ps(C00, A0, B0);
        _tile_dpbf16ps(C10, A1, B0);
        //_tile_loadd(B1, pB0, 64);  pB0 += 16*32;
        _tile_dpbf16ps(C01, A0, B0);
        _tile_dpbf16ps(C11, A1, B0);
    }
    _tile_stored(C00, &C(0,0), C.stride);
    _tile_stored(C01, &C(0,16), C.stride);
    _tile_stored(C10, &C(16,0), C.stride);
    _tile_stored(C11, &C(16,16), C.stride);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}

void unittest_Wint8(tensor2D<bfloat16> & A,
                tensor2D<int8_t> & B,
                tensor2D<float> & C) {
    static tensor2D<bfloat16> Bbf16(16*32, 32);
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;

    // int8
    // load B using avx and de-quant
    int Bi8_stride = B.stride;

    float dequant_scale = 0.2f;
    auto dq_scale = _mm512_set1_ps(dequant_scale);

    auto dequant_16x32 = [&](int8_t * src, bfloat16 * dst) {
        //_mm_prefetch(src + Bi8_stride*64, _MM_HINT_NTA);
        for (int k = 0; k < 16; k++) {
            //_mm_prefetch(src + Bi8_stride*32, _MM_HINT_NTA);
            auto a = _mm_load_si128((__m128i*)src);
            auto b = _mm_load_si128((__m128i*)(src + 16));
            auto a_512 = _mm512_cvtepi8_epi32(a);
            auto b_512 = _mm512_cvtepi8_epi32(b);
            auto a_f = _mm512_cvtepi32_ps(a_512);
            auto b_f = _mm512_cvtepi32_ps(b_512);
            a_f = _mm512_mul_ps(a_f, dq_scale);
            b_f = _mm512_mul_ps(b_f, dq_scale);
            auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f);
            _mm512_store_epi32(dst, (__m512i)reg_out);
            src += Bi8_stride;
            dst += 32;
        }
    };

    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pB0 = &B(0,0);
    auto * pBbf160 = &Bbf16(0, 0);
    auto * pBbf161 = pBbf160 + 16*32;
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
    _tile_loadd(A0, pA0, 64); pA0 += 16*32;
    _tile_loadd(A1, pA1, 64); pA1 += 16*32;
    for(int k = 0; k < K; k+=32) {
        dequant_16x32(pB0, pBbf160); pB0 += 16*32;
        _tile_loadd(B0, pBbf160, 64);
        _tile_dpbf16ps(C00, A0, B0);
        _tile_dpbf16ps(C10, A1, B0);

        dequant_16x32(pB0, pBbf161); pB0 += 16*32;
        _tile_loadd(B1, pBbf161, 64);
        _tile_dpbf16ps(C01, A0, B1);
        _tile_dpbf16ps(C11, A1, B1);
    }

    _tile_stored(C00, &C(0,0), C.stride);
    _tile_stored(C01, &C(0,16), C.stride);
    _tile_stored(C10, &C(16,0), C.stride);
    _tile_stored(C11, &C(16,16), C.stride);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}

void unittest_WFakeint8(tensor2D<bfloat16> & A,
                         tensor2D<bfloat16> & B,
                         tensor2D<float> & C) {
    static tensor2D<bfloat16> Bbf16(16*32, 32);
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;

    // int8
    // load B using avx and de-quant
    int B_stride = B.stride;
    float dequant_scale = 0.2f;
    bfloat16 v0 = B(0,0) * 0.00001f;
    auto delta = _mm512_set1_epi16(*reinterpret_cast<int16_t*>(&v0));

    auto fake_dequant_i8_16x32 = [&](int8_t * & src, bfloat16 * dst) {
        for (int k = 0; k < 16; k+=2) {
            auto a = _mm512_load_si512((__m512i*)src);
            _mm512_store_si512(dst, a);
            _mm512_store_si512(dst + 32, a);
            src += B_stride;
            dst += 32*2;
        }
    };

    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pBi8 = reinterpret_cast<int8_t*>(&B(0,0));
    auto * pBbf160 = &Bbf16(0, 0);
    auto * pBbf161 = pBbf160 + 16*32;
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
    _tile_loadd(A0, pA0, 64); pA0 += 16*32;
    _tile_loadd(A1, pA1, 64); pA1 += 16*32;
    for(int k = 0; k < K; k+=32) {
        fake_dequant_i8_16x32(pBi8, pBbf160);
        _tile_loadd(B0, pBbf160, 64);
        _tile_dpbf16ps(C00, A0, B0);
        _tile_dpbf16ps(C10, A1, B0);

        fake_dequant_i8_16x32(pBi8, pBbf161);
        _tile_loadd(B1, pBbf161, 64);
        _tile_dpbf16ps(C01, A0, B1);
        _tile_dpbf16ps(C11, A1, B1);
    }

    _tile_stored(C00, &C(0,0), C.stride);
    _tile_stored(C01, &C(0,16), C.stride);
    _tile_stored(C10, &C(16,0), C.stride);
    _tile_stored(C11, &C(16,16), C.stride);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}



void unittest_WFakeint4(tensor2D<bfloat16> & A,
                         tensor2D<bfloat16> & B,
                         tensor2D<float> & C) {
    static tensor2D<bfloat16> Bbf16(16*32, 32);
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;

    // int8
    // load B using avx and de-quant
    int B_stride = B.stride;
    float dequant_scale = 0.2f;
    bfloat16 v0 = B(0,0) * 0.00001f;
    auto delta = _mm512_set1_epi16(*reinterpret_cast<int16_t*>(&v0));

    auto fake_dequant_i4_16x32 = [&](int8_t * & src, bfloat16 * dst) {
        for (int k = 0; k < 16; k+=4) {
            auto a = _mm512_load_si512((__m512i*)src);  // read 32 bf16
            _mm512_store_si512(dst, a);
            _mm512_store_si512(dst + 32, a);
            _mm512_store_si512(dst + 32*2, a);
            _mm512_store_si512(dst + 32*3, a);
            src += B_stride;
            dst += 32*4;
        }
    };

    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pBi4 = reinterpret_cast<int8_t*>(&B(0,0));
    auto * pBbf160 = &Bbf16(0, 0);
    auto * pBbf161 = pBbf160 + 16*32;
    _tile_zero(C00);
    _tile_zero(C01);
    _tile_zero(C10);
    _tile_zero(C11);
    _tile_loadd(A0, pA0, 64); pA0 += 16*32;
    _tile_loadd(A1, pA1, 64); pA1 += 16*32;
    for(int k = 0; k < K; k+=32) {
        fake_dequant_i4_16x32(pBi4, pBbf160);
        _tile_loadd(B0, pBbf160, 64);
        _tile_dpbf16ps(C00, A0, B0);
        _tile_dpbf16ps(C10, A1, B0);

        fake_dequant_i4_16x32(pBi4, pBbf161);
        _tile_loadd(B1, pBbf161, 64);
        _tile_dpbf16ps(C01, A0, B1);
        _tile_dpbf16ps(C11, A1, B1);
    }

    _tile_stored(C00, &C(0,0), C.stride);
    _tile_stored(C01, &C(0,16), C.stride);
    _tile_stored(C10, &C(16,0), C.stride);
    _tile_stored(C11, &C(16,16), C.stride);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}


void unittest_avx512(tensor2D<bfloat16> & A,
                            tensor2D<bfloat16> & B,
                            tensor2D<float> & C) {
    static tensor2D<bfloat16> Bbf16(16*32, 32);
    int K = A.dims[1];
    assert(A.dims[0] == 32);
    assert(B.dims[1] == 32);
    assert(B.dims[0] == K);
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;

    // int8
    // load B using avx and de-quant
    int B_stride = B.stride;
    float dequant_scale = 0.2f;
    bfloat16 v0 = B(0,0) * 0.00001f;
    auto delta = _mm512_set1_epi16(*reinterpret_cast<int16_t*>(&v0));

    auto fake_dequant_i4_16x32 = [&](int8_t * & src, bfloat16 * dst) {
        for (int k = 0; k < 16; k+=4) {
            auto a = _mm512_load_si512((__m512i*)src);  // read 32 bf16
            _mm512_store_si512(dst, a);
            _mm512_store_si512(dst + 32, a);
            _mm512_store_si512(dst + 32*2, a);
            _mm512_store_si512(dst + 32*3, a);
            src += B_stride;
            dst += 32*4;
        }
    };

    auto * pA0 = &A(0,0);
    auto * pA1 = &A(16,0);
    auto * pB0 = &B(0,0);
    auto * pBi4 = reinterpret_cast<int8_t*>(&B(0,0));
    auto * pBbf160 = &Bbf16(0, 0);
    auto * pBbf161 = pBbf160 + 16*32;
    
    __m512 rC[16];
    for(int i = 0; i < 16; i++)  rC[i] = _mm512_setzero();

    auto avx512_bf16_dp = [&rC](bfloat16 * srcA, bfloat16 * srcB) {
        for(int i = 0; i < 16; i++) {
            // load A, B
            _mm_prefetch(srcB + (16 + i)*32, _MM_HINT_NTA);
            auto rA = _mm512_load_si512(srcA);
            auto rB = _mm512_load_si512(srcB);
            rC[i] = _mm512_dpbf16_ps(rC[i], (__m512bh)rA, (__m512bh)rB);
            srcA += 32;
            srcB += 32;
        }
    };
    for(int k = 0; k < K; k+=32) {
        // B0 B1 requires 32x32*sizeof(bfloat16) = 2KB
        // 50% L2 cache: 1MB = 512
        
        //fake_dequant_i4_16x32(pBi4, pBbf160);
        //avx512_bf16_dp(pA0, pBbf160);
        avx512_bf16_dp(pA0, pB0); pB0 += 16*32;
        avx512_bf16_dp(pA1, pB0); pB0 += 16*32;

        //fake_dequant_i4_16x32(pBi4, pBbf161);
        //avx512_bf16_dp(pA1, pBbf161);
    }
    for(int i = 0; i < 16; i++)
        _mm512_store_ps(&C(i,0), rC[i]);
    if(!C.is_normal()) std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}


int amx_unit_test_int8(int K) {

    tensor2D<bfloat16> A(32, K);
    tensor2D<bfloat16> B(K, 32);    // assume in each 32x32 unit, it's already packed as 2x(16x(16x2))
    tensor2D<int8_t> Bi8(K, 32, true);
    tensor2D<float> C(32, 32);
    tileconfig_t tfg(1, 0, 8, 16, 64);

    std::cout << "# K = " << K/32 << "*32 = " << K << ", sizeof(B)=" << double(B.capacity) / 1024/1024 << " MB" << std::endl;

    timer.tag(__func__, "base ")(-1000, [&](){
        unittest_base(A, B, C);
    },
    double(32 * K) * sizeof(bfloat16), // only B memory is accessed
    18e9, "B/s"
    );


    timer.tag(__func__, "halfB")(-1000, [&](){
        unittest_halfB(A, B, C);
    },
    double(32 * K) * sizeof(int8_t), // only B memory is accessed
    18e9, "B/s"
    );

    timer.tag(__func__, "Wint8")(-1000, [&](){
        unittest_Wint8(A, Bi8, C);
    },
    double(32 * K) * sizeof(int8_t), // memory accessed
    18e9, "B/s"
    );

    timer.tag(__func__, "WFakeint8")(-1000, [&](){
        unittest_WFakeint8(A, B, C);
    },
    double(32 * K) * sizeof(int8_t), // memory accessed
    18e9, "B/s"
    );


    timer.tag(__func__, "WFakeint4")(-1000, [&](){
        unittest_WFakeint4(A, B, C);
    },
    double(32 * K) * sizeof(int8_t)/2, // memory accessed
    18e9, "B/s"
    );


    timer.tag(__func__, "avx512")(-1000, [&](){
        unittest_avx512(A, B, C);
    },
    double(32 * K) * sizeof(bfloat16), // memory accessed
    18e9, "B/s"
    );


    return 0;
}

//=====================================================================================================
int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);
    //thp.Start();

    //test_all_bw(3.0); return 0;
    //test_parallel_FC();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    // K=12*32, A+B fits L1D, gives 100% HW usage
    // K=80*32  A+B fits L2, gives 70% HW usage
    // K=512*32 A+B fits L2, gives 30% HW usage
    //K = 80*32;
    //K = 51200*32;
    amx_unit_test_int8(51200*32);
    amx_unit_test_int8(80*32);
    return 0;
    //test_acc();
    //test_perf();

    precision = amx_bf16::Matmul::Weight_BF16;
    amx_FC_acc(2, 10*32 + 17, 256 + 15);
    precision = amx_bf16::Matmul::Weight_INT8;
    amx_FC_acc(2, 10*32 + 17, 256 + 15);

    precision = amx_bf16::Matmul::Weight_BF16;
    amx_MatmulMT_perf(32, 2560, 320, false, -10000);
    amx_MatmulMT_perf(32, 2560, 320, false, -10000);

    precision = amx_bf16::Matmul::Weight_INT8;
    amx_MatmulMT_perf(32, 2560, 320, false, -10000);
    amx_MatmulMT_perf(32, 2560, 320, false, -10000);

    return 0;

    precision = amx_bf16::Matmul::Weight_BF16;
    amx_MatmulMT_perf(2, 2560, 10752, false, -1000);
    amx_MatmulMT_perf(2, 2560, 10752, false, -1000);
    //amx_MatmulMT_perf(2, 2560, 10752, false, -1000);
    precision = amx_bf16::Matmul::Weight_INT8;
    amx_MatmulMT_perf(2, 2560, 10752, false, -1000);
    amx_MatmulMT_perf(2, 2560, 10752, false, -1000);

    precision = amx_bf16::Matmul::Weight_BF16;
    amx_FC_MTML_perf(2, 2560, 10752, 20, -10000);
    amx_FC_MTML_perf(2, 2560, 10752, 20, -10000);
    precision = amx_bf16::Matmul::Weight_INT8;
    amx_FC_MTML_perf(2, 2560, 10752, 20, -10000);
    amx_FC_MTML_perf(2, 2560, 10752, 20, -10000);
    return 0;
    // return 0;

    //test_bf16(); return 0;
    //amx_Matmul_perf(12, 256, 32, true); return 0;

    amx_Matmul_perf_float(16, 256, 256);
    amx_Matmul_perf_float(224, 256, 256);
    amx_Matmul_perf_float(512, 256, 256);
    
    //amx_Matmul_perf(32, 120, 5, true); return 0;
    //amx_Matmul_perf(32, 18, 5, true); return 0;

    //amx_FC_perf(32, 5120, 32, -1000); return 0;
    //amx_Matmul_perf(928, 96, 928, true); return 0;

    amx_MatmulMT_BiasGelu_acc(88, 77, 66, false);
    amx_MatmulMT_perf(2*901, 2560, 7680, false);
    amx_MatmulMT_BiasGelu_perf(2*901, 2560, 7680, false);


    amx_Matmul_perf(928, 96, 928, true);
    amx_Matmul_perf(901, 80, 901, true);
    amx_Matmul_perf(901, 901, 80, false); 

    test_blk_loops();

    amx_unit_test_gemAvB(901, 80);
    return 0;
}
