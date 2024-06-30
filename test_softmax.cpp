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


// vfmadd132ps ymm(8 floats)  Throughput (CPI)=0.5
const double vfmaddOpsPerCycle = 16;


int OMP_NT = omp_thread_count();

int test_softmax() {
    tensor2D<float> x;
    tensor2D<float> y0;
    tensor2D<ov::bfloat16> y1;
    float scale = 0.25f;
    float eps = 1e-5;
    bool inside_sqrt = true;

    auto ref = [&](tensor2D<float>& x, float scale) {
        auto len = x.dims[1];
        auto a = &x[0];
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] * scale - max * scale);
            sum += a[i];
        }
        float s = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= s;
        }
    };
    int errors = 0;
    for(int N = 1; N < 129; N++) {
        x.resize(1, N);
        y1.resize(1, N);
        x.fill_rnd();
        y0 = x.clone();
        ref(x, scale);
        attn_softmax_kernel(&y0[0],
                            &y1[0],
                            scale,
                            0,
                            0,
                            0,
                            false,
                            N,
                            N,
                            false);
        for(int i=0;i<N;i++) {
            if (abs((x[i] - y1[i])/x[i]) > 0.01f) {
                errors ++;
                std::cout << "#" << i << "/" << N << ":  " <<x[i] << " vs " << y1[i] << " diff " << (x[i] - y1[i]) << std::endl;
            }
        }
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }
    {
        tensor2D<float> x;
        int N = 1024;
        x.resize(32, N);
        x = 0.1f;
        benchmark.tag(__func__, N, "softmax")(1000, [&](){
            for (int i = 0; i < 32; i++)
                attn_softmax_kernel(&x(i, 0),
                                    &x(i, 0),
                                    scale,
                                    0,
                                    0,
                                    0,
                                    false,
                                    x.dims[1],
                                    x.dims[1],
                                    false);
        });
    }
    return 0;
}

int main(int argc, const char *argv[]) {
    benchmark.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    test_softmax();

    return 0;
}
