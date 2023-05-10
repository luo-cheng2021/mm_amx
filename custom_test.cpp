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
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"
#include "tensor2D.hpp"
#include "fc_custom.hpp"
#include "dnnl_thread.hpp"

timeit timer;

using ov::bfloat16;

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();
int OMP_NT = dnnl_get_max_threads();
// std::ostream & operator<<(std::ostream & os, Matmul::WeightPrecision & prec) {
//     static const char* names_prec[] = {
//     "bf16",
//     "int8",
//     "int4"
//     };
//     os << names_prec[(int)prec];
//     return os;
// }

void amx_FC_MTML_perf_gpt_i8(int M, int K, int N, int layer_num, int times = -1000) {
    struct Layer {
        tensor2D<int8_t> A0;
        tensor2D<int8_t> A1;
        tensor2D<int8_t> A2;
        tensor2D<int8_t> A3;
        tensor2D<int8_t> B0;
        tensor2D<int8_t> B1;
        tensor2D<int8_t> B2;
        tensor2D<int8_t> B3;
        tensor2D<int8_t> C0;
        tensor2D<int8_t> C1;
        tensor2D<int8_t> C2;
        tensor2D<int8_t> C3;
        std::vector<FC> fc;
        tensor2D<float> q;
        Layer(int M) : A0(2, 2560), A1(2, 2560), A2(2, 2560), A3(2, 2560), 
                       B0(2560, 2560), B1(2560, 2560), B2(2560, 2560), B3(2560, 2560), 
                       C0(2, 2560), C1(2, 2560), C2(2, 2560), C3(2, 2560), q(1, 2560)
        {

        }
    };

    std::vector<Layer> layers;
    for (int i = 0; i < layer_num; i++) {
        layers.emplace_back(M);
        Layer& layer = layers.back();
        layer.fc.emplace_back();
        layer.fc.back().init(OMP_NT, FC::FCType_S8);
        layer.fc.emplace_back();
        layer.fc.back().init(OMP_NT, FC::FCType_S8);
        layer.fc.emplace_back();
        layer.fc.back().init(OMP_NT, FC::FCType_S8);
        layer.fc.emplace_back();
        layer.fc.back().init(OMP_NT, FC::FCType_S8);
    }
    size_t b_size = (layers[0].B0.capacity + layers[0].B1.capacity + layers[0].B2.capacity + layers[0].B3.capacity) * layer_num * 90 +
                    (layers[0].A0.capacity + layers[0].A1.capacity + layers[0].A2.capacity + layers[0].A3.capacity +
                     layers[0].C0.capacity + layers[0].C1.capacity + layers[0].C2.capacity + layers[0].C3.capacity) * layer_num * 90;

    timer.tag(__func__, M, K, N, "int8")(times, [&](){
        for (int k = 0; k < 90; k++) {
            for (int j = 0; j < layer_num; j++) {
                // amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::QUANT> pp_qkv(layers[j].C0);
                // amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::QUANT> pp_dense(layers[j].C1);
                // amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::QUANT> pp_h_to_4h(layers[j].C2);
                // amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::QUANT> pp_4h_to_h(layers[j].C3);
                // {
                //     pp_qkv.set_q_scale(0.2);
                //     pp_dense.set_q_scale(0.2);
                //     pp_h_to_4h.set_q_scale(0.2);
                //     pp_4h_to_h.set_q_scale(0.2);
                // }
                layers[j].fc[0].fc_s8s8s8_dq_q(layers[j].A0.data.get(), layers[j].B0.data.get(), layers[j].C0.data.get(), layers[j].A0.dims[0], layers[j].B0.dims[0], layers[j].A0.dims[1], layers[j].q.data.get(), layers[j].q.data.get());
                layers[j].fc[1].fc_s8s8s8_dq_q(layers[j].A1.data.get(), layers[j].B1.data.get(), layers[j].C1.data.get(), layers[j].A1.dims[0], layers[j].B1.dims[0], layers[j].A1.dims[1], layers[j].q.data.get(), layers[j].q.data.get());
                layers[j].fc[2].fc_s8s8s8_dq_q(layers[j].A2.data.get(), layers[j].B2.data.get(), layers[j].C2.data.get(), layers[j].A2.dims[0], layers[j].B2.dims[0], layers[j].A2.dims[1], layers[j].q.data.get(), layers[j].q.data.get());
                layers[j].fc[3].fc_s8s8s8_dq_q(layers[j].A3.data.get(), layers[j].B3.data.get(), layers[j].C3.data.get(), layers[j].A3.dims[0], layers[j].B3.dims[0], layers[j].A3.dims[1], layers[j].q.data.get(), layers[j].q.data.get());
                // layers[j].fc[1](layers[j].A1, layers[j].B1, pp_dense);
                // layers[j].fc[2](layers[j].A2, layers[j].B2, pp_h_to_4h);
                // layers[j].fc[3](layers[j].A3, layers[j].B3, pp_4h_to_h);
            }
        }
    },
    (double(b_size)),
    1e12,
    "Byte/s");
}

//=====================================================================================================
int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);
    //thp.Start();
    //test_all_bw(3.0); return 0;
    //test_parallel_FC();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    //std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();
    // amx_FC_MTML_perf_gpt_bf16(2, 1, 1, 32, 10);
    // precision = Matmul::Weight_INT8;
    // amx_FC_MTML_perf_gpt_w8(2, 1, 1, 32, 10);
    amx_FC_MTML_perf_gpt_i8(2, 1, 1, 32, 100);
    amx_FC_MTML_perf_gpt_i8(2, 1, 1, 32, 100);
    //amx_FC_MTML_perf_gpt_i8(2, 1, 1, 32, 10);
    // amx_FC_MTML_perf_gpt_bf16(2, 1, 1, 32, 10);
    // precision = Matmul::Weight_INT8;
    // amx_FC_MTML_perf_gpt_w8(2, 1, 1, 32, 10);
    return;
}
