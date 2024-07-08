import argparse
import os
import subprocess

# configuration from SPR 56 cores(US SPR: 10.242.51.94)
# bandwidth(bytes/cycle) from test `build test_linear_mk.cpp` which buffer size is 512 KB
# L2
L2_read = 50
L2_write = 16
L2_copy = 13
# L3 from test `build test_linear_mk.cpp` which buffer size is 512*100 KB, also use `./mlc --loaded_latency -d0 -b60m -u -T -k0-55`
L3_read = 11.45
L3_write = 9.57
AMX_ops = 512           # for bf16 case
# DDR
DDR_read = 2.57

L1_size = 48 * 1024
L2_size = 2 * 1024 * 1024
L3_size = 105 * 1024 * 1024
Core_number = 56
Core_freq = 1_800_000_000
L2_size_safe = L2_size * 0.6  # safe size for buffer

AB_element_size = 2     # for bf16 case
C_element_size = 4
# 2x2 register blocking aka 32x32
N_blk = 32
M_blk = 32

def rnd_up(v, aligned):
    return (v + aligned - 1) // aligned * aligned

# compute a block cost(register block is 2x2), loop order:
# then [32, K] * [K, 32]                                  K
# then if A > B, should loop N, or loop M                 N(A>B), M(A<B)
# then M or N                                             M(A>B), N(A<B)
def get_cost_kernel(M, N, K, K_sub=None, show_warn=False):
    if not K_sub:
        K_sub = K
    A_size = M * K * AB_element_size
    B_size = N * K * AB_element_size
    A_size_sub = M * K_sub * AB_element_size
    B_size_sub = N * K_sub * AB_element_size
    C_size = M * N * C_element_size
    # The goal of blocking:
    # 1, A(32*K) + B(K*32) + C(32*32) should fit in L2
    # 2.1, if M > N, B will be reused many times and should sit in L2
    # 2.2  if M < N, A will be reused many times and should sit in L2 
    assert M_blk * K_sub * AB_element_size + N_blk * K_sub * AB_element_size + M_blk * N_blk * C_element_size < L2_size, f"kernel size must fit in L2, problem {M=} {N=} {K=} {K_sub=}"
    if M >= N:
        assert M_blk * K_sub * AB_element_size + N * K_sub * AB_element_size + M_blk * N_blk * C_element_size < L2_size, f"when loop N first, B should fit in L2 to fully reuse, problem {N=} {K_sub=}"
    else:
        assert N_blk * K_sub * AB_element_size + M * K_sub * AB_element_size + M_blk * N_blk * C_element_size < L2_size, f"when loop M first, A should fit in L2 to fully reuse, problem {M=} {K_sub=}"

    # default cost
    A_readtimes = N / N_blk
    B_readtimes = M / M_blk
    AB_cost_L3 = (A_size + B_size) / L3_read
    AB_cost_L2 = A_size * (A_readtimes - 1) / L2_read + B_size * (B_readtimes - 1) / L2_read
    C_cost_L2 = C_size / L2_write
    cost = AB_cost_L3 + AB_cost_L2 + C_cost_L2
    # fix: A+B fits in L2
    if A_size > L1_size and B_size > L1_size and A_size + B_size <= L2_size:
        if show_warn:
            print('bottleneck is L2, the cost will be computed using accessed L2 size')
        if A_size_sub + B_size_sub + C_size > L2_size:
            if show_warn:
                print(f'WARNING: A+B+C size:{(A_size_sub + B_size_sub + C_size) / 1024:,.0f}KB is larger than L2 size:{L2_size / 1024:,.0f}KB, the estimation may be smaller than the real')

        A_cost = A_size * A_readtimes / L2_read
        B_cost = B_size * B_readtimes / L2_read
        C_cost = C_size / L2_write
        cost = A_cost + B_cost + C_cost
    # fix: one of A or B is larger than L2, the other is less than L2
    elif (A_size >= L2_size and L1_size < B_size < L2_size and L2_size < A_size + B_size + C_size < L3_size) or \
         (B_size >= L2_size and L1_size < A_size < L2_size and L2_size < A_size + B_size + C_size < L3_size):
        if show_warn:
            print('bottleneck is L2 and L3, the cost will be computed using accessed L2+L3 size.')
            print('WARNING: when A+B size is close to L2 size, the exact size in L2/L3 is unknown but it will greatly affect the estimation of accuary')
            print('WARNING: when A+B size is larger than L2 size, L2 to L1 may be overlapped with L3 to L2, this will make the estimation be larger than the real')
        if M >= N:
            # loop order: K, N, M
            A_cost_L3 = A_size / L3_read                        # when A is close to L2, some part may be in L2!
            A_cost_L2 = A_size * (A_readtimes - 1) / L2_read    # after first iter, A will be hit in L2
            A_cost = A_cost_L2 + A_cost_L3
            B_cost = B_size * B_readtimes / L2_read
        else:
            # loop order: K, M, N
            B_cost_L3 = B_size / L3_read
            B_cost_L2 = B_size * (B_readtimes - 1) / L2_read
            B_cost = B_cost_L2 + B_cost_L3
            A_cost = A_size * A_readtimes / L2_read
        C_cost = C_size / L3_write
        cost = A_cost + B_cost + C_cost
    # fix: one of A or B is less than L2
    elif A_size_sub < L1_size and L2_size > B_size_sub > L1_size:
        if show_warn:
            print('bottleneck is computation')
            print('WARNING: part of write time will be overlapped with computation and it may impact the estimation accuracy')
        computation_cost = M * N * K / AMX_ops
        C_cost = C_size / L2_write
        cost = computation_cost + C_cost
    else:
        if show_warn:
            print(f'WARNING: unknown case with {M=}, {N=}, {K=}, {K_sub=}')

    if K != K_sub:
        K_block_no = K // K_sub
        cost += C_size * (K_block_no - 1) / L2_read + C_size * (K_block_no - 1) / L2_write
    return cost

def run_case(M, N, K, M_sub, N_sub, K_sub):
    # amx_mm
    subprocess.call(f'BM={M_sub} BN={N_sub} BK={K} KS={K_sub} numactl -m1 -C60 ./a.out', shell=True)
    # onednn: numactl -C0-55 ./benchdnn --mode=p --ip --dir=FWD_I --dt=bf16:bf16:f32  mb8192ic1024oc1024_n
    subprocess.call(f'numactl -C56-111 -m1 ./thirdparty/oneDNN/build/tests/benchdnn/benchdnn --mode=p --ip --dir=FWD_I --dt=bf16:bf16:f32 mb{M}ic{K}oc{N}_n', shell=True)
    print('\n')

def get_cost(B, M, N, K, run_test=False):
    print(f'{B=} {M=} {N=} {K=} with {Core_number=}')
    # ideal block number of M for min load latency
    M_block_no_min_latency = (B * M / (N / Core_number)) ** 0.5

    K_sub_block = K
    cost_min = 1e9
    K_sub_best = K_sub_block
    M_best = M
    N_best = N
    while True:
        # min block number of M if sub A should fit in L2
        def get_min_m_block_no_fit_A_in_L2():
            return (B * M * K_sub_block * AB_element_size) / (L2_size_safe - N_blk * K_sub_block * AB_element_size - M_blk * N_blk * C_element_size)
        M_block_no_min_fit_A_in_L2 = get_min_m_block_no_fit_A_in_L2()
        # min block number of N if sub B should fit in L2
        def get_max_m_block_no_fit_B_in_L2():
            return (L2_size_safe - M_blk * K_sub_block * AB_element_size - M_blk * N_blk * C_element_size) * Core_number / (N * K_sub_block * AB_element_size)
        M_block_no_max_fit_B_in_L2 = get_max_m_block_no_fit_B_in_L2()
        def try_m_block(M_block_no):
            nonlocal K_sub_best, M_best, N_best, cost_min
            M_block = B * M // M_block_no
            N_block = N // (Core_number // M_block_no)
            M_block = rnd_up(M_block, 32)
            N_block = rnd_up(N_block, 32)
            K_block = rnd_up(K, 32)
            cost = get_cost_kernel(M_block, N_block, K_block, K_sub_block)
            if cost < cost_min:
                cost_min = cost
                K_sub_best = K_sub_block
                M_best = M_block
                N_best = N_block
            print(f'{M_block=} {N_block=} realN {N / (Core_number / M_block_no):.0f} {M_block * N_block * K * 2} {cost} ')
            return cost, M_block * N_block * K * 2 / cost

        # prefer loop N
        M_block_no_prefer_N = -1000000
        diff = 1e9
        for i in range(int(M_block_no_max_fit_B_in_L2), 0, -1):
            if B * M % i == 0:
                if abs(i - M_block_no_min_latency) < diff:
                    M_block_no_prefer_N = i
                    diff = abs(i - M_block_no_min_latency)
        cost = -1
        ops = 0
        if M_block_no_prefer_N > 0:
            cost, ops = try_m_block(M_block_no_prefer_N)
        def log_m_block(m_block, hint):
            print(f'{K_sub_block=} expected M_block={M_block_no_min_latency:.1f} sub A M [{M_block_no_min_fit_A_in_L2:.1f}, {Core_number}] sub B M [1, {M_block_no_max_fit_B_in_L2:.1f}], M_block({hint})={m_block}, {cost=:,.0f}, {ops=:.0f}')
        log_m_block(M_block_no_prefer_N, 'prefer N')

        M_block_no_prefer_M = -1000000
        diff = 1e9
        for i in range(int(M_block_no_min_fit_A_in_L2 + 0.9999999), Core_number + 1):
            if B * M % i == 0:
                if abs(i - M_block_no_min_latency) < diff:
                    M_block_no_prefer_M = i
                    diff = abs(i - M_block_no_min_latency)
        cost = -1
        ops = 0
        if M_block_no_prefer_M > 0:
            cost, ops = try_m_block(M_block_no_prefer_M)
        log_m_block(M_block_no_prefer_M, 'prefer M')

        K_sub_block //= 2
        if K_sub_block < 256:
            break
    
    if cost_min >= 1e9 - 1:
        print(f'could not get the blocking scheme')
    else:
        print(f'Best: {M_best=} {N_best=} {K=} {K_sub_best=} cost {cost_min:,.0f} cycles, ops {M_best * N_best * K * 2 / cost_min:.0f}, {cost_min / Core_freq * 1000:.3f} ms@{Core_freq / 1000 / 1000 /1000 :.2f}GHz\n')
        if run_test:
            run_case(B * M, N, K, M_best, N_best, K_sub_best)

def compute_mt_cases(run_test):
    shapes = [
        # BM=1024 BN=448 BK=1024 build tests/amx-mm.cpp, 0.807ms
        # estimate: 0.837ms
        {"B": 8, "M": 1024, "N" : 3072, "K": 1024},
        # BLOCK_DETAIL=1 ONEDNN_VERBOSE=0 numactl -C0-55 ./benchdnn --mode=p --ip --dir=FWD_I --dt=bf16:bf16:f32  mb8192ic1024oc1024_n, 0.332ms
        # BM=1024 BN=160 BK=1024 build tests/amx-mm.cpp, 0.325ms
        # estimate: 0.35ms
        {"B": 8, "M": 1024, "N" : 1024, "K": 1024},
        {"B": 8, "M": 1024, "N" : 4096, "K": 1024},
        {"B": 8, "M": 1024, "N" : 1024, "K": 4096},
        # {"B": 1, "M": 256, "N" : 1024*4*3, "K": 1024*4},
    ]
    for shape in shapes:
        get_cost(shape['B'], shape['M'], shape['N'], shape['K'], run_test)

def compute_st_cases():
    shapes = [
        {"M": 256, "N" : 224, "K": 4096, "K_sub": 2048},

    ]
    for shape in shapes:
        cost = get_cost_kernel(shape['M'], shape['N'], shape['K'], shape['K_sub'], True)
        print(f"M={shape['M']} N={shape['N']} K={shape['K']} KS={shape['K_sub']} {cost=:,.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--mt', nargs='?', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--test', nargs='?', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    args = parser.parse_args()
    if args.mt:
        compute_mt_cases(args.test)
    else:
        compute_st_cases()

# TODO: extend more case for get_cost