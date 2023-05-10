if ! which icx > /dev/null; then
source ~/intel/oneapi/setvars.sh
fi

source=custom_test.cpp


#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# target=a`git rev-parse --short HEAD`.out
target=t.out

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

# omp
#COMMON_OPTS="-DDNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_OMP -DENABLE_NUMA -I$SCRIPT_DIR/include -L. -lcustom -lpthread -march=native -std=c++14 -lstdc++ -lnuma -qopenmp"
# tbb
COMMON_OPTS="-DDNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_TBB -DENABLE_NUMA -I$SCRIPT_DIR/tbb/include -I$SCRIPT_DIR/include -L. -L`pwd`/tbb/lib/ -ltbb -lcustom -lpthread -march=native -std=c++14 -lstdc++ -lnuma"

icx $source -O2 $COMMON_OPTS -o $target &&
icx $source -O0 $COMMON_OPTS -g -o debug-test.out &&
echo $target is generated &&
echo debug-test.out is generated &&
echo ======== test begin========== &&
echo LD_LIBRARY_PATH=.:./tbb/lib/:\$LD_LIBRARY_PATH numactl --localalloc -C 0-55 ./$target &&
LD_LIBRARY_PATH=.:./tbb/lib/:$LD_LIBRARY_PATH numactl --localalloc -C 0-55 ./$target
