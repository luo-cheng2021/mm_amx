if ! which icx > /dev/null; then
source ~/intel/oneapi/setvars.sh
fi

source=fc.cpp

if ! test -f "${source}"; then
    echo "cannot find input source cpp file: '$source'"
    exit 1
fi

#https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# target=a`git rev-parse --short HEAD`.out
target=libcustom.so

# g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

# omp
#COMMON_OPTS="-DDNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_OMP -DENABLE_NUMA -I$SCRIPT_DIR/include -I$SCRIPT_DIR/include -lpthread -march=native -std=c++14 -lstdc++ -lnuma -qopenmp"
# tbb
COMMON_OPTS="-DDNNL_CPU_THREADING_RUNTIME=DNNL_RUNTIME_TBB -DENABLE_NUMA -I$SCRIPT_DIR/include -I$SCRIPT_DIR/tbb/include -L$SCRIPT_DIR/tbb/lib -ltbb -lpthread -march=native -std=c++14 -lstdc++ -lnuma"

icx $source -O2 $COMMON_OPTS -S -masm=intel -fverbose-asm  -o _shared.s &&
cat _shared.s | c++filt > shared.s &&
icx $source -O2 $COMMON_OPTS -fPIC -shared -o $target &&
icx $source -O0 $COMMON_OPTS -fPIC -shared -g -o debug-s.out &&
echo $target is generated &&
echo shared.s is generated &&
echo debug-s.out is generated
#echo ======== test begin========== &&
#echo numactl --localalloc -C 0-55 ./$target &&
#numactl --localalloc -C 0-55 ./$target