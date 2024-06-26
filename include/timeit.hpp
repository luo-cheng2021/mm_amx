#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <stdio.h>
#include <thread>
#include <tuple>
#include <vector>

// _rdpmc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

uint64_t rdtsc_calibrate(int seconds = 1) {

    uint64_t start_ticks;
    std::cout << "rdtsc is calibrating ... " << std::flush;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    auto tsc_diff = (__rdtsc() - start_ticks);
    std::cout << "done." << std::endl;
    return tsc_diff / seconds;
}

struct RDTSC {
    uint64_t tsc_ticks_per_second;
    RDTSC() {
        tsc_ticks_per_second = rdtsc_calibrate();
        name = nullptr;
    }

    uint64_t st;
    const char* name;
    void start(const char* _name = nullptr) {
        if (name) {
            double dt = (__rdtsc() - st) * 1.0 / tsc_ticks_per_second;
            std::cout << " [RDTSC] : " << name << " took " << dt * 1e6 << " us" << std::endl;
            name = nullptr;
        }
        name = _name;
        st = __rdtsc();
    }
    void end() { start(nullptr); }
};

uint64_t get_tsc_ticks_per_second() {
    static auto tsc_ticks_per_second = rdtsc_calibrate();
    return tsc_ticks_per_second;
}
double tsc2second(uint64_t diff) { return diff * 1.0 / get_tsc_ticks_per_second(); }

uint64_t second2tsc(double sec) { return sec * get_tsc_ticks_per_second(); }

// performance counter
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define HW_PERF_COUNTER
__attribute__((weak)) int perf_event_open(struct perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags) { return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags); }

struct linux_perf_event {
    uint32_t type;
    uint32_t config;
    int fd;
    struct perf_event_mmap_page* buf;
    const char* name;

    linux_perf_event(const linux_perf_event&) = delete;
    linux_perf_event(linux_perf_event&&) = delete;

    linux_perf_event(uint32_t type, uint32_t config, const char* name) : type(type), config(config), fd(-1), buf(nullptr), name(name) {
        struct perf_event_attr attr = {};
        attr.type = type;
        attr.size = PERF_ATTR_SIZE_VER0;
        attr.config = config;
        attr.sample_type = PERF_SAMPLE_READ;
        attr.exclude_kernel = 1;

        fd = perf_event_open(&attr, 0, -1, -1, 0);
        if (fd < 0) {
            perror("perf_event_open, consider:  echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid");
            abort();
            return;
        }
        buf = (struct perf_event_mmap_page*)mmap(NULL, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, fd, 0);
        if (buf == MAP_FAILED) {
            perror("mmap");
            close(fd);
            fd = -1;
            abort();
            return;
        }
        // std::ios::fmtflags f(std::cout.flags());
        // std::cout << std::hex << "Linux perf event " << name << " (type=" << type << ",config=" << config << ")" << " is opened!" << std::endl;
        // std::cout.flags(f);
    }
    ~linux_perf_event() {
        if (fd > 0) {
            close(fd);
            munmap(buf, sysconf(_SC_PAGESIZE));
        }
    }
    uint64_t rdpmc_read() {
        uint64_t val, offset;
        uint32_t seq, index;

        do {
            seq = buf->lock;
            std::atomic_thread_fence(std::memory_order_acquire);
            index = buf->index;   //
            offset = buf->offset; // used to compensate the initial counter value
            if (index == 0) {     /* rdpmc not allowed */
                val = 0;
                std::cout << "rdpmc" << std::endl;
                break;
            }
            val = _rdpmc(index - 1);
            std::atomic_thread_fence(std::memory_order_acquire);
        } while (buf->lock != seq);
        uint64_t ret = (val + offset) & 0xffffffffffff;
        return ret;
    }
};

struct perf_log {
    std::vector<std::shared_ptr<linux_perf_event>> events;

    std::vector<uint64_t> m_counters;
    std::atomic<int> m_count;
    const int m_events;

    void reserve(int count) {
        m_counters.resize(count * (2 + events.size()), 0);
        m_count = 0;
    }

    perf_log(const std::vector<std::tuple<uint32_t, uint32_t, const char*>>& type_config_names = {}) : m_events(type_config_names.size()) {
        int nthr = 0;
        get_tsc_ticks_per_second();
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            if (ithr == 0)
                nthr = omp_get_num_threads();
            for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp barrier
                if (i == ithr) {
                    for (auto& tc : type_config_names)
                        events.emplace_back(new linux_perf_event(std::get<0>(tc), std::get<1>(tc), std::get<2>(tc)));
                }
            }
        }
        reserve(128 * nthr);
    }

    template <typename Callable>
    void operator()(const Callable& c, uint64_t opsPerCall = 0) {
        int ithr = omp_get_thread_num();
        auto ev_offset = ithr * m_events;

        auto* log = &m_counters[m_count.fetch_add(3 + m_events)];
        for (int i = 0; i < m_events; i++)
            log[i] = events[i + ev_offset]->rdpmc_read();

        auto start = __rdtsc();
        c();
        auto finish = __rdtsc();

        for (int i = 0; i < m_events; i++)
            log[i] = events[i + ev_offset]->rdpmc_read() - log[i];

        log[m_events] = finish - start;
        log[m_events + 1] = ithr;
        log[m_events + 2] = opsPerCall;
    }

    // add empty separator log
    void operator()(const char * info = "-------------------------------------------") {
        int ithr = omp_get_thread_num();
        auto ev_offset = ithr * m_events;

        auto* log = &m_counters[m_count.fetch_add(3 + m_events)];
        log[m_events] = 0;
        log[m_events + 1] = ithr;
        log[m_events + 2] = reinterpret_cast<uintptr_t>(info);
    }

    ~perf_log() {
        #pragma omp critical
        {
            auto ansi_color = ANSIcolor("0;33");
            if (_color)
                ansi_color = ANSIcolor(_color);
            
            std::cout << ansi_color << "================[ perf_log : " << _tag << " ]================" << ANSIcolor() << std::endl;
            std::cout << std::fixed << std::setprecision(2) << ansi_color << std::setw(4) << "#" << std::setw(6) << "  thr " << std::setw(10) << " latency";
            for (int k = 0; k < m_events; k++) {
                std::cout << "," << std::setw(10) << events[k]->name;
                if (std::string("HW_CYCLES") == events[k]->name) {
                    std::cout << "," << std::setw(10) << "CPU(GHz)";
                    std::cout << "," << std::setw(10) << "Ops/cycle";
                    std::cout << "," << std::setw(10) << "GFLOPS";
                }
            }
            std::cout << ANSIcolor() << std::endl;

            for (int i = 0, round = 0; i < m_count; i += (3 + m_events), round++) {
                auto dt = tsc2second(m_counters[i + m_events]);
                auto ithr = m_counters[i + m_events + 1];
                auto opsPerCall = m_counters[i + m_events + 2];

                std::cout << std::fixed << std::setprecision(2) << ansi_color << std::setw(4) << round << std::setw(6) << ithr << std::setw(8);
                if (dt == 0) {
                    std::cout << reinterpret_cast<const char*>(opsPerCall) << ANSIcolor() << std::endl;
                    continue;
                }

                if (dt > 1e-3)
                    std::cout << dt * 1e3 << "ms";
                else
                    std::cout << dt * 1e6 << "us";
                for (int k = 0; k < m_events; k++) {
                    auto count = m_counters[i + k];
                    std::cout << "," << std::setw(10) << count;
                    if (std::string("HW_CYCLES") == events[k]->name) {
                        std::cout << "," << std::setw(10) << (count * 1e-9 / dt);
                        std::cout << "," << std::setw(10) << opsPerCall / count;
                        std::cout << "," << std::setw(10) << opsPerCall * 1e-9 / dt;
                    }
                }
                std::cout << ANSIcolor() << std::endl;
            }

            // if (opsPerCall > 0)
            // std::cout << " " << autoUnit(opsPerCall/ avg_latency, "Ops/s");
            // }
        }
    }

    std::string _tag = "";

    template <typename... Ts>
    perf_log& tag(Ts... args) {
        std::stringstream ss;
        int dummy[sizeof...(Ts)] = {(ss << args << "_", 0)...};
        _tag = ss.str();
        if (_tag.size() > 1 && _tag.back() == '_')
            _tag.pop_back();
        return *this;
    }

    const char * _color = nullptr;
    perf_log& color(const char * code) {
        _color = code;
        return *this;
    }
};

// timeit will record best latency for each problem in a csv log file
// and it will also show hint about whether it's improved or descreased
// over changes
struct timeit {
    const char* app_version;
    std::vector<std::shared_ptr<linux_perf_event>> events;
    int override_expect_times_milliseconds;

    std::vector<uint64_t> pmu_cnt;
    timeit(const std::vector<std::tuple<uint32_t, uint32_t, const char*>>& type_config_names = {}) {
        override_expect_times_milliseconds = 0;

        if (std::getenv("TIMES")) {
            override_expect_times_milliseconds = atoi(std::getenv("TIMES"));
        }
        for (auto& tc : type_config_names) {
            events.emplace_back(new linux_perf_event(std::get<0>(tc), std::get<1>(tc), std::get<2>(tc)));
        }
        pmu_cnt.resize(events.size(), 0);
        // std::cout << ANSIcolor("0;33") << "   Test name    :   AvgLatency x repeats "
        //           << "  HW usage (measured / theoretical_peak) , PMU0 = value, .... " << ANSIcolor() << std::endl;
    }

    void set_perf_counters(bool initial) {
        if (initial) {
            for (int i = 0; i < events.size(); i++)
                pmu_cnt[i] = events[i]->rdpmc_read();
        } else {
            for (int i = 0; i < events.size(); i++)
                pmu_cnt[i] = events[i]->rdpmc_read() - pmu_cnt[i];
        }
    }

    std::vector<uint64_t> get_perf_counters() {
        std::vector<uint64_t> ret(events.size(), 0);
        for (int i = 0; i < events.size(); i++) {
            ret[i] = events[i]->rdpmc_read();
        }
        return ret;
    }

    void set_app(const char* _app_version) { app_version = _app_version; }

    struct ANSIcolor {
        const char* code;
        ANSIcolor(const char* code = "0") : code(code) {}
        friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
            out << "\033[" << obj.code << "m";
            return out;
        }
    };

    std::string _tag = "";

    template <typename... Ts>
    timeit& tag(Ts... args) {
        std::stringstream ss;
        int dummy[sizeof...(Ts)] = {(ss << args << "_", 0)...};
        _tag = ss.str();
        if (_tag.size() > 1 && _tag.back() == '_')
            _tag.pop_back();
        return *this;
    }

    std::string _color;
    timeit& color(std::string code) {
        _color = code;
        return *this;
    }

    EnvVar _clflush{"CLFLUSH"};

    int preset_expect_times_milliseconds;
    void set_time_ms(int time_ms) { preset_expect_times_milliseconds = time_ms; }

    const char* preset_unit = "";
    void set_unit(const char* unit) { preset_unit = unit; }

    double preset_peakOpsPerSecond;
    void set_peak_metric_per_second(double peak_per_second) { preset_peakOpsPerSecond = peak_per_second; }

    std::map<std::string, uint64_t> perf_counters;

    template <typename Callable>
    double operator()(const Callable& c, double opsPerCall = 0, double peakOpsPerSecond = 0, const char* unit = nullptr) {
        if (peakOpsPerSecond == 0)
            peakOpsPerSecond = preset_peakOpsPerSecond;
        if (unit == nullptr)
            unit = preset_unit;
        return operator()(preset_expect_times_milliseconds, c, opsPerCall, peakOpsPerSecond, preset_unit);
    }

    std::function<void(void)> hook_clear_cache;

    void clear_cache() {
        static int value = 0;
        if (hook_clear_cache) {
            hook_clear_cache();
            return;
        }
#pragma omp parallel
        {
            std::vector<uint8_t> _cache_data;
            std::vector<uint8_t> _cache_data2;
            _cache_data.resize(32 * 1024 * 1024);
            _cache_data2.resize(32 * 1024 * 1024);

            memset(&_cache_data[0], value++, _cache_data.size());
            memcpy(&_cache_data2[0], &_cache_data[0], _cache_data.size());
            memset(&_cache_data[0], value++, _cache_data.size());
            memcpy(&_cache_data2[0], &_cache_data[0], _cache_data.size());
        }
    }

    template <typename Callable>
    double operator()(int expect_times_milliseconds, const Callable& c, double opsPerCall = 0, double peakOpsPerSecond = 0, const char* unit = "Ops") {
        if (override_expect_times_milliseconds)
            expect_times_milliseconds = override_expect_times_milliseconds;
        int times;
        // cache warm-up
        if (expect_times_milliseconds != 1) {
            // one-time trigger must avoid warm-up
            std::cout << "warm-up..." << std::flush;
            c();
            c();
            std::cout << "done\r" << std::flush;
        }
        // determine times
        if (expect_times_milliseconds > 0) {
            times = expect_times_milliseconds;
        } else {
            double expect_duration = -expect_times_milliseconds * 0.001;
            // estimate how many times required to reach the duration
            auto start = __rdtsc();
            c();
            auto oneshot = __rdtsc() - start;
            times = second2tsc(expect_duration) / oneshot;
        }
        assert(times > 0);
        std::cout << "start..." << std::flush;
        // profiling
        std::vector<uint64_t> perf_counter(events.size(), 0);
        double total_latency = 0;
        if (_clflush.v_int) {
            std::vector<uint64_t> perf_counter0(perf_counter.size(), 0);
            int i;
            auto start0 = __rdtsc();
            for (i = 0; i < times; i++) {
                clear_cache();
                for (int i = 0; i < perf_counter.size(); i++)
                    perf_counter0[i] = events[i]->rdpmc_read();
                auto start = __rdtsc();
                c();
                for (int i = 0; i < perf_counter.size(); i++)
                    perf_counter[i] += events[i]->rdpmc_read() - perf_counter0[i];
                auto finish = __rdtsc();
                total_latency += tsc2second(finish - start);
                if (expect_times_milliseconds < 0 && tsc2second(finish - start0) * 1e3 > (-expect_times_milliseconds))
                    break;
            }
            times = i;
        } else {
            auto perf_counters0 = get_perf_counters();
            auto start = __rdtsc();
            for (int i = 0; i < times; i++) {
                c();
            }
            // auto finish = std::chrono::high_resolution_clock::now();
            auto finish = __rdtsc();
            perf_counter = get_perf_counters();
            for (int i = 0; i < perf_counter.size(); i++) {
                perf_counter[i] -= perf_counters0[i];
            }
            // std::chrono::duration<double> diff = finish-start;
            // total_latency = diff.count();
            total_latency = tsc2second(finish - start);
        }

        std::cout << "done\r                                \r" << std::flush;
        auto avg_latency = total_latency / times;

        std::string ansi_color = _color;
        if (ansi_color.size() == 0)
            ansi_color = "0;33";
        std::cout << std::fixed << std::setprecision(2) << ANSIcolor(ansi_color.c_str()) << _tag << "\t: " << avg_latency * 1e6 << " us x " << times;
        if (opsPerCall > 0 && peakOpsPerSecond > 0) {
            std::cout << ", " << static_cast<int>(100 * (opsPerCall / avg_latency) / (peakOpsPerSecond)) << "% (" << opsPerCall / avg_latency / (1e9) << " G" << unit << " /" << peakOpsPerSecond / 1e9 << " G" << unit << ")";
        }

        perf_counters.clear();
        for (int i = 0; i < perf_counter.size(); i++) {
            auto avg_counter = perf_counter[i] / times;
            perf_counters[events[i]->name] = avg_counter;
            std::cout << ", " << events[i]->name << "=" << avg_counter;

            if (std::string("HW_CYCLES") == events[i]->name) {
                std::cout << " CPU~" << (avg_counter * 1e-9 / avg_latency) << "GHz";
                if (opsPerCall > 0) {
                    // return average HW cycles instead
                    std::cout << " " << opsPerCall / avg_counter << "(Ops/cycle)";
                }
            }
        }

        if (opsPerCall > 0)
            std::cout << " " << autoUnit(opsPerCall / avg_latency, "Ops/s");

        std::cout << ANSIcolor() << std::endl;
        return avg_latency;
    }

    std::string autoUnit(double v, const std::string& unit) {
        auto to_fixed_str = [](double value) {
            char buf[256];
            std::sprintf(buf, "%.3f", value);
            return std::string(buf);
        };
        if (v >= 1e12)
            return to_fixed_str(v / 1e12) + "(T" + unit + ")";
        if (v >= 1e9)
            return to_fixed_str(v / 1e9) + "(G" + unit + ")";
        if (v >= 1e6)
            return to_fixed_str(v / 1e6) + "(M" + unit + ")";
        if (v >= 1e3)
            return to_fixed_str(v / 1e3) + "(K" + unit + ")";
        return to_fixed_str(v) + "(" + unit + ")";
    }
};

//=============================================================
// BF16-amx Peak (Gops)
// c += a*b is counted as 2 Ops
//

constexpr double AMXBf16OpsPerTDP = (16 * 16 * 32) * 2;
constexpr double AMXBf16TDPThrouput = 16;
constexpr double AMXBf16OpsPerCycleCore = AMXBf16OpsPerTDP / AMXBf16TDPThrouput;
constexpr double AMXBf16FreqGHz = 2.05;
constexpr double AMXBf16Freq2GHz = 3; // 2.32;
constexpr double AMXBf16PeakGopsPerCore = AMXBf16OpsPerCycleCore * AMXBf16FreqGHz;
constexpr double AMXBf16PeakGops2PerCore = AMXBf16OpsPerCycleCore * AMXBf16Freq2GHz;

constexpr double AVX512FreqGHz = 3;        // 2.32;
constexpr double FP32OpsPerCycleCore = 64; // 2 AVX512_FMAs/cycle/core = 2*(16+16) Ops/cycle/core
constexpr double FP32PeakGopsPerCore = FP32OpsPerCycleCore * AVX512FreqGHz;
