// clang-format off
#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;

const int INF = 1047483647;
using ll = int_fast64_t;
using ull = uint_fast64_t;
#define myall(x) (x).begin(), (x).end()
#define MyWatch(x) (double)(clock() - (x)) / CLOCKS_PER_SEC
#define istrue ? assert(true) : assert(false && "istrue")
#define _rep0(a) for (uint_fast64_t _tmp_i = 0; _tmp_i < UINT_FAST64_MAX; ++_tmp_i, assert(_tmp_i < INT_MAX))
#define _rep1(a) for (int_fast64_t _tmp_i = 0; _tmp_i < (int_fast64_t)(a); ++_tmp_i)
#define _rep2(i, a) for (int_fast64_t i = 0; i < (int_fast64_t)(a); ++i)
#define _rep3(i, a, b) for (int_fast64_t i = (int_fast64_t)(a); i < (int_fast64_t)(b); ++i)
#define _print0(a) cout << endl
#define _print1(a) cout << (a) << endl
#define _print2(a, b) cout << (a) << ", " << (b) << endl
#define _print3(a, b, c) cout << (a) << ", " << (b) << ", " << (c) << endl
#define _overload(a, b, c, d, e ...) d
#define rep(...) _overload(__VA_ARGS__ __VA_OPT__(,)  _rep3, _rep2, _rep1, _rep0)(__VA_ARGS__)
#define print(...) _overload(__VA_ARGS__ __VA_OPT__(,)  _print3, _print2, _print1, _print0)(__VA_ARGS__)

template<class T>bool chmax(T &a, const T &b) { if (a<b) { a=b; return 1; } return 0; }
template<class T>bool chmin(T &a, const T &b) { if (b<a) { a=b; return 1; } return 0; }
// clang-format on


#include <NvInfer.h>
// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0


#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// // Logger for TensorRT
// class Logger : public ILogger {
// public:
//     void log(ILogger::Severity severity, const char* msg) override
//     {
//         // suppress information level log
//         if (severity == Severity::kINFO) return;
//         std::cout << msg << std::endl;
//     }
// };
int device_count;
int device = 1;

int main() {
    print(cudaGetDeviceCount(&device_count));
    // setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    if (cudaSetDevice(device) != cudaSuccess)
{
    cout << "cudaSetDevice failed" << std::endl;
    return 0 ;
}
    cout << "test4" << endl;
    return 0;
}