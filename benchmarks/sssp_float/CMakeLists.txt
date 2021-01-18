cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (sssp_float-benchmark-module sssp_float-bench "main.cu;")

make_benchmark_choice (sssp_float-benchmark sssp_float-benchmark-smartio sssp_float-benchmark-module)
