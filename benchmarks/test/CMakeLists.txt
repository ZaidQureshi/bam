cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (test-benchmark-module test-bench "main.cu;")

make_benchmark_choice (test-benchmark test-benchmark-smartio test-benchmark-module)
