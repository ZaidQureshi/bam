cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (test2-benchmark-module test2-bench "main.cu;")

make_benchmark_choice (test2-benchmark test2-benchmark-smartio test2-benchmark-module)
