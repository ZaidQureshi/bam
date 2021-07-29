cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (iodepth-block-benchmark-module iodepth-block-bench "main.cu;")

make_benchmark_choice (iodepth-block-benchmark iodepth-block-benchmark-smartio iodepth-block-benchmark-module)
