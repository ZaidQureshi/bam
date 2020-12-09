cmake_minimum_required (VERSION 3.10)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 10.0 REQUIRED)



make_benchmark (readwrite-benchmark-module readwrite-bench "main.cu;")

make_benchmark_choice (readwrite-benchmark readwrite-benchmark-smartio readwrite-benchmark-module)
