cmake_minimum_required (VERSION 3.10)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 10.0 REQUIRED)



make_benchmark (readwrite_stripe-benchmark-module readwrite_stripe-bench "main.cu;")

make_benchmark_choice (readwrite_stripe-benchmark readwrite_stripe-benchmark-smartio readwrite_stripe-benchmark-module)
