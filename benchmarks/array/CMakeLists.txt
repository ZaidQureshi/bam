cmake_minimum_required (VERSION 3.5)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package(CUDAToolkit REQUIRED)



make_benchmark (array-benchmark-module array-bench "main.cu;")

make_benchmark_choice (array-benchmark array-benchmark-smartio array-benchmark-module)
