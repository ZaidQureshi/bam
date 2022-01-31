cmake_minimum_required (VERSION 3.10)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 10.0 REQUIRED)


make_benchmark (sectorvalid-benchmark-module sectorvalid-bench "main.cu;")

make_benchmark_choice (sectovalid-benchmark sectorvalid-benchmark-smartio sectorvalid-benchmark-module)