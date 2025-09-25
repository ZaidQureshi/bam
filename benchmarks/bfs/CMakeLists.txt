cmake_minimum_required (VERSION 3.5)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package(CUDAToolkit REQUIRED)



make_benchmark (bfs-benchmark-module bfs-bench "main.cu;")

make_benchmark_choice (bfs-benchmark bfs-benchmark-smartio bfs-benchmark-module)
