# ASPLOS AOE 

BaM work is submitted for the Artifact Available and Functional ACM Badges. 
Reproducible Badge requires access to proprietary codebase and large datasets that may not feasible to complete during the AoE review process. 
As BaM requires customization across the hardware and software stack, for AOE we provide remote access to our mini-prototype machine. 
Due to differences in system capabilities, it is important to understand not all results presented in the BaM paper will be reproducible. 
However, we have worked hard to enable the reviewers to validate the BaM prototype's functional capabilities.

The AoE evaluation is performed based on the contributions claimed (and the relevant experiments) in the paper. 
Primarily, we want to establish following key things:

1) BaM is functional with a single SSD and supports the contributions described in the paper
    * Evaluation for Figure 4 as an example
2) BaM is functional with multiple SSDs - up to 2 SSD can be tested on this system
    * Evaluation for Figures 4 and 7 as examples
3) BaM is functional with different types of SSDs - consumer-grade Samsung 980 Pro and datacenter-grade Intel Optane SSDs are available in the system. 
    * Evaluation for Figure 9 as an example
4) BaM can be used by different applications - microbenchmarks and graph applications (w/ datasets) are provided for AoE.
    * Evaluation for Figures 5, 6, and 7 as examples 

This README.md provides necessarily command line arguments required to establish above mentioned goals. 
The README.md is structured such that we go over each Figures in the paper individually, describe if they are runnable in this mini-prototype system and what to expect to see as output from the experiments. 
If you run into any troubles during the AoE process, please reach out over comments in HotCRP page. 

The expected output for each command (or group of commands) is provided as a separate log file and linked appropriately.
Everything in the log should match except performance numbers as we are not going for the resutls reproduced badege.

## Goal 1: BaM is functional with a single Samsung 980 Pro SSD - the codebase builds and the components (I/O stack, cache, application APIs) are functionally usable

### Building the codebase
First, we get the required software dependencies and run `cmake`.
```
$ git submodule update --init --recursive
$ mkdir -p build; cd build
$ cmake ..
```
The cmake log is available [here](./cmake.log).


Next, we build the core host library.
```
$ make libnvm -j                         # builds library
```
The relevant log is available [here](./build_libnvm.log).


Next, we build the benchmarks.
```
$ make benchmarks -j                     # builds benchmark program
```
The relevant log is available [here](./build_benchmarks.log).

Next, we build the kernel module.
```
$ cd module
$ make -j                                # build kernel module
$ sudo make reload                       # we have already loaded the driver, this will reload the drivers again. 
```
The relevant log is available [here](./build_kernel_modules.log).

### Running the I/O stack component
From the `build` directory, we can run the I/O stack benchmark, similar to an `fio` benchmark.
```
$ sudo ./bin/nvm-block-bench --threads=262144 --blk_size=64 --reqs=1 --pages=262144 --queue_depth=1024  --page_size=512 --num_blks=2097152 --gpu=0 --n_ctrls=1 --num_queues=128 --random=true
```
Here, we launch a GPU kernel with 262144 threads to access 1 Samsung 980 Pro SSD, where each thread makes one I/O request for a 512-byte block, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_block_bench_1_sam.log).

### Running with the complete BaM stack (high-level `array` abstraction, cache and I/O stack)
From the `build` directory, we can run the abstraction and cache benchmark.
```
$ sudo ./bin/nvm-array-bench --threads=$((1024*1024)) --blk_size=64 --reqs=1 --pages=$((1024*1024)) --queue_depth=1024  --page_size=512 --gpu=0 --n_ctrls=1 --num_queues=128 --random=false
```
Here, we launch a GPU kernel with 1 Million threads to access 1 Samsung 980 Pro SSD through a 512MB cache made up of 512-byte cache-lines, where each thread makes an access for a 64-bit value, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_array_bench_1_sam.log).

## Goal 2: BaM is functional with multiple SSDs - up to 2 SSDs can be used in the provided system

### Running the I/O stack component with 2 Samsung 980 Pro SSDs
From the `build` directory, we can run the I/O stack benchmark, similar to an `fio` benchmark.
```
$ sudo ./bin/nvm-block-bench --threads=262144 --blk_size=64 --reqs=1 --pages=262144 --queue_depth=1024  --page_size=512 --num_blks=2097152 --gpu=0 --n_ctrls=2 --num_queues=128 --random=true
```
Here, we launch a GPU kernel with 262144 threads to access 2 Samsung 980 Pro SSDs, where each thread makes one I/O request for a 512-byte block, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_block_bench_2_sam.log).

### Running the complete BaM stack (high-level `array` abstraction, cache and I/O stack) with 2 Samsung 980 Pro SSDs
From the `build` directory, we can run the stack benchmark.
```
$ sudo ./bin/nvm-array-bench --threads=$((1024*1024)) --blk_size=64 --reqs=1 --pages=$((1024*1024)) --queue_depth=1024  --page_size=512 --gpu=0 --n_ctrls=2 --num_queues=128 --random=false
```
Here, we launch a GPU kernel with 1 Million threads to access 2 Samsung 980 Pro SSDs through a 512MB cache made up of 512-byte cache-lines, where each thread makes an access for a 64-bit value, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_array_bench_2_sam.log).

## Goal 3: BaM is functional with different types of SSDs -  the provided system consists of 2 different types of SSDs

The previous benchmarks all were using consumer grade Samsung 980 Pro SSDs. 
Now we show BaM running benchmarks with the datacenter grade Intel Optane SSDs, by setting the value for the `--ssd`for each application to `1`.

### Running the I/O stack component with a Intel Optane SSD
From the `build` directory, we can run the I/O stack benchmark, similar to an `fio` benchmark.
```
$ sudo ./bin/nvm-block-bench --threads=262144 --blk_size=64 --reqs=1 --pages=262144 --queue_depth=1024  --page_size=512 --num_blks=2097152 --gpu=0 --n_ctrls=1 --num_queues=128 --random=true --ssd=1
```
Here, we launch a GPU kernel with 262144 threads to access a Intel Optane SSD, where each thread makes one I/O request for a 512-byte block, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_block_bench_1_intel.log).

### Running the complete BaM stack (high-level `array` abstraction, cache and I/O stack) with a Intel Optane SSD
From the `build` directory, we can run the stack benchmark.
```
$ sudo ./bin/nvm-array-bench --threads=$((1024*1024)) --blk_size=64 --reqs=1 --pages=$((1024*1024)) --queue_depth=1024  --page_size=512 --gpu=0 --n_ctrls=1 --num_queues=128 --random=false --ssd=1
```
Here, we launch a GPU kernel with 1 Million threads to access a Intel Optane through a 512MB cache made up of 512-byte cache-lines, where each thread makes an access for a 64-bit value, where we have 128 NVMe queues with each queue is 1024 deep.
The expected log is available [here](./nvm_array_bench_1_intel.log).


## Goal 4: BaM can be used by different applications - the graph analytics applications (BFS and CC) are provided
**Note: The data analytics application implementation in BaM is proprietary so it is not shared**

The previous applications all were microbenchmarks for different components of the BaM stack.
Now we show that BaM can be used by real applications, namely the breadth-first search (BFS) and connected components (CC) graph analytics applications, with real world datasets.
The datasets are already loaded on the SSDs in the provided system to make it easy for the reviewers to evaluate.
For these experiments, BaM is using 4KB cache-lines and an 8GB cache.

### Running the BFS application on the MOLIERE_2016 dataset with 1 Intel Optane SSD
From the `build` directory, we can run the BFS benchmark.
```
$ sudo ./bin/nvm-bfs-bench -f /home/vsm2/bafsdata/MOLIERE_2016.bel  -l 240518168576 --impl_type 20 --memalloc 6 --src 13229860 --n_ctrls 1 -p 4096 --gpu 0 --threads 128 -C 8 -M $((8*1024*1024*1024)) --ssd=1
```
The expected log is available [here](./nvm_bfs_bench_1_intel.log).

### Running the CC application on the GAP_kron dataset with 1 Intel Optane SSD
From the `build` directory, we can run the CC benchmark.
```
$ sudo ./bin/nvm-cc-bench -f /home/vsm2/bafsdata/GAP-kron.bel -l 0 --impl_type 20 --memalloc 6 --src 58720242 --n_ctrls 1 -p 4096 --gpu 0 --threads 128 -M $((8*1024*1024*1024)) -P 128 -C 8 --ssd=1
```
The expected log is available [here](./nvm_cc_bench_1_intel.log). 
