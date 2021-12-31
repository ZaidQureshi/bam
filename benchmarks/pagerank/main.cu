/* References:
 *
 *      Baseline
 *          Harish, Pawan, and P. J. Narayanan.
 *          "Accelerating large graph algorithms on the GPU using CUDA."
 *          International conference on high-performance computing.
 *          Springer, Berlin, Heidelberg, 2007.
 *
 *      Coalesce
 *          Hong, Sungpack, et al.
 *          "Accelerating CUDA graph algorithms at maximum warp."
 *          Acm Sigplan Notices 46.8 (2011): 267-276.
 *
 */

#include <cuda.h>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
//#include "helper_cuda.h"
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>
#include <math.h>
#include <chrono>
#include <ctime>
#include <ratio>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>

#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <buffer.h>
#include "settings.h"
#include <ctrl.h>
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>

using error = std::runtime_error;
using std::string;
//const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro1", "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5", "/dev/libnvmpro6", "/dev/libnvmpro7"};
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7"};
//const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5", "/dev/libnvmpro6", "/dev/libnvmpro7"};
//const char* const ctrls_paths[] = {"/dev/libnvmpro0"};


#define MYINFINITY 0xFFFFFFFF

#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_SIZE 1024ULL

#define MAXWARP 64

typedef uint64_t EdgeT;

typedef float ValueT;

typedef enum {
    BASELINE = 0,
    COALESCE = 1,
    COALESCE_CHUNK = 2,
    BASELINE_PC = 3,
    COALESCE_PC = 4,
    COALESCE_CHUNK_PC =5,
    BASELINE_HASH= 6, 
    COALESCE_HASH= 7, 
    COALESCE_CHUNK_HASH= 8, 
    BASELINE_HASH_PC= 9, 
    COALESCE_HASH_PC= 10, 
    COALESCE_CHUNK_HASH_PC= 11, 

} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT= 6,
} mem_type;

__global__ void initialize(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT alpha) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count) {
        value[tid] = 1.0f - alpha;
        delta[tid] = (1.0f - alpha) * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
	}
}

__global__ void kernel_baseline(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(tid < vertex_count && label[tid]) {
        const uint64_t start = vertexList[tid];
        // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[tid+1];

        for(uint64_t i = start; i < end; i += 1){
            // if (i >= start){
                EdgeT next = edgeList[i];
                atomicAdd(&residual[next], delta[tid]);
            // }
        }
        label[tid] = false;
    }
}

__global__ __launch_bounds__(1024,2)
void kernel_baseline_pc(array_d_t<uint64_t>* da, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    // array_d_t<uint64_t> d_array = *da;

    if(tid < vertex_count && label[tid]) {
        const uint64_t start = vertexList[tid];
        // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[tid+1];

        for(uint64_t i = start; i < end; i += 1){
            // if (i >= start){
                // EdgeT next = edgeList[i];
                EdgeT next = da->seq_read(i);
                atomicAdd(&residual[next], delta[tid]);
            // }
        }
        label[tid] = false;
    }
}



__global__ void kernel_baseline_hash(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, int sm_count) {
    const uint64_t oldtid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = sm_count * MAXWARP; 

    if(oldtid < vertex_count){ 
        uint64_t tid; 
        const uint64_t nep = vertex_count/STRIDE;
        if(oldtid <(STRIDE*nep)){
            tid = (oldtid/nep) + ((oldtid % nep)*STRIDE);
        }
        else{
            tid = oldtid; 
        }
 

        if(label[tid]) {
            const uint64_t start = vertexList[tid];
            // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[tid+1];

            for(uint64_t i = start; i < end; i += 1){
                // if (i >= start){
                    EdgeT next = edgeList[i];
                    atomicAdd(&residual[next], delta[tid]);
                // }
            }
            label[tid] = false;
        }
    }
}

__global__ //__launch_bounds__(1024,2)
void kernel_baseline_hash_pc(array_d_t<uint64_t>* da, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, int sm_count) {
     const uint64_t oldtid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    // const uint64_t warpIdx = tid >> WARP_SHIFT;
    // const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = sm_count * MAXWARP; 

    if(oldtid < vertex_count){ 
        uint64_t tid; 
        const uint64_t nep = vertex_count/STRIDE;
        if(oldtid <(STRIDE*nep)){
            tid = (oldtid/nep) + ((oldtid % nep)*STRIDE);
        }
        else{
            tid = oldtid; 
        }
 

        if(label[tid]) {
            const uint64_t start = vertexList[tid];
            // const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[tid+1];

            for(uint64_t i = start; i < end; i += 1){
                // if (i >= start){
                    // EdgeT next = edgeList[i];
                    EdgeT next = da->seq_read(i);
                    atomicAdd(&residual[next], delta[tid]);
                // }
            }
            label[tid] = false;
        }
    }
}



__global__ void kernel_coalesce(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start){
                EdgeT next = edgeList[i];
                atomicAdd(&residual[next], delta[warpIdx]);
            }

        label[warpIdx] = false;
    }
}

__global__ //__launch_bounds__(128,16)
void kernel_coalesce_pc(array_d_t<uint64_t>* da, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    // array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx]) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
            if (i >= start){
                // EdgeT next = edgeList[i];
                // EdgeT next = d_array.seq_read(i);
                EdgeT next = da->seq_read(i);
                atomicAdd(&residual[next], delta[warpIdx]);
            }

        label[warpIdx] = false;
    }
}


__global__ void kernel_coalesce_hash(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, int sm_count) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = sm_count * MAXWARP; 

    if(oldwarpIdx < vertex_count){
        uint64_t warpIdx; 
        const uint64_t nep = vertex_count/STRIDE; 
        if(oldwarpIdx <(STRIDE*nep)){
                warpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*STRIDE);
        }
        else{
                warpIdx = oldwarpIdx; 
        }

        if(label[warpIdx]) {
            const uint64_t start = vertexList[warpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[warpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
                if (i >= start){
                    EdgeT next = edgeList[i];
                    atomicAdd(&residual[next], delta[warpIdx]);
                }

            label[warpIdx] = false;
        }
    }
}

__global__ //__launch_bounds__(128,16)
void kernel_coalesce_hash_pc(array_d_t<uint64_t>* da, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, int sm_count) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = sm_count * MAXWARP; 

    if(oldwarpIdx < vertex_count){
        uint64_t warpIdx; 
        const uint64_t nep = vertex_count/STRIDE; 
        if(oldwarpIdx <(STRIDE*nep)){
                warpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*STRIDE);
        }
        else{
                warpIdx = oldwarpIdx; 
        }

        if(label[warpIdx]) {
            const uint64_t start = vertexList[warpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[warpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE)
                if (i >= start){
                    // EdgeT next = edgeList[i];
                    // EdgeT next = d_array.seq_read(i);
                    EdgeT next = da->seq_read(i);
                    atomicAdd(&residual[next], delta[warpIdx]);
                }

            label[warpIdx] = false;
        }
    }
}


__global__ void kernel_coalesce_chunk(bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;

    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE)
                if (j >= start){
                    EdgeT next = edgeList[j];
                    atomicAdd(&residual[next], delta[i]);
                }

            label[i] = false;
        }
    }
}

__global__ //__launch_bounds__(1024,2)
void kernel_coalesce_chunk_pc(array_d_t<uint64_t>* da, bool* label, ValueT *delta, ValueT *residual, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;
    // array_d_t<uint64_t> d_array = *da;
    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i]) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE)
                if (j >= start){
                    // EdgeT next = edgeList[j];
                    // EdgeT next = d_array.seq_read(j);
                    EdgeT next = da->seq_read(j);
                    atomicAdd(&residual[next], delta[i]);
                }

            label[i] = false;
        }
    }
}




__global__ void update(bool *label, ValueT *delta, ValueT *residual, ValueT *value, const uint64_t vertex_count, const uint64_t *vertexList, ValueT tolerance, ValueT alpha, bool *changed) {
    const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < vertex_count && residual[tid] > tolerance) {
        value[tid] += residual[tid];
        delta[tid] = residual[tid] * alpha / (vertexList[tid+1] - vertexList[tid]);
        residual[tid] = 0.0f;
        label[tid] = true;
        *changed = true;
	}
}

__global__ void throttle_memory(uint32_t *pad) {
    pad[1] = pad[0];
}

int main(int argc, char *argv[]) {
    using namespace std::chrono; 

    Settings settings; 
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    std::ifstream file;
    std::string vertex_file, edge_file;
    std::string filename;

    bool changed_h, *changed_d, *label_d;
    // int c, arg_num = 0;
    impl_type type;
    mem_type mem;
    ValueT *delta_d, *residual_d, *value_d, *value_h, *delta_h;
    ValueT tolerance, alpha;
    uint32_t iter;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t numblocks, numblocks_update, numthreads;
    uint64_t typeT;
    size_t freebyte, totalbyte;
    uint32_t *pad;
    EdgeT *edgeList_dtmp;


    float milliseconds;
    double avg_milliseconds;

    uint64_t pc_page_size;
    uint64_t pc_pages; 

    alpha = 0.85;
    tolerance = 0.001;
    
    try{
        //prepare from settings
        filename = std::string(settings.input); 

        // if(settings.src == 0) {
        //         num_run = settings.repeat; 
        //         src = 0;
        // }
        // else {
        //         num_run = 1; 
        //         src = settings.src; 
        // }

        type = (impl_type) settings.type; 
        mem = (mem_type) settings.memalloc; 

        pc_page_size = settings.pageSize; 
        pc_pages = ceil((float)settings.maxPageCacheSize/pc_page_size);

        numthreads = settings.numThreads;
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
    
        cudaEvent_t start, end;
        cuda_err_chk(cudaEventCreate(&start));
        cuda_err_chk(cudaEventCreate(&end));

        vertex_file = filename + ".col";
        edge_file = filename + ".dst";

        std::cout << filename << std::endl;
        fprintf(stderr, "File %s\n", filename.c_str());
        // Read files
        file.open(vertex_file.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            fprintf(stderr, "Vertex file open failed\n");
            exit(1);
        };

        file.read((char*)(&vertex_count), 8);
        file.read((char*)(&typeT), 8);

        vertex_count--;

        printf("Vertex: %llu, ", vertex_count);
        vertex_size = (vertex_count+1) * sizeof(uint64_t);

        vertexList_h = (uint64_t*)malloc(vertex_size);

        file.read((char*)vertexList_h, vertex_size);
        file.close();

        file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            fprintf(stderr, "Edge file open failed\n");
            exit(1);
        };

        file.read((char*)(&edge_count), 8);
        file.read((char*)(&typeT), 8);

        printf("Edge: %llu\n", edge_count);
        fflush(stdout);
        edge_size = edge_count * sizeof(EdgeT);
        edge_size = edge_size + (4096 - (edge_size & 0xFFFULL));

        edgeList_h = NULL;

        // Allocate memory for GPU
        cuda_err_chk(cudaMalloc((void**)&label_d, vertex_count * sizeof(bool)));
        cuda_err_chk(cudaMalloc((void**)&vertexList_d, vertex_size));
        cuda_err_chk(cudaMalloc((void**)&changed_d, sizeof(bool)));
        cuda_err_chk(cudaMalloc((void**)&delta_d, vertex_count * sizeof(ValueT)));
        cuda_err_chk(cudaMalloc((void**)&residual_d, vertex_count * sizeof(ValueT)));
        cuda_err_chk(cudaMalloc((void**)&value_d, vertex_count * sizeof(ValueT)));

        value_h = (ValueT*)malloc(vertex_count * sizeof(ValueT));
        delta_h = (ValueT*)malloc(vertex_count * sizeof(ValueT));

        switch (mem) {
            case GPUMEM:
                cuda_err_chk(cudaMalloc((void**)&edgeList_d, edge_size));
                break;
            case UVM_READONLY:
                cuda_err_chk(cudaMallocManaged((void**)&edgeList_d, edge_size));
                file.read((char*)edgeList_d, edge_size);
                cuda_err_chk(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));

                cuda_err_chk(cudaMemGetInfo(&freebyte, &totalbyte));
                if (totalbyte < 16*1024*1024*1024ULL)
                    printf("total memory sizeo of current GPU is %llu byte, no need to throttle\n", totalbyte);
                else {
                    printf("total memory sizeo of current GPU is %llu byte, throttling %llu byte.\n", totalbyte, totalbyte - 16*1024*1024*1024ULL);
                    cuda_err_chk(cudaMalloc((void**)&pad, totalbyte - 16*1024*1024*1024ULL));
                    throttle_memory<<<1,1>>>(pad);
                }
                break;
            case UVM_DIRECT:
                {/*
                cuda_err_chk(cudaMallocManaged((void**)&edgeList_d, edge_size));
                file.read((char*)edgeList_d, edge_size);
                cuda_err_chk(cudaMemAdvise(edgeList_d, edge_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                break;
                */
                file.close();
                for (uint64_t i = 0; i < vertex_count + 1; i++) {
                    vertexList_h[i] += 2;
                }   
                int fd = open(edge_file.c_str(), O_RDONLY | O_DIRECT);
                FILE *file_temp = fdopen(fd, "rb");
                if ((file_temp == NULL) || (fd == -1)) {
                    printf("edge file fd open failed\n");
                    exit(1);
                }   
                uint64_t edge_count_4k_aligned = ((edge_count + 2 + 4096 / sizeof(uint64_t)) / (4096 / sizeof(uint64_t))) * (4096 / sizeof(uint64_t));
                uint64_t edge_size_4k_aligned = edge_count_4k_aligned * sizeof(uint64_t);
                cuda_err_chk(cudaMallocManaged((void**)&edgeList_d, edge_size_4k_aligned));
                cuda_err_chk(cudaMemAdvise(edgeList_d, edge_size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                high_resolution_clock::time_point ft1 = high_resolution_clock::now();
                      
                if (fread(edgeList_d, sizeof(uint64_t), edge_count_4k_aligned, file_temp) != edge_count + 2) {
                    printf("edge file fread failed\n");
                    exit(1);
                }   
                                                                                                                                            fclose(file_temp);                                                                                                              
                close(fd);
                high_resolution_clock::time_point ft2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
                std::cout<< "edge file read time: "<< time_span.count() <<std::endl;
                      
                file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
                if (!file.is_open()) {
                    printf("edge file open failed\n");
                    exit(1);
                }   
                break;
            }
            case BAFS_DIRECT:
                 cuda_err_chk(cudaMemGetInfo(&freebyte, &totalbyte));
                 if (totalbyte < 16*1024*1024*1024ULL)
                     printf("total memory sizeo of current GPU is %llu byte, no need to throttle\n", totalbyte);
                 else {
                     printf("total memory sizeo of current GPU is %llu byte, throttling %llu byte.\n", totalbyte, totalbyte - 16*1024*1024*1024ULL);
                     cuda_err_chk(cudaMalloc((void**)&pad, totalbyte - 16*1024*1024*1024ULL));
                     throttle_memory<<<1,1>>>(pad);
                 }
                 break;                
        }
        file.close();

        printf("Allocation finished\n");
        fflush(stdout);

        // Initialize values
        cuda_err_chk(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

        if (mem == GPUMEM)
            cuda_err_chk(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));

        switch (type) {
            case BASELINE:
            case BASELINE_PC:
            case BASELINE_HASH:
            case BASELINE_HASH_PC:
                numblocks = ((vertex_count+numthreads)/numthreads);
                break;
            case COALESCE:
            case COALESCE_PC:
            case COALESCE_HASH:
            case COALESCE_HASH_PC:
               numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
                break;
            case COALESCE_CHUNK:
            case COALESCE_CHUNK_PC:
            case COALESCE_CHUNK_HASH:
            case COALESCE_CHUNK_HASH_PC:
                numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                break;
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }

        numblocks_update = ((vertex_count + numthreads) / numthreads);

        dim3 blockDim(BLOCK_SIZE, (numblocks+BLOCK_SIZE)/BLOCK_SIZE);
        dim3 blockDim_update(BLOCK_SIZE, (numblocks_update+BLOCK_SIZE)/BLOCK_SIZE);
        std::vector<Controller*> ctrls(settings.n_ctrls);
        if(mem == BAFS_DIRECT){
            cuda_err_chk(cudaSetDevice(settings.cudaDevice));
            for (size_t i = 0 ; i < settings.n_ctrls; i++)
                ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
            printf("Controllers Created\n");
        }
        printf("Initialization done\n");
        fflush(stdout);

        avg_milliseconds = 0.0f;
        iter = 0;

        if((type == BASELINE_PC) || (type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC) || (type == BASELINE_HASH_PC) || (type == COALESCE_HASH_PC) ||(type == COALESCE_CHUNK_HASH_PC)){
            printf("page size: %d, pc_entries: %llu\n", pc_page_size, pc_pages);
        }


        page_cache_t* h_pc; 
        range_t<uint64_t>* h_range;
        std::vector<range_t<uint64_t>*> vec_range(1);
        array_t<uint64_t>* h_array; 
        uint64_t n_pages = ceil(((float)edge_size)/pc_page_size); 

        if((type == BASELINE_PC) || (type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC) || (type == BASELINE_HASH_PC) || (type == COALESCE_HASH_PC) ||(type == COALESCE_CHUNK_HASH_PC)){
            h_pc =new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
            h_range = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)edge_count, (uint64_t) (ceil(settings.ofileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); //, (uint8_t*)edgeList_d);
            vec_range[0] = h_range; 
            h_array = new array_t<uint64_t>(edge_count, settings.ofileoffset, vec_range, settings.cudaDevice);
            
            printf("Page cache initialized\n");
            fflush(stdout);
         }

        cuda_err_chk(cudaEventRecord(start, 0));
        // printf("*****baseaddr: %p\n", h_pc->pdt.base_addr);
        // fflush(stdout);
        initialize <<<blockDim_update, numthreads>>> (label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, alpha);

        // Run pagerank
        do {
            changed_h = false;
            cuda_err_chk(cudaMemcpy(changed_d, &changed_h, sizeof(bool), cudaMemcpyHostToDevice));

            auto itrstart = std::chrono::system_clock::now();
            switch (type) {
                case BASELINE:
                    kernel_baseline<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                    break;
                case COALESCE:
                    kernel_coalesce<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                    break;
                case COALESCE_CHUNK:
                    kernel_coalesce_chunk<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                    break;
                case BASELINE_PC:
                    kernel_baseline_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);                    
                    break;
                case COALESCE_PC:
                    kernel_coalesce_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                    break;
                case COALESCE_CHUNK_PC:
                    kernel_coalesce_chunk_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d);
                    break;
                case BASELINE_HASH:
                    kernel_baseline_hash<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);
                    break;
                case COALESCE_HASH:
                    kernel_coalesce_hash<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);
                    break;
                // case COALESCE_HASH_CHUNK:
                //     kernel_coalesce_hash_chunk<<<blockDim, numthreads>>>(label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);
                //     break;
                case BASELINE_HASH_PC:
                    kernel_baseline_hash_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);                    
                    break;
                case COALESCE_HASH_PC:
                    kernel_coalesce_hash_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);
                    break;
                // case COALESCE_CHUNK_HASH_PC:
                //     kernel_coalesce_chunk_hash_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, delta_d, residual_d, vertex_count, vertexList_d, edgeList_d, properties.multiProcessorCount);
                //     break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            cuda_err_chk(cudaDeviceSynchronize());

            iter++;
            auto itrend = std::chrono::system_clock::now();
            update <<<blockDim_update, numthreads>>> (label_d, delta_d, residual_d, value_d, vertex_count, vertexList_d, tolerance, alpha, changed_d);

            cuda_err_chk(cudaMemcpy(&changed_h, changed_d, sizeof(bool), cudaMemcpyDeviceToHost));

	        //std::chrono::duration<double> elapsed_seconds = itrend-itrstart;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(itrend - itrstart);
            if(mem == BAFS_DIRECT) {
                  h_array->print_reset_stats();
		          //std::cout<< "itr time: "<< elapsed.count() << " ms" <<std::endl;
                  printf("\nPageRank SSD: %d \t PageSize: %d \t ItrTime %f ms\n", settings.n_ctrls, settings.pageSize,(double)elapsed.count());
                  for (auto ct : ctrls) {
                      ct->print_reset_stats();
                  }
            }
        } while(changed_h && iter < 100);
        

        cuda_err_chk(cudaEventRecord(end, 0));
        cuda_err_chk(cudaEventSynchronize(end));
        cuda_err_chk(cudaEventElapsedTime(&milliseconds, start, end));

        printf("pg iteration %*u, ", 3, iter);
//        printf("time %*f ms\n", 12, milliseconds);
        printf("\nPageRank Graph:%s \t Impl: %d \t SSD: %d \t PageSize: %d \t TotalTime %f ms\n", filename.c_str(), type, settings.n_ctrls, settings.pageSize, milliseconds); 
        fflush(stdout);

        avg_milliseconds += (double)milliseconds;
        
        cuda_err_chk(cudaMemcpy(delta_h, delta_d, vertex_count * sizeof(ValueT), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemcpy(value_h, value_d, vertex_count * sizeof(ValueT), cudaMemcpyDeviceToHost));

        // for(uint64_t tid = 0; tid<32; tid++){
        //         printf("Value of tid: %llu is %f and delta is %f\n", tid, value_h[tid], delta_h[tid]); 
        // }

        if((type == BASELINE_PC) || (type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC) || (type == BASELINE_HASH_PC) || (type == COALESCE_HASH_PC) ||(type == COALESCE_CHUNK_HASH_PC)){
            delete h_pc;
            delete h_range;
            delete h_array;
        }
        free(value_h);
        cuda_err_chk(cudaFree(label_d));
        cuda_err_chk(cudaFree(changed_d));
        cuda_err_chk(cudaFree(vertexList_d));
        cuda_err_chk(cudaFree(delta_d));
        cuda_err_chk(cudaFree(residual_d));
        cuda_err_chk(cudaFree(value_d));

        if(mem!=BAFS_DIRECT)
            cuda_err_chk(cudaFree(edgeList_d));
         
         for (size_t i = 0 ; i < settings.n_ctrls; i++)
             delete ctrls[i];
    }
    catch (const error& e){
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }
    return 0;
}
