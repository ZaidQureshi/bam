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
#include <chrono>
#include <iostream>

using error = std::runtime_error;
using std::string;
//const char* const ctrls_paths[] = {"/dev/libnvm0","/dev/libnvm1",   "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
const char* const intel_ctrls_paths[] = {"/dev/libinvm0", "/dev/libinvm1", "/dev/libinvm4", "/dev/libinvm9", "/dev/libinvm2", "/dev/libinvm3", "/dev/libinvm5", "/dev/libinvm6", "/dev/libinvm7", "/dev/libinvm8"};
//const char* const ctrls_paths[] = {"/dev/libnvm0"};
#define UINT64MAX 0xFFFFFFFFFFFFFFFF

#define MYINFINITY 0xFFFFFFFF

#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_NUM 1024ULL

typedef uint64_t EdgeT;

typedef enum {
    BASELINE = 0,
    COALESCE = 1,
    COALESCE_CHUNK = 2,
    BASELINE_PC = 3,
    COALESCE_PC = 4, 
    COALESCE_CHUNK_PC = 5,
    FRONTIER_BASELINE = 6,
    FRONTIER_COALESCE = 7,
    FRONTIER_BASELINE_PC = 8,
    FRONTIER_COALESCE_PC = 9,
    BASELINE_PTR_PC = 10,
    COALESCE_PTR_PC = 11,
    COALESCE_CHUNK_PTR_PC = 12,
    FRONTIER_BASELINE_PTR_PC = 13,
    FRONTIER_COALESCE_PTR_PC = 14,
    COALESCE_HASH = 15,
    COALESCE_HASH_PTR_PC = 16,
    COALESCE_COARSE = 18, 
    COALESCE_HASH_COARSE = 19, 
    COALESCE_COARSE_PTR_PC = 20, 
    COALESCE_HASH_COARSE_PTR_PC = 21, 
    COALESCE_HASH_HALF = 22, 
    COALESCE_HASH_HALF_PTR_PC = 23, 
    OPTIMIZED=26,
    OPTIMIZED_PC=27,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT = 6,
} mem_type;



//TODO: Templatize
//TODO: winnerList is initialized to UINT64MAX
// launch params - number of vertices and each thread does a scatter operation. 
__global__ __launch_bounds__(128,16)
void kernel_first_vertex_step1(uint64_t vertex_count, uint64_t *vertexList, uint64_t num_elems_in_cl, unsigned long long int *winnerList){
   const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x; 

   if(tid < vertex_count){
       unsigned long long int clid = (unsigned long long int) vertexList[tid]/(unsigned long long int)num_elems_in_cl;
       atomicMin(&(winnerList[clid]), tid);
   }
}



//TODO: Templatize the kernel for clstart and clend.
//TODO: launch param: number of CL lines in the data. 
__global__ __launch_bounds__(128,16)
void kernel_first_vertex_step2(uint64_t n_cachelines, uint64_t *vertexList, unsigned long long int *winnerList, uint64_t num_elems_in_cl, uint64_t *firstVertexList){

    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x; 
    
    //const uint64_t clstart = tid*num_elems_in_cl; 
    //const uint64_t clend   = (tid+1)*num_elems_in_cl; 

    //check if the cacheline is filled by backtracking. If the winner array has value of RANDMAX, then it should be filled by the backtracking.
    if(tid < n_cachelines){
        uint64_t wid = winnerList[tid]; 
        if(wid!=UINT64MAX){

            uint64_t winVertval = vertexList[wid];

            uint64_t fringes = winVertval % num_elems_in_cl;
            if((fringes == 0)){
                firstVertexList[tid] = wid;
                bool backtrack = false; 
                uint64_t i =1; 
                while(!backtrack){
     //               printf("I got called : %llu\n", i);
                    backtrack = true; 
                    int64_t tmptid = (int64_t)tid- (int64_t)i;
                    if(tmptid>=0){
                        uint64_t pwid = winnerList[tid-i]; 
                        if(pwid == UINT64MAX){
                            backtrack  = false; 
                            firstVertexList[tid-i] = wid-1; 
                        }
                        else {
                            backtrack = true; 
                        }
                    }
                    i++; 
                }
            } else {
                wid                  = wid - 1; 
                uint64_t currVertval = vertexList[wid]; 
                uint64_t nsize       = winVertval - currVertval - fringes; 

                uint64_t backtrackItr = (nsize + num_elems_in_cl)/num_elems_in_cl; 
                for(uint64_t i = 0; i < backtrackItr ; i++){
                    if( (tid-i)>=0 )
                       firstVertexList[tid-i] = wid; //TODO: does this required to be atomicMin?  
                }
            }
        }
    }
}

__global__ __launch_bounds__(128,16)
void kernel_verify(uint64_t count, unsigned long long int *list, uint64_t condval, uint8_t type){
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x; 
    
    if(tid < count){
        switch(type){
            case 1: {
                    if(list[tid] != condval){
                        printf("index %llu is incorrect and has value :%llu \n",(unsigned long long int) tid, list[tid]);
                    }
                    break;
            }
            case 2: {
                    if(list[tid] == condval){
                        printf("index %llu is incorrect and has value :%llu \n",(unsigned long long int) tid, list[tid]);
                    }
                    break;
            }
        }
    }
}






__global__ __launch_bounds__(128,16)
void kernel_frontier_baseline(unsigned int *label, const unsigned int level, const uint64_t vertex_count,
                                const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long int *changed,
                                const uint32_t *curr_frontier, uint32_t *next_frontier) {
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    //const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint32_t laneIdx = tid &  ((1 << WARP_SHIFT) - 1);

    if (tid < curr_frontier_size) {
        const uint32_t nid = curr_frontier[tid];
        const uint64_t start = vertexList[nid];
        const uint64_t end = vertexList[nid+1];

        for (uint64_t i = start; i < end; i++) {
            const EdgeT next = edgeList[i];

            if(label[next] == MYINFINITY) {
                //unsigned int prev = atomicExch(label+next, level+1);
                //if (prev == MYINFINITY) {
                    //performance code
                    // unsigned int pre_val = atomicCAS(&(label[next]),(unsigned int)MYINFINITY,(unsigned int)(level+1));
                    // if(pre_val == MYINFINITY){
                    //     atomicAdd(&globalvisitedcount_d[0], (unsigned long long int)(vertexList[next+1] - vertexList[next]));
                    // }
                    // *changed = true;

                    // uint32_t mask = __activemask();

                    // int leader = __ffs(mask) - 1;
                    // unsigned long long int pos;
                    // if (laneIdx == leader) {
                    //     pos = atomicAdd(changed, (unsigned long long int)__popc(mask));

                    // }
                    label[next] = level + 1;
                    uint64_t mypos = atomicAdd(changed, 1);
                    //pos = __shfl_sync(mask, pos, leader);

                    //unsigned long long int mypos = (pos) + __popc(mask & ((1 << laneIdx) - 1));

                    next_frontier[mypos] = next;

                //}

            }
        }
    }


}

__global__ __launch_bounds__(128,16)
void kernel_frontier_coalesce(unsigned int *label, const unsigned int level, const uint64_t vertex_count,
                                const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long int *changed,
                                const uint32_t *curr_frontier, uint32_t *next_frontier) {
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < curr_frontier_size) {
        const uint32_t nid = curr_frontier[warpIdx];
        const uint64_t start = vertexList[nid];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[nid+1];


        for (uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                const EdgeT next = edgeList[i];

                if(label[next] == MYINFINITY) {
                    uint32_t prev = atomicExch(label+next, level+1);
                    if (prev == MYINFINITY) {


                        //label[next] = level + 1;
                        uint32_t mask = __activemask();
                        uint32_t leader = __ffs(mask) - 1;
                        unsigned long long pos;
                        if (laneIdx == leader)
                            pos = atomicAdd(changed, __popc(mask));
                        pos = __shfl_sync(mask, pos, leader);
                        uint64_t mypos = pos + __popc(mask & ((1 << laneIdx) - 1));

                        //uint64_t mypos = atomicAdd(changed, 1);
                        next_frontier[mypos] = next;
                    }

                }
            }
        }
    }


}

__global__ __launch_bounds__(128,16)
void kernel_frontier_baseline_pc(unsigned int *label, const unsigned int level, const uint64_t vertex_count,
                                const uint64_t *vertexList, array_d_t<uint64_t>* da, const uint64_t curr_frontier_size, unsigned long long int *changed,
                                const uint32_t *curr_frontier, uint32_t *next_frontier) {
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    //const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint32_t laneIdx = tid &  ((1 << WARP_SHIFT) - 1);

    if (tid < curr_frontier_size) {
        const uint32_t nid = curr_frontier[tid];
        const uint64_t start = vertexList[nid];
        const uint64_t end = vertexList[nid+1];

        for (uint64_t i = start; i < end; i++) {
            const EdgeT next = da->seq_read(i);

            if(label[next] == MYINFINITY) {
                unsigned int prev = atomicExch(label+next, level+1);
                if (prev == MYINFINITY) {
                    //performance code
                    // unsigned int pre_val = atomicCAS(&(label[next]),(unsigned int)MYINFINITY,(unsigned int)(level+1));
                    // if(pre_val == MYINFINITY){
                    //     atomicAdd(&globalvisitedcount_d[0], (unsigned long long int)(vertexList[next+1] - vertexList[next]));
                    // }
                    // *changed = true;

                    uint32_t mask = __activemask();

                    int leader = __ffs(mask) - 1;
                    unsigned long long int pos;
                    if (laneIdx == leader) {
                        pos = atomicAdd(changed, (unsigned long long int)__popc(mask));

                    }
                    label[next] = level + 1;
                    //uint64_t mypos = atomicAdd(changed, 1);
                    pos = __shfl_sync(mask, pos, leader);

                    unsigned long long int mypos = (pos) + __popc(mask & ((1 << laneIdx) - 1));

                    next_frontier[mypos] = next;

                }

            }
        }
    }


}

__global__ __launch_bounds__(128,16)
void kernel_frontier_coalesce_pc(unsigned int *label, const unsigned int level, const uint64_t vertex_count,
                                const uint64_t *vertexList, array_d_t<uint64_t>* da, const uint64_t curr_frontier_size, unsigned long long int *changed,
                                const uint32_t *curr_frontier, uint32_t *next_frontier) {
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < curr_frontier_size) {
        const uint32_t nid = curr_frontier[warpIdx];
        const uint64_t start = vertexList[nid];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[nid+1];


        for (uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                const EdgeT next = da->seq_read(i);

                if(label[next] == MYINFINITY) {
                    uint32_t prev = atomicExch(label+next, level+1);
                    if (prev == MYINFINITY) {


                        //label[next] = level + 1;
                        uint32_t mask = __activemask();
                        uint32_t leader = __ffs(mask) - 1;
                        unsigned long long pos;
                        if (laneIdx == leader)
                            pos = atomicAdd(changed, __popc(mask));
                        pos = __shfl_sync(mask, pos, leader);
                        uint64_t mypos = pos + __popc(mask & ((1 << laneIdx) - 1));

                        //uint64_t mypos = atomicAdd(changed, 1);
                        next_frontier[mypos] = next;
                    }

                }
            }
        }
    }
}


__global__ __launch_bounds__(128,16)
void kernel_frontier_coalesce_ptr_pc(unsigned int *label, const unsigned int level, const uint64_t vertex_count,
                                const uint64_t *vertexList, array_d_t<uint64_t>* da, const uint64_t curr_frontier_size, unsigned long long int *changed,
                                const uint32_t *curr_frontier, uint32_t *next_frontier) {
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < curr_frontier_size) {
        bam_ptr<uint64_t> ptr(da);
        const uint32_t nid = curr_frontier[warpIdx];
        const uint64_t start = vertexList[nid];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[nid+1];


        for (uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                //const EdgeT next = da->seq_read(i);
                const EdgeT next = ptr[i];

                if(label[next] == MYINFINITY) {
                    uint32_t prev = atomicExch(label+next, level+1);
                    if (prev == MYINFINITY) {
                        //label[next] = level + 1;
                        uint32_t mask = __activemask();
                        uint32_t leader = __ffs(mask) - 1;
                        unsigned long long pos;
                        if (laneIdx == leader)
                            pos = atomicAdd(changed, __popc(mask));
                        pos = __shfl_sync(mask, pos, leader);
                        uint64_t mypos = pos + __popc(mask & ((1 << laneIdx) - 1));

                        //uint64_t mypos = atomicAdd(changed, 1);
                        next_frontier[mypos] = next;
                    }

                }
            }
        }
    }
}



__global__ void kernel_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count, 
                        const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, unsigned long long int *globalvisitedcount_d, unsigned long long int *vertexVisitCount_d
    ) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

    // if(tid==0)
    //         printf("Warning: The code is not optimal because of additional counters added for profiling\n");

    if(tid < vertex_count && label[tid] == level) {
        const uint64_t start = vertexList[tid];
        const uint64_t end = vertexList[tid+1];

        for(uint64_t i = start; i < end; i++) {
            const EdgeT next = edgeList[i];
            //performance code
            // atomicAdd(&vertexVisitCount_d[next], 1);

            if(label[next] == MYINFINITY) {
                //performance code
                // unsigned int pre_val = atomicCAS(&(label[next]),(unsigned int)MYINFINITY,(unsigned int)(level+1));
                // if(pre_val == MYINFINITY){
                //     atomicAdd(&globalvisitedcount_d[0], (unsigned long long int)(vertexList[next+1] - vertexList[next]));
                // }
                // *changed = true;

                label[next] = level + 1;
                *changed = true;
            }
        }
    }
}



__global__ __launch_bounds__(128,16)
void kernel_baseline_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                        const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, unsigned long long int *globalvisitedcount_d, unsigned long long int *vertexVisitCount_d
    ) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;

//    array_d_t<uint64_t> d_array = *da;
    // if(tid==0)
    //         printf("Warning: The code is not optimal because of additional counters added for profiling\n");

    if(tid < vertex_count && label[tid] == level) {
        const uint64_t start = vertexList[tid];
        const uint64_t end = vertexList[tid+1];

        for(uint64_t i = start; i < end; i++) {
            //EdgeT next = da->seq_read(i);
            EdgeT next = da->seq_read(i);
//                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);
            //performance code
            // atomicAdd(&vertexVisitCount_d[next], 1);

            if(label[next] == MYINFINITY) {
                //performance code
                // unsigned int pre_val = atomicCAS(&(label[next]),(unsigned int)MYINFINITY,(unsigned int)(level+1));
                // if(pre_val == MYINFINITY){
                //     atomicAdd(&globalvisitedcount_d[0], (unsigned long long int)(vertexList[next+1] - vertexList[next]));
                // }
                // *changed = true;

                label[next] = level + 1;
                *changed = true;
            }
        }
    }
}





__global__ void kernel_coalesce(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

            if (i >= start) {
                const EdgeT next = edgeList[i];
  //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {

                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}




__global__ __launch_bounds__(128,16)
void kernel_coalesce_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    //array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                //const EdgeT next = edgeList[i];
                //EdgeT next = da->seq_read(i);
                EdgeT next = da->seq_read(i);
//                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {
                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}

__global__ __launch_bounds__(128,16)
void kernel_coalesce_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    //array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        bam_ptr<uint64_t> ptr(da);
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                //const EdgeT next = edgeList[i];
                //EdgeT next = da->seq_read(i);
                EdgeT next = ptr[i];
//                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {
                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}



__global__ __launch_bounds__(128,16)
void kernel_coalesce_coarse(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t coarse) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    
    for(uint64_t j = 0; j < coarse; j++){
        uint64_t cwarpIdx = warpIdx * coarse + j;
        if(cwarpIdx < vertex_count && label[cwarpIdx] == level) {
            const uint64_t start = vertexList[cwarpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[cwarpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
    //        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

                if (i >= start) {
                    const EdgeT next = edgeList[i];
    //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                    if(label[next] == MYINFINITY) {

                    //    if(level ==0)
                    //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}




__global__ __launch_bounds__(128,16)
void kernel_coalesce_coarse_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t coarse) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    bam_ptr<uint64_t> ptr(da);

    for(uint64_t j = 0; j < coarse; j++){
        uint64_t cwarpIdx = warpIdx * coarse + j;
        if(cwarpIdx < vertex_count && label[cwarpIdx] == level) {
            const uint64_t start = vertexList[cwarpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[cwarpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
    //        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

                if (i >= start) {
                    // const EdgeT next = edgeList[i];
                    EdgeT next = ptr[i];
    //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);
                    if(label[next] == MYINFINITY) {
                    //    if(level ==0)
                    //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}






__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = oldtid >> WARP_SHIFT;
    const uint64_t laneIdx = oldtid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = stride; 
    
    const uint64_t nep = (vertex_count+(STRIDE))/(STRIDE); 
    uint64_t warpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*(STRIDE));
    
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
//        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

            if (i >= start) {
                const EdgeT next = edgeList[i];
  //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {

                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}



__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = oldtid >> WARP_SHIFT;
    const uint64_t laneIdx = oldtid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = stride; 
    const uint64_t nep = (vertex_count+(STRIDE))/(STRIDE); 
    uint64_t warpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*(STRIDE));

    //array_d_t<uint64_t> d_array = *da;
    if(warpIdx < vertex_count && label[warpIdx] == level) {
        bam_ptr<uint64_t> ptr(da);
        const uint64_t start = vertexList[warpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[warpIdx+1];

        for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
            if (i >= start) {
                //const EdgeT next = edgeList[i];
                //EdgeT next = da->seq_read(i);
                EdgeT next = ptr[i];
//                printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {
                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu\n", tid, (unsigned long long)level, (unsigned long long)next);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}




__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_coarse(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed,uint64_t coarse, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = oldtid >> WARP_SHIFT;
    const uint64_t laneIdx = oldtid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = stride; 
    
    const uint64_t nep = (vertex_count+(STRIDE*coarse))/(STRIDE*coarse); 
    uint64_t cwarpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*(STRIDE));
    
    for(uint64_t j=0; j<coarse; j++){
        uint64_t warpIdx = cwarpIdx*coarse+j;
        if(warpIdx < vertex_count && label[warpIdx] == level) {
            const uint64_t start = vertexList[warpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[warpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
    //        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

                if (i >= start) {
                    const EdgeT next = edgeList[i];
    //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                    if(label[next] == MYINFINITY) {

                    //    if(level ==0)
                    //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        } 
    }
}




__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_coarse_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t coarse, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldwarpIdx = oldtid >> WARP_SHIFT;
    const uint64_t laneIdx = oldtid & ((1 << WARP_SHIFT) - 1);
    uint64_t STRIDE = stride; 
    bam_ptr<uint64_t> ptr(da);
    
    const uint64_t nep = (vertex_count+(STRIDE*coarse))/(STRIDE*coarse); 
    uint64_t cwarpIdx = (oldwarpIdx/nep) + ((oldwarpIdx % nep)*(STRIDE));
    
    for(uint64_t j=0; j<coarse; j++){
        uint64_t warpIdx = cwarpIdx*coarse+j;
        if(warpIdx < vertex_count && label[warpIdx] == level) {
            const uint64_t start = vertexList[warpIdx];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[warpIdx+1];

            for(uint64_t i = shift_start + laneIdx; i < end; i += WARP_SIZE) {
    //        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

                if (i >= start) {
                    // const EdgeT next = edgeList[i];
                    EdgeT next = ptr[i];
                    //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                    if(label[next] == MYINFINITY) {

                    //    if(level ==0)
                    //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        } 
    }
}
 



__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_half(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldhalfwarpIdx = oldtid >> 4;
    const uint64_t halflaneIdx = oldtid & ((1 << 4) - 1);
    uint64_t STRIDE = stride; 
    
    const uint64_t nep = (vertex_count+(STRIDE))/(STRIDE); 
    uint64_t halfwarpIdx = (oldhalfwarpIdx/nep) + ((oldhalfwarpIdx % nep)*(STRIDE));
    
    if(halfwarpIdx < vertex_count && label[halfwarpIdx] == level) {
        const uint64_t start = vertexList[halfwarpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[halfwarpIdx+1];

        for(uint64_t i = shift_start + halflaneIdx; i < end; i += 16) {
//        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

            if (i >= start) {
                const EdgeT next = edgeList[i];
  //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {

                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}



__global__ __launch_bounds__(128,16)
void kernel_coalesce_hash_half_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t stride) {
    const uint64_t oldtid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t oldhalfwarpIdx = oldtid >> 4;
    const uint64_t halflaneIdx = oldtid & ((1 << 4) - 1);
    uint64_t STRIDE = stride; 
    
    const uint64_t nep = (vertex_count+(STRIDE))/(STRIDE); 
    uint64_t halfwarpIdx = (oldhalfwarpIdx/nep) + ((oldhalfwarpIdx % nep)*(STRIDE));
    
    if(halfwarpIdx < vertex_count && label[halfwarpIdx] == level) {
        const uint64_t start = vertexList[halfwarpIdx];
        const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
        const uint64_t end = vertexList[halfwarpIdx+1];
        bam_ptr<uint64_t> ptr(da);

        for(uint64_t i = shift_start + halflaneIdx; i < end; i += 16) {
//        printf("Inside kernel %llu %llu %llu\n", (unsigned long long) i, (unsigned long long)start, (unsigned long long) (end-start));

            if (i >= start) {
                EdgeT next = ptr[i];
  //printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                if(label[next] == MYINFINITY) {

                //    if(level ==0)
                //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                    label[next] = level + 1;
                    *changed = true;
                }
            }
        }
    }
}




__global__ void kernel_coalesce_chunk(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
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
        if(label[i] == level) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    const EdgeT next = edgeList[j];
          
                    if(label[next] == MYINFINITY) {
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}


__global__  __launch_bounds__(1024,2)
void kernel_coalesce_chunk_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed) {
    const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t warpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE;
    uint64_t chunk_size = CHUNK_SIZE;
    bam_ptr<uint64_t> ptr(da);

    //array_d_t<uint64_t> d_array = *da;
    if((chunkIdx + CHUNK_SIZE) > vertex_count) {
        if ( vertex_count > chunkIdx )
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for(uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++) {
        if(label[i] == level) {
            const uint64_t start = vertexList[i];
            const uint64_t shift_start = start & 0xFFFFFFFFFFFFFFF0;
            const uint64_t end = vertexList[i+1];

            for(uint64_t j = shift_start + laneIdx; j < end; j += WARP_SIZE) {
                if (j >= start) {
                    // const EdgeT next = edgeList[j];
                    //EdgeT next = da->seq_read(j);
                    EdgeT next = ptr[j];
                    // printf("tid: %llu, idx: %llu next: %llu\n", (unsigned long long) tid, (unsigned long long) i, (unsigned long long) next);

                    if(label[next] == MYINFINITY) {
                        label[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}



//TODO: change launch parameters. The number of warps to be launched equal to the number of cachelines. Each warp works on a cacheline. 
//TODO: make it templated. 
__global__ __launch_bounds__(128,16)
void kernel_optimized(uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t* first_vertex, uint64_t num_elems_in_cl, uint64_t n_pages, uint64_t stride){
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t cwarpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);


    uint64_t nep = (n_pages+stride)/stride; 
    uint64_t warpIdx = (cwarpIdx/nep) + ((cwarpIdx % nep)*stride);

    if(warpIdx < n_pages){
        const uint64_t clstart = warpIdx*num_elems_in_cl; 
        const uint64_t clend   = (warpIdx+1)*num_elems_in_cl; 
        //start vertex
        uint64_t cur_vertexid = first_vertex[warpIdx]; 

        //for the first vertex iterator start is the clstart point while the end is either the clend or the cur_vertexid neighborlist end
        uint64_t start = clstart; 
        uint64_t end   = vertexList[cur_vertexid+1];
        bool stop      = false;

        // uint64_t itr = 0;
        if((cur_vertexid < vertex_count) && (warpIdx < n_pages) ) {
           //printf("warpidx: %llu laneidx:%llu clstart: %llu clend: %llu start: %llu end: %llu cur_vertexid : %llu\n", warpIdx, laneIdx, clstart, clend, start, end, cur_vertexid);
            while(!stop){
            //if (cur_vertexid < vertex_count && curr_visit[cur_vertexid] == true) {
                //check if the fetched end of cur_vertexid is beyond clend. If yes, then trim end to clend and this is the last while loop iteration.
                if(end >= clend){
                    end  = clend;
                    stop = true;
                    //           printf("called end >=clend and end is:%llu\n", end);
                }

                if(label[cur_vertexid] == level){
                   for(uint64_t i = start + laneIdx; i < end; i += WARP_SIZE){
                    //            uint64_t val = (uint64_t)atomicAdd(&(totalcount_d[0]), 1);
                    //            printf("itr:%llu i:%llu laneIdx: %llu starts:%llu end:%llu cur_vertexid: %llu pre_atomicval:%llu\n",itr, i, laneIdx,start,  end, cur_vertexid, val);
                       EdgeT next = edgeList[i];
                       if(label[next] == MYINFINITY) {
                    //    if(level ==0)
                    //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                            label[next] = level + 1;
                            *changed = true;
                        }
                   }
                }
                // itr++; 
                
                //this implies there are more vertices to compute in the cacheline. So repeat the loop.
                if(end < clend){
                    cur_vertexid = cur_vertexid + 1; //go to next elem in the vertexlist
                    for (;(cur_vertexid < vertex_count) && (end == vertexList[cur_vertexid+1]);cur_vertexid++) {}
                    if(cur_vertexid < vertex_count){
                        start        = vertexList[cur_vertexid]; 
                        end          = vertexList[cur_vertexid+1]; 
                    }
                    else {
                        stop = true; 
                    }
                }
            }
        }
    }
}



//TODO: change launch parameters. The number of warps to be launched equal to the number of cachelines. Each warp works on a cacheline. 
//TODO: make it templated. 
__global__ __launch_bounds__(128,16)
void kernel_optimized_ptr_pc(array_d_t<uint64_t>* da, uint32_t *label, const uint32_t level, const uint64_t vertex_count, const uint64_t *vertexList, const EdgeT *edgeList, uint64_t *changed, uint64_t* first_vertex, uint64_t num_elems_in_cl, uint64_t n_pages, uint64_t stride, uint64_t iter){
    //const uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    const uint64_t cwarpIdx = tid >> WARP_SHIFT;
    const uint64_t laneIdx = tid & ((1 << WARP_SHIFT) - 1);
    
    uint64_t nep = (n_pages+stride)/stride; 
    uint64_t warpIdx = (cwarpIdx/nep) + ((cwarpIdx % nep)*stride);

    if(warpIdx < n_pages){
         const uint64_t clstart = warpIdx*num_elems_in_cl; 
         const uint64_t clend   = (warpIdx+1)*num_elems_in_cl; 
         //start vertex
         uint64_t cur_vertexid = first_vertex[warpIdx]; 

         //for the first vertex iterator start is the clstart point while the end is either the clend or the cur_vertexid neighborlist end
         uint64_t start = clstart; 
         uint64_t end   = vertexList[cur_vertexid+1];
         bool stop      = false;
         bam_ptr<uint64_t> ptr(da);

         //uint64_t itr = 0;
         if((cur_vertexid < vertex_count) && (warpIdx < n_pages) ) {
            //printf("warpidx: %llu laneidx:%llu clstart: %llu clend: %llu start: %llu end: %llu cur_vertexid : %llu\n", warpIdx, laneIdx, clstart, clend, start, end, cur_vertexid);
             while(!stop){
             //if (cur_vertexid < vertex_count && curr_visit[cur_vertexid] == true) {
                 //check if the fetched end of cur_vertexid is beyond clend. If yes, then trim end to clend and this is the last while loop iteration.
                 if(end >= clend){
                     end  = clend;
                     stop = true;
                     //           printf("called end >=clend and end is:%llu\n", end);
                 }

                 if(label[cur_vertexid] == level){
                    for(uint64_t i = start + laneIdx; i < end; i += WARP_SIZE){
                     //            uint64_t val = (uint64_t)atomicAdd(&(totalcount_d[0]), 1);
                            //printf("itr:%llu i:%llu laneIdx: %llu starts:%llu end:%llu cur_vertexid: %llu pre_atomicval:%llu\n",itr, i, laneIdx,start,  end, cur_vertexid, 0);
                        EdgeT next = ptr[i];
                        //EdgeT next = da->seq_read(i); //[i];
                        if(label[next] == MYINFINITY) {
                     //    if(level ==0)
                     //            printf("tid:%llu, level:%llu, next: %llu start:%llu end:%llu\n", tid, (unsigned long long)level, (unsigned long long)next, (unsigned long long)start, (unsigned long long)end);
                             label[next] = level + 1;
                             *changed = true;
                         }
                    }
                 }
                 //itr++; 
                 
                 //this implies there are more vertices to compute in the cacheline. So repeat the loop.
                 if(end < clend){
                     cur_vertexid = cur_vertexid + 1; //go to next elem in the vertexlist
                     for (;(cur_vertexid < vertex_count) && (end == vertexList[cur_vertexid+1]);cur_vertexid++) {}
                     if(cur_vertexid < vertex_count){
                         start        = vertexList[cur_vertexid]; 
                         end          = vertexList[cur_vertexid+1]; 
                     }
                     else {
                         stop = true; 
                     }
                 }
             }
         }
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

    uint64_t changed_h, *changed_d;// no_src = false;
    int num_run = 0;// arg_num = 0;
    int total_run = 1;// arg_num = 0;
    impl_type type;
    mem_type mem;
    uint32_t *pad;
    uint32_t *label_d, level, zero, iter;
    uint64_t *vertexList_h, *vertexList_d;
    EdgeT *edgeList_h, *edgeList_d;
    uint64_t vertex_count, edge_count, vertex_size, edge_size;
    uint64_t typeT, src;
    uint64_t numblocks, numthreads;
    size_t freebyte, totalbyte;

    float milliseconds;
    double avg_milliseconds;

    uint64_t pc_page_size; 
    uint64_t pc_pages;
   
    try{
         //prepare from settings
         filename = std::string(settings.input); 

         if(settings.src == 0) {
                 total_run = settings.repeat; 
                 src = 0;
         }
         else {
                 total_run = 2; 
                 src = settings.src; 
         }

         type = (impl_type) settings.type; 
         mem = (mem_type) settings.memalloc; 

         pc_page_size = settings.pageSize; 

         uint64_t tile_size = 4096;
         if(settings.tsize == 0)
             tile_size = pc_page_size; 
         else
             tile_size = settings.tsize; 

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

        /*        
        vertex_count = 19;
        edge_count = 41; 
        */
         vertex_count--;

         printf("Vertex: %llu, ", vertex_count);
         vertex_size = (vertex_count+1) * sizeof(uint64_t);

         vertexList_h = (uint64_t*)malloc(vertex_size);

         file.read((char*)vertexList_h, vertex_size);
         file.close();

        /*
        uint64_t  edgesample[] ={1,2,0,2,3,4,0,1,3,4,1,2,4,1,2,3,4,6,0,0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,0,1,2,3}; 
        uint64_t  vertsample[] ={0,2,6,10,13,13,13,16,18,19,19,19,19,19,19,19,19,19,37,41} ;

        for(int vi = 0; vi < vertex_count+1; vi++)
            vertexList_h[vi] = vertsample[vi];
        */

         file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
         if (!file.is_open()) {
             fprintf(stderr, "Edge file open failed\n");
             exit(1);
         };

         file.read((char*)(&edge_count), 8);
         file.read((char*)(&typeT), 8);

         printf("Edge: %llu\n", edge_count);
         fflush(stdout);
         edge_size = edge_count * sizeof(EdgeT); //4096 padding for weights and edges. 
         edge_size = edge_size + (4096 - (edge_size & 0xFFFULL));

         edgeList_h = NULL;
         edgeList_d = NULL;

         // Allocate memory for GPU
         cuda_err_chk(cudaMalloc((void**)&vertexList_d, vertex_size));
         cuda_err_chk(cudaMalloc((void**)&label_d, vertex_count * sizeof(uint32_t)));
         cuda_err_chk(cudaMalloc((void**)&changed_d, sizeof(uint64_t)));
     
         std::vector<unsigned long long int> vertexVisitCount_h;
         unsigned long long int* vertexVisitCount_d;
         unsigned long long int globalvisitedcount_h;
         unsigned long long int* globalvisitedcount_d;
     
         vertexVisitCount_h.resize(vertex_count);
         cuda_err_chk(cudaMalloc((void**)&globalvisitedcount_d, sizeof(unsigned long long int)));
         cuda_err_chk(cudaMemset(globalvisitedcount_d, 0, sizeof(unsigned long long int)));
         cuda_err_chk(cudaMalloc((void**)&vertexVisitCount_d, vertex_count*sizeof(unsigned long long int)));
         cuda_err_chk(cudaMemset(vertexVisitCount_d, 0, vertex_count*sizeof(unsigned long long int)));

         switch (mem) {
             case GPUMEM:
                 edgeList_h = (EdgeT*)malloc(edge_size);
                 file.read((char*)edgeList_h, edge_size);
                /*
                for(int ei = 0; ei < edge_count; ei++)
                    edgeList_h[ei] = edgesample[ei];
                */
                 cuda_err_chk(cudaMalloc((void**)&edgeList_d, edge_size));
                 file.close();
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
                 file.close();
                 break;
             case UVM_DIRECT:
             {
             /*    cuda_err_chk(cudaMallocManaged((void**)&edgeList_d, edge_size));
                 // printf("Address is %p   %p\n", edgeList_d, &edgeList_d[0]); 
                 high_resolution_clock::time_point ft1 = high_resolution_clock::now();
                 file.read((char*)edgeList_d, edge_size);
                 file.close();
                 high_resolution_clock::time_point ft2 = high_resolution_clock::now();
                 duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
                 std::cout<< "edge file read time: "<< time_span.count() <<std::endl;
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
                 std::cout<< "Edge file read time: "<< time_span.count() <<std::endl;
                       
                 file.open(edge_file.c_str(), std::ios::in | std::ios::binary);
                 if (!file.is_open()) {
                     printf("edge file open failed\n");
                     exit(1);
                 }   
                 break;
             }
             case BAFS_DIRECT: 
                 //cuda_err_chk(cudaMemGetInfo(&freebyte, &totalbyte));
                 //if (totalbyte < 16*1024*1024*1024ULL)
                 //    printf("total memory sizeo of current GPU is %llu byte, no need to throttle\n", totalbyte);
                 //else {
                 //    printf("total memory sizeo of current GPU is %llu byte, throttling %llu byte.\n", totalbyte, totalbyte - 16*1024*1024*1024ULL);
                 //    cuda_err_chk(cudaMalloc((void**)&pad, totalbyte - 16*1024*1024*1024ULL));
                 //    throttle_memory<<<1,1>>>(pad);
                 //}
                 break;
              default: 
                 printf("ERROR: Invalid Mem type specified\n");
                 break;
         }
     
     
         printf("Allocation finished\n");
         fflush(stdout);
         uint64_t n_pages = ceil(((float)edge_size)/pc_page_size);
         uint64_t n_tiles = ceil(((float)edge_size)/tile_size);
         // Initialize values
         cuda_err_chk(cudaMemcpy(vertexList_d, vertexList_h, vertex_size, cudaMemcpyHostToDevice));

         if (mem == GPUMEM){
             cuda_err_chk(cudaMemcpy(edgeList_d, edgeList_h, edge_size, cudaMemcpyHostToDevice));
         }
    

         switch (type) {
             case BASELINE:
             case BASELINE_PC:
                 numblocks = ((vertex_count + numthreads) / numthreads);
                 break;
             case COALESCE:
             case COALESCE_HASH:
             case COALESCE_PC:
             case COALESCE_PTR_PC:
             case COALESCE_HASH_PTR_PC:
                 numblocks = ((vertex_count * WARP_SIZE + numthreads) / numthreads);
                 break;
             case COALESCE_CHUNK:
             case COALESCE_CHUNK_PC:
                 numblocks = ((vertex_count * (WARP_SIZE / CHUNK_SIZE) + numthreads) / numthreads);
                 break;
             case COALESCE_COARSE:
             case COALESCE_HASH_COARSE:
             case COALESCE_COARSE_PTR_PC:
             case COALESCE_HASH_COARSE_PTR_PC:
                   numblocks = ((vertex_count * (WARP_SIZE / settings.coarse) + numthreads) / numthreads);
                   break;
             case COALESCE_HASH_HALF:
             case COALESCE_HASH_HALF_PTR_PC:
                   numblocks = ((vertex_count * (WARP_SIZE / 2) + numthreads) / numthreads);
                   break;

             case FRONTIER_BASELINE:
             case FRONTIER_COALESCE:
             case FRONTIER_BASELINE_PC:
             case FRONTIER_COALESCE_PC:
             case FRONTIER_COALESCE_PTR_PC:
                 break;
            
             case OPTIMIZED:
             case OPTIMIZED_PC:
                  numblocks = (n_tiles*WARP_SIZE+numthreads)/numthreads;
                  break; 
             default:
                 fprintf(stderr, "Invalid type\n");
                 exit(1);
                 break;
         }
    
         //TODO : FIX THIS. 
         dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);

         avg_milliseconds = 0.0f;


         if((type==BASELINE_PC)||(type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC)||(type==FRONTIER_BASELINE_PC)||(type == FRONTIER_COALESCE_PC) || (type== FRONTIER_COALESCE_PTR_PC) ||(type == COALESCE_COARSE_PTR_PC) ||(type == COALESCE_HASH_PTR_PC) || (type == COALESCE_HASH_COARSE_PTR_PC) || (type == COALESCE_HASH_HALF_PTR_PC ) || (type == OPTIMIZED_PC)){
                printf("page size: %d, pc_entries: %llu tile_size:%llu\n", pc_page_size, pc_pages, tile_size);
                fflush(stdout);
         }


         std::vector<Controller*> ctrls(settings.n_ctrls);
         if(mem == BAFS_DIRECT){
             cuda_err_chk(cudaSetDevice(settings.cudaDevice));
             for (size_t i = 0 ; i < settings.n_ctrls; i++)
                 ctrls[i] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
             printf("Controllers Created\n");
         }
         char gdevst[15];
         cuda_err_chk(cudaDeviceGetPCIBusId(gdevst, 15, settings.cudaDevice));
         std::cout << "GPUID: "<< gdevst << std::endl;

         printf("Initialization done.\n");
         fflush(stdout);
         
         page_cache_t* h_pc; 
         range_t<uint64_t>* h_range;
         std::vector<range_t<uint64_t>*> vec_range(1);
         array_t<uint64_t>* h_array; 
         uint32_t* curr_frontier_d;
         uint32_t* next_frontier_d;
         
         if((type==BASELINE_PC)||(type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC)||(type==FRONTIER_BASELINE_PC)||(type == FRONTIER_COALESCE_PC) || (type== FRONTIER_COALESCE_PTR_PC) ||(type == COALESCE_COARSE_PTR_PC) ||(type == COALESCE_HASH_PTR_PC) || (type == COALESCE_HASH_COARSE_PTR_PC) || (type == COALESCE_HASH_HALF_PTR_PC ) || (type == OPTIMIZED_PC)){            
            h_pc =new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
            h_range = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)edge_count, (uint64_t) (ceil(settings.ofileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); //, (uint8_t*)edgeList_d);
            vec_range[0] = h_range; 
            h_array = new array_t<uint64_t>(edge_count, settings.ofileoffset, vec_range, settings.cudaDevice);
            
            printf("Page cache initialized\n");
            fflush(stdout);
         }
         if ((type==FRONTIER_BASELINE)||(type == FRONTIER_COALESCE) ||(type==FRONTIER_BASELINE_PC)||(type == FRONTIER_COALESCE_PC)||(type==FRONTIER_COALESCE_PTR_PC)){
             cuda_err_chk(cudaMalloc((void**)&curr_frontier_d,  vertex_count * sizeof(uint32_t)));
             cuda_err_chk(cudaMalloc((void**)&next_frontier_d,  vertex_count * sizeof(uint32_t)));
         }
         uint32_t* tmp_front;




        uint64_t *firstVertexList_d;
        unsigned long long  int *winnerList_d; 
        uint64_t num_elems_in_cl = (uint32_t) tile_size / (sizeof(uint64_t));
        //preprocessing for the optimized implementation.


        if((type == OPTIMIZED_PC) || (type == OPTIMIZED)){
            
            printf("n_tiles: %llu num_elems_in_tile:%llu \n", n_tiles, num_elems_in_cl);
            n_tiles = (edge_count+num_elems_in_cl) / num_elems_in_cl;
            printf("n_tiles: %llu num_elems_in_tile:%llu \n", n_tiles, num_elems_in_cl);
            cuda_err_chk(cudaMalloc((void**)&winnerList_d,   (n_tiles)* sizeof(unsigned long long int)));
            cuda_err_chk(cudaMemset(winnerList_d, UINT64MAX, (n_tiles)* sizeof(unsigned long long int)));
            //printf("UNIT64MAX is: %llu\n", UINT64MAX);
            //uint64_t nblocks = (n_tiles+numthreads)/numthreads;
            //dim3 verifyBlockDim(nblocks); 
            //kernel_verify<<<verifyBlockDim,numthreads>>>(n_tiles,winnerList_d, UINT64MAX, 1);

            // cuda_err_chk(cudaDeviceSynchronize());
            //num_elems_in_cl  = 6;
            printf("Allocating %f MB for FirstVertexList with n_tiles: %llu numelemspercl: %llu\n", ((double)n_tiles*sizeof(uint64_t)/(1024*1024)), n_tiles, num_elems_in_cl);
            cuda_err_chk(cudaMalloc((void**)&firstVertexList_d, n_tiles * sizeof(uint64_t)));

            uint64_t nblocks_step1 = (vertex_count+1+numthreads)/numthreads; 
            uint64_t nblocks_step2 = (n_tiles+numthreads)/numthreads; 
            printf("Launching step1 in generation of FirstVertexList: numblocks: %llu numthreads: %llu\n", nblocks_step1, numthreads);
            dim3 step1blockdim(nblocks_step1);
            dim3 step2blockdim(nblocks_step2);
            kernel_first_vertex_step1<<<step1blockdim,numthreads>>>(vertex_count+1, vertexList_d, num_elems_in_cl, winnerList_d);
            //uint64_t *winnerList_h; 
            //uint64_t copysize = (n_tiles) * sizeof(unsigned long long int);
            //winnerList_h = (uint64_t*)malloc(copysize);
            //cuda_err_chk(cudaMemcpy((void**)winnerList_h, (void**)winnerList_d, copysize, cudaMemcpyDeviceToHost));
            //printf("winnerlist values: \n");
            //for(uint64_t i=0; i< n_tiles; i++){
            //      printf("i: %llu, winner: %llu\n",i, winnerList_h[i]);
            //}
            //printf("\n");
            kernel_first_vertex_step2<<<step2blockdim,numthreads>>>(n_tiles, vertexList_d, winnerList_d, num_elems_in_cl, firstVertexList_d);
            
            //uint64_t *firstVertexList_h; 
            //uint64_t copysize2 = n_tiles * sizeof(unsigned long long int);
            //firstVertexList_h = (uint64_t*) malloc(copysize2); 
            //cuda_err_chk(cudaMemcpy((void**)firstVertexList_h, (void**)firstVertexList_d, copysize2, cudaMemcpyDeviceToHost));
            //printf("Firstvertex values\n");
            //for(uint64_t i=0; i< n_tiles; i++){
            //    printf("%llu\n", firstVertexList_h[i]);
            //}
            uint64_t nblocks = (n_tiles+numthreads)/numthreads;
            //dim3 verifyBlockDim(nblocks); 
            //kernel_verify<<<verifyBlockDim,numthreads>>>(n_tiles, (unsigned long long int*) firstVertexList_d, UINT64MAX, 2);
            
            cuda_err_chk(cudaDeviceSynchronize());
            cuda_err_chk(cudaFree(winnerList_d));
            //free(winnerList_h);
        }

 
         // Set root
         for (int i = 0; i < total_run; i++) {
             zero = 0;
             cuda_err_chk(cudaMemset(label_d, 0xFF, vertex_count * sizeof(uint32_t)));
             cuda_err_chk(cudaMemcpy(&label_d[src], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
             if ((type==FRONTIER_BASELINE)||(type == FRONTIER_COALESCE) ||(type==FRONTIER_BASELINE_PC)||(type == FRONTIER_COALESCE_PC)||(type==FRONTIER_COALESCE_PTR_PC)){
                 cuda_err_chk(cudaMemcpy(curr_frontier_d, &src, sizeof(uint32_t), cudaMemcpyHostToDevice));
             }

             level = 0;
             iter = 0;

             cuda_err_chk(cudaEventRecord(start, 0));
   // printf("*****baseaddr: %p\n", h_pc->pdt.base_addr);
   //          fflush(stdout);

             // Run BFS
             changed_h = 1;

             //printf("Hash Stride: %llu Coarse: %llu\n", (settings.stride), settings.coarse);
             
             do {
                 uint64_t active = changed_h;
                 changed_h = 0;
                 cuda_err_chk(cudaMemcpy(changed_d, &changed_h, sizeof(uint64_t), cudaMemcpyHostToDevice));
                 auto start = std::chrono::system_clock::now();
                 switch (type) {
                     case BASELINE:
                         kernel_baseline<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, globalvisitedcount_d, vertexVisitCount_d);
                         break;
                     case COALESCE:
                         kernel_coalesce<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                         break;
                     case COALESCE_HASH:
                         //TODO: fix the stride 
                         kernel_coalesce_hash<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.stride);
                         break;
                     case COALESCE_CHUNK:
                         kernel_coalesce_chunk<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                         break;
                     case BASELINE_PC:
                         //printf("Calling Page cache enabled baseline kernel\n");
                         kernel_baseline_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, globalvisitedcount_d, vertexVisitCount_d);
                         break;
                     case COALESCE_PC:
                         //printf("Calling Page cache enabled coalesce kernel\n");
                         kernel_coalesce_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                         break;
                     case COALESCE_PTR_PC:
                         //printf("Calling Page cache enabled coalesce kernel\n");
                         kernel_coalesce_ptr_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                         break;
                     case COALESCE_HASH_PTR_PC:
                         {
                            //  //TODO: fix the stride
                            //  //printf("Calling transposed kernel\n");
                            //  uint64_t stride = settings.stride; 
                            //  if(iter == 6){
                            //      printf("changing stride\n");
                            //      fflush(stdout);
                            //      stride = 768; 
                            //  }
                             kernel_coalesce_hash_ptr_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.stride);
                             break;
                         }
                    case COALESCE_CHUNK_PC:
                         //printf("Calling Page cache enabled coalesce chunk kernel\n");
                         kernel_coalesce_chunk_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d);
                         break;

                    case COALESCE_COARSE:
                         {
                             kernel_coalesce_coarse<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.coarse);
                             break;
                         }
                    case COALESCE_HASH_COARSE:
                         {
                            kernel_coalesce_hash_coarse<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.coarse, settings.stride);
                             break;
                         }
                    case COALESCE_COARSE_PTR_PC:
                         {
                             kernel_coalesce_coarse_ptr_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.coarse);
                             break;
                         }
                    case COALESCE_HASH_COARSE_PTR_PC:
                         {
                             kernel_coalesce_hash_coarse_ptr_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.coarse, settings.stride);
                             break;
                         }
                    case COALESCE_HASH_HALF:
                         {
                             kernel_coalesce_hash_half<<<blockDim, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.stride);
                             break;
                         }
                    case COALESCE_HASH_HALF_PTR_PC:
                         {
                             kernel_coalesce_hash_half_ptr_pc<<<blockDim, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d, settings.stride);
                             break;
                         }
                     case FRONTIER_BASELINE:
                          // kernel_frontier_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                          //       const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long *changed,
                          //                          const uint32_t *curr_frontier, uint32_t *next_frontier)
                         numblocks = ((active + numthreads) / numthreads);
                         assert(numblocks <= 0xFFFFFFFF);
                         kernel_frontier_baseline<<<numblocks, numthreads>>>((unsigned int*)label_d, (unsigned int) level, vertex_count, vertexList_d, edgeList_d, active,(unsigned long long int*)changed_d, curr_frontier_d, next_frontier_d);
                         tmp_front = curr_frontier_d;
                         curr_frontier_d = next_frontier_d;
                         next_frontier_d = tmp_front;
                         break;
                     case FRONTIER_COALESCE:
                         // kernel_frontier_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                         //       const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long *changed,
                         //                          const uint32_t *curr_frontier, uint32_t *next_frontier)
                         numblocks = ((active * WARP_SIZE + numthreads) / numthreads);
                         assert(numblocks <= 0xFFFFFFFF);
                         //printf("numblocks: %llu\n", numblocks);
                         kernel_frontier_coalesce<<<numblocks, numthreads>>>((unsigned int*)label_d, (unsigned int) level, vertex_count, vertexList_d, edgeList_d, active,(unsigned long long int*)changed_d, curr_frontier_d, next_frontier_d);
                         tmp_front = curr_frontier_d;
                         curr_frontier_d = next_frontier_d;
                         next_frontier_d = tmp_front;
                         break;
                     case FRONTIER_BASELINE_PC:
                          // kernel_frontier_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                          //       const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long *changed,
                          //                          const uint32_t *curr_frontier, uint32_t *next_frontier)
                         numblocks = ((active + numthreads) / numthreads);
                         assert(numblocks <= 0xFFFFFFFF);
                         kernel_frontier_baseline_pc<<<numblocks, numthreads>>>((unsigned int*)label_d, (unsigned int) level, vertex_count, vertexList_d, h_array->d_array_ptr, active,(unsigned long long int*)changed_d, curr_frontier_d, next_frontier_d);
                         tmp_front = curr_frontier_d;
                         curr_frontier_d = next_frontier_d;
                         next_frontier_d = tmp_front;
                         break;
                     case FRONTIER_COALESCE_PC:
                         // kernel_frontier_baseline(uint32_t *label, const uint32_t level, const uint64_t vertex_count,
                         //       const uint64_t *vertexList, const EdgeT *edgeList, const uint64_t curr_frontier_size, unsigned long long *changed,
                         //                          const uint32_t *curr_frontier, uint32_t *next_frontier)
                         numblocks = ((active * WARP_SIZE + numthreads) / numthreads);
                         assert(numblocks <= 0xFFFFFFFF);
                         //printf("numblocks: %llu\t", numblocks);
                         kernel_frontier_coalesce_pc<<<numblocks, numthreads>>>((unsigned int*)label_d, (unsigned int) level, vertex_count, vertexList_d, h_array->d_array_ptr, active,(unsigned long long int*)changed_d, curr_frontier_d, next_frontier_d);
                         tmp_front = curr_frontier_d;
                         curr_frontier_d = next_frontier_d;
                         next_frontier_d = tmp_front;
                         break;
                     case FRONTIER_COALESCE_PTR_PC:
                         numblocks = ((active * WARP_SIZE + numthreads) / numthreads);
                         assert(numblocks <= 0xFFFFFFFF);
                         //printf("numblocks: %llu\t", numblocks);
                         kernel_frontier_coalesce_ptr_pc<<<numblocks, numthreads>>>((unsigned int*)label_d, (unsigned int) level, vertex_count, vertexList_d, h_array->d_array_ptr, active,(unsigned long long int*)changed_d, curr_frontier_d, next_frontier_d);
                         tmp_front = curr_frontier_d;
                         curr_frontier_d = next_frontier_d;
                         next_frontier_d = tmp_front;
                         break;
                     case OPTIMIZED: 
                            kernel_optimized<<<numblocks, numthreads>>>(label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d,firstVertexList_d, num_elems_in_cl, n_tiles,settings.stride);
                            break;
                     case OPTIMIZED_PC: 
                           kernel_optimized_ptr_pc<<<numblocks, numthreads>>>(h_array->d_array_ptr, label_d, level, vertex_count, vertexList_d, edgeList_d, changed_d,firstVertexList_d, num_elems_in_cl, n_tiles, settings.stride, iter);
                           break;
                     default:
                         fprintf(stderr, "Invalid type\n");
                         exit(1);
                         break;
                 }

                 iter++;
                 level++;

                 cuda_err_chk(cudaMemcpy(&changed_h, changed_d, sizeof(uint64_t), cudaMemcpyDeviceToHost));
                 //auto end = std::chrono::system_clock::now();
                 //if(mem == BAFS_DIRECT) {
                 //     h_array->print_reset_stats();

                 // }
                 //auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                 //std::cout << "Iter "<< iter << " Time: " << elapsed.count() << " ms" << std::endl;

                 //break;
             } while(changed_h);

             cuda_err_chk(cudaEventRecord(end, 0));
             cuda_err_chk(cudaEventSynchronize(end));
             cuda_err_chk(cudaEventElapsedTime(&milliseconds, start, end));
             if(iter > 1){
                 printf("run %*d: ", 3, i);
                 printf("src %*u, ", 10, src);
                 printf("iteration %*u, ", 3, iter);
                 printf("time %*f ms\n", 12, milliseconds);
                 if(mem == BAFS_DIRECT) {
                     h_array->print_reset_stats();
                 }
                 fflush(stdout);
                 avg_milliseconds += (double)milliseconds;
				 num_run++; 
			 }
			 else {
                 avg_milliseconds += 0;
			 }
            
             if(settings.src == 0)
                   src += vertex_count / total_run;
             printf("\nBFS-%d Graph:%s \t Impl: %d \t SSD: %d \t CL: %d \t Cache: %llu \t Stride: %llu \t Coarse: %d \t AvgTime %f ms\n", i, filename.c_str(), type, settings.n_ctrls, settings.pageSize, settings.maxPageCacheSize, settings.stride, settings.coarse, milliseconds);
         }
         
         free(vertexList_h);
         if((type==BASELINE_PC)||(type == COALESCE_PC) ||(type == COALESCE_CHUNK_PC)||(type==FRONTIER_BASELINE_PC)||(type == FRONTIER_COALESCE_PC) || (type== FRONTIER_COALESCE_PTR_PC) ||(type == COALESCE_COARSE_PTR_PC) ||(type == COALESCE_HASH_PTR_PC) || (type == COALESCE_HASH_COARSE_PTR_PC) || (type == COALESCE_HASH_HALF_PTR_PC ) || (type == OPTIMIZED_PC)){            delete h_pc; 
            delete h_range; 
            delete h_array;
         }
         if (edgeList_h)
             free(edgeList_h);
         cuda_err_chk(cudaFree(vertexList_d));
         cuda_err_chk(cudaFree(label_d));
         cuda_err_chk(cudaFree(changed_d));

         cuda_err_chk(cudaFree(globalvisitedcount_d));
         cuda_err_chk(cudaFree(vertexVisitCount_d));
         vertexVisitCount_h.clear();

         if (edgeList_d)
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
