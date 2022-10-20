/* References:
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

#include <iterator> 
#include <functional>

#define UINT64MAX 0xFFFFFFFFFFFFFFFF

using error = std::runtime_error;
using std::string;
//const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro1", "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5", "/dev/libnvmpro6", "/dev/libnvmpro7"};
//const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};

#define WARP_SHIFT 5
#define WARP_SIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_NUM 1024ULL

#define MAXWARP 64

typedef uint64_t EdgeT;

typedef enum {
    BASELINE = 0,
    OPTIMIZED= 1, 
    BASELINE_PC = 2,
    OPTIMIZED_PC= 3,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    UVM_READONLY_NVLINK = 3,
    UVM_DIRECT_NVLINK = 4,
    BAFS_DIRECT= 6,
} mem_type;


__global__ //__launch_bounds__(64,32)
void kernel_baseline(uint64_t n_elems, uint64_t *A, uint64_t *B, unsigned long long int *sum){
    //uint64_t tid = blockDim.x * BLOCK_NUM * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid<n_elems){
       sum[tid]= A[tid] + B[tid];  
       //uint64_t val = A[tid] + B[tid];  
       //atomicAdd(&sum[0], val);
       //printf("tid: %llu A:%llu B:%llu \n",tid,  A[tid], B[tid]);
    }
}

__global__ __launch_bounds__(64,32)
void kernel_baseline_ptr_pc(array_d_t<uint64_t>* da, array_d_t<uint64_t>* db, uint64_t n_elems, array_d_t<uint64_t>* dc,  unsigned long long int *sum){
    uint64_t  tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    bam_ptr<uint64_t> Aptr(da);
    bam_ptr<uint64_t> Bptr(db);
    bam_ptr<uint64_t> Cptr(dc);

    if(tid<n_elems){
       Cptr[tid] = Aptr[tid] + Bptr[tid];  
       //uint64_t val = Aptr[tid] + Bptr[tid];  
       //uint64_t val = A[tid] + B[tid];  
       //sum[tid] = val; 
       //atomicAdd(&sum[0], val);
    }
}


template<typename T>
__global__ __launch_bounds__(64,32)
void kernel_sequential_warp(T *A, T *B, uint64_t n_elems,  uint64_t n_pages_per_warp, unsigned long long* sum,  uint64_t n_warps, size_t page_size) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T val = 0;
    uint64_t idx=0; 

    if(tid ==0)
        printf("n_elems_per_page: %llu\n", n_elems_per_page);
    if (warp_id < n_warps) {
        size_t start_page = n_pages_per_warp * warp_id;;
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE) {
               idx = start_idx + j; 
               if(idx < n_elems){
                   val  = A[idx] + B[idx];
                   sum[idx] = val;
                   //atomicAdd(&sum[0], val);
       //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid], B[tid]);
               }
            }
        }
    }
}

template<typename T>
__global__ //__launch_bounds__(64,32)
void kernel_sequential_warp_ptr_pc(array_d_t<T> *da, array_d_t<T> *db, uint64_t n_elems,  uint64_t n_pages_per_warp, array_d_t<T> *dc, unsigned long long* sum,  uint64_t n_warps, size_t page_size, uint64_t stride) {

    bam_ptr<uint64_t> Aptr(da);
    bam_ptr<uint64_t> Bptr(db);
    bam_ptr<uint64_t> Cptr(dc);

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T val = 0;
    uint64_t idx=0; 
    uint64_t nep = (n_warps+stride-1)/stride; 
    uint64_t warp_id = (old_warp_id/nep) + ((old_warp_id % nep)* stride);

    if (warp_id < n_warps) {
        size_t start_page = n_pages_per_warp * warp_id;;
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARP_SIZE) {
               idx = start_idx + j; 
               if(idx < n_elems){
                   val  = Aptr[idx] + Bptr[idx];
                   Cptr[idx] = val; 
                   //sum[idx] = val;
                   //atomicAdd(&sum[0], val);
       //            printf("tid: %llu A:%llu B:%llu \n",idx,  A[tid], B[tid]);
               }
            }
        }
        //sum[0] =val;
    }
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

    std::ifstream filea, fileb;
    std::string a_file, b_file;
    std::string a_file_bin, b_file_bin;
    std::string filename;

    impl_type type;
    mem_type mem;
    uint32_t *pad;
    uint64_t *a_h, *a_d;
    uint64_t *b_h, *b_d;
    uint64_t *c_h, *c_d;
    uint64_t n_elems, n_size;
    uint64_t typeT;
    uint64_t numblocks, numthreads;
    size_t freebyte, totalbyte;

    float milliseconds;

    uint64_t pc_page_size;
    uint64_t pc_pages; 

    try{

        a_file = std::string(settings.input_a); 
        b_file = std::string(settings.input_b); 
        
        type = (impl_type) settings.type; 
        mem = (mem_type) settings.memalloc; 

        pc_page_size = settings.pageSize; 
        pc_pages = ceil((float)settings.maxPageCacheSize/pc_page_size);

        numthreads = settings.numThreads;
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        
        cudaEvent_t start, end, tstart, tend;
        cuda_err_chk(cudaEventCreate(&start));
        cuda_err_chk(cudaEventCreate(&end));
        cuda_err_chk(cudaEventCreate(&tstart));
        cuda_err_chk(cudaEventCreate(&tend));

        a_file_bin = a_file + ".dst";
        b_file_bin = b_file + ".dst";

        std::cout << "A: " << a_file_bin << " B: " << b_file_bin << std::endl;

        uint64_t n_elems = settings.n_elems;
        uint64_t n_elems_size = n_elems * sizeof(uint64_t);
        printf("Total elements: %llu \n", n_elems);
        uint64_t tmp; 
        
        // Read files
        filea.open(a_file_bin.c_str(), std::ios::in | std::ios::binary);
        if (!filea.is_open()) {
            printf("A file open failed\n");
            exit(1);
        };

        filea.read((char*)(&tmp), 16);
        if(mem != BAFS_DIRECT)
            a_h = (uint64_t*)malloc(n_elems_size);
        if((mem!=BAFS_DIRECT) &&  (mem != UVM_DIRECT)){
             //printf("before mem switch\n");
             //fflush(stdout); 
             filea.read((char*)a_h, n_elems_size);
             filea.close();
        }

        // Read files
        fileb.open(b_file_bin.c_str(), std::ios::in | std::ios::binary);
        if (!fileb.is_open()) {
            printf("A file open failed\n");
            exit(1);
        };

        fileb.read((char*)(&tmp), 16);
        if(mem != BAFS_DIRECT)
            b_h = (uint64_t*)malloc(n_elems_size);
        if((mem!=BAFS_DIRECT) && (mem != UVM_DIRECT)){
            fileb.read((char*)b_h, n_elems_size);
            fileb.close();
        }


        switch (mem) {
            case GPUMEM:
                {  
                cuda_err_chk(cudaMalloc((void**)&a_d, n_elems_size));
                cuda_err_chk(cudaMalloc((void**)&b_d, n_elems_size));
                high_resolution_clock::time_point mc1 = high_resolution_clock::now();
                cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
                cuda_err_chk(cudaMemcpy(b_d, b_h, n_elems_size, cudaMemcpyHostToDevice));
                high_resolution_clock::time_point mc2 = high_resolution_clock::now();
                duration<double> mc_time_span = duration_cast<duration<double>>(mc2 -mc1);
                std::cout<< "Memcpy time for loading the inputs: "<< mc_time_span.count() <<std::endl;
                break;
                }
            case UVM_READONLY:
                {
                cuda_err_chk(cudaMallocManaged((void**)&a_d, n_elems_size));
                cuda_err_chk(cudaMallocManaged((void**)&b_d, n_elems_size));
                cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
                cuda_err_chk(cudaMemcpy(b_d, b_h, n_elems_size, cudaMemcpyHostToDevice));
                //TODO: we can move that read op here.
                //file.read((char*)edgeList_d, edge_size);
                cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                cuda_err_chk(cudaMemGetInfo(&freebyte, &totalbyte));
                break;
                }
            case UVM_DIRECT:
                {
                filea.close();
                fileb.close();
                int fda = open(a_file_bin.c_str(), O_RDONLY | O_DIRECT); 
                int fdb = open(b_file_bin.c_str(), O_RDONLY | O_DIRECT); 
                FILE *fa_tmp= fdopen(fda, "rb");
                if ((fa_tmp == NULL) || (fda == -1)) {
                    printf("A file fd open failed\n");
                    exit(1);
                }   
                FILE *fb_tmp= fdopen(fdb, "rb");
                if ((fb_tmp == NULL) || (fdb == -1)) {
                    printf("A file fd open failed\n");
                    exit(1);
                }   
                
                uint64_t count_4k_aligned = ((n_elems + 2 + 4096 / sizeof(uint64_t)) / (4096 / sizeof(uint64_t))) * (4096 / sizeof(uint64_t));
                //uint64_t count_4k_aligned = n_elems; 
                uint64_t size_4k_aligned = count_4k_aligned * sizeof(uint64_t);

                cuda_err_chk(cudaMallocManaged((void**)&a_d, size_4k_aligned));
                cuda_err_chk(cudaMallocManaged((void**)&b_d, size_4k_aligned));
                cuda_err_chk(cudaMemAdvise(a_d, size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                cuda_err_chk(cudaMemAdvise(b_d, size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                high_resolution_clock::time_point ft1 = high_resolution_clock::now();
               
                if (fread(a_d, sizeof(uint64_t), count_4k_aligned, fa_tmp) <0) {
                    printf("A file fread failed: %llu \t %llu\n", count_4k_aligned, n_elems+2);
                    exit(1);
                }   
                fclose(fa_tmp);                                                                                                              
                close(fda);
                
                if (fread(b_d, sizeof(uint64_t), count_4k_aligned, fb_tmp) <0) {
                    printf("B file fread failed\n");
                    exit(1);
                }   
                fclose(fb_tmp);                                                                                                              
                close(fdb);


                a_d = a_d + 2;
                b_d = b_d + 2;

                high_resolution_clock::time_point ft2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
                std::cout<< "file read time: "<< time_span.count() <<std::endl;
                
                /* //THIS DOES NOT WORK
                high_resolution_clock::time_point ft1 = high_resolution_clock::now();
                cuda_err_chk(cudaMallocManaged((void**)&a_d, n_elems_size));
                cuda_err_chk(cudaMallocManaged((void**)&b_d, n_elems_size));
                filea.read((char*)a_d, n_elems_size);
                fileb.read((char*)b_d, n_elems_size);
                cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                //cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                //cuda_err_chk(cudaMemAdvise(b_d, n_elems_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                high_resolution_clock::time_point ft2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
                std::cout<< "file read time: "<< time_span.count() <<std::endl;
                */


                break;
                }
            case BAFS_DIRECT: 
                {
                break;
                }
        }

        
        uint64_t n_pages = ceil(((float)n_elems_size)/pc_page_size); 

        // Allocate memory for GPU
        unsigned long long int *sum_d;
        unsigned long long int *sum_h;
        sum_h = (unsigned long long int*) malloc(n_elems*sizeof(unsigned long long int));
    
        cuda_err_chk(cudaMalloc((void**)&sum_d, n_elems*sizeof(unsigned long long int)));

		printf("Allocation finished\n");
        fflush(stdout);

        uint64_t n_warps;

        switch (type) {
            case BASELINE:
            case BASELINE_PC:
                numblocks = ((n_elems+numthreads-1)/numthreads);
                break;
            case OPTIMIZED: 
            case OPTIMIZED_PC:{
                uint64_t n_elems_per_page = pc_page_size/sizeof(uint64_t); 
                n_warps = (n_elems + n_elems_per_page-1)/n_elems_per_page; 
                numblocks = (n_warps * WARP_SIZE + numthreads-1) / numthreads; 
                break;
                           }
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }
        
        //dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);
        dim3 blockDim(numblocks);

        if((type == BASELINE_PC) || (type==OPTIMIZED_PC)) {
                printf("page size: %d, pc_entries: %llu\n", pc_page_size, pc_pages);
        }
        std::vector<Controller*> ctrls(settings.n_ctrls);
        if(mem == BAFS_DIRECT){
            cuda_err_chk(cudaSetDevice(settings.cudaDevice));
            for (size_t i = 0 ; i < settings.n_ctrls; i++)
                ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
            printf("Controllers Created\n");
        }
        printf("Initialization done\n");
        fflush(stdout);

        page_cache_t* h_pc;
        range_t<uint64_t>* h_Arange;
        range_t<uint64_t>* h_Brange;
        range_t<uint64_t>* h_Crange;
        std::vector<range_t<uint64_t>*> vec_Arange(1);
        std::vector<range_t<uint64_t>*> vec_Brange(1);
        std::vector<range_t<uint64_t>*> vec_Crange(1);
        array_t<uint64_t>* h_Aarray;
        array_t<uint64_t>* h_Barray;
        array_t<uint64_t>* h_Carray;


        uint64_t cfileoffset = 720*1024*1024*1024;
        if((type == BASELINE_PC) || (type == OPTIMIZED_PC)) {
            //TODO: fix for 2 arrays
            h_pc =new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
            h_Arange = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)n_elems, (uint64_t) (ceil(settings.afileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); 
            h_Brange = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)n_elems, (uint64_t) (ceil(settings.bfileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); 
            h_Crange = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)n_elems, (uint64_t) (ceil(cfileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); 
            vec_Arange[0] = h_Arange; 
            vec_Brange[0] = h_Brange; 
            vec_Crange[0] = h_Crange; 
            h_Aarray = new array_t<uint64_t>(n_elems, settings.afileoffset, vec_Arange, settings.cudaDevice);
            h_Barray = new array_t<uint64_t>(n_elems, settings.bfileoffset, vec_Brange, settings.cudaDevice);
            h_Carray = new array_t<uint64_t>(n_elems, cfileoffset, vec_Crange, settings.cudaDevice);

            printf("Page cache initialized\n");
            fflush(stdout);
        }

        //cuda_err_chk(cudaMalloc((void**)&c_d, n_elems_size));

        for(int titr=0; titr<1; titr+=1){
            cuda_err_chk(cudaEventRecord(start, 0));
                
            auto itrstart = std::chrono::system_clock::now();
            cuda_err_chk(cudaMemset(sum_d, 0, n_elems*sizeof(unsigned long long int)));

            switch (type) {
                case BASELINE:
                    printf("launching baseline: blockDim.x :%llu blockDim.y :%llu numthreads:%llu\n", blockDim.x, blockDim.y, numthreads);
                    kernel_baseline<<<blockDim, numthreads>>>(n_elems, a_d, b_d, sum_d);
                    //kernel_baseline<<<blockDim, numthreads>>>(n_elems,  sum_d);
                    break;
                case BASELINE_PC:
                    printf("launching baseline_pc: blockDim.x :%llu blockDim.y :%llu numthreads:%llu\n", blockDim.x, blockDim.y, numthreads);
                    kernel_baseline_ptr_pc<<<blockDim, numthreads>>>(h_Aarray->d_array_ptr, h_Barray->d_array_ptr, n_elems, h_Carray->d_array_ptr, sum_d);
                    h_pc->flush_cache(); 
                    //uint64_t n_pages = pc_pages; 
                    //numblocks = (n_pages+numthreads-1)/numthreads; 
                    //cache_flush<uint64_t><<<numblocks, numthreads>>>(h_Carray->d_array_ptr, n_pages, settings.pageSize);
                    break;

                case OPTIMIZED:
                    printf("launching optimized: blockDim.x :%llu numthreads:%llu\n", blockDim.x, numthreads);
                    kernel_sequential_warp<uint64_t><<<blockDim, numthreads>>>(a_d, b_d, n_elems, 1, sum_d, n_warps, settings.pageSize);
                    break;

                case OPTIMIZED_PC:
                    printf("launching optimized: blockDim.x :%llu numthreads:%llu\n", blockDim.x, numthreads);
                    kernel_sequential_warp_ptr_pc<uint64_t><<<blockDim, numthreads>>>(h_Aarray->d_array_ptr, h_Barray->d_array_ptr, n_elems, 1, h_Carray->d_array_ptr, sum_d, n_warps, settings.pageSize, settings.stride);
                    h_pc->flush_cache(); 
                    break;
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            cuda_err_chk(cudaEventRecord(end, 0));
            cuda_err_chk(cudaEventSynchronize(end));
            cuda_err_chk(cudaEventElapsedTime(&milliseconds, start, end));
            
            //cuda_err_chk(cudaMemcpy(c_d, sum_h, n_elems_size, cudaMemcpyDeviceToHost));
            cuda_err_chk(cudaMemcpy(sum_h, sum_d, n_elems*sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            printf("sum: %llu\n", sum_h[0]);

            auto itrend = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(itrend - itrstart);

            //if(mem == BAFS_DIRECT) {
            //         h_Aarray->print_reset_stats();
            //         h_Barray->print_reset_stats();
            //         h_Carray->print_reset_stats();
		    // printf("VA SSD: %d PageSize: %d itrTime: %f\n", settings.n_ctrls, settings.pageSize, (double)elapsed.count()); 
            //}



            if(mem == BAFS_DIRECT) {
                 h_Aarray->print_reset_stats();
                 h_Barray->print_reset_stats();
                 cuda_err_chk(cudaDeviceSynchronize());
            }
            printf("\nVA %d A:%s \t B:%s Impl: %d \t SSD: %d \t CL: %d \t Cache: %llu \t Stride: %llu \t TotalTime %f ms\n", titr, a_file_bin.c_str(), b_file_bin.c_str(), type, settings.n_ctrls, settings.pageSize,settings.maxPageCacheSize, settings.stride, milliseconds); 
            fflush(stdout);
        }

        if(mem!=BAFS_DIRECT){
           free(a_h);
           free(b_h);
         }

        if((type == BASELINE_PC) || (type == OPTIMIZED_PC)) {
            //TODO: Fix this
            delete h_pc;
            delete h_Arange;
            delete h_Brange;
            delete h_Aarray;
            delete h_Barray;
        }

        cuda_err_chk(cudaFree(sum_d));
        if(mem!=BAFS_DIRECT){
            if(mem==UVM_DIRECT){
              a_d = a_d-2; 
              b_d = b_d-2;
            }
            cuda_err_chk(cudaFree(a_d));
            cuda_err_chk(cudaFree(b_d));
        }
            
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
             delete ctrls[i];

    }
    catch (const error& e){
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    return 0;
}
