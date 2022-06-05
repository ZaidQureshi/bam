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
#include <numeric> 
#include <functional>
#include <cub/cub.cuh>


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

typedef uint64_t ElemT;

typedef enum {
    BASELINE = 0,
    OPTIMIZED=1,
    BASELINE_PC = 2,
    OPTIMIZED_PC=3,
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    BAFS_DIRECT= 6,
} mem_type;



__global__ 
void kernel_reduce_baseline(ElemT *input, ElemT *output, uint64_t len){

    extern __shared__ ElemT sharedMem[]; 

    uint64_t tid = threadIdx.x; 
    uint64_t start = 2*blockDim.x*blockIdx.x; 

    if((start+tid) < len)
        sharedMem[tid] = input[start+tid];
    else 
        sharedMem[tid] = 0; 

    if((blockDim.x+start+tid) < len)
        sharedMem[blockDim.x+tid] = input[blockDim.x+start+tid];
    else 
        sharedMem[blockDim.x+tid] = 0; 

    for(uint64_t stride = blockDim.x; stride>0; stride >>=1){
        __syncthreads();

        if( tid < stride)
            sharedMem[tid] += sharedMem[stride+tid];
    }

    if(threadIdx.x == 0)
        output[blockIdx.x] = sharedMem[0];

}

__global__ 
void kernel_reduce_optimized(ElemT *input, ElemT *output, uint64_t len, uint64_t page_size, uint64_t n_warps){
    extern __shared__ ElemT sharedMem[];

    uint64_t tid = blockDim.x*blockIdx.x+threadIdx.x; 
    uint64_t laneid = tid % WARP_SIZE;
    uint64_t warp_id = tid / WARP_SIZE;
    uint64_t num_elems_per_cl = page_size/sizeof(ElemT); 
    
    uint64_t start = 2*warp_id * num_elems_per_cl;  

    for(size_t i=laneid; i<num_elems_per_cl;i+=WARP_SIZE){

        if(((start+i)<len))
            sharedMem[i] = input[start+i];
        else 
            sharedMem[i] =0;

        if((num_elems_per_cl+start+i)<len)
            sharedMem[num_elems_per_cl+i] = input[num_elems_per_cl+start+i];
        else 
            sharedMem[i] =0;

        __syncwarp(); 
    }

    for(uint64_t stride = num_elems_per_cl; stride>0; stride>>=1){
        __syncwarp(); 

        for(size_t i =0; i<num_elems_per_cl; i+=WARP_SIZE){
            uint64_t idx = i*WARP_SIZE + laneid; 
        
            if(idx <stride)
                sharedMem[idx] += sharedMem[stride+idx]; 
        }
    }

    if(laneid == 0)
        output[warp_id] = sharedMem[0];
}



__global__ 
void kernel_reduce_baseline_ptr_pc(array_d_t<ElemT> *da, ElemT *output, uint64_t len){

    extern __shared__ ElemT sharedMem[]; 

    bam_ptr<ElemT> input(da);

    uint64_t tid = threadIdx.x; 
    uint64_t start = 2*blockDim.x*blockIdx.x; 

    if((start+tid) < len)
        sharedMem[tid] = input[start+tid];
    else 
        sharedMem[tid] = 0; 

    if((blockDim.x+start+tid) < len)
        sharedMem[blockDim.x+tid] = input[blockDim.x+start+tid];
    else 
        sharedMem[blockDim.x+tid] = 0; 

    for(uint64_t stride = blockDim.x; stride>0; stride >>=1){
        __syncthreads();

        if( tid < stride)
            sharedMem[tid] += sharedMem[stride+tid];
    }

    if(threadIdx.x == 0)
        output[blockIdx.x] = sharedMem[0];

}

__global__ 
void kernel_reduce_optimized_ptr_pc(array_d_t<ElemT> *da, ElemT *output, uint64_t len, uint64_t page_size, uint64_t n_warps){
    extern __shared__ ElemT sharedMem[];
    bam_ptr<ElemT> input(da);

    uint64_t tid = blockDim.x*blockIdx.x+threadIdx.x; 
    uint64_t laneid = tid % WARP_SIZE;
    uint64_t warp_id = tid / WARP_SIZE;
    uint64_t num_elems_per_cl = page_size/sizeof(ElemT); 
    
    uint64_t start = 2*warp_id * num_elems_per_cl;  

    for(size_t i=laneid; i<num_elems_per_cl;i+=WARP_SIZE){

        if(((start+i)<len))
            sharedMem[i] = input[start+i];
        else 
            sharedMem[i] =0;

        if((num_elems_per_cl+start+i)<len)
            sharedMem[num_elems_per_cl+i] = input[num_elems_per_cl+start+i];
        else 
            sharedMem[i] =0;

        __syncwarp(); 
    }

    for(uint64_t stride = num_elems_per_cl; stride>0; stride>>=1){
        __syncwarp(); 

        for(size_t i =0; i<num_elems_per_cl; i+=WARP_SIZE){
            uint64_t idx = i*WARP_SIZE + laneid; 
        
            if(idx <stride)
                sharedMem[idx] += sharedMem[stride+idx]; 
        }
    }

    if(laneid == 0)
        output[warp_id] = sharedMem[0];
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
    std::string a_file;
    std::string a_file_bin;
    std::string filename;

    impl_type type;
    mem_type mem;
    uint64_t *a_h, *a_d;
    uint64_t numblocks, numthreads;

    float milliseconds;

    uint64_t pc_page_size;
    uint64_t pc_pages; 

    try{

        a_file = std::string(settings.input_a); 
        
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

        std::cout << "A: " << a_file_bin  << std::endl;

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
            a_h = (uint64_t*)calloc(n_elems_size, sizeof(uint64_t));
        if((mem!=BAFS_DIRECT) &&  (mem != UVM_DIRECT)){
             filea.read((char*)a_h, n_elems_size);
             filea.close();
        }

        switch (mem) {
            case GPUMEM:
                {  
                cuda_err_chk(cudaMalloc((void**)&a_d, n_elems_size));
                high_resolution_clock::time_point mc1 = high_resolution_clock::now();
                cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
                high_resolution_clock::time_point mc2 = high_resolution_clock::now();
                duration<double> mc_time_span = duration_cast<duration<double>>(mc2 -mc1);
                std::cout<< "Memcpy time for loading the inputs: "<< mc_time_span.count() <<std::endl;
                break;
                }
            case UVM_READONLY:
                {
                cuda_err_chk(cudaMallocManaged((void**)&a_d, n_elems_size));
                cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
                cuda_err_chk(cudaMemAdvise(a_d, n_elems_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                break;
                }
            case UVM_DIRECT:
                {
                filea.close();
                fileb.close();
                int fda = open(a_file_bin.c_str(), O_RDONLY | O_DIRECT); 
                FILE *fa_tmp= fdopen(fda, "rb");
                if ((fa_tmp == NULL) || (fda == -1)) {
                    printf("A file fd open failed\n");
                    exit(1);
                }   
                
                uint64_t count_4k_aligned = ((n_elems + 2 + 4096 / sizeof(uint64_t)) / (4096 / sizeof(uint64_t))) * (4096 / sizeof(uint64_t));
                //uint64_t count_4k_aligned = n_elems; 
                uint64_t size_4k_aligned = count_4k_aligned * sizeof(uint64_t);

                cuda_err_chk(cudaMallocManaged((void**)&a_d, size_4k_aligned));
                cuda_err_chk(cudaMemAdvise(a_d, size_4k_aligned, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                high_resolution_clock::time_point ft1 = high_resolution_clock::now();
               
                if (fread(a_d, sizeof(uint64_t), count_4k_aligned, fa_tmp)) {
                    printf("A file fread failed: %llu \t %llu\n", count_4k_aligned, n_elems+2);
                    exit(1);
                }   
                fclose(fa_tmp);                                                                                                              
                close(fda);
                
                a_d = a_d + 2;

                high_resolution_clock::time_point ft2 = high_resolution_clock::now();
                duration<double> time_span = duration_cast<duration<double>>(ft2 -ft1);
                std::cout<< "file read time: "<< time_span.count() <<std::endl;
                
                break;
                }
            case BAFS_DIRECT: 
                {
                break;
                }
        }

        
        uint64_t n_pages = ceil(((float)n_elems_size)/pc_page_size); 

        uint64_t n_warps = 0; 

        switch (type) {
            case BASELINE:
            case BASELINE_PC:{
                numblocks = ((n_elems/(2*numthreads)) + 1);
                break;
            }
            case OPTIMIZED: 
            case OPTIMIZED_PC:{
                uint64_t num_elems_per_cl = pc_page_size/sizeof(ElemT); 
                n_warps = ceil(n_pages/2);
                numblocks = ceil(n_warps*WARP_SIZE/numthreads); 
                break;
            }

            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }
        
        //dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);
        dim3 blockDim((numblocks));
        if((type == BASELINE_PC)) {
                printf("page size: %d, pc_entries: %llu\n", pc_page_size, pc_pages);
        }
        
        // Allocate memory for GPU
        ElemT *result_h;
        ElemT *result_d;
        ElemT *dev2out_d; 
        ElemT *int_d; 

        cuda_err_chk(cudaMalloc((void**)&int_d, (numblocks)*sizeof(ElemT)));
        cuda_err_chk(cudaMalloc((void**)&dev2out_d, (numblocks)*sizeof(ElemT)));
        cuda_err_chk(cudaMalloc((void**)&result_d, (n_elems+1)*sizeof(ElemT)));
        result_h = (ElemT*) malloc(n_elems* sizeof(ElemT)); 
        cuda_err_chk(cudaMemset(result_d, 0, (n_elems+1)*sizeof(ElemT)));

		printf("Allocation finished\n");
        fflush(stdout);

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
        std::vector<range_t<uint64_t>*> vec_Arange(1);
        array_t<uint64_t>* h_Aarray;


        if((type == BASELINE_PC) || (type == OPTIMIZED_PC)) {
            //TODO: fix for 2 arrays
            h_pc =new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
            h_Arange = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)n_elems, (uint64_t) (ceil(settings.afileoffset*1.0/pc_page_size)),(uint64_t)n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); 
            vec_Arange[0] = h_Arange; 
            h_Aarray = new array_t<uint64_t>(n_elems, settings.afileoffset, vec_Arange, settings.cudaDevice);

            printf("Page cache initialized\n");
            fflush(stdout);
        }

        void *d_tmp = NULL;
        size_t tmp_size =0; 

        for(int titr=0; titr<1; titr+=1){
            cuda_err_chk(cudaEventRecord(start, 0));
                
            auto itrstart = std::chrono::system_clock::now();

            switch (type) {
                case BASELINE:{
                    printf("launching PC: blockDim.x :%llu blockDim.y :%llu numthreads:%llu sharedMemSize: %llu\n", blockDim.x, blockDim.y, numthreads, 2*numblocks*sizeof(ElemT));
                    kernel_reduce_baseline<<<blockDim, numthreads, 2*numthreads*sizeof(ElemT)>>>(a_d, result_d, n_elems);
                    break;
                }
                case OPTIMIZED:{
                    printf("launching PC: blockDim.x :%llu blockDim.y :%llu numthreads:%llu sharedMemSize: %llu\n", blockDim.x, blockDim.y, numthreads, 2*pc_page_size);
                    kernel_reduce_optimized<<<blockDim, numthreads, 2*pc_page_size>>>(a_d, result_d, n_elems, settings.pageSize, n_warps);
                    break;

                    }
                case BASELINE_PC:{

                    printf("launching PC: blockDim.x :%llu blockDim.y :%llu numthreads:%llu\n", blockDim.x, blockDim.y, numthreads);
                    kernel_reduce_baseline_ptr_pc<<<blockDim, numthreads, 2*numthreads*sizeof(ElemT)>>>(h_Aarray->d_array_ptr, result_d, n_elems);
                    break;
                }
                case OPTIMIZED_PC:{
                    printf("launching PC: blockDim.x :%llu blockDim.y :%llu numthreads:%llu sharedMemSize: %llu\n", blockDim.x, blockDim.y, numthreads, 2*pc_page_size);
                    kernel_reduce_optimized_ptr_pc<<<blockDim, numthreads, 2*pc_page_size>>>(h_Aarray->d_array_ptr, result_d, n_elems, settings.pageSize, n_warps);
                    break;
                }
                default:
                    fprintf(stderr, "Invalid type\n");
                    exit(1);
                    break;
            }
            cuda_err_chk(cudaEventRecord(end, 0));
            cuda_err_chk(cudaEventSynchronize(end));
            cuda_err_chk(cudaEventElapsedTime(&milliseconds, start, end));
            
            cuda_err_chk(cudaMemcpy(result_h, (result_d), (n_elems+1)*sizeof(ElemT), cudaMemcpyDeviceToHost));
            //printf("\n******\n");
            //fflush(stdout);
            //if(mem != BAFS_DIRECT){
            //   printf("Input list::");
            //   for (uint64_t i=n_elems-100; i< n_elems; i++)
            //       printf("%llu\t", a_h[i]);
            //}
            //printf("\n\nScan result:");
            //for (uint64_t i=n_elems-100; i< n_elems; i++)
            //    printf("%llu\t", result_h[i]);
            //printf("\n******\n");
           
            //std::vector<uint64_t> a_h_vec (a_h, a_h+n_elems);
            //uint64_t total = std::accumulate(a_h_vec.begin(), a_h_vec.begin()+n_elems, 0, std::plus<uint64_t>());
            uint64_t total = 0;
            if(mem != BAFS_DIRECT){
                for(uint64_t count=0; count<n_elems; count++)
                    total+=a_h[count];
                printf("total in cpu: %llu \n", total);
            }

            total =0; 
            for(uint64_t count=0; count<numblocks ;count++){
                total += result_h[count];
            }
            printf("total in gpu: %llu \n ", total);
            auto itrend = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(itrend - itrstart);

            //if(mem == BAFS_DIRECT) {
            //         h_Aarray->print_reset_stats();
		    // printf("VA SSD: %d PageSize: %d itrTime: %f\n", settings.n_ctrls, settings.pageSize, (double)elapsed.count()); 
            //}



            if(mem == BAFS_DIRECT) {
                 h_Aarray->print_reset_stats();
                 cuda_err_chk(cudaDeviceSynchronize());
            }
            printf("\nVA %d A:%s Impl: %d \t SSD: %d \t CL: %d \t Cache: %llu \t TotalTime %f ms\n", titr, a_file_bin.c_str(), type, settings.n_ctrls, settings.pageSize,settings.maxPageCacheSize, milliseconds); 
            fflush(stdout);
        }

        if(mem!=BAFS_DIRECT){
           free(a_h);
         }
        free(result_h);

        if((type == BASELINE_PC) || (type == OPTIMIZED_PC)) {
            //TODO: Fix this
            delete h_pc;
            delete h_Arange;
            delete h_Aarray;
        }

        if(mem!=BAFS_DIRECT){
            if(mem==UVM_DIRECT){
              a_d = a_d-2; 
            }
            cuda_err_chk(cudaFree(a_d));
            cuda_err_chk(cudaFree(int_d));
            cuda_err_chk(cudaFree(dev2out_d));
            cuda_err_chk(cudaFree(result_d));
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
