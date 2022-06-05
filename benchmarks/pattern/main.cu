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

#include "zip.h"


#define UINT64MAX 0xFFFFFFFFFFFFFFFF
#define Align(size,alignment) (size+alignment-1) & ~(alignment-1)

using error = std::runtime_error;
using std::string;
//const char* const ctrls_paths[] = {"/dev/libnvmpro0", "/dev/libnvmpro1", "/dev/libnvmpro2", "/dev/libnvmpro3", "/dev/libnvmpro4", "/dev/libnvmpro5", "/dev/libnvmpro6", "/dev/libnvmpro7"};
//const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};

#define WARP_SHIFT 5
#define WARPSIZE 32

#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)

#define BLOCK_NUM 1024ULL

#define MAXWARP 64

typedef uint64_t EdgeT;

typedef enum {
    SEQUENTIAL         = 0,
    SEQUENTIAL_PC      = 1,
    SEQUENTIAL_WARP    = 2,
    SEQUENTIAL_WARP_PC = 3,
    RANDOM             = 4,
    RANDOM_PC          = 5,
    RANDOM_WARP        = 6,
    RANDOM_WARP_PC     = 7,
    STRIDE             = 8, 
    STRIDE_PC          = 9, 
    STRIDE_WARP        = 10, 
    STRIDE_WARP_PC     = 11, 
    POWERLAW_WARP      = 13, 
    POWERLAW_WARP_PC   = 15, 
} impl_type;

typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    BAFS_DIRECT= 6,
} mem_type;


template<typename T>
__global__ __launch_bounds__(64,32)
void kernel_sequential(T *input, uint64_t num_elems, unsigned long long int* output){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

    uint64_t val=0; 
    for(uint64_t i=tid; i < num_elems; i+= blockDim.x*gridDim.x){
        val += input[i];
    }

    if(threadIdx.x ==0)
        //atomicAdd(&(output[0]), val);
        output[0] = val;
}

template<typename T>
__global__ //__launch_bounds__(64,32)
void kernel_sequential_pc(array_d_t<T>* dr, T *input, uint64_t num_elems, unsigned long long int* output){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
	bam_ptr<T> ptr(dr);

    uint64_t val=0; 
    for(uint64_t i=tid; i < num_elems; i+= blockDim.x*gridDim.x){
        val += ptr[i];
        //atomicAdd(&(output[0]), ptr[i]);
    }

    if(threadIdx.x ==0)
        //atomicAdd(&(output[0]), val);
        output[0] = val;
}


template<typename T>
__global__ __launch_bounds__(64,32)
void kernel_random(T *input, uint64_t* assignment, uint64_t num_elems, unsigned long long int* output){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; 

    uint64_t val=0; 
    for(uint64_t i=tid; i < num_elems; i+= blockDim.x*gridDim.x){
        if(i < num_elems)
            val += input[assignment[i]];
    }

    if(threadIdx.x ==0)
        output[0] = val;
}

template<typename T>
__global__ //__launch_bounds__(64,32)
void kernel_random_pc(array_d_t<T>* dr, T *input, uint64_t* assignment, uint64_t num_elems, unsigned long long int* output){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x; 
	bam_ptr<T> ptr(dr);
    //uint64_t stride = 8192; 
    //uint64_t nep = (num_elems+stride)/stride; 
    //uint64_t ntid = (tid/nep) + ((tid % nep)* stride);
    uint64_t ntid = tid;  
    uint64_t val=0; 
    for(uint64_t i=ntid; i < num_elems; i+= blockDim.x*gridDim.x){
        if(i < num_elems)
            val += ptr[assignment[i]];
    }

    if(threadIdx.x ==0)
        output[0] = val;
}



template<typename T>
__global__ __launch_bounds__(64,32)
//launch 2*threads as blocksize- more of scan pattern. 
void kernel_stride(T *input, uint64_t num_elems, unsigned long long int* output){

    uint64_t stride = blockDim.x; 
    uint64_t tid = 2*blockIdx.x * blockDim.x + threadIdx.x; 


    uint64_t val=0; 
    if(tid <num_elems)
        val = input[tid];

    if((tid+stride)<num_elems)
        val += input[tid+stride];

    __syncthreads(); 

    if(threadIdx.x ==0)
        output[0] = val;
}


template<typename T>
__global__ __launch_bounds__(64,32)
void kernel_sequential_warp(T *input, uint64_t n_elems,  uint64_t n_pages_per_warp, unsigned long long* sum,  uint64_t n_warps, size_t page_size) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T v = 0;
    uint64_t idx=0; 

    if (warp_id < n_warps) {
        size_t start_page = n_pages_per_warp * warp_id;;
        //if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            //printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARPSIZE) {
                    //printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                    idx = start_idx + j; 
                    if(idx < n_elems)
                        v += input[idx];
            }

        }
          sum[0] = v;
        //atomicAdd(&sum[0], v);
    }

}

template<typename T>
__global__ //__launch_bounds__(64,32)
void kernel_sequential_warp_pc(array_d_t<T>* dr, T *input, uint64_t n_elems, uint64_t n_pages_per_warp, unsigned long long* sum,  uint64_t n_warps, size_t page_size, uint64_t stride) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T v = 0;
    uint64_t idx =0;
    uint64_t nep = (n_warps+stride-1)/stride; 
    uint64_t warp_id = (old_warp_id/nep) + ((old_warp_id % nep)* stride);

    if (warp_id < n_warps) {
		bam_ptr<T> ptr(dr);
        size_t start_page = n_pages_per_warp * warp_id;;
        //	if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            //	    printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += WARPSIZE) {
                    //printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                    idx = start_idx + j; 
                    if(idx < n_elems)
                        v += ptr[idx];
                        //v = ptr[idx];
                        //atomicAdd(&sum[0], v);
            }
        }
        sum[0] = v;
    }
}

template<typename T>
__global__ __launch_bounds__(64,32)
void kernel_random_warp(T *input,uint64_t n_elems, uint64_t n_pages_per_warp, unsigned long long* sum,  uint64_t* assignment, uint64_t n_warps, size_t page_size, uint64_t stride) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    
    uint64_t nep = (n_warps+stride-1)/stride; 
    uint64_t warp_id = (old_warp_id/nep) + ((old_warp_id % nep)* stride);

    T v = 0;
    uint64_t idx=0; 
    if (warp_id < n_warps) {
        size_t start_page = assignment[warp_id];
        //	if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            //	    printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += 32) {
            //		printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                    idx = start_idx + j; 
                    if(idx < n_elems)
                        v += input[idx];
            }

        }
        *sum = v;
    }

}


template<typename T>
__global__ //__launch_bounds__(64,32)
void kernel_random_warp_pc(array_d_t<T>* dr, T *input, uint64_t n_elems, uint64_t n_pages_per_warp, unsigned long long* sum,  uint64_t* assignment, uint64_t n_warps, size_t page_size, uint64_t stride) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    
    uint64_t nep = (n_warps+stride-1)/stride; 
    uint64_t warp_id = (old_warp_id/nep) + ((old_warp_id % nep)* stride);
    //uint64_t warp_id = old_warp_id; 
    T v = 0;
    if (warp_id < n_warps) {
		bam_ptr<T> ptr(dr);
        size_t start_page = assignment[warp_id];
        //	if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
            //	    printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += 32) {
            //		printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                        v += ptr[start_idx+j];
            }
        }
        *sum = v;
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

    std::ifstream filea;
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
                cuda_err_chk(cudaMemcpy(a_d, a_h, n_elems_size, cudaMemcpyHostToDevice));
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

        
        uint64_t n_pc_pages = ceil(((float)n_elems_size)/pc_page_size); 
        uint64_t blocksize = 64; 
        uint64_t n_warps =0;
        
    

        switch (type) {
            case SEQUENTIAL:
            case RANDOM:
            case SEQUENTIAL_PC:
            case RANDOM_PC:
            //case POWERLAW:
            {    
				numblocks = ((numthreads+blocksize-1)/blocksize);
                printf("numblocks:%llu \n", numblocks);
                break;
            }
			case SEQUENTIAL_WARP:
            case RANDOM_WARP:
            case POWERLAW_WARP:
            case SEQUENTIAL_WARP_PC:
            case RANDOM_WARP_PC:
            case POWERLAW_WARP_PC:
			{
                 numblocks = (numthreads + blocksize - 1)/blocksize;//80*16;
                 n_warps = blocksize * numblocks/ WARPSIZE; 
                 if(n_warps > n_pc_pages){
                     printf("Error: Cannot have n_warps greater than n_elems.\n");
                     printf("n_warps: %llu \t n_pc_pages:%llu \n", n_warps, n_pc_pages);
                 }
                 printf("n_warps: %llu \t numblocks:%llu \n", n_warps, numblocks);
                break;
			}
            default:
                fprintf(stderr, "Invalid type\n");
                exit(1);
                break;
        }
        
        //dim3 blockDim(BLOCK_NUM, (numblocks+BLOCK_NUM)/BLOCK_NUM);
        dim3 blockDim((numblocks));
        if((type == SEQUENTIAL_PC) || (type == SEQUENTIAL_WARP_PC) || (type == RANDOM_PC) || (type == RANDOM_WARP_PC) || (type == STRIDE_PC) || (type == STRIDE_WARP_PC) || (type == POWERLAW_WARP_PC)) {
                printf("page size: %d, pc_entries: %llu\n", pc_page_size, pc_pages);
        }
        
        // Allocate memory for GPU
        unsigned long long int *output_h;
        output_h = (unsigned long long int*) malloc(sizeof(unsigned long long int)); 
		output_h[0] = 0;
        uint64_t* assignment_h; 
        uint64_t* assignment_d; 

        uint64_t n_data_pages = (uint64_t) n_elems_size/pc_page_size;  
        
        if((type == RANDOM) || (type == RANDOM_PC)){
            printf("I am called %llu\n", n_data_pages);
            assignment_h = (uint64_t*) malloc (numthreads*sizeof(uint64_t));
            for(uint64_t i=0; i< numthreads; i++){
                uint64_t page = rand() % n_data_pages; 
                assignment_h[i] = page; 
            }
            cuda_err_chk(cudaMalloc(&assignment_d, numthreads*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(assignment_d, assignment_h, numthreads*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        if((type == RANDOM_WARP) || (type == RANDOM_WARP_PC)){
            assignment_h = (uint64_t*) malloc (n_warps*sizeof(uint64_t));
            //std::random_device rd;  //Will be used to obtain a seed for the random number engine
            //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            //std::uniform_int_distribution<uint64_t> distrib(0, n_data_pages);

            for(uint64_t i=0; i< n_warps; i++){
                //uint64_t page = distrib(gen); 
                uint64_t page = rand() % n_data_pages; 
                assignment_h[i] = page; 
                //printf("%llu \t", page);
            }
            cuda_err_chk(cudaMalloc(&assignment_d, n_warps*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(assignment_d, assignment_h, n_warps*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }


        if((type == POWERLAW_WARP) || (type == POWERLAW_WARP_PC)){
            std::random_device rd;
            unsigned seed; 
            if(settings.seed >0)
               seed = settings.seed; 
            else 
                seed = rd(); 
            std::mt19937 gen(seed);
            uint64_t unique_keys = n_data_pages;//pc_pages*8; // 8 times is picked for making sure all keys do not fall within the cache and hence there are misses. 
			//if(n_warps < unique_keys)
			//	printf("WARNING: powerlaw pattern requires unique keys (%llu) to be smaller than the n_warps (%llu). Either reduce the cache size or increase n_warps.\n", unique_keys, n_warps);
			//TODO: Control alpha or zipf coefficient. 
			zipf_distribution<uint64_t> zipf(unique_keys, 1.8);
            assignment_h = (uint64_t*) malloc (n_warps*sizeof(uint64_t));
            for(uint64_t i=0; i< n_warps; i++){
                assignment_h[i] = zipf(gen); 
            }
            cuda_err_chk(cudaMalloc(&assignment_d, n_warps*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(assignment_d, assignment_h, n_warps*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

		printf("Allocation finished\n");
        fflush(stdout);

        std::vector<Controller*> ctrls(settings.n_ctrls);
        if(mem == BAFS_DIRECT){
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


        if((type == SEQUENTIAL_PC) || (type == SEQUENTIAL_WARP_PC) || (type == RANDOM_PC) || (type == RANDOM_WARP_PC) || (type == STRIDE_PC) || (type == STRIDE_WARP_PC) || (type == POWERLAW_WARP_PC)) {
            h_pc =new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
            h_Arange = new range_t<uint64_t>((uint64_t)0 ,(uint64_t)n_elems, (uint64_t) (ceil(settings.afileoffset*1.0/pc_page_size)),(uint64_t)n_pc_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice); 
            vec_Arange[0] = h_Arange; 
            h_Aarray = new array_t<uint64_t>(n_elems, settings.afileoffset, vec_Arange, settings.cudaDevice);

            printf("Page cache initialized\n");
            fflush(stdout);
        }

		uint64_t n_pages_per_warp = settings.coarse;
		uint64_t n_elems_per_page = pc_page_size/sizeof(uint64_t); 
	
		unsigned long long int* output_d;
        cuda_err_chk(cudaMalloc(&output_d, sizeof(unsigned long long)));
	    	

        float totaltime = 0; 
        float avgtime = 0; 

        for(int titr=0; titr<11; titr+=1){
            cuda_err_chk(cudaEventRecord(start, 0));
        	cuda_err_chk(cudaMemset(output_d, 0, sizeof(unsigned long long)));
                
            auto itrstart = std::chrono::system_clock::now();

            switch (type) {
                case SEQUENTIAL:{
					kernel_sequential<uint64_t><<<blockDim, blocksize>>>(a_d, numthreads, output_d);
                    break;
                }
                case SEQUENTIAL_PC:{
                    //printf("blockDim.x is %llu \t blocksize: %llu\n", blockDim.x, blocksize );
                    kernel_sequential_pc<uint64_t><<<blockDim, blocksize>>>(h_Aarray->d_array_ptr,a_d, numthreads, output_d);
                    break;
                }
                case SEQUENTIAL_WARP:{
                    kernel_sequential_warp<uint64_t><<<blockDim, blocksize>>>(a_d,n_elems, n_pages_per_warp, output_d, n_warps, pc_page_size);
                    break;
                }
                case SEQUENTIAL_WARP_PC:{
                    //printf("blockDim.x is %llu \t blocksize: %llu\n", blockDim.x, blocksize );
                    kernel_sequential_warp_pc<uint64_t><<<blockDim, blocksize>>>(h_Aarray->d_array_ptr, a_d,n_elems, n_pages_per_warp, output_d, n_warps, pc_page_size, settings.stride);
                    break;
                }

                case RANDOM:{
                    //printf("blockDim.x is %llu \t blocksize: %llu\n", blockDim.x, blocksize );
                    kernel_random<uint64_t><<<blockDim, blocksize>>>(a_d, assignment_d, numthreads, output_d);
                    break;
                }
                case RANDOM_PC:{
                    //printf("blockDim.x is %llu \t blocksize: %llu\n", blockDim.x, blocksize );
                    kernel_random_pc<uint64_t><<<blockDim, blocksize>>>(h_Aarray->d_array_ptr, a_d, assignment_d, numthreads, output_d);
                    break;
                }
                case RANDOM_WARP:{
                    kernel_random_warp<uint64_t><<<blockDim, blocksize>>>(a_d,n_elems, n_pages_per_warp, output_d, assignment_d, n_warps, pc_page_size, settings.stride);
                    break;
                }
                case RANDOM_WARP_PC:{
                    //printf("blockDim.x is %llu \t blocksize: %llu\n", blockDim.x, blocksize );
                    kernel_random_warp_pc<uint64_t><<<blockDim, blocksize>>>(h_Aarray->d_array_ptr, a_d,n_elems,  n_pages_per_warp, output_d, assignment_d, n_warps, pc_page_size, settings.stride);
                    break;
                }
                case POWERLAW_WARP:{
                    kernel_random_warp<uint64_t><<<blockDim, blocksize>>>(a_d, n_elems, n_pages_per_warp, output_d, assignment_d, n_warps, pc_page_size, settings.stride);
                    break;
                }
                case POWERLAW_WARP_PC:{
                    kernel_random_warp_pc<uint64_t><<<blockDim, blocksize>>>(h_Aarray->d_array_ptr, a_d, n_elems, n_pages_per_warp, output_d, assignment_d, n_warps, pc_page_size, settings.stride);
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
            
            if(titr>0){
                totaltime +=milliseconds; 
                avgtime = totaltime/(titr); 
            }

            cuda_err_chk(cudaMemcpy(output_h, (output_d), (1)*sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            //printf("\n******\n");
            //fflush(stdout);
            //if(mem != BAFS_DIRECT){
            //   printf("Input list::");
            //   for (uint64_t i=n_elems-100; i< n_elems; i++)
            //       printf("%llu\t", a_h[i]);
            //}
           
            //if(mem != BAFS_DIRECT){
            //    uint64_t total = 0;
            //    for(uint64_t count=0; count<n_elems; count++)
            //        total+=a_h[count];
            //    printf("total in cpu: %llu \n", total);
            //}
            printf("val in gpu: %llu \n", output_h[0]);
            auto itrend = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(itrend - itrstart);

            uint64_t ios =numthreads; 
			uint64_t data = ios*sizeof(uint64_t);
			double iops = ((double) ios*1000/ (milliseconds)); 
			double bandwidth = (((double) data*1000/(milliseconds))/(1024ULL*1024ULL*1024ULL));
			
			double avgiops = ((double) ios*1000/ (avgtime)); 
			double avgbandwidth = (((double) data*1000/(avgtime))/(1024ULL*1024ULL*1024ULL));
            if((type == SEQUENTIAL_WARP) || (type == SEQUENTIAL_WARP_PC) || (type == RANDOM_WARP) || (type == RANDOM_WARP_PC) || (type == POWERLAW_WARP) || (type == POWERLAW_WARP_PC)){
				
				//ios = n_warps*n_pages_per_warp*n_elems_per_page; 
				ios = n_warps*n_pages_per_warp*n_elems_per_page; 
                iops = ((double) ios*1000/ (milliseconds)); 
				data = ios*sizeof(uint64_t); 
				bandwidth = (((double) data*1000/(milliseconds))/(1024ULL*1024ULL*1024ULL));
			    avgiops = ((double) ios*1000/ (avgtime)); 
			    avgbandwidth = (((double) data*1000/(avgtime))/(1024ULL*1024ULL*1024ULL));
            }

			if(mem == BAFS_DIRECT) {
                 h_Aarray->print_reset_stats();
                 cuda_err_chk(cudaDeviceSynchronize());
            }
			printf("P:%d Impl: %llu \t SSD: %llu \t n_warps:%llu \t n_pages_per_warp: %llu \t n_elems_per_page:%llu \t ios: %llu \t IOPs: %f \t data:%llu \t bandwidth: %f GBps \t avgiops: %f \t avgbandwidth: %f \n",titr, type, settings.n_ctrls, n_warps, n_pages_per_warp, n_elems_per_page, ios, iops, data, bandwidth, avgiops, avgbandwidth ); 
            //printf("\nVA %d A:%s Impl: %d \t SSD: %d \t CL: %d \t Cache: %llu \t TotalTime %f ms\n", titr, a_file_bin.c_str(), type, settings.n_ctrls, settings.pageSize,settings.maxPageCacheSize, milliseconds); 
            fflush(stdout);
        }

        if(mem!=BAFS_DIRECT){
           free(a_h);
         }
        free(output_h);

        if((type == SEQUENTIAL_PC) || (type == SEQUENTIAL_WARP_PC) || (type == RANDOM_PC) || (type == RANDOM_WARP_PC) || (type == STRIDE_PC) || (type == STRIDE_WARP_PC) || (type == POWERLAW_WARP_PC)) {
            delete h_pc;
            delete h_Arange;
            delete h_Aarray;
        }

        if(mem!=BAFS_DIRECT){
            if(mem==UVM_DIRECT){
              a_d = a_d-2; 
            }
            cuda_err_chk(cudaFree(a_d));
            cuda_err_chk(cudaFree(output_d));
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
