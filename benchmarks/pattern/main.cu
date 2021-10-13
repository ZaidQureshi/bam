#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;


#define ARRAYTYPE uint64_t

#define Align(size,alignment) (size+alignment-1) & ~(alignment-1)

const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};

typedef enum {
    SEQUENTIAL        =0,
    RANDOM            =1, 
    GRID_STREAMING    =2,
    BLOCK_STREAMING   =3,
    WARP_RANDOM       =4, 
    BLOCK_RANDOM      =5,
    SEQUENTIAL_PC     =6,
    RANDOM_PC         =7, 
    GRID_STREAMING_PC =8,
    BLOCK_STREAMING_PC=9,
    WARP_RANDOM_PC    =10, 
    BLOCK_RANDOM_PC   =11
} impl_type;


//TODO: Static partition case to add. 
typedef enum {
    GPUMEM = 0,
    UVM_READONLY = 1,
    UVM_DIRECT = 2,
    // UVM_READONLY_NVLINK = 3,
    // UVM_DIRECT_NVLINK = 4,
    DRAGON_MAP = 5,
    BAFS_DIRECT = 6,
} mem_type;




//Honestly this is grid streaming and can be removed entirely
__global__
void sequential_access_kernel(ARRAYTYPE* dr, uint64_t num_elems, unsigned long long* output) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(uint64_t i=tid; i < num_elems; i+=blockDim.x*gridDim.x){
        output[0] += dr[i];
        __syncthreads(); 
    }
}


__global__
void random_access_kernel(ARRAYTYPE* dr, uint64_t num_elems, unsigned long long* output, uint64_t* assignment) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i=tid; i < num_elems; i+=blockDim.x*gridDim.x){
        output[0] += dr[assignment[i]];
    }
}

//Intial access pattern extracted from https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/unified-memory-oversubscription/uvm_oversubs.cu

// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_grid_streaming(data_type *ptr, const size_t size, unsigned long long* output)
{
    size_t n = size / sizeof(data_type);
    data_type accum = 0;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (1) {
      if ((tid - threadIdx.x) > n) {
        break;
      }
      if (tid < n)
        accum += ptr[tid];
      tid += (blockDim.x * gridDim.x);
      __syncthreads();
    }
    if (threadIdx.x == 0)
      output[0] = accum;
}


// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_block_streaming(data_type *ptr, const size_t size, unsigned long long* output)
{
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  size_t elements_per_block = ((n + (gridDim.x - 1)) / gridDim.x) + 1;
  size_t startIdx = elements_per_block * blockIdx.x;

  size_t rid = threadIdx.x + startIdx;
  while (1) {
    if ((rid - threadIdx.x - startIdx) > elements_per_block) {
      break;
    }

    if (rid < n) {
       accum += ptr[rid];
    }
    rid += blockDim.x;
    __syncthreads();
  }

  if (threadIdx.x == 0)
    output[0] = accum;
}
// each thread reads 4B streaming making a warp copy 128B of data. Each warp picks a random input. 
template<typename data_type>
__global__ void read_cta_random_warp_streaming(data_type *ptr, size_t feat_size, 
                                               const size_t size, size_t num_pages,
                                               size_t page_size, uint64_t* assignment, 
                                               unsigned long long* output)
{
  size_t n = size / feat_size; //num features
  int loop_count = n / (blockDim.x * gridDim.x); // loop across entire dataset. 

  size_t dtype_per_page = feat_size/sizeof(data_type); //num element in feature
//size_t lane0_idx_mod = dtype_per_page - warpSize;   // so that warp doesnt overshoot page boundary

  int lane_id = threadIdx.x & 31;
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  data_type accum = 0;

  //uint64_t nRandom = 0; 
  int cpyitr = feat_size/(warpSize*sizeof(data_type)); //asumes that the feat_size is multiple of warpsize 

  for (int i = 0; i < loop_count; i++) {
//    nRandom = assignment[idx];
    uint64_t pageframe_number = assignment[idx]; // each warp handles a feature // nRandom % num_pages;

    // warp lane 0 broadcast page number to all other warp lanes
    pageframe_number = __shfl_sync(0xffffffff, pageframe_number, 0);

    // coalesced 128 byte access within page - not aligned
    // maybe access two cache lines instead of one
    //uint64_t page_idx = nRandom % lane0_idx_mod;
    //page_idx = __shfl_sync(0xffffffff, page_idx, 0);
    //page_idx += lane_id;
    //uint64_t page_idx = lane_id;
    for(int j = 0; j< cpyitr; j++){
         accum += ptr[pageframe_number * dtype_per_page + (cpyitr*j) +lane_id];
    }
    idx += blockDim.x * gridDim.x;
  }

  if (threadIdx.x == 0)
    output[0] = accum;
}


// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_grid_streaming_pc(array_d_t<uint64_t> *ptr, size_t size, unsigned long long* output)
{
    size_t n = size / sizeof(data_type);
    data_type accum = 0;

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (1) {
      if ((tid - threadIdx.x) > n) {
        break;
      }
      if (tid < n)
        accum += ptr->seq_read(tid);
      tid += (blockDim.x * gridDim.x);
      __syncthreads();
    }
    if (threadIdx.x == 0)
      output[0] = accum;
}


// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_block_streaming_pc(array_d_t<uint64_t> *ptr, size_t size, unsigned long long* output)
{
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  size_t elements_per_block = ((n + (gridDim.x - 1)) / gridDim.x) + 1;
  size_t startIdx = elements_per_block * blockIdx.x;

  size_t rid = threadIdx.x + startIdx;
  while (1) {
    if ((rid - threadIdx.x - startIdx) > elements_per_block) {
      break;
    }

    if (rid < n) {
       accum += ptr->seq_read(rid);
    }
    rid += blockDim.x;
    __syncthreads();
  }

  if (threadIdx.x == 0)
    output[0] = accum;
}
// each thread reads 4B streaming making a warp copy 128B of data. Each warp picks a random input. 
template<typename data_type>
__global__ void read_cta_random_warp_streaming_pc(array_d_t<uint64_t> *ptr, size_t feat_size, 
                                                  size_t size, size_t num_pages,
                                                  size_t page_size, uint64_t* assignment, 
                                                  unsigned long long* output)
{
  size_t n = size / feat_size; //num features
  int loop_count = n / (blockDim.x * gridDim.x); // loop across entire dataset. 

  size_t dtype_per_page = feat_size/sizeof(data_type); //num element in feature
//size_t lane0_idx_mod = dtype_per_page - warpSize;   // so that warp doesnt overshoot page boundary

  int lane_id = threadIdx.x & 31;
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  data_type accum = 0;

  //uint64_t nRandom = 0; 
  int cpyitr = feat_size/(warpSize*sizeof(data_type)); //asumes that the feat_size is multiple of warpsize 

  for (int i = 0; i < loop_count; i++) {
//    nRandom = assignment[idx];
    uint64_t pageframe_number = assignment[idx]; // each warp handles a feature // nRandom % num_pages;

    // warp lane 0 broadcast page number to all other warp lanes
    pageframe_number = __shfl_sync(0xffffffff, pageframe_number, 0);

    // coalesced 128 byte access within page - not aligned
    // maybe access two cache lines instead of one
    //uint64_t page_idx = nRandom % lane0_idx_mod;
    //page_idx = __shfl_sync(0xffffffff, page_idx, 0);
    //page_idx += lane_id;
    //uint64_t page_idx = lane_id;
    for(int j = 0; j< cpyitr; j++){
         uint64_t idxtmp = pageframe_number * dtype_per_page + (cpyitr*j) +lane_id;
         accum += ptr->seq_read(idxtmp);
    }
    idx += blockDim.x * gridDim.x;
  }

  if (threadIdx.x == 0)
    output[0] = accum;
}




__global__
void sequential_access_kernel_pc(array_d_t<uint64_t>* dr, uint64_t num_elems, unsigned long long* output) {    
    // uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < n_reqs) {
    //     for (size_t i = 0; i < reqs_per_thread; i++)
    //         output += (*dr)[(tid)];
    // }
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(uint64_t i=tid; i < num_elems; i+=blockDim.x*gridDim.x){
            output[0] += dr->seq_read(tid);
    }
}



__global__
void random_access_kernel_pc(array_d_t<uint64_t>* dr, uint64_t num_elems, unsigned long long* output, uint64_t* assignment) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i=tid; i < num_elems; i+=blockDim.x*gridDim.x){
            output[0] += dr->seq_read(assignment[i]);
    }
}

int main(int argc, char** argv) {

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

    try {
        impl_type type; 
        mem_type mem; 

        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << "GPUID: " << st << std::endl;
        
        type = (impl_type) settings.type; 
        mem  = (mem_type) settings.memalloc; 

        cudaDeviceProp prop; 
        cuda_err_chk(cudaGetDeviceProperties(&prop, settings.cudaDevice)); 

        std::vector<Controller*> ctrls(settings.n_ctrls);
        if(mem==BAFS_DIRECT){
            for (size_t i = 0 ; i < settings.n_ctrls; i++)
                ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
        }

        ARRAYTYPE *d_array_ptr; //*h_array_ptr
        size_t array_size; 
        size_t tensor_size; 
        if((type==WARP_RANDOM) || (type==WARP_RANDOM_PC) ){
            tensor_size = settings.tensor_size;
        }else {
            tensor_size =1;
        }
        
        uint64_t n_elems = settings.numElems;
        array_size = n_elems*tensor_size*sizeof(ARRAYTYPE);
        std::cout << "Tensorsize: " << tensor_size << std::endl; 
        array_size = Align(array_size, settings.pageSize);
        std::cout << "Arraysize: " << array_size/(1024ULL * 1024ULL * 1024ULL) << " GBytes" << std::endl; 
        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor /b_size;
        uint64_t n_threads = b_size * g_size;

     
        //uint64_t n_pages = total_cache_size/page_size;
        //if (n_pages < n_threads) {
        //    std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
        //    exit(1);
        //}


        uint64_t* h_rand_assignment;
        uint64_t* d_rand_assignment;
        if ((type==RANDOM) || (type==RANDOM_PC)) {
            h_rand_assignment = (uint64_t*) malloc(n_elems*sizeof(uint64_t));
            for (size_t i = 0; i< n_elems; i++)
                h_rand_assignment[i] = rand() % (n_elems);
            
            cuda_err_chk(cudaMalloc(&d_rand_assignment, n_elems*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_rand_assignment, h_rand_assignment,  n_elems*sizeof(uint64_t), cudaMemcpyHostToDevice));
            std::cout << "Random assignment complete"<< std::endl; 
        }
        if ((type==WARP_RANDOM) || (type==WARP_RANDOM_PC)) {
            uint64_t n_keys = ceil((float)n_elems/tensor_size);
            h_rand_assignment = (uint64_t*) malloc(n_keys*sizeof(uint64_t));
            for (size_t i = 0; i< n_keys; i++)
                h_rand_assignment[i] = rand() % (n_keys);
            
            cuda_err_chk(cudaMalloc(&d_rand_assignment, n_keys*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_rand_assignment, h_rand_assignment,  n_keys*sizeof(uint64_t), cudaMemcpyHostToDevice));
            std::cout << "Random key assignment complete"<< std::endl; 
        }
        

        

        void* map_in; 
        int fd_in; 
        struct stat sb_in; 
        
        if(mem!=BAFS_DIRECT){
            //TODO: Generate array data for representation per se. We can load a binary file so that it is consistent with all others. 
            // we need sort of oversubscription ratio to cache size ratio variation to show the effect of perf on bandwidth with different access pattern. 
            // loading a file at arbitary size need to be checked. if feasible then create a very large file, mmap it and memcpy
            //move to settings.h file
            const char* input_f;
            
            if(settings.input == nullptr){
                fprintf(stderr, "Input file required\n");
                return 1;
            }
            else {
                input_f = settings.input; 
                printf("File is : %s\n",input_f);
            }

            if((fd_in = open(input_f, O_RDONLY)) == -1){
                fprintf(stderr, "Input file cannot be opened\n");
                return 1;
            }
            
            fstat(fd_in, &sb_in);
            
            map_in = mmap(NULL, sb_in.st_size, PROT_READ, MAP_SHARED, fd_in, 0);
            
            if((map_in == (void*)-1)){
                    fprintf(stderr,"Input file map failed %d\n",map_in);
                    return 1;
            }
        }


        switch(mem){
            case GPUMEM: {
                         //TODO: Fill this array with some initial data values.
                         cuda_err_chk(cudaMalloc((void**)&d_array_ptr, array_size));
                         cuda_err_chk(cudaMemcpy(d_array_ptr, map_in, array_size, cudaMemcpyHostToDevice)); //this is optional. done for correctness check across all combinations.
                         break;
                         }
            case UVM_READONLY: {
                         cuda_err_chk(cudaMallocManaged((void**)&d_array_ptr, array_size));
                         cuda_err_chk(cudaMemcpy(d_array_ptr, map_in, array_size, cudaMemcpyHostToDevice)); //this is optional. done for correctness check across all combinations. 
                         cuda_err_chk(cudaMemAdvise(d_array_ptr, array_size, cudaMemAdviseSetReadMostly, settings.cudaDevice));
                         break;
                         }
            case UVM_DIRECT: {
                         cuda_err_chk(cudaMallocManaged((void**)&d_array_ptr, array_size));
                         cuda_err_chk(cudaMemAdvise(d_array_ptr, array_size, cudaMemAdviseSetAccessedBy, settings.cudaDevice));
                        //  cuda_err_chk(cudaMemcpy(d_array_ptr, map_in, array_size, cudaMemcpyHostToDevice)); //this is optional. done for correctness check across all combinations. 
                         break;
                         }
            case BAFS_DIRECT: {
                         break;
                         }
            default:
                        fprintf(stderr, "Invalid mem picked\n"); 
                        exit(1);
                        break;
        }        
        
        if(mem!=BAFS_DIRECT){
            if(munmap(map_in, sb_in.st_size) == -1) 
                fprintf(stderr,"munmap error input file\n");
            close(fd_in);
        }

        unsigned long long* d_output;
        cuda_err_chk(cudaMalloc((void**)&d_output, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_output, 0, sizeof(unsigned long long)));
        
        page_cache_t* h_pc; 
        range_t<uint64_t>* h_range; 
        std::vector<range_t<uint64_t>*> vec_range(1);
        array_t<uint64_t>* h_array; 
        uint64_t pc_page_size = settings.pageSize;
        uint64_t file_n_pages = ceil(((float)array_size)/pc_page_size);
        uint64_t pc_pages = ceil((float)settings.maxPageCacheSize/pc_page_size);
        uint64_t total_cache_size = (pc_page_size * pc_pages); 


        if(mem==BAFS_DIRECT){
                h_pc = new page_cache_t(pc_page_size, pc_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
                h_range = new range_t<uint64_t>((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)file_n_pages, (uint64_t)0, (uint64_t)pc_page_size, h_pc, settings.cudaDevice);
                vec_range[0] = h_range; 
                h_array = new array_t<uint64_t>(n_elems, 0, vec_range, settings.cudaDevice);
                std::cout << "finished creating cache of size: "<< total_cache_size/(1024ULL * 1024ULL * 1024ULL) << " GB"<<std::endl;
        }
        // page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        // range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;
        fflush(stdout);


        
        //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        

        printf("Launching kernels: %d blocks and %d threads\n", g_size, b_size); 
        fflush(stdout);
        
        Event before;
        
        switch(type){
            case SEQUENTIAL: { 
                        sequential_access_kernel<<<g_size, b_size>>>(d_array_ptr, n_elems, d_output);
                        break;
                        }
            case RANDOM:{
                        random_access_kernel<<<g_size, b_size>>>(d_array_ptr, n_elems, d_output, d_rand_assignment);
                        break;
                        }
            case GRID_STREAMING:{
                        read_grid_streaming<ARRAYTYPE><<<g_size, b_size>>>(d_array_ptr, array_size, d_output); 
                        break;
                        }
            case BLOCK_STREAMING:{
                        read_block_streaming<ARRAYTYPE><<<g_size, b_size>>>(d_array_ptr, array_size, d_output); 
                        break;
                        }
            case WARP_RANDOM:{
                        read_cta_random_warp_streaming<ARRAYTYPE><<<g_size, b_size>>>(d_array_ptr, settings.tensor_size, array_size, file_n_pages, pc_page_size, d_rand_assignment, d_output);
                        break;
                        }
            case BLOCK_RANDOM:{
                        printf("FATAL: NOT YET IMPLEMENTED\n");
                        exit(1);
                        break;
                        }
            case SEQUENTIAL_PC: { 
                        sequential_access_kernel_pc<<<g_size, b_size>>>(h_array->d_array_ptr, n_elems, d_output);
                        break;
                        }
            case RANDOM_PC:{
                        random_access_kernel_pc<<<g_size, b_size>>>(h_array->d_array_ptr, n_elems, d_output, d_rand_assignment);
                        break;
                        }
            case GRID_STREAMING_PC:{
                        read_grid_streaming_pc<ARRAYTYPE><<<g_size, b_size>>>(h_array->d_array_ptr, array_size, d_output);
                        break;
                        }
            case BLOCK_STREAMING_PC:{
                        read_block_streaming_pc<ARRAYTYPE><<<g_size, b_size>>>(h_array->d_array_ptr, array_size, d_output);
                        break;
                        }
            case WARP_RANDOM_PC:{
                        read_cta_random_warp_streaming_pc<ARRAYTYPE><<<g_size, b_size>>>(h_array->d_array_ptr, settings.tensor_size, array_size, file_n_pages, pc_page_size, d_rand_assignment, d_output);
                        break;
                        }
            case BLOCK_RANDOM_PC:{
                        printf("FATAL: NOT YET IMPLEMENTED\n");
                        exit(1);
                        break;
                        }
            default:
                        fprintf(stderr, "Invalid type\n"); 
                        exit(1);
                        break;
        }

        Event after;
        cuda_err_chk(cudaDeviceSynchronize());
        std::cout << "Kernel execution complete\n" ; 

        double elapsed = after - before;

        uint64_t ios = n_elems; 
        uint64_t data = array_size;
        
        if(tensor_size>1){
            ios =  ceil((float)n_elems/tensor_size);
            data = ios*tensor_size;
        }
            

        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        if(type==BAFS_DIRECT)
            h_array->print_reset_stats();
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Read Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Read Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        //std::ofstream ofile("../data", std::ios::binary | std::ios::trunc);
        //ofile.write((char*)ret_array, data);
        //ofile.close();
        if(mem==BAFS_DIRECT){
            for (size_t i = 0 ; i < settings.n_ctrls; i++)
                delete ctrls[i];
        }        //hexdump(ret_array, n_pages*page_size);

        if(mem!=BAFS_DIRECT){
            cuda_err_chk(cudaFree(d_array_ptr)); 
        }
        if((mem==RANDOM) || (mem==RANDOM_PC) || (mem==WARP_RANDOM) || (mem==WARP_RANDOM_PC))
            cuda_err_chk(cudaFree(d_rand_assignment));
        //std::cout << "END\n";

        //std::cout << RAND_MAX << std::endl;

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
