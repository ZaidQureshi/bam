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
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
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

typedef  float TYPE;


//uint32_t n_ctrls = 1;
const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
const char* const intel_ctrls_paths[] = {"/dev/libinvm0", "/dev/libinvm1", "/dev/libinvm4", "/dev/libinvm9", "/dev/libinvm2", "/dev/libinvm3", "/dev/libinvm5", "/dev/libinvm6", "/dev/libinvm7", "/dev/libinvm8"};


template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    uint64_t num_idx, int cache_dim,
                                     uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr, uint32_t* wb_id_array, uint32_t q_depth) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    wb_bam_ptr<T> ptr(dr);
        ptr.set_wb(wb_queue_counter, wb_depth, queue_ptr, wb_id_array, q_depth);

       	  uint64_t row_index = index_ptr[idx_idx];
 
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
        T temp = ptr[(row_index) * cache_dim + tid];
        out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}

template <typename T = float>
__global__ void evict_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    uint64_t num_idx, int cache_dim,
                                     uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr, uint32_t* wb_id_array, uint32_t q_depth) {
    
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    wb_bam_ptr<T> ptr(dr);
        ptr.set_wb(wb_queue_counter, wb_depth, queue_ptr, wb_id_array, q_depth);

       	  uint64_t row_index = index_ptr[idx_idx];
 
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
	    T temp = ptr[(row_index) * cache_dim + tid];
	  out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;

    }
  }

}

template <typename T = float>
__global__ void print_array( void* ptr_t, uint32_t* id_array, uint32_t test_wb_depth, uint32_t q_depth){
    T* ptr = (T*) ptr_t;
    if(threadIdx.x == 0){
        for(int k = 0; k < test_wb_depth; k++){
            for(int j = 0; j < 2; j++){
                printf("cache line: %i\n",j);
                printf("cache id: %i\n",id_array[j+(q_depth*k)]);

                for(int i = 0; i < 1024; i++){
                    printf("%f ", ptr[j*1024+i+(1024*k*q_depth)]);
                }
                printf("\n");
            }
        }
    }
}




template <typename T = float>
__global__
void update_wb_counters(array_d_t<T> *dr,uint32_t** batch_arrays, uint32_t* batch_size_array, uint64_t wb_size){
    //x dim: each node in batch
    //y dim: depth of wb
    uint32_t cur_iter = threadIdx.y;

    uint32_t* cur_batch = batch_arrays[cur_iter];
    
    int32_t cur_batch_node = threadIdx.x + blockIdx.x * blockDim.x;
    
    uint32_t my_batch_len = batch_size_array[cur_iter];
    wb_bam_ptr<T> ptr(dr);

    for(uint32_t i = cur_batch_node; i < my_batch_len; i+=blockDim.x){
        uint32_t cur_node = cur_batch[i];
        printf("cur node: %u cur iter: %u\n", cur_node, cur_iter);
        ptr.update_wb(cur_node,  cur_iter);
    }
}

template <typename T = float>
__global__
void check_wb_counters(array_d_t<T> *dr){
    //x dim: each node in batch
    //y dim: depth of wb
    wb_bam_ptr<T> ptr(dr);
    if(threadIdx.x == 0){
    
        for(int i = 0; i < 10; i++){
            uint8_t reuse_val = ptr.check_reuse_val(i);
            printf("page: %i reuse_val: %u\n", i, reuse_val);
        }
    
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
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);


        uint64_t b_size = 32;
        uint64_t g_size = (settings.numReqs)/(b_size/32);//80*16;
        uint64_t n_threads = b_size * g_size;
        std::cout << "bsize: " << b_size << " g_size: " << g_size << std::endl;
        
        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);

        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        //QueuePair* d_qp;
        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        
        uint64_t n_elems = settings.numElems;
        uint64_t t_size = n_elems * sizeof(TYPE);

        range_t<TYPE> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<TYPE>* d_range = (range_t<TYPE>*) h_range.d_range_ptr;

        std::vector<range_t<TYPE>*> vr(1);
        vr[0] = & h_range;
        //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        array_t<TYPE> a(n_elems, 0, vr, settings.cudaDevice);

        std::cout << "finished creating range\n";

        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;
   
        uint64_t num_idx = (uint64_t) (settings.numReqs);

        TYPE* d_out;
        cuda_err_chk(cudaMalloc(&d_out, sizeof(float) * 1024 * num_idx));
        int64_t *index_ptr = (int64_t*) malloc(sizeof(int64_t) * num_idx); 
        for(auto i = 0; i < num_idx; i++){
               index_ptr[i] = i;
        }
        int64_t *d_index_ptr;
        cuda_err_chk(cudaMalloc(&d_index_ptr, sizeof(int64_t) * num_idx));
        cuda_err_chk(cudaMemcpy(d_index_ptr, index_ptr, sizeof(int64_t) * num_idx, cudaMemcpyHostToDevice));
                     
        
        int dim = 1024;
        int cache_dim = 1024;
        
        uint64_t  wb_depth = 128;
        uint64_t  queue_depth = 128 * 1024;
        uint32_t* wb_queue_counter;
        TYPE* queue_ptr;
        TYPE* h_queue_ptr;

        uint32_t* h_wb_id_array;
        uint32_t* wb_id_array;
        
        cuda_err_chk(cudaMalloc(&wb_queue_counter, sizeof(uint32_t) * wb_depth));
        cuda_err_chk(cudaMalloc(&wb_id_array, sizeof(uint32_t) * wb_depth * queue_depth));

        cuda_err_chk(cudaHostAlloc((TYPE **)&h_queue_ptr, page_size * wb_depth * queue_depth, cudaHostAllocMapped));
        cudaHostGetDevicePointer((TYPE **)&queue_ptr, (TYPE *)h_queue_ptr, 0);
        
        cuda_err_chk(cudaHostAlloc((uint32_t **)&h_wb_id_array, sizeof(uint32_t) * wb_depth * queue_depth, cudaHostAllocMapped));
        cudaHostGetDevicePointer((uint32_t **)&wb_id_array, (uint32_t *)h_wb_id_array, 0);

        std::cout << "num_reqs: " << num_idx << std::endl;
        
        cudaEvent_t before, after, wb_before, wb_after;
        cudaEventCreate(&before);
        cudaEventCreate(&after);
        cudaEventCreate(&wb_before);
        cudaEventCreate(&wb_after);
        
        cudaEventRecord(before);
        read_feature_kernel<TYPE><<<g_size, b_size>>>(a.d_array_ptr, d_out, d_index_ptr, dim, num_idx, cache_dim, 
                                                      wb_queue_counter,  wb_depth, queue_ptr, wb_id_array, queue_depth);
        cudaEventRecord(after);
        cudaEventSynchronize(after);
  
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, before, after);

        uint64_t ios = g_size * (b_size / 32);
        uint64_t data = ios*page_size;
        
        double bandwidth = (((double)data)/(elapsed/1000))/(1024ULL*1024ULL*1024ULL);
        a.print_reset_stats();
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Read Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

        std::cout << std::dec << "Window Buffer Update" << std::endl;
        
        uint32_t** batch_arrays;
        uint32_t test_wb_depth = 2;
        cuda_err_chk(cudaMalloc(&batch_arrays, sizeof(uint32_t*) * test_wb_depth));
        
        uint32_t* h_batch_arrays [test_wb_depth];

        uint32_t batch_size = 4;
        for(int i =0; i < test_wb_depth; i++){
                uint32_t* cur_batch_array;
                cuda_err_chk(cudaMalloc(&cur_batch_array, sizeof(uint32_t) * batch_size));
                h_batch_arrays[i] = cur_batch_array;
        }
        
        cuda_err_chk(cudaMemcpy(batch_arrays, h_batch_arrays, sizeof(uint32_t*) * test_wb_depth,cudaMemcpyHostToDevice));

        uint32_t h_batch_size_array[test_wb_depth];
        for(int i = 0; i < test_wb_depth; i++){
            h_batch_size_array[i] = batch_size;
        }
        
        uint32_t* batch_size_array;
        cuda_err_chk(cudaMalloc(&batch_size_array, sizeof(uint32_t) * test_wb_depth));
        cuda_err_chk(cudaMemcpy(batch_size_array, h_batch_size_array, sizeof(uint32_t) * test_wb_depth,cudaMemcpyHostToDevice));


        uint32_t h_batch[batch_size];
        for(int i = 0; i < batch_size; i++){
            h_batch[i] = i * 2;
        }
        cuda_err_chk(cudaMemcpy(h_batch_arrays[0], h_batch, sizeof(uint32_t) * batch_size,cudaMemcpyHostToDevice));
        
        for(int i = 0; i < batch_size; i++){
            h_batch[i] = i;
        }
        cuda_err_chk(cudaMemcpy(h_batch_arrays[1], h_batch, sizeof(uint32_t) * batch_size,cudaMemcpyHostToDevice));
        
        dim3 b_block(32, test_wb_depth, 1);

        cudaEventRecord(wb_before);
        update_wb_counters<TYPE><<<1, b_block>>>(a.d_array_ptr, batch_arrays, batch_size_array, (uint64_t)test_wb_depth);
        cudaEventRecord(wb_after);
        cudaEventSynchronize(wb_after);

        check_wb_counters<TYPE><<<1,32>>>(a.d_array_ptr);
        cuda_err_chk(cudaDeviceSynchronize());
        
        uint32_t num_evict = num_idx;

        for(auto i = 0; i < num_evict; i++){
               index_ptr[i] = i+num_idx;
        }
        
        int64_t *d_index_evict_ptr;
        cuda_err_chk(cudaMalloc(&d_index_evict_ptr, sizeof(int64_t) * num_evict));
        cuda_err_chk(cudaMemcpy(d_index_evict_ptr, index_ptr, sizeof(int64_t) * num_evict, cudaMemcpyHostToDevice));
            
        
        evict_feature_kernel<TYPE><<<num_evict, b_size>>>(a.d_array_ptr, d_out, d_index_evict_ptr, dim, num_evict, cache_dim, 
                                                      wb_queue_counter,  wb_depth, queue_ptr, wb_id_array, queue_depth);
                                                      
       cuda_err_chk(cudaDeviceSynchronize());
                         
      print_array<float><<<1, b_size>>>(queue_ptr, wb_id_array, test_wb_depth, queue_depth);
      cuda_err_chk(cudaDeviceSynchronize());
      printf("done\n");
    
     cudaFreeHost(h_queue_ptr);
     cudaFreeHost(h_wb_id_array);
     cudaFree(wb_queue_counter);
     for(int i =0; i < test_wb_depth; i++){
         cudaFree(h_batch_arrays[i]);
     }
     cudaFree(batch_arrays);
     cudaFree(batch_size_array);
     

     for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
