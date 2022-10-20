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





//uint32_t n_ctrls = 1;
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};


template<typename T>
__global__ __launch_bounds__(64,32)
void random_access_warp(array_d_t<T>* dr, uint64_t n_pages_per_warp, unsigned long long* sum, uint64_t type, uint64_t* assignment, uint64_t n_warps, size_t page_size, uint64_t stride) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t old_warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    
    uint64_t nep = (n_warps+stride)/stride; 
    uint64_t warp_id = (old_warp_id/nep) + ((old_warp_id % nep)* stride);

    T v = 0;
    if (warp_id < n_warps) {
	    bam_ptr<T> ptr(dr);
        size_t start_page = assignment[warp_id];//n_pages_per_warp * warp_id;//assignment[warp_id];
//	if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
//	    printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += 32) {
//		printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                if (type == ORIG) {
                    v += (*dr)[start_idx + j];
                }
                else {
                    v += ptr[start_idx + j];
                }
            }

        }
        *sum = v;
    }

}

template<typename T>
__global__ __launch_bounds__(64,32)
void sequential_access_warp(array_d_t<T>* dr, uint64_t n_pages_per_warp, unsigned long long* sum, uint64_t type, uint64_t* assignment, uint64_t n_warps, size_t page_size) {

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lane = tid % 32;
    const uint64_t warp_id = tid / 32;
    const uint64_t n_elems_per_page = page_size / sizeof(T);
    T v = 0;
    if (warp_id < n_warps) {
	bam_ptr<T> ptr(dr);
        size_t start_page = n_pages_per_warp * warp_id;;
//	if (lane == 0) printf("start_page: %llu\n", (unsigned long long) start_page);
        for (size_t i = 0; i < n_pages_per_warp; i++) {
            size_t cur_page = start_page + i;
//	    printf("warp_id: %llu\tcur_page: %llu\n", (unsigned long long) warp_id, (unsigned long long) cur_page);
            size_t start_idx = cur_page * n_elems_per_page + lane;

            for (size_t j = 0; j < n_elems_per_page; j += 32) {
//		printf("startidx: %llu\n", (unsigned long long) (start_idx+j));
                if (type == ORIG) {
                    v += (*dr)[start_idx + j];
                }
                else {
                    v += ptr[start_idx + j];
                }
            }

        }
        *sum = v;
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
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        uint64_t b_size = 64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;
        uint64_t n_warps = n_threads/32;


        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        //uint64_t n_pages = total_cache_size/page_size;


        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        #define TYPE uint64_t
        uint64_t n_elems = settings.numElems;
        uint64_t t_size = n_elems * sizeof(TYPE);
        uint64_t n_data_pages =  (uint64_t)(t_size/page_size);

        range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, n_data_pages, (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint64_t>*> vr(1);
        vr[0] = & h_range;
        array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);


        std::cout << "finished creating range\n";


        uint64_t n_pages_per_warp = settings.numReqs;
        uint64_t gran = settings.gran; //(settings.gran == WARP) ? 32 : b_size;
        uint64_t type = settings.type;

        uint64_t n_elems_per_page = page_size / sizeof(uint64_t);
        std::cout << "n_elems_per_page: " << n_elems_per_page << std::endl;
        unsigned long long* d_req_count;
        cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
        std::cout << "atlaunch kernel\n";
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;
        uint64_t* assignment;
        uint64_t* d_assignment;
        if (settings.random) {
            assignment = (uint64_t*) malloc(n_warps*sizeof(uint64_t));
            for (size_t i = 0; i < n_warps; i++) {
                uint64_t page = rand() % (n_data_pages);
                assignment[i] = page;
            }
            cuda_err_chk(cudaMalloc(&d_assignment, n_warps*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_warps*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        
        for(uint64_t id=0; id<2;id++){

        Event before;
                
        if (settings.random) {
                //printf("blockDim.x is %llu \t blocksize: %llu\n", g_size, b_size );
                random_access_warp<TYPE><<<g_size, b_size>>>(a.d_array_ptr, n_pages_per_warp, d_req_count, type, d_assignment, n_warps, page_size, settings.stride);
        }
        else {
                sequential_access_warp<TYPE><<<g_size, b_size>>>(a.d_array_ptr, n_pages_per_warp, d_req_count, type, d_assignment, n_warps, page_size);
        }
        Event after;

        cuda_err_chk(cudaDeviceSynchronize());

        double elapsed = after - before;
        uint64_t ios = n_warps*n_pages_per_warp*n_elems_per_page;
        uint64_t data = ios*sizeof(uint64_t);
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        a.print_reset_stats();
        std::cout << std::dec << "Itr:" << id << " type: "<< settings.random <<" Elapsed Time: " << elapsed << "\tNumber of Read Ops: "<< ios << "\tData Size (bytes): " << data ;
        std::cout << std::dec << "Read Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

		printf("ID:%d \t type:%d \t n_warps:%llu \t n_pages_per_warp: %llu \t n_elems_per_page:%llu \t ios: %llu \t IOPs: %f \t data:%llu \t bandwidth: %f GBps \t time: %f\n",id, settings.random,  n_warps, n_pages_per_warp, n_elems_per_page, ios, iops, data, bandwidth, elapsed); 
        }
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
