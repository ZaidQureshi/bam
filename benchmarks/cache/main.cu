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
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7"};


__global__ __launch_bounds__(64,32)
void sequential_access_warp(array_d_t<uint64_t>* dr, uint64_t n_reqs, unsigned long long* sum, uint64_t type) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;
    uint64_t n_warps = grid_size / 32;
    uint64_t lane = tid % 32;
    uint64_t warp_id = tid / 32;

    uint64_t reqs_per_thread = n_reqs / grid_size;
    uint64_t reqs_per_warp   = n_reqs / n_warps;
    uint64_t reqs_per_block = n_reqs / gridDim.x;

    uint64_t start_req = warp_id * reqs_per_warp + lane;
    uint64_t end_req = (warp_id + 1) * reqs_per_warp;



    uint64_t acc = 0;
    if (type == ORIG) {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += 32)
            acc += (*dr)[start_req];
    }
    else {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += 32)
            acc += (dr)->seq_read((start_req));
    }

    if (threadIdx.x == 0)
        *sum = acc;



}


__global__ __launch_bounds__(64,32)
void sequential_access_blk(array_d_t<uint64_t>* dr, uint64_t n_reqs, unsigned long long* sum, uint64_t type) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;
    uint64_t n_warps = grid_size / 32;
    uint64_t lane = tid % 32;
    uint64_t warp_id = tid / 32;

    uint64_t reqs_per_thread = n_reqs / grid_size;
    uint64_t reqs_per_warp   = n_reqs / n_warps;
    uint64_t reqs_per_block = n_reqs / gridDim.x;

    uint64_t start_req = blockIdx.x * reqs_per_block + threadIdx.x;
    uint64_t end_req = (blockIdx.x + 1) * reqs_per_block;

    uint64_t acc = 0;
    if (type == ORIG) {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += blockDim.x)
            acc += (*dr)[((start_req))];
    }
    else {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += blockDim.x)
            acc += (dr)->seq_read((start_req));
    }

    if (threadIdx.x == 0)
        *sum = acc;

}


__global__ __launch_bounds__(64,32)
void random_access_warp(array_d_t<uint64_t>* dr, uint64_t n_reqs, unsigned long long* sum, uint64_t type, uint64_t* assignment) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;
    uint64_t n_warps = grid_size / 32;
    uint64_t lane = tid % 32;
    uint64_t warp_id = tid / 32;

    uint64_t reqs_per_thread = n_reqs / grid_size;
    uint64_t reqs_per_warp   = n_reqs / n_warps;
    uint64_t reqs_per_block = n_reqs / gridDim.x;

    uint64_t start_req = warp_id * reqs_per_warp + lane;
    uint64_t end_req = (warp_id + 1) * reqs_per_warp;

    uint64_t acc = 0;
    if (type == ORIG) {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += 32)
            acc += (*dr)[assignment[start_req]];
    }
    else {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += 32)
            acc += (dr)->seq_read(assignment[start_req]);
    }

    if (threadIdx.x == 0)
        *sum = acc;



}


__global__ __launch_bounds__(64,32)
void random_access_blk(array_d_t<uint64_t>* dr, uint64_t n_reqs, unsigned long long* sum, uint64_t type, uint64_t* assignment) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;
    uint64_t n_warps = grid_size / 32;
    uint64_t lane = tid % 32;
    uint64_t warp_id = tid / 32;

    uint64_t reqs_per_thread = n_reqs / grid_size;
    uint64_t reqs_per_warp   = n_reqs / n_warps;
    uint64_t reqs_per_block = n_reqs / gridDim.x;

    uint64_t start_req = blockIdx.x * reqs_per_block + threadIdx.x;
    uint64_t end_req = (blockIdx.x + 1) * reqs_per_block;

    uint64_t acc = 0;
    if (type == ORIG) {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += blockDim.x)
            acc += (*dr)[assignment[start_req]];
    }
    else {
        for (; (start_req < n_reqs) && (start_req < end_req); start_req += blockDim.x)
            acc += (dr)->seq_read(assignment[start_req]);
    }

    if (threadIdx.x == 0)
        *sum = acc;

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
        //Controller ctrl(settings.controllerPath, settings.nvmNamespace, settings.cudaDevice);

        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);

        //auto dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(64*1024*10, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);

        //std::cout << dma.get()->vaddr << std::endl;
        //QueuePair h_qp(ctrl, settings, 1);
        //std::cout << "in main: " << std::hex << h_qp.sq.cid << "raw: " << h_qp.sq.cid<< std::endl;
        //std::memset(&h_qp, 0, sizeof(QueuePair));
        //prepareQueuePair(h_qp, ctrl, settings, 1);
        //const uint32_t ps, const uint64_t np, const uint64_t c_ps, const Settings& settings, const Controller& ctrl)
        //
        /*
        Controller** d_ctrls;
        cuda_err_chk(cudaMalloc(&d_ctrls, n_ctrls*sizeof(Controller*)));
        for (size_t i = 0; i < n_ctrls; i++)
            cuda_err_chk(cudaMemcpy(d_ctrls+i, &(ctrls[i]->d_ctrl), sizeof(Controller*), cudaMemcpyHostToDevice));
        */
        uint64_t b_size = 64;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;


        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        //uint64_t n_pages = total_cache_size/page_size;


        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        //QueuePair* d_qp;
        page_cache_t* d_pc = (page_cache_t*) (h_pc.d_pc_ptr);
        #define TYPE uint64_t
        uint64_t n_elems = settings.numElems;
        uint64_t t_size = n_elems * sizeof(TYPE);
        uint64_t n_data_pages =  (uint64_t)(t_size/page_size);

        range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, n_data_pages, (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint64_t>*> vr(1);
        vr[0] = & h_range;
        //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);


        std::cout << "finished creating range\n";


        uint64_t n_reqs = settings.numReqs;
        uint64_t gran = settings.gran; //(settings.gran == WARP) ? 32 : b_size;
        uint64_t type = settings.type;

        uint64_t n_elems_per_page = page_size / sizeof(uint64_t);

        uint64_t n_reqs_pages = (n_reqs * sizeof(uint64_t)) / page_size;
        std::cout << "n_reqs: " << n_reqs << std::endl;
        std::cout << "n_reqs_pages: " << n_reqs_pages << std::endl;
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
            assignment = (uint64_t*) malloc(n_reqs*sizeof(uint64_t));
            if (settings.trueRandom) {
                for (size_t i = 0; i< n_reqs; i++)
                    assignment[i] = rand() % (n_elems);
            }
            else {
                for (size_t i = 0; i < n_reqs_pages; i++) {
                    uint64_t page = rand() % (n_data_pages);
                    uint64_t page_starting_idx = page*n_elems_per_page;
                    for(size_t j = 0; j < n_elems_per_page; j++)
                        assignment[i*n_elems_per_page + j] = page_starting_idx + j;
                }

            }

            cuda_err_chk(cudaMalloc(&d_assignment, n_reqs*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_reqs*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        Event before;
        //access_kernel<<<g_size, b_size>>>(h_pc.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, settings.numReqs);
        if (settings.random) {
            if (gran == WARP)
                random_access_warp<<<g_size, b_size>>>(a.d_array_ptr, n_reqs, d_req_count, type, d_assignment);
            else
                random_access_blk<<<g_size, b_size>>>(a.d_array_ptr, n_reqs, d_req_count, type, d_assignment);

        }
        else {
            if (gran == WARP)
                sequential_access_warp<<<g_size, b_size>>>(a.d_array_ptr, n_reqs, d_req_count, type);
            else
                sequential_access_blk<<<g_size, b_size>>>(a.d_array_ptr, n_reqs, d_req_count, type);
        }
        Event after;
        //new_kernel<<<1,1>>>();
        //uint8_t* ret_array = (uint8_t*) malloc(n_pages*page_size);

        //cuda_err_chk(cudaMemcpy(ret_array, h_pc.base_addr,page_size*n_pages, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaDeviceSynchronize());


        double elapsed = after - before;
        uint64_t ios = n_reqs;
        uint64_t data = ios*sizeof(uint64_t);
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        a.print_reset_stats();
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Read Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Read Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        //std::ofstream ofile("../data", std::ios::binary | std::ios::trunc);
        //ofile.write((char*)ret_array, data);
        //ofile.close();

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
        //hexdump(ret_array, n_pages*page_size);
/*
        cudaFree(d_qp);
        cudaFree(d_pc);
        cudaFree(d_req_count);
        free(ret_array);
*/

        //std::cout << "END\n";

        //std::cout << RAND_MAX << std::endl;

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
