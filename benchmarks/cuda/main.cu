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


__global__
void new_kernel(ulonglong4* dst, ulonglong4* src, size_t num) {
    warp_memcpy<ulonglong4>(dst, src, num);

}
/*
__device__ void read_data(page_cache_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) threadIdx.x, (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    sq_dequeue(&qp->sq, sq_pos);
    cq_dequeue(&qp->cq, cq_pos);


    put_cid(&qp->sq, cid);


}

*/
__global__
void access_kernel(Controller** ctrls, page_cache_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t* assignment, uint64_t reqs_per_thread) {
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl = (tid/32) % (num_ctrls);
    uint32_t queue = (tid/32) % (ctrls[ctrl]->n_qps);


    if (tid < n_reqs) {
        uint64_t start_block = (assignment[tid]*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
        //start_block = tid;
        uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
        //printf("tid: %llu\tstart_block: %llu\tn_blocks: %llu\n", (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks);

        for (size_t i = 0; i < reqs_per_thread; i++)
            read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //__syncthreads();
        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
        //printf("tid: %llu finished\n", (unsigned long long) tid);

    }

}
__global__
void access_kernel(array_t<uint64_t>* dr, uint64_t n_reqs, unsigned long long* req_count, uint64_t* assignment) {
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    //uint32_t bid = blockIdx.x;
    //uint32_t smid = get_smid();

    //uint32_t ctrl = bid & (num_ctrls-1);
    //uint32_t queue = smid & (ctrls[ctrl].n_qps-1);


    if (tid < n_reqs) {
        //req_count += (*dr)[tid];
        req_count += dr->seq_read(tid);
        //uint64_t start_block = (assignment[tid]*req_size) >> ctrls[ctrl].d_qps[queue].block_size_log;
        //uint64_t n_blocks = req_size >> ctrls[ctrl].d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;

        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block, n_blocks, tid);
        //__syncthreads();
        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
        //printf("tid: %llu finished\n", (unsigned long long) tid);

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
        uint64_t b_size = settings.blkSize;//64;
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
        uint64_t n_elems = (1024ULL)*(1024ULL)*(2);
        uint64_t t_size = n_elems * sizeof(TYPE);

        range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;

        std::vector<range_t<uint64_t>*> vr(1);
        vr[0] = & h_range;
        //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);


        std::cout << "finished creating range\n";

        uint64_t* assignment = (uint64_t*) malloc(n_threads*sizeof(uint64_t));
        for (size_t i = 0; i< n_threads; i++)
            assignment[i] = rand() % (n_elems);

        uint64_t* d_assignment;
        cuda_err_chk(cudaMalloc(&d_assignment, n_threads*sizeof(uint64_t)));
        cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice));


        unsigned long long* d_req_count;
        cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
        std::cout << "atlaunch kernel\n";
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        Event before;
        access_kernel<<<g_size, b_size>>>(h_pc.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, settings.numReqs);
        //access_kernel<<<g_size, b_size>>>(a.d_array_ptr, n_threads, d_req_count, d_assignment);
        Event after;
        //new_kernel<<<1,1>>>();
        uint8_t* ret_array = (uint8_t*) malloc(n_pages*page_size);

        cuda_err_chk(cudaMemcpy(ret_array, h_pc.base_addr,page_size*n_pages, cudaMemcpyDeviceToHost));



        double elapsed = after - before;
        uint64_t ios = g_size*b_size*settings.numReqs;
        uint64_t data = ios*page_size;
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        std::cout << std::dec << "Elapsed: " << elapsed << "\tIOS: "<< ios << "\tData: " << data << std::endl;
        std::cout << std::dec << "IOPs: " << iops << "\tBandwidth(GB/S): " << bandwidth << std::endl;
        std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        std::ofstream ofile("../data", std::ios::binary | std::ios::trunc);
        ofile.write((char*)ret_array, data);
        ofile.close();

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
        //hexdump(ret_array, n_pages*page_size);
/*
        cudaFree(d_qp);
        cudaFree(d_pc);
        cudaFree(d_req_count);
        free(ret_array);
*/

        std::cout << "END\n";

        std::cout << RAND_MAX << std::endl;

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
