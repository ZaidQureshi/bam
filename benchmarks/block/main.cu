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
#include <byteswap.h>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;


//uint32_t n_ctrls = 1;
const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
const char* const intel_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9"};
#ifdef __GRAID__
const std::pair<unsigned, unsigned> graid_devs[] = { {0, 0}, {1, 0}, {2, 0}, {3, 0} };
#endif
//const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9", "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13", "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17", "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21", "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25", "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29", "/dev/libnvm30", "/dev/libnvm31"};


#define SIZE (8*4096)

__global__
void print_cache_kernel(page_cache_d_t* pc) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        hexdump(pc->base_addr, SIZE);
    }
}

__global__
void new_kernel(ulonglong4* dst, ulonglong4* src, size_t num) {
    warp_memcpy<ulonglong4>(dst, src, num);

}

__global__ //__launch_bounds__(64,32)
void sequential_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t total_blocks, uint64_t reqs_per_thread, uint32_t access_type, uint8_t* access_type_assignment) {
    //printf("in threads\n");
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    uint32_t laneid = lane_id();
    //uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t qhash;
    if (laneid == 0) {
        ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        //qhash = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        //qhash = smid % (ctrls[ctrl]->n_qps);
        qhash = warp_id % (ctrls[ctrl]->n_qps);
    }
    ctrl =  __shfl_sync(0xFFFFFFFF, ctrl, 0);
    qhash =  __shfl_sync(0xFFFFFFFF, qhash, 0);

    if (tid < n_reqs) {
        Controller *_ctrl = ctrls[ctrl];
        const uint64_t block_size_log = _ctrl->d_qps[qhash].block_size_log;
        uint64_t start_block = (tid*req_size*reqs_per_thread) >> block_size_log;
        uint64_t n_blocks = req_size >> block_size_log;
        uint8_t opcode;
        for (size_t i = 0; i < reqs_per_thread; i++) {
            opcode = access_type == MIXED ? access_type_assignment[tid] :
                     access_type == READ ? NVM_IO_READ : NVM_IO_WRITE;
            access_data(_ctrl, qhash, start_block, n_blocks, opcode, pc, tid);
            start_block += n_blocks;
            if (start_block >= total_blocks)
                start_block = 0;
        }
    }
}

__global__ //__launch_bounds__(64,32)
void random_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t* assignment, uint64_t total_blocks, uint64_t reqs_per_thread, uint32_t access_type, uint8_t* access_type_assignment) {
    //printf("in threads\n");
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    uint32_t laneid = lane_id();
    //uint32_t bid = blockIdx.x;
    uint32_t smid = get_smid();

    uint32_t ctrl;
    uint32_t qhash;
    if (laneid == 0) {
	//ctrl = smid % (pc->n_ctrls);
        ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
        //qhash = ctrls[ctrl]->queue_counter.fetch_add(1, simt::memory_order_relaxed) %  (ctrls[ctrl]->n_qps);
        //qhash = smid % (ctrls[ctrl]->n_qps);
        qhash = warp_id % (ctrls[ctrl]->n_qps);
    }
    ctrl =  __shfl_sync(0xFFFFFFFF, ctrl, 0);
    qhash =  __shfl_sync(0xFFFFFFFF, qhash, 0);


    if (tid < n_reqs) {
        Controller *_ctrl = ctrls[ctrl];
        const uint64_t block_size_log = _ctrl->d_qps[qhash].block_size_log;
        uint64_t start_block = (assignment[tid]*req_size) >> block_size_log;
        uint64_t n_blocks = req_size >> block_size_log; /// _ctrl.ns.lba_data_size;;

        uint8_t opcode;
        for (size_t i = 0; i < reqs_per_thread; i++) {
            opcode = access_type == MIXED ? access_type_assignment[tid] :
                     access_type == READ ? NVM_IO_READ : NVM_IO_WRITE;
            access_data(_ctrl, qhash, start_block, n_blocks, opcode, pc, tid);
            start_block += n_blocks;
            if (start_block >= total_blocks)
                start_block = 0;
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

        const char* input_f;

        input_f = settings.input;

        void* map_in = nullptr;
        int fd_in;
        struct stat sb_in;

        if (input_f != nullptr) {
            if((fd_in = open(input_f, O_RDWR)) == -1){
                fprintf(stderr, "Input file cannot be opened\n");
                return 1;
            }

            fstat(fd_in, &sb_in);

            map_in = mmap(NULL, sb_in.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_in, 0);

            if((map_in == (void*)-1)){
                fprintf(stderr,"Input file map failed\n");
                return 1;
            }

            close(fd_in);
        }
        //Controller ctrl(settings.controllerPath, settings.nvmNamespace, settings.cudaDevice);
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++) {
            Controller *ctrl = NULL;
            switch (settings.ssdtype) {
            case 0:
                ctrl = new Controller(sam_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
                break;
            case 1:
                ctrl = new Controller(intel_ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
		break;
            case 2:
#ifdef __GRAID__
                ctrl = new Controller(graid_devs[i], settings.cudaDevice, settings.numQueues);
#else
                fprintf(stderr,"GRAID Controller not supported\n");
                return 1;
#endif
		break;
            };
            ctrls[i] = ctrl;
	}

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
        //
        if (n_pages < n_threads) {
            std::cerr << "Please provide enough pages. Number of pages must be greater than or equal to the number of threads!\n";
            exit(1);
        }


        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        //QueuePair* d_qp;
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        #define TYPE uint64_t
        uint64_t n_blocks = settings.numBlks;
        //uint64_t t_size = n_blocks * sizeof(TYPE);

        // range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        // range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;

        // std::vector<range_t<uint64_t>*> vr(1);
        // vr[0] = & h_range;
        // //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        // array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);


        //std::cout << "finished creating range\n";




        unsigned long long* d_req_count;
        cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));

        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;
        uint64_t* assignment;
        uint64_t* d_assignment;
        if (settings.random) {
            assignment = (uint64_t*) malloc(n_threads*sizeof(uint64_t));
            for (size_t i = 0; i< n_threads; i++)
                assignment[i] = rand() % (n_blocks);


            cuda_err_chk(cudaMalloc(&d_assignment, n_threads*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment,  n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        Event before;

        uint8_t* access_assignment;
        uint8_t* d_access_assignment = NULL;
        if (settings.accessType == 2) {
            access_assignment = (uint8_t*) malloc(n_threads*sizeof(uint8_t));
            for (size_t i = 0; i < n_threads; i++)
                access_assignment[i] = ((((unsigned)rand() % 100u) + 1u) <= settings.ratio) ? NVM_IO_READ : NVM_IO_WRITE;

            cuda_err_chk(cudaMalloc(&d_access_assignment, n_threads*sizeof(uint8_t)));
            cuda_err_chk(cudaMemcpy(d_access_assignment, access_assignment, n_threads*sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
        std::cout << "atlaunch kernel\n";
        if (settings.random)
            random_access_kernel<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, n_blocks, settings.numReqs, settings.accessType, d_access_assignment);
        else
            sequential_access_kernel<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, n_blocks, settings.numReqs, settings.accessType, d_access_assignment);
        Event after;

#ifdef __GRAID__
	const bool giioq_debug = false;
	if (ctrls[0]->d_giioqs && giioq_debug)
		giioq_print_counters<<<1,1>>>(ctrls[0]->d_giioqs);
#endif
        //print_cache_kernel<<<1,1>>>(d_pc);
        //new_kernel<<<1,1>>>();
        //uint8_t* ret_array = (uint8_t*) malloc(n_pages*page_size);

        //cuda_err_chk(cudaMemcpy(ret_array, h_pc.base_addr,page_size*n_pages, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaDeviceSynchronize());
        if (input_f != nullptr && map_in != nullptr) {
            cuda_err_chk(cudaMemcpy(map_in, h_pc.pdt.base_addr,  std::min((uint64_t)sb_in.st_size, total_cache_size), cudaMemcpyDeviceToHost));
            munmap(map_in, sb_in.st_size);
        }

        double elapsed = after - before;
        uint64_t ios = g_size*b_size*settings.numReqs;
        uint64_t data = ios*page_size;
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
        h_pc.print_reset_stats();
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
