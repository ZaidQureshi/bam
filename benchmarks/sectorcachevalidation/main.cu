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

 //const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6"};
const char* const ctrls_paths[] = {"/dev/libnvm1"};

__global__
void sequential_access_write_kernel(array_d_t<uint64_t>* dr, uint64_t n_reqs, uint64_t n_pages, uint64_t page_size, int* counter) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_reqs) {
        dr->seq_write((size_t)tid, (uint64_t)tid);
        atomicAdd(counter, 1);
    }
    //__syncthreads();
     
    //printf ("tid: %llu counter: %llu\n", (unsigned long long)tid, (unsigned long long)(*counter));
    /*bool hastoflushpage = true;
    while (hastoflushpage && (tid < n_pages)) {
        if ((*counter) == (int)n_reqs)
        {
            //dr->flushcache(tid, page_size);
            hastoflushpage = false;
            printf ("tid: %llu counter: %llu\n", (unsigned long long)tid, (unsigned long long)(*counter));
        }
    }*/
    

}

__global__
void sequential_access_flush_kernel(array_d_t<uint64_t>* dr, uint64_t n_pages, int64_t page_size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_pages) {
            dr->flushcache(tid, page_size);
            //printf ("tid: %llu counter: %llu\n", (unsigned long long)tid, (unsigned long long)(*counter));
    }
    

}

__global__
void sequential_access_read_kernel(array_d_t<uint64_t>* dr, uint64_t n_reqs, uint64_t* device_buffer, uint64_t n_pages, uint64_t page_size, int* counter) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x*blockDim.x;
    //printf("stride %llu\n", (unsigned long long)stride);
    bam_ptr<uint64_t> ptr(dr);
    if (tid < n_reqs) {
        
        //device_buffer[tid]= (*dr)[(tid)];
        device_buffer[tid]= ptr[tid];
        //printf("tid = %llu\t device_buffer %llu\n", (unsigned long long)tid, (unsigned long long)device_buffer[tid]);
        //device_buffer[stride + tid]= ptr[tid];

    }
    //__syncthreads();

}

__global__
void random_access_read_kernel(array_d_t<uint64_t>* dr, uint64_t n_reqs, uint64_t* device_buffer, uint64_t* assignment_buffer, uint64_t n_pages, uint64_t page_size, int* counter) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    bam_ptr<uint64_t> ptr(dr);
    if (tid < n_reqs) {
        device_buffer[tid]= ptr[assignment_buffer[tid]];
        //device_buffer[tid]= (*dr)[(assignment_buffer[tid])];

        /*uint64_t result;
        //result = (*dr)[(tid)];
        result = dr->seq_read((size_t)tid);
        printf("%llu : %16x\n", (unsigned long long)tid, result);*/

    }
    __syncthreads();

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
        printf("controller created\n");
        fflush(stderr);
        fflush(stdout);

        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = settings.numThreads;//b_size * g_size;
        std::cout << n_threads <<std::endl;

        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        uint64_t n_blocks = settings.numBlks;
        /*uint64_t total_cache_size = settings.cacheSize;
        uint64_t n_pages = ceil(total_cache_size/page_size);*/
        uint32_t sector_size = settings.sectorSize;

        /*if(total_cache_size > (sb_in.st_size - settings.ifileoffset)){
                n_pages = ceil((sb_in.st_size - settings.ifileoffset)/(1.0*settings.pageSize));
                total_cache_size = n_pages*page_size; 
        }*/

        page_cache_t h_pc(page_size, n_pages, sector_size, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n Total Cache size (MBs):" << ((float)total_cache_size/(1024*1024)) <<std::endl;
        fflush(stderr);
        fflush(stdout);

        //QueuePair* d_qp;
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);

        #define TYPE uint64_t
        uint64_t n_elems = (settings.numBlks*512)/sizeof(TYPE);
        uint64_t t_size = n_elems * sizeof(TYPE);
        std::cout << "started creating range\n";
        range_t<uint64_t> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<uint64_t>* d_range = (range_t<uint64_t>*) h_range.d_range_ptr;
        std::cout << "finished creating range\n";

        std::vector<range_t<uint64_t>*> vr(1);
        vr[0] = & h_range;        
        array_t<uint64_t> a(n_elems, 0, vr, settings.cudaDevice);
        std::cout << "finished creating array\n";
         
            /*printf("Writing contents to NVMe Device at %llu\n", settings.ofileoffset); 

               //uint64_t cpysize = 16*total_cache_size;
               //std::cout << "cpysize = " << cpysize << std::endl;
               fflush(stderr);
               fflush(stdout);

               int* counter_d;
               cuda_err_chk(cudaMalloc((void**)(&counter_d), sizeof(int)));
               cuda_err_chk(cudaMemset(counter_d, 0, sizeof(int)));

               Event before;
               sequential_access_write_kernel<<<g_size, b_size>>>(a.d_array_ptr, n_threads, n_pages, page_size, counter_d); 
               
               Event after;
               cuda_err_chk(cudaDeviceSynchronize());
               sequential_access_flush_kernel<<<1, n_pages>>>(a.d_array_ptr, n_pages, page_size);
               cuda_err_chk(cudaDeviceSynchronize());
               

               double elapsed = after - before;

               std::cout << "Completed Time:" <<elapsed << std::endl;

               // uint64_t ios = g_size*b_size*settings.numReqs;
               // uint64_t data = ios*page_size;
               // double iops = ((double)ios)/(elapsed/1000000);
               // double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
               // std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
               // std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
                */
                printf("Reading NVMe contents from %llu", settings.ofileoffset);                  
                fflush(stderr);
                fflush(stdout);

                    uint64_t cpysize = n_threads*sizeof(TYPE); 

                    cuda_err_chk(cudaMemset(h_pc.pdt.base_addr, 0, total_cache_size));

                    uint64_t* tmprbuff; 
                    tmprbuff = (uint64_t*) malloc(cpysize);
                    memset(tmprbuff, 0, (cpysize));
                    
                    uint64_t* device_buffer;
                    cuda_err_chk(cudaMalloc((void**)(&device_buffer), cpysize));
                    cuda_err_chk(cudaMemset(device_buffer, 0, cpysize));
                    int* counter_d;
                    cuda_err_chk(cudaMalloc((void**)(&counter_d), sizeof(int)));
                    cuda_err_chk(cudaMemset(counter_d, 0, sizeof(int)));
                    uint64_t* assignment;
                    assignment = (uint64_t*) malloc(n_threads*sizeof(uint64_t));
                    //memset(assignment, 0, cpysize);
                    for (size_t i = 0; i< n_threads; i++)
                        assignment[i] = rand() % (n_threads);
                    uint64_t* assignment_d;
                    cuda_err_chk(cudaMalloc((void**)(&assignment_d),n_threads*sizeof(uint64_t)));
                    cuda_err_chk(cudaMemcpy(assignment_d, assignment, n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice ));

                    Event rbefore; 
                    sequential_access_read_kernel<<<g_size, b_size>>>(a.d_array_ptr, n_threads, device_buffer, n_pages, page_size, counter_d);
                    //random_access_read_kernel<<<g_size, b_size>>>(a.d_array_ptr, n_threads, device_buffer, assignment_d, n_pages, page_size, counter_d);
                    Event rafter;
                    cuda_err_chk(cudaDeviceSynchronize());
                    cuda_err_chk(cudaMemcpy(tmprbuff,device_buffer, cpysize, cudaMemcpyDeviceToHost));
                    
                    double relapsed = rafter - rbefore;
                    std::cout << "Read Completed  Read Time:" <<relapsed << std::endl;

                    int errorcnt=0;
                    
                    for (uint64_t i=0; i<n_threads; i++) {
                        //std::cout << i << "   :   " << tmprbuff[i] << std::endl;
                        if (i != tmprbuff[(size_t)i]) {
                            errorcnt++;
                            std::cout << "Error: threadID : " << i << "\tValue : " << tmprbuff[(size_t)i] <<std::endl;
                        }
                    }
                    std::cout << "Total error count : " << errorcnt <<std::endl; 


        

 
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
