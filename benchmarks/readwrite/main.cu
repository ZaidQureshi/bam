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
const char* const ctrls_paths[] = {"/dev/libnvm0"};

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
void sequential_access_kernel(Controller** ctrls, page_cache_d_t* pc,  uint32_t req_size, uint32_t n_reqs, //unsigned long long* req_count,
                                uint32_t num_ctrls, uint64_t reqs_per_thread, uint32_t access_type, uint64_t s_offset, uint64_t o_offset){
    //printf("in threads\n");
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // uint32_t bid = blockIdx.x;
    // uint32_t smid = get_smid();

    uint32_t ctrl = (tid/32) % (num_ctrls);
    uint32_t queue = (tid/32) % (ctrls[ctrl]->n_qps);
    uint64_t itr=0; 

//    printf("Num pages: %llu, s_offset: %llu n_reqs: %llu\t req_size: %llu\n", (unsigned long long int) pc->n_pages, (unsigned long long int) s_offset, (unsigned long long int) n_reqs, (unsigned long long) req_size); 
    for (;tid < pc->n_pages; tid = tid+n_reqs){
            uint64_t start_block = (o_offset+s_offset + tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log ;
            uint64_t pc_idx = (tid);
            //uint64_t start_block = (tid*req_size) >> ctrls[ctrl]->d_qps[queue].block_size_log;
            //start_block = tid;
            uint64_t n_blocks = req_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
//            printf("itr:%llu\ttid: %llu\tstart_block: %llu\tn_blocks: %llu\tpc_idx: %llu\n", (unsigned long long)itr, (unsigned long long) tid, (unsigned long long) start_block, (unsigned long long) n_blocks, (unsigned long long) pc_idx);
            itr = itr+1; 
            // uint8_t opcode;
            // for (size_t i = 0; i < reqs_per_thread; i++) {
                if (access_type == READ) {
                    read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, pc_idx);
                    //if(tid ==( pc->n_pages - 1)){
                    //        printf("I am here\n");
                    //        hexdump(pc->base_addr+tid*req_size, 4096); 
                    //}
                }
                else {
                    write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, pc_idx);
                }

            
            // }
            //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            //__syncthreads();
            //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
            //printf("tid: %llu finished\n", (unsigned long long) tid);
    }
}

/*__global__
void random_access_kernel(Controller** ctrls, page_cache_t* pc,  uint32_t req_size, uint32_t n_reqs, unsigned long long* req_count, uint32_t num_ctrls, uint64_t* assignment, uint64_t reqs_per_thread, uint32_t access_type, uint8_t* access_type_assignment) {
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

        uint8_t opcode;
        for (size_t i = 0; i < reqs_per_thread; i++) {
            if (access_type == MIXED) {
                opcode = access_type_assignment[tid];
                access_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid, opcode);
            }
            else if (access_type == READ) {
                read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);

            }
            else {
                write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
            }
        }
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //read_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
        //__syncthreads();
        //read_data(pc, (ctrls[ctrl].d_qps)+(queue),start_block*2, n_blocks, tid);
        //printf("tid: %llu finished\n", (unsigned long long) tid);

    }

}
*/

__global__ 
void verify_kernel(uint64_t* orig_h, uint64_t* nvme_h, uint64_t n_elems,uint32_t n_reqs){
        uint64_t tid = blockIdx.x*blockDim.x + threadIdx.x; 

        for (;tid < n_elems; tid = tid+n_reqs){
           uint64_t orig_val = orig_h[tid]; 
           uint64_t nvme_val = nvme_h[tid]; 
           if(orig_val != nvme_val)
              printf("MISMATCH: at %llu\torig_val:%llu\tnvme_val:%llu\tn_reqs:%lu\tn_elms:%llu\n",tid, (unsigned long long)orig_val, (unsigned long long)nvme_h, n_reqs, n_elems);
        }
        __syncthreads();//really not needed. 
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
    std::string read_f_tmp = std::string(input_f) + ".nvme";
    const char* read_f = read_f_tmp.c_str();

    try {
        void* map_in;
        int fd_in;
        struct stat sb_in;
        
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

        // uint64_t* dummyArray= NULL; 
        // cuda_err_chk(cudaHostAlloc(&dummyArray, 1024, cudaHostAllocDefault)); 
        // cuda_err_chk(cudaMemcpy(dummyArray, map_in+16, 1024, cudaMemcpyHostToDevice));

        // printf("value at 0: %llu\n", (uint64_t)dummyArray[0]);
        // fflush(stderr);
        // fflush(stdout);
        
        cuda_err_chk(cudaSetDevice(settings.cudaDevice));
        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
        printf("controller created\n");
        fflush(stderr);
        fflush(stdout);

        // unsigned long long* d_req_count;
        // cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        // cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;

        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;

        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        uint64_t n_blocks = settings.numBlks;

        if(total_cache_size > (sb_in.st_size - settings.ifileoffset)){
                n_pages = ceil((sb_in.st_size - settings.ifileoffset)/(1.0*settings.pageSize));
                total_cache_size = n_pages*page_size; 
        }


        //if((settings.accessType == WRITE) || (settings.accessType == READ))
        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n Total Cache size (MBs):" << ((float)total_cache_size/(1024*1024)) <<std::endl;
        fflush(stderr);
        fflush(stdout);

        //QueuePair* d_qp;
        page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc.d_pc_ptr);
        
        uint32_t n_tsteps = ceil((float)(sb_in.st_size-settings.ifileoffset)/(float)total_cache_size);  
        uint64_t n_telem = ((sb_in.st_size-settings.ifileoffset)/sizeof(uint64_t)); 
        uint64_t s_offset = 0; 
        
        printf("n_tsteps: %lu, n_telem: %llu, n_pages:%llu\n", n_tsteps, n_telem, n_pages); 

        if(settings.accessType == WRITE){
            printf("Writing contents from %s to NVMe Device at %llu\n", input_f, settings.ofileoffset); 
            for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {

               uint64_t cpysize = std::min(total_cache_size, (sb_in.st_size-settings.ifileoffset-s_offset));
               printf("cstep: %lu  s_offset: %llu   cpysize: %llu pcaddr:%p, block size: %llu, grid size: %llu\n", cstep, s_offset, cpysize, h_pc.pdt.base_addr, b_size, g_size);
               fflush(stderr);
               fflush(stdout);
               cuda_err_chk(cudaMemcpy(h_pc.pdt.base_addr, map_in+s_offset+settings.ifileoffset, cpysize, cudaMemcpyHostToDevice));
               Event before; 
               sequential_access_kernel<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, //d_req_count,
                                                               settings.n_ctrls, settings.numReqs, settings.accessType, s_offset, settings.ofileoffset);
               Event after;
               cuda_err_chk(cudaDeviceSynchronize());

               float completed = 100*(total_cache_size*(cstep+1))/(sb_in.st_size-settings.ifileoffset);
               double elapsed = after - before;

               s_offset = s_offset + cpysize; 

               std::cout << "Completed:" << completed << "%   Time:" <<elapsed << std::endl;

               // uint64_t ios = g_size*b_size*settings.numReqs;
               // uint64_t data = ios*page_size;
               // double iops = ((double)ios)/(elapsed/1000000);
               // double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
               // std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
               // std::cout << std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;
                        
            }
            if(munmap(map_in, sb_in.st_size) == -1) 
                fprintf(stderr,"munmap error input file\n");
            close(fd_in);
        }
        else if(settings.accessType == READ){
                int fd_out, ft;
                void* map_out; 
                if((fd_out = open(read_f, O_RDWR)) == -1){
                    fprintf(stderr, "NVMe Output file cannot be opened\n");
                    return 1;
                }
                
                if( (ft =ftruncate(fd_out,(sb_in.st_size-settings.ifileoffset))) == -1){
                    fprintf(stderr, "Truncating NVMe Output file failed\n");
                    return 1;
                }
                
                map_out = mmap(NULL, sb_in.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_out, 0);
                
                if((map_out == (void*)-1)){
                        fprintf(stderr,"Output file map failed: %d\n",map_out);
                        return 1;
                }

                uint8_t* tmprbuff; 
                tmprbuff = (uint8_t*) malloc(sb_in.st_size-settings.ifileoffset);
                memset(tmprbuff, 0, (sb_in.st_size-settings.ifileoffset));
                
                printf("Reading NVMe contents from %llu to file\n", settings.ofileoffset, read_f); 
                
                for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {

                    uint64_t cpysize = std::min(total_cache_size, (sb_in.st_size-settings.ifileoffset-s_offset));
                    //uint64_t cpysize = (total_cache_size); //, (sb_in.st_size-settings.ifileoffset-s_offset));
                    printf("cstep: %lu  s_offset: %llu   cpysize: %llu pcaddr:%p, block size: %llu, grid size: %llu\n", cstep, s_offset, cpysize, h_pc.pdt.base_addr, b_size, g_size);
                    fflush(stderr);
                    fflush(stdout);
//                    for(size_t wat=0; wat<32; wat++)
//                            std::cout << std::hex << tmprbuff[wat]; 
//                    std::cout<<std::endl;
                    cuda_err_chk(cudaMemset(h_pc.pdt.base_addr, 0, total_cache_size));

                    Event rbefore; 
                    sequential_access_kernel<<<g_size, b_size>>>(h_pc.pdt.d_ctrls, d_pc, page_size, n_threads, //d_req_count,
                                                                    settings.n_ctrls, settings.numReqs, settings.accessType, s_offset, settings.ofileoffset);
                    Event rafter;
                    cuda_err_chk(cudaDeviceSynchronize());
                //    cuda_err_chk(cudaMemcpy(map_out+s_offset,h_pc.pdt.base_addr, cpysize, cudaMemcpyDeviceToHost));
                    cuda_err_chk(cudaMemcpy(tmprbuff+s_offset,h_pc.pdt.base_addr, cpysize, cudaMemcpyDeviceToHost));
                    //cuda_err_chk(cudaMemcpy(tmprbuff+s_offset,h_pc.pdt.base_addr, cpysize, cudaMemcpyDeviceToHost));
                   

                    //if(msync(map_out, cpysize, MS_SYNC) == -1) {
                    //        fprintf(stderr, "msync failed in cstep:%d\n", cstep);
                    //}
//                    for(size_t wat=0; wat<32; wat++)
//                            std::cout << std::hex << tmprbuff[wat]; 
//                    std::cout<<std::endl;
                    s_offset = s_offset + cpysize; 
                    float rcompleted = 100*(total_cache_size*(cstep+1))/(sb_in.st_size-settings.ifileoffset);
                    double relapsed = rafter - rbefore;
                    std::cout << "Read Completed:" << rcompleted << "%   Read Time:" <<relapsed << std::endl;

                }
                uint64_t sz =0; 
                while(sz !=(sb_in.st_size-settings.ifileoffset)){
                        sz +=write(fd_out, tmprbuff+sz, (sb_in.st_size-sz-settings.ifileoffset));  
                }
                printf("Written file size: %llu\n", sz);
                free(tmprbuff);

                //if(munmap(map_out, sb_in.st_size) == -1) 
                //        fprintf(stderr,"munmap error output file\n");
                close(fd_out);
                if(munmap(map_in, sb_in.st_size) == -1) 
                    fprintf(stderr,"munmap error input file\n");
                close(fd_in);

        }
        else if (settings.accessType == VERIFY){
               uint64_t* orig_h;
               uint64_t* nvme_h;
        
               if(munmap(map_in, sb_in.st_size) == -1) 
                  fprintf(stderr,"munmap error input file\n");
               close(fd_in);
               
               void* map_orig;
               int fd_orig;
               struct stat sb_orig;
               
               printf("reading first file %s\n", input_f);
               if((fd_orig= open(input_f, O_RDWR)) == -1){
                   fprintf(stderr, "Orig file cannot be opened\n");
                   return 1;
               }
               
               fstat(fd_orig, &sb_orig);
               
               map_orig = mmap(NULL, sb_orig.st_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd_orig, 0);
               
               if((map_orig == (void*)-1)){
                       fprintf(stderr,"Input file map failed %d\n",map_orig);
                       return 1;
               }
               
               size_t orig_sz = sb_orig.st_size - settings.ifileoffset; 

               cuda_err_chk(cudaMallocManaged((void**)&orig_h, orig_sz)); 
               cuda_err_chk(cudaMemAdvise(orig_h, orig_sz, cudaMemAdviseSetAccessedBy, 0));
//               cuda_err_chk(cudaMemset(orig_h, 0, orig_sz)); 
               memcpy(orig_h, map_orig+settings.ifileoffset, orig_sz);

               void* map_nvme;
               int fd_nvme;
               struct stat sb_nvme;
               
               if((fd_nvme= open(read_f, O_RDONLY)) == -1){
                   fprintf(stderr, "NVMe file cannot be opened\n");
                   return 1;
               }
               
               fstat(fd_nvme, &sb_nvme);
               printf("reading second file %s\n", read_f);
               
               map_nvme = mmap(NULL, sb_nvme.st_size, PROT_READ, MAP_SHARED, fd_nvme, 0);
               
               if((map_nvme == (void*)-1)){
                       fprintf(stderr,"Input file map failed %d\n",map_nvme);
                       return 1;
               }
               
               size_t nvme_sz = sb_nvme.st_size - settings.ifileoffset; 
               if(orig_sz != nvme_sz){
                   fprintf(stderr,"Orig and NVMe file are of different sizes: orig: %llu and nvme: %llu\n Continuing...\n",orig_sz, nvme_sz);
                   return 1;
               }
               cuda_err_chk(cudaMallocManaged((void**)&nvme_h, nvme_sz)); 
               cuda_err_chk(cudaMemAdvise(nvme_h, nvme_sz, cudaMemAdviseSetAccessedBy, 0));
  //             cuda_err_chk(cudaMemset(nvme_h, 0, nvme_sz)); 
               memcpy(nvme_h, map_nvme+settings.ifileoffset, nvme_sz);
               printf("Launching verification kernel");
                
               
//                for(int ver=0; ver<100; ver++){
//                        printf("id:%u \t orig: %x \t nvme: %x\n", (uint64_t)ver, (uint8_t)(orig_h[ver] & 0xFF), (uint8_t)((nvme_h[ver])&0xFF));
//                }
//

               //cuda_err_chk(cudaMallocManaged((void**)&result_h, orig_sz)); 
               //cuda_err_chk(cudaMemset((void**)&result_h, 0, orig_sz)); 
               verify_kernel<<<g_size, b_size>>>(orig_h,nvme_h, orig_sz, n_threads); 
               cuda_err_chk(cudaDeviceSynchronize());
               std::cout<< "Completed. Check for mismatches" <<std::endl;

        }

 
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
