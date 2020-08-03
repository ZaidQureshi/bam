#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__

#include <nvm_types.h>
#include <cstdint>
#include "buffer.h"
#include "settings.h"
#include "ctrl.h"
#include <cuda.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_admin.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include "util.h"

using error = std::runtime_error;
using std::string;

struct QueuePairMeta {
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    BufferPtr           cq_tail_mark;
};

struct QueuePair
{
    uint32_t            pageSize;
    uint32_t            block_size;
    uint32_t            block_size_log;
    uint32_t            block_size_minus_1;
    uint32_t            nvmNamespace;
    //void*               prpList;
    //uint64_t*           prpListIoAddrs;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
    uint16_t            qp_id;
    QueuePairMeta*      meta;
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    BufferPtr           cq_tail_mark;




#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)

    void init_gpu_specific_struct(const Settings& settings) {
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->sq_cid = createBuffer(65536 * sizeof(padded_struct), settings.cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();
        this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
        std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
        this->sq.qs_minus_1 = this->sq.qs - 1;
        this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);


        this->cq_tickets = createBuffer(this->cq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->cq_tail_mark = createBuffer(this->cq.qs * sizeof(padded_struct), settings.cudaDevice);
        this->cq.tickets = (padded_struct*) this->cq_tickets.get();
        this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
        this->cq.tail_mark = (padded_struct*) this->cq_tail_mark.get();
        this->cq.qs_minus_1 = this->cq.qs - 1;
        this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);

    }

    

    QueuePair( const Controller& ctrl, const Settings& settings, const uint16_t qp_id)
    {
        //this->meta = (QueuePairMeta*) malloc(sizeof(QueuePairMeta));
        init_gpu_specific_struct(settings);
        std::cout << "in preparequeuepair: " << std::hex << this->sq.cid << std::endl;

        std::cout << "HERE\n";
        uint64_t cap = ((volatile uint64_t*) ctrl.ctrl->mm_ptr)[0];
        bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;

        uint64_t sq_size = (cqr) ?
            ((MAX_SQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) )) ? MAX_SQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) ) ) :
            ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) );
        uint64_t cq_size = (cqr) ?
            ((MAX_CQ_ENTRIES_64K <= ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) )) ? MAX_CQ_ENTRIES_64K :  ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) ) ) :
            ((((volatile uint16_t*) ctrl.ctrl->mm_ptr)[0] + 1) );

        bool sq_need_prp = (!cqr) || (sq_size > MAX_SQ_ENTRIES_64K);
        bool cq_need_prp = (!cqr) || (cq_size > MAX_CQ_ENTRIES_64K);

        size_t sq_mem_size =  sq_size * sizeof(nvm_cmd_t) + sq_need_prp*(64*1024);
        size_t cq_mem_size =  cq_size * sizeof(nvm_cpl_t) + cq_need_prp*(64*1024);

        std::cout << sq_size << "\t" << sq_mem_size << std::endl;
        //size_t queueMemSize = ctrl.info.page_size * 2;
        //size_t prpListSize = ctrl.info.page_size * settings.numThreads * (settings.doubleBuffered + 1);
        size_t prp_mem_size = sq_size * (4096) * 2;
        std::cout << "Started creating DMA\n";
        // qmem->vaddr will be already a device pointer after the following call
        this->sq_mem = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16), settings.cudaDevice);
        std::cout << "Finished creating sq dma\n";
        this->cq_mem = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16), settings.cudaDevice);
        //this->prp_mem = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_mem_size, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);
        std::cout << "Finished creating DMA\n";

        // Set members
        this->pageSize = ctrl.info.page_size;
        this->block_size = ctrl.ns.lba_data_size;
        this->block_size_minus_1 = ctrl.ns.lba_data_size-1;
        this->block_size_log = std::log2(ctrl.ns.lba_data_size);
        this->nvmNamespace = ctrl.ns.ns_id;

        //this->prpList = NVM_DMA_OFFSET(this->prp_mem, 0);
        //this->prpListIoAddrs = this->prp_mem->ioaddrs;
        this->qp_id = qp_id;

        if (cq_need_prp) {
            size_t iters = (size_t)ceil(((float)cq_size*sizeof(nvm_cpl_t))/((float)ctrl.ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl.ctrl->page_size);
                cpu_vaddrs[i] = this->cq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl.ctrl->page_size);
            }

            if (this->cq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->cq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->cq_mem.get()->vaddr = (void*)((uint64_t)this->cq_mem.get()->vaddr + 64*1024);

            free(cpu_vaddrs);
        }

        if (sq_need_prp) {
            size_t iters = (size_t)ceil(((float)sq_size*sizeof(nvm_cpl_t))/((float)ctrl.ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl.ctrl->page_size);
                cpu_vaddrs[i] = this->sq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl.ctrl->page_size);
            }

            if (this->sq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->sq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->sq_mem.get()->vaddr = (void*)((uint64_t)this->sq_mem.get()->vaddr + 64*1024);

            free(cpu_vaddrs);
        }

        // Create completion queue
        // (nvm_aq_ref ref, nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)
        int status = nvm_admin_cq_create(ctrl.aq_ref, &this->cq, qp_id, this->cq_mem.get(), 0, cq_size, cq_need_prp);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
        }

        // Get a valid device pointer for CQ doorbell
        void* devicePtr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) this->cq.db, 0);
        if (err != cudaSuccess)
        {
            throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->cq.db = (volatile uint32_t*) devicePtr;

        // Create submission queue
        //  nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp = false)
        status = nvm_admin_sq_create(ctrl.aq_ref, &this->sq, &this->cq, qp_id, this->sq_mem.get(), 0, sq_size, sq_need_prp);
        if (!nvm_ok(status))
        {
            throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
        }


        // Get a valid device pointer for SQ doorbell
        err = cudaHostGetDevicePointer(&devicePtr, (void*) this->sq.db, 0);
        if (err != cudaSuccess)
        {
            throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->sq.db = (volatile uint32_t*) devicePtr;
        std::cout << "Finish Making Queue\n";
        return;



    }

};
#endif
