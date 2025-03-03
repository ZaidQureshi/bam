#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <algorithm>
#include <cstdint>
#include "buffer.h"
#include "ctrl.h"
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include "nvm_admin.h"
#include "nvm_queue.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <functional>
#include <cmath>
#include "util.h"

#ifdef __GRAID__
#include "dmapool.h"
#endif

using error = std::runtime_error;
using std::string;

struct QueuePair;
typedef std::function<int(enum QueueType, const nvm_ctrl_t *, const uint32_t, nvm_aq_ref &, DmaPtr &, nvm_queue_t *, nvm_queue_t *, const uint16_t, const uint64_t, string &)> CreateQueueFunc;

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
    DmaPtr              sq_mem;
    DmaPtr              cq_mem;
    DmaPtr              prp_mem;
    BufferPtr           sq_tickets;
    //BufferPtr           sq_head_mark;
    BufferPtr           sq_tail_mark;
    BufferPtr           sq_cid;
    //BufferPtr           cq_tickets;
    BufferPtr           cq_head_mark;
    //BufferPtr           cq_tail_mark;
    BufferPtr           cq_pos_locks;
    //BufferPtr           cq_clean_cid;




#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)

    inline void init_gpu_specific_struct( const uint32_t cudaDevice) {
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        //this->sq_head_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_cid = createBuffer(65536 * sizeof(padded_struct), cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();
        //this->sq.head_mark = (padded_struct*) this->sq_head_mark.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
    //    std::cout << "init_gpu_specific: " << std::hex << this->sq.cid <<  std::endl;
        this->sq.qs_minus_1 = this->sq.qs - 1;
        this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);


        //this->cq_tickets = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq_tail_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        //this->cq.tickets = (padded_struct*) this->cq_tickets.get();
        this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
        //this->cq.tail_mark = (padded_struct*) this->cq_tail_mark.get();
        this->cq.qs_minus_1 = this->cq.qs - 1;
        this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);
        this->cq_pos_locks = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq.pos_locks = (padded_struct*) this->cq_pos_locks.get();

        //this->cq_clean_cid = createBuffer(this->cq.qs * sizeof(uint16_t), cudaDevice);
       // this->cq.clean_cid = (uint16_t*) this->cq_clean_cid.get();
    }

    static inline
    int nvm_create_queue(enum QueueType qt, const nvm_ctrl_t *ctrl, const uint32_t cudaDevice,
                         nvm_aq_ref &aq_ref, DmaPtr &q_mem,
                         nvm_queue_t *cq, nvm_queue_t *sq,
                         const uint16_t qp_id, const uint64_t queueDepth,
                         string &errmsg)
    {
        const bool cqr = ctrl->cqr;
        const uint32_t max_qs = (qt == QT_SQ) ? MAX_SQ_ENTRIES_64K : MAX_CQ_ENTRIES_64K;
        uint64_t q_size = (cqr) ? std::min(max_qs, ctrl->max_qs) : ctrl->max_qs;
        q_size = std::min(queueDepth, q_size);
        bool q_need_prp = false;//(!cqr) || (q_size > MAX_[SC]Q_ENTRIES_64K);
        const size_t q_item_size = (qt == QT_SQ) ? sizeof(nvm_cmd_t) : sizeof(nvm_cpl_t);

        const size_t q_mem_size = q_size * q_item_size + q_need_prp*(64*1024);
        q_mem = createDma(ctrl, NVM_PAGE_ALIGN(q_mem_size, 1UL << 16), cudaDevice);

        // Fill queue PRP list
        if (q_need_prp) {
            size_t iters = (size_t)ceil(((float)q_size*q_item_size)/((float)ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl->page_size);
                cpu_vaddrs[i] = q_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl->page_size);
            }

            if (q_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(q_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            q_mem.get()->vaddr = (void*)((uint64_t)q_mem.get()->vaddr + 64*1024);

            free(cpu_vaddrs);
        }

        // Issue create queue admin command
        int status;
        if (qt == QT_CQ) {
            status = nvm_admin_cq_create(aq_ref, cq, qp_id, q_mem.get(), 0, q_size, q_need_prp);
        } else {
            status = nvm_admin_sq_create(aq_ref, sq, cq, qp_id, q_mem.get(), 0, q_size, q_need_prp);
        }
        if (!nvm_ok(status))
        {
            errmsg = nvm_strerror(status);
            return EIO;
        }
        return 0;
    }

#ifdef __GRAID__
    static inline
    int graid_create_queue(enum QueueType qt, const nvm_ctrl_t *ctrl, const uint32_t cudaDevice,
                           nvm_aq_ref &aq_ref, DmaPtr &q_mem,
                           nvm_queue_t *cq, nvm_queue_t *sq,
                           const uint16_t qp_id, const uint64_t queueDepth,
                           string &errmsg)
    {
        UNUSED(cudaDevice);
        UNUSED(aq_ref);
        UNUSED(q_mem);
        UNUSED(queueDepth);
        void *q_vaddr = qt == QT_CQ ? graid_ctrl_cq_vaddr(ctrl, qp_id) :
                                      graid_ctrl_sq_vaddr(ctrl, qp_id);

	uint64_t q_ioaddr = qt == QT_CQ ? graid_ctrl_cq_ioaddr(ctrl, qp_id) :
                                          graid_ctrl_sq_ioaddr(ctrl, qp_id);

        nvm_queue_t *q = qt == QT_CQ ? cq : sq;

        if (nvm_queue_clear(q, ctrl, qt, qp_id, NVMeMaxDGQDepth, true, q_vaddr, q_ioaddr)) {
            errmsg = string("Graid device: Failed to initialize queue structure");
            return EIO;
        }

        printf("%s Queue %d: q_vaddr=%016lx, q_paddr=%016lx\n",
                qt == QT_CQ ? "Completion" : "Submission",
                qp_id, (uint64_t)q_vaddr, q_ioaddr);
        return 0;
    }
#endif

    inline QueuePair(const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, nvm_aq_ref& aq_ref, const uint16_t qp_id, const uint64_t queueDepth, CreateQueueFunc create_queue)
    {
        // Set members
        this->pageSize = info.page_size;
        this->block_size = ns.lba_data_size;

        this->block_size_minus_1 = ns.lba_data_size-1;
        this->block_size_log = std::log2(ns.lba_data_size);
//        std::cout << "block size: " << this->block_size << "\tblock_size_log: " << this->block_size_log << std::endl ;
        this->nvmNamespace = ns.ns_id;

        //this->prpList = NVM_DMA_OFFSET(this->prp_mem, 0);
        //this->prpListIoAddrs = this->prp_mem->ioaddrs;
        this->qp_id = qp_id;

        string errmsg;
        if (create_queue(QT_CQ, ctrl, cudaDevice, aq_ref, this->cq_mem, &this->cq, &this->sq, qp_id, queueDepth, errmsg)) {
            throw error(string("Failed to create submission queue: ") + errmsg);
        }

        if (create_queue(QT_SQ, ctrl, cudaDevice, aq_ref, this->sq_mem, &this->cq, &this->sq, qp_id, queueDepth, errmsg)) {
            throw error(string("Failed to create completion queue: ") + errmsg);
        }

//        std::cout << "Finish Making Queue\n";

        init_gpu_specific_struct(cudaDevice);
       // std::cout << "in preparequeuepair: " << std::hex << this->sq.cid << std::endl;
        return;



    }

};
#endif
