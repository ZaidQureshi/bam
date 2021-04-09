#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#ifndef __device__ 
#define __device__
#endif 
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__  
#define __forceinline__ inline
#endif


//#ifndef __CUDACC__
//#define __device__
//#define __host__
//#define __forceinline__ inline
//#endif

#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include <simt/atomic>
#define LOCKED   1
#define UNLOCKED 0

__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    return (x >> y) * 2;  // (x/2^y) *2
}



inline __device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, simt::memory_order_relaxed) & (65535);
        //printf("in thread: %p\n", (void*) ((sq->cid)+id));
        uint64_t old = sq->cid[id].val.exchange(LOCKED, simt::memory_order_acquire);
        not_found = old == LOCKED;
    } while (not_found);


    return id;



}

inline __device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, simt::memory_order_release);
}

inline __device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail) {
    uint32_t count = 0;




    bool pass = true;
    while (pass ) {
        pass = (q->tail_mark[(cur_tail+count++)&q->qs_minus_1].val.exchange(UNLOCKED, simt::memory_order_acq_rel)) == LOCKED;

    }
    return (count-1);
}

inline __device__
uint32_t move_head_cq(nvm_queue_t* q, uint32_t cur_head) {
    uint32_t count = 0;


    bool pass = true;
    //uint32_t old_head;
    while (pass) {
        uint64_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_acq_rel)) == LOCKED;



    }
    return (count-1);

}

inline __device__
uint32_t move_head_sq(nvm_queue_t* q, uint32_t in_cur_head) {
    uint32_t count = 0;
    uint32_t cur_head = q->head.load(simt::memory_order_acquire);

    bool pass = true;
    //uint32_t old_head;
    while (pass) {
        count++;

        uint64_t loc = (cur_head)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_acq_rel)) == LOCKED;
        if (pass) {
            cur_head = q->head.fetch_add(1, simt::memory_order_acq_rel);
            q->tickets[loc].val.fetch_add(2, simt::memory_order_release);

        }


    }
    return (count-1);

}
inline __device__
uint16_t sq_enqueue(nvm_queue_t* sq, nvm_cmd_t* cmd) {

    //uint32_t mask = __activemask();
    //uint32_t active_count = __popc(mask);
    //uint32_t leader = __ffs(mask) - 1;
    //uint32_t lane = lane_id();
    uint32_t ticket;
    ticket = sq->in_ticket.fetch_add(1, simt::memory_order_acquire);
    /* if (lane == leader) { */
    /*     ticket = sq->in_ticket.fetch_add(active_count, simt::memory_order_acquire); */
    /* } */

    /* ticket = __shfl_sync(mask, ticket, leader); */
    /* ticket += __popc(mask & ((1 << lane) - 1)); */

    uint32_t pos = ticket & (sq->qs_minus_1);
    uint64_t id = get_id(ticket, sq->qs_log2);

    //uint64_t k = 0;
    while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id) ) {
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(simt::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(simt::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(simt::memory_order_acquire) & (sq->qs_minus_1)));
                   }*/
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(100);
#endif
    }

    while (((pos+1) & sq->qs_minus_1) == (sq->head.load(simt::memory_order_acquire) & (sq->qs_minus_1))) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(100);
#endif
    }

    ulonglong4* queue_loc = ((ulonglong4*)(sq->vaddr)) + pos;
    //printf("+++tid: %llu\tcid: %llu\tsq_loc: %llx\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) (cmd->dword[0] >> 16), (uint64_t) queue_loc);

    //printf("sq->loc: %p\n", queue_loc);
    queue_loc[0] =   *((ulonglong4*) (cmd->dword+0));
    queue_loc[1] =   *((ulonglong4*) (cmd->dword+8));
    //queue_loc->dword[0] = cmd->dword[0];
    //queue_loc->dword[1] = cmd->dword[1];
    //queue_loc->dword[6] = cmd->dword[6];
    //queue_loc->dword[7] = cmd->dword[7];

    //*((ulonglong4*) (queue_loc->dword+8)) =   *((ulonglong4*) (cmd->dword+8));
    //queue_loc->dword[8] = cmd->dword[8];
    //queue_loc->dword[9] = cmd->dword[9];
    //queue_loc->dword[10] = cmd->dword[10];
    //queue_loc->dword[11] = cmd->dword[11];
    //queue_loc->dword[12] = cmd->dword[12];

/* #pragma unroll */
/*     for (uint32_t i = 0; i < 16; i++) { */
/*         queue_loc->dword[i] = cmd->dword[i]; */
/*     } */




    //uint32_t new_tail = pos;
    /*
    bool proceed = false;
    do {

        uint32_t cur_head = sq->head.load(simt::memory_order_acquire) & (sq->qs_minus_1);

        uint32_t check = (cur_head - 1)  & (sq->qs_minus_1);
        //uint32_t cur_head_mod = cur_head & (sq->qs_minus_1);
        //uint32_t size = (cur_head > new_tail) ? (sq->qs - cur_head + new_tail) : (new_tail - cur_head);
        proceed = check != pos;
        printf("here pos: %llu\n", (unsigned long long) pos);
    } while(!proceed);
    */
    //sq->tickets[pos].val.store(id + 1, simt::memory_order_release);
    sq->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);
    bool cont = true;
    while(cont) {
        cont = sq->tail_mark[pos].val.load(simt::memory_order_acquire) == LOCKED;
        if (cont) {
            /* cont = sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acq_rel) == LOCKED; */
            /* if(!cont) { */
                uint32_t cur_tail = sq->tail.load(simt::memory_order_acquire);

                uint32_t tail_move_count = move_tail(sq, cur_tail);

                if (tail_move_count) {
                    uint32_t new_tail = cur_tail + tail_move_count;
                    uint32_t new_db = (new_tail) & (sq->qs_minus_1);
                    *(sq->db) = new_db;
                    //printf("wrote sq_db: %llu\tsq_tail: %llu\tsq_head: %llu\n", (unsigned long long) new_db, (unsigned long long) (new_tail),  (unsigned long long)(sq->head.load(simt::memory_order_acquire)));
                    sq->tail.store(new_tail, simt::memory_order_release);
                }
                //sq->tail_lock.store(UNLOCKED, simt::memory_order_release);
            //}
        }
    }



    return pos;

}

inline __device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {

    sq->head_mark[pos].val.store(LOCKED, simt::memory_order_release);
    bool cont = true;
    while (cont) {
        cont = sq->head_mark[pos].val.load(simt::memory_order_acquire) == LOCKED;
        if (cont) {
            cont = sq->head_lock.exchange(LOCKED, simt::memory_order_acq_rel) == LOCKED;
            if (!cont){
                uint32_t cur_head = sq->head.load(simt::memory_order_acquire);;

                uint32_t head_move_count = move_head_sq(sq, cur_head);
                (void) head_move_count;
                //printf("sq head_move_count: %llu\n", (unsigned long long) head_move_count);
                /* if (head_move_count) { */
                /*     uint32_t new_head = cur_head + head_move_count; */
                /*     //printf("sq new_head: %llu\n", (unsigned long long) new_head); */
                /*     sq->head.store(new_head, simt::memory_order_release); */
                /* } */
                sq->head_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }
    }



}

inline __device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid) {
    uint64_t j = 0;
    //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("---tid: %llu\tcid: %llu\tcq_start: %llx\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) (search_cid), (uint64_t) cq->vaddr);
    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_acquire);

        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((volatile nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
            /* if (j % 100 == 0) */

            /*     printf("qs_log2: %llu\thead: %llu\tcur_head: %llu\tsearch_cid: %llu\tsearch_phase: %llu\tcq->loc: %p\tcq->qs: %llu\ti: %llu\tj: %llu\tcid: %llu\tphase:%llu\tmark: %llu\n", */
            /*            (unsigned long long) cq->qs_log2, */
            /*            (unsigned long long)head, (unsigned long long) cur_head, (unsigned long long) search_cid, (unsigned long long) search_phase, ((volatile nvm_cpl_t*)cq->vaddr)+loc, */
            /*            (unsigned long long) cq->qs, (unsigned long long) i, (unsigned long long) j, (unsigned long long) cid, (unsigned long long) phase, */
            /*            (unsigned long long) cq->head_mark[loc].val.load(simt::memory_order_acquire)); */

            if ((cid == search_cid) && (phase == search_phase)){
                 //if ((cpl_entry >> 17) != 0)
                      //printf("NVM Error: %llx\tcid: %llu\n", (unsigned long long) (cpl_entry >> 17), (unsigned long long) search_cid);
                return loc;
            }
            if (phase != search_phase)
                break;
            //__nanosleep(1000);
        }
        j++;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(100);
#endif
    }
}

inline __device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos) {

    //uint32_t pos = cq_poll(cq, cid);

    cq->head_mark[pos].val.store(LOCKED, simt::memory_order_release);
    bool cont = true;
    while (cont) {
        cont = cq->head_mark[pos].val.load(simt::memory_order_acquire) == LOCKED;
        if (cont) {
            /* cont = cq->head_lock.fetch_or(LOCKED, simt::memory_order_acq_rel) == LOCKED; */
            /* if (!cont) { */
                uint32_t cur_head = cq->head.load(simt::memory_order_acquire);;

                uint32_t head_move_count = move_head_cq(cq, cur_head);
                //printf("cq head_move_count: %llu\n", (unsigned long long) head_move_count);

                if (head_move_count) {
                    uint32_t new_head = cur_head + head_move_count;

                    uint32_t new_db = (new_head) & (cq->qs_minus_1);

                    *(cq->db) = new_db;
                    //printf("wrote cq_db: %llu\tcq_head: %llu\tcq_tail: %llu\n", (unsigned long long) new_db, (unsigned long long) (new_head),  (unsigned long long)(cq->tail.load(simt::memory_order_acquire)));
                    cq->head.store(new_head, simt::memory_order_release);
                }
                //cq->head_lock.store(UNLOCKED, simt::memory_order_release);
            //}
        }
    }



}

//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __NVM_PARALLEL_QUEUE_H_
