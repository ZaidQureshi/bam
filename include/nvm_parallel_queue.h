#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#include "nvm_types.h"
#include "nvm_util.h"
#include <simt/atomic>
#define LOCKED   1
#define UNLOCKED 0

__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    return (x >> y) * 2;  // (x/2^y) *2
}



__device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, simt::memory_order_acquire) & (65535);
        //printf("in thread: %p\n", (void*) ((sq->cid)+id));
        uint64_t old = sq->cid[id].val.exchange(LOCKED, simt::memory_order_acquire);
        not_found = old == LOCKED;
    } while (not_found);


    return id;



}

__device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, simt::memory_order_release);
}

__device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail, uint32_t pos) {
    uint32_t count = 0;


    q->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool pass = true;
    while (pass && (count <= q->qs_minus_1)) {
        pass = (q->tail_mark[(cur_tail+count++)&q->qs_minus_1].val.fetch_and(UNLOCKED, simt::memory_order_acquire)) == LOCKED;

    }
    return (count-1);
}

__device__
uint32_t move_head(nvm_queue_t* q, uint32_t cur_head, uint32_t pos, bool is_sq) {
    uint32_t count = 0;


    q->head_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool pass = true;
    while (pass) {
        uint64_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.fetch_and(UNLOCKED, simt::memory_order_acquire)) == LOCKED;
        if (pass && is_sq) {
            q->tickets[loc].val.fetch_add(1, simt::memory_order_release);
            nvm_cmd_t* queue_loc = ((nvm_cmd_t*)(q->vaddr)) + loc;
            uint16_t cid = queue_loc->dword[0] >> 16;
            put_cid(q, cid);
        }

    }
    return (count-1);

}
__device__
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

    uint64_t k = 0;
    while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id) || (((pos+1) & sq->qs_minus_1) == (sq->head.load(simt::memory_order_acquire) & (sq->qs_minus_1)))) {
        /*if (k++ % 100 == 0)   {
            printf("tid: %llu\tpos: %llu\tticket: %llu\tid: %llu\ttickets_pos: %llu\tqueue_head: %llu\tqueue_tail: %llu\n",
                   (unsigned long long) threadIdx.x, (unsigned long long) pos,
                   (unsigned long long)ticket, (unsigned long long)id, (unsigned long long) (sq->tickets[pos].val.load(simt::memory_order_acquire)),
                   (unsigned long long)(sq->head.load(simt::memory_order_acquire) & (sq->qs_minus_1)), (unsigned long long)(sq->tail.load(simt::memory_order_acquire) & (sq->qs_minus_1)));
        }*/
        __nanosleep(100);
    }

    volatile nvm_cmd_t* queue_loc = ((volatile nvm_cmd_t*)(sq->vaddr)) + pos;

    //printf("sq->loc: %p\n", queue_loc);
#pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        queue_loc->dword[i] = cmd->dword[i];
    }




    uint32_t new_tail = pos;
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
    sq->tickets[pos].val.store(id + 1, simt::memory_order_release);

    uint32_t cur_tail = sq->tail.load(simt::memory_order_acquire);

    uint32_t tail_move_count = move_tail(sq, cur_tail, pos);

    if (tail_move_count) {
        uint32_t new_tail = cur_tail + tail_move_count;
        uint32_t new_db = (new_tail) & (sq->qs_minus_1);
        /*
        uint32_t cur_head = sq->head.load(simt::memory_order_acquire);
        uint32_t cur_head_mod = cur_head & (sq->qs_minus_1);
        

        while (new_db == cur_head_mod) {
            __nanosleep(100);
            cur_head = sq->head.load(simt::memory_order_acquire);
            cur_head_mod = cur_head & (sq->qs_minus_1);

        }
        */

        *(sq->db) = new_db;
        //printf("wrote sq_db: %llu\tsq_head: %llu\n", (unsigned long long) new_db,  (unsigned long long)(sq->head.load(simt::memory_order_acquire)));
        sq->tail.store(new_tail, simt::memory_order_release);
    }

    return pos;

}

__device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {


    uint32_t cur_head = sq->head.load(simt::memory_order_acquire);;

    uint32_t head_move_count = move_head(sq, cur_head, pos, true);
     //printf("sq head_move_count: %llu\n", (unsigned long long) head_move_count);
    if (head_move_count) {
        uint32_t new_head = cur_head + head_move_count;
        //printf("sq new_head: %llu\n", (unsigned long long) new_head);
        sq->head.store(new_head, simt::memory_order_release);
    }


}

__device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid) {
    uint64_t j = 0;
    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_acquire);

        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((volatile nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
            if (j % 100 == 0)
                printf("qs_log2: %llu\thead: %llu\tcur_head: %llu\tsearch_cid: %llu\tsearch_phase: %llu\tcq->loc: %p\tcq->qs: %llu\ti: %llu\tj: %llu\tcid: %llu\tphase:%llu\n",
                       (unsigned long long) cq->qs_log2,
                       (unsigned long long)head, (unsigned long long) cur_head, (unsigned long long) search_cid, (unsigned long long) search_phase, ((volatile nvm_cpl_t*)cq->vaddr)+loc,
                       (unsigned long long) cq->qs, (unsigned long long) i, (unsigned long long) j, (unsigned long long) cid, (unsigned long long) phase);
            if ((cid == search_cid) && (phase == search_phase)){
                /* if ((cpl_entry >> 17) != 0) */
                /*     printf("NVM Error!!!\n"); */
                return loc;
            }
            if (phase != search_phase)
                break;
        }
        j++;

        __nanosleep(100);
    }
}

__device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos) {

    //uint32_t pos = cq_poll(cq, cid);



    uint32_t cur_head = cq->head.load(simt::memory_order_acquire);;

    uint32_t head_move_count = move_head(cq, cur_head, pos, false);
    //printf("cq head_move_count: %llu\n", (unsigned long long) head_move_count);

    if (head_move_count) {
        uint32_t new_head = cur_head + head_move_count;

        uint32_t new_db = (new_head) & (cq->qs_minus_1);
        //printf("cq new_head: %llu\t new_db: %llu\n", (unsigned long long) new_head, (unsigned long long) new_db);
        *(cq->db) = new_db;
        cq->head.store(new_head, simt::memory_order_release);
    }

}

#endif // __NVM_PARALLEL_QUEUE_H_
