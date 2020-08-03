#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#include "nvm_types.h"
#include <simt/atomic>
#define LOCKED   1
#define UNLOCKED 0

__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    return (x >> y) * 2;
}

__forceinline__ __device__ uint32_t lane_id()
{
    uint32_t ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, simt::memory_order_acquire) & (65535);
        printf("id: %u\n", (unsigned int) id);
        uint64_t old = sq->cid[id].val.exchange(LOCKED, simt::memory_order_acquire);
        not_found = old == LOCKED;
    } while (not_found);


    return id;



}

__device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, simt::memory_order_acquire);
}

__device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail, uint32_t pos) {
    uint32_t count = 0;


    q->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool pass = true;
    while (pass) {
        pass = (q->tail_mark[(cur_tail+count++)&q->qs_minus_1].val.fetch_and(UNLOCKED, simt::memory_order_acquire)) == LOCKED;

    }
    return (count == 0) ? 0 : (count-1);
}

__device__
uint32_t move_head(nvm_queue_t* q, uint32_t cur_head, uint32_t pos, bool is_sq) {
    uint32_t count = 0;


    q->head_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool pass = true;
    while (pass) {
        uint64_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.fetch_and(UNLOCKED, simt::memory_order_acquire)) == LOCKED;
        if (pass && is_sq)
            q->tickets[loc].val.fetch_or(1, simt::memory_order_release);

    }
    return (count == 0) ? 0 : (count-1);

}
__device__
uint16_t sq_enqueue(nvm_queue_t* sq, nvm_cmd_t* cmd) {

    uint32_t mask = __activemask();
    uint32_t active_count = __popc(mask);
    uint32_t leader = __ffs(mask) - 1;
    uint32_t lane = lane_id();
    uint32_t ticket;
    if (lane == leader) {
        ticket = sq->in_ticket.fetch_add(active_count, simt::memory_order_acquire);
    }

    ticket = __shfl_sync(mask, ticket, leader);
    ticket += __popc(mask & ((1 << lane) - 1));

    uint32_t pos = ticket & (sq->qs_minus_1);
    uint64_t id = get_id(ticket, sq->qs_log2);

    while (sq->tickets[pos].val.load(simt::memory_order_acquire) != id)
        __nanosleep(100);

    volatile nvm_cmd_t* queue_loc = ((volatile nvm_cmd_t*)(sq->vaddr)) + pos;
    for (uint32_t i = 0; i < 16; i++)
        queue_loc->dword[i] = cmd->dword[i];


    sq->tickets[pos].val.store(id + 1, simt::memory_order_release);


    uint32_t cur_tail = sq->tail.load(simt::memory_order_acquire);

    uint32_t tail_move_count = move_tail(sq, cur_tail, pos);

    if (tail_move_count) {
        uint32_t new_tail = cur_tail + tail_move_count;
        uint32_t new_db = (new_tail) & (sq->qs_minus_1);
        uint32_t cur_head = sq->head.load(simt::memory_order_acquire);
        uint32_t cur_head_mod = cur_head & (sq->qs_minus_1);
        

        while (new_db == cur_head_mod) {
            __nanosleep(100);
            cur_head = sq->head.load(simt::memory_order_acquire);
            cur_head_mod = cur_head & (sq->qs_minus_1);

        }

        *(sq->db) = new_db;
        sq->tail.store(new_tail, simt::memory_order_release);
    }

    return pos;

}

__device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {


    uint32_t cur_head = sq->head.load(simt::memory_order_acquire);;

    uint32_t head_move_count = move_head(sq, cur_head, pos, true);

    if (head_move_count) {
        uint32_t new_head = cur_head + head_move_count;

        sq->head.store(new_head, simt::memory_order_release);
    }


}

__device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid) {
    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_acquire);

        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ~((cur_head >> cq->qs_log2) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((volatile nvm_cpl_t*)cq->vaddr)[loc].dword[0];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;
            if ((cid == search_cid) && (phase == search_phase)){
                return loc;
            }
        }

        __nanosleep(100);
    }
}

__device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos) {

    //uint32_t pos = cq_poll(cq, cid);



    uint32_t cur_head = cq->head.load(simt::memory_order_acquire);;

    uint32_t head_move_count = move_head(cq, cur_head, pos, false);

    if (head_move_count) {
        uint32_t new_head = cur_head + head_move_count;

        uint32_t new_db = (new_head) & (cq->qs_minus_1);
        *(cq->db) = new_db;
        cq->head.store(new_head, simt::memory_order_release);
    }

}

#endif // __NVM_PARALLEL_QUEUE_H_
