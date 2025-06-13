#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

#include "window_buffer.h"

#define FREE 2

#define no_reuse 0xFFFF000000000000

enum data_dist_t {REPLICATE = 0, STRIPE = 1};



#define ALL_CTRLS 0xffffffffffffffff

//broken
#define VALID_DIRTY (1ULL << 31)
#define USE_DIRTY (VALID_DIRTY | USE)

#define INVALID 0x00000000U
#define VALID   0x80000000U
#define BUSY    0x40000000U
#define DIRTY   0x20000000U
#define CNT_SHIFT (29ULL)
#define CNT_MASK 0x1fffffffU
#define VALID_MASK 0x7
#define BUSY_MASK 0xb
#define DISABLE_BUSY_ENABLE_VALID 0xc0000000U
#define DISABLE_BUSY_MASK 0xbfffffffU
#define NV_NB 0x00U
#define NV_B 0x01U
#define V_NB 0x02U
#define V_B 0x03U


struct page_cache_t;

struct page_cache_d_t;

//typedef padded_struct_pc* page_states_t;


template <typename T>
struct range_t;

template<typename T>
struct array_d_t;

template <typename T>
struct range_d_t;


typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  state;                                                      //
    uint32_t offset;

    simt::atomic<uint8_t,  simt::thread_scope_device> prefetch_count; 
    simt::atomic<uint8_t,  simt::thread_scope_device> prefetch_counter;
   // uint8_t pad[32-4-4];

    uint32_t cpu_feature_offset;

} __attribute__((aligned (32))) data_page_t;

typedef data_page_t* pages_t;

template<typename T>
struct returned_cache_page_t {
    T* addr;
    uint32_t size;
    uint32_t offset;

    T operator[](size_t i) const {
        if (i < size)
            return addr[i];
        else
            return 0;
    }

    T& operator[](size_t i) {
        if (i < size)
            return addr[i];
        else
            return addr[0];
    }
};
#define THREAD_ 0
#define SHARED_ 1
#define GLOBAL_ 2

//#ifdef __CUDACC__
#define TID ( (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)))
//#else
//#define TID 0
//#endif

//#ifdef __CUDACC__
#define BLKSIZE ( (blockDim.x * blockDim.y * blockDim.z) )
//#else
//#define BLKSIZE 1
//#endif

#ifdef __CUDACC__
#define SYNC (loc != THREAD_ ? __syncthreads() : (void)0)
#else
#define SYNC (void)0
#endif




#define INVALID_ 0x00000000
#define VALID_ 0x80000000
#define BUSY_ 0x40000000
#define CNT_MASK_ 0x3fffffff
#define INVALID_MASK_ 0x7fffffff
#define DISABLE_BUSY_MASK_ 0xbfffffff





template<simt::thread_scope _scope = simt::thread_scope_device>
struct tlb_entry {
    uint64_t global_id;
    simt::atomic<uint32_t, _scope> state;
    data_page_t* page = nullptr;

    __forceinline__
    __host__ __device__
    tlb_entry() { init(); }

    __forceinline__
    __host__ __device__
    void init() {
        global_id = 0;
        state.store(0, simt::memory_order_relaxed);
        page = nullptr;
    }

    __forceinline__
    __device__
    void release(const uint32_t count) {
//		    if (global_id == 515920192)
//			printf("--(2)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) count);

        state.fetch_sub(count, simt::memory_order_release); }

    __forceinline__
    __device__
    void release() { if (page != nullptr)  {
//		    if (global_id == 515920192)
//			printf("--(1)st: %llx\tcount: %llu\n", (unsigned long long) state.load(simt::memory_order_relaxed), (unsigned long long) 1);

            page->state.fetch_sub(1, simt::memory_order_release); }}



};




template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct tlb {
    tlb_entry<_scope> entries[n];
    array_d_t<T>* array = nullptr;

    __forceinline__
    __host__ __device__
    tlb() {}

    __forceinline__
    __device__
    void init(array_d_t<T>* a) {
        //SYNC;
        if (n) {
            size_t tid = TID;
            if (tid == 0)
                array = a;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].init();
        }

    }

    __forceinline__
    __device__
    void fini() {


        if (n) {
            size_t tid = TID;
            for (; tid < n; tid+=BLKSIZE)
                entries[tid].release();
        }

    }

    __forceinline__
    __device__
    ~tlb() {  }

    __forceinline__
    __device__
    T* acquire(const size_t i, const size_t gid, size_t& start, size_t& end, range_d_t<T>* range, const size_t page_) {

        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);
        uint64_t base_master, base;
        if (lane == master) {
            uint64_t c = 0;
            bool cont = false;
            uint32_t st;
            do {

                //lock;
                do {
                    st = entry->state.fetch_or(VALID_, simt::memory_order_acquire);
                    if ((st & VALID_) == 0)
                        break;
                    __nanosleep(100);
                } while (true);

                if ((entry->page != nullptr) && (gid == entry->global_id)) {
//		    if (gid == 515920192)
//			printf("++(1)st: %llx\tst&Val: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) count);

                    st += count;

                    base_master = (uint64_t) range->get_cache_page_addr(entry->page->offset);

                    entry->state.store(st, simt::memory_order_release);
                    break;
                }
                else if(((entry->page == nullptr)) || (((st & 0x3fffffff) == 0))) {
//		    if (gid == 515920192)
//			printf("++(2)st: %llx\tst&Val: %llx\tVal: %llx\tcount: %llu\n", (unsigned long long) st, (unsigned long long) (st & VALID_), (unsigned long long) VALID_, (unsigned long long) count);
//
                    if (entry->page != nullptr)
                        entry->page->state.fetch_sub(1, simt::memory_order_release);
                    data_page_t* page = nullptr;// = (data_page_t*)0xffffffffffffffff;
                    base_master = (uint64_t) array->acquire_page_(i, page, start, end, range, page_);
                    if (((uint64_t) page == 0xffffffffffffffff) || (page == nullptr))
                        printf("failure\n");
                    entry->page = page;
                    entry->global_id = gid;
                    st += count;
                    entry->state.store(st, simt::memory_order_release);
                    break;

                }
                else {
                    if (++c % 100000 == 0)
                        printf("c: %llu\ttid: %llu\twanted_gid: %llu\tgot_gid: %llu\tst: %llx\tst&0x7: %llx\n", (unsigned long long) c, (unsigned long long) (TID), (unsigned long long) gid, (unsigned long long) entry->global_id, (unsigned long long) st, (unsigned long long) (st & 0x7fffffff));
                    entry->state.store(st, simt::memory_order_relaxed);
                    __nanosleep(100);

                }

            } while(true);


        }

        base_master = __shfl_sync(eq_mask,  base_master, master);

        return (T*) base_master;

    }

    __forceinline__
    __device__
    void release(const size_t gid) {
        //size_t gid = array->get_page_gid(i);
        uint32_t lane = lane_id();
        uint32_t mask = __activemask();
        uint32_t eq_mask = __match_any_sync(mask, gid);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        uint32_t master = __ffs(eq_mask) - 1;
        uint32_t count = __popc(eq_mask);

        size_t ent = gid % n;
        tlb_entry<_scope>* entry = entries + ent;
        if (lane == master)
            entry->release(count);
        __syncwarp(eq_mask);

    }

};

template<typename T, size_t n = 32, simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
struct bam_ptr_tlb {
    tlb<T,n,_scope,loc>* tlb_ = nullptr;
    array_d_t<T>* array = nullptr;
    range_d_t<T>* range;
    size_t page;
    size_t start = 0;
    size_t end = 0;
    size_t gid = 0;
    //int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr_tlb(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { init(a, t); }

    __forceinline__
    __device__
    ~bam_ptr_tlb() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a, tlb<T,n,_scope,loc>* t) { array = a; tlb_ = t; }

    __forceinline__
    __device__
    void fini(void) {
        if (addr) {

            tlb_->release(gid);
            addr = nullptr;
        }

    }

    __forceinline__
    __device__
    void update_page(const size_t i) {
      fini(); //destructor
        array->get_page_gid(i, range, page, gid);
        addr = (T*) tlb_->acquire(i, gid, start, end, range, page);
    }

    __forceinline__
    __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
            update_page(i);
        }
        return addr[i-start];
    }

    __forceinline__
    __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            range->mark_page_dirty(page);
        }
        return addr[i-start];
    }
};


template<typename T>
struct bam_ptr {
    data_page_t* page = nullptr;
    array_d_t<T>* array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    bam_ptr(array_d_t<T>* a) { init(a); }

    __forceinline__
    __host__ __device__
    ~bam_ptr() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a) { array = a; }

    __forceinline__
    __host__ __device__
    void fini(void) {
        if (page) {
            array->release_page(page, range_id, start);
            page = nullptr;
        }
    }

    __forceinline__
    __host__ __device__
    T* update_page(const size_t i) {

        fini(); //destructor
        addr = (T*) array->acquire_page(i, page, start, end, range_id);

        return addr;
    }




    __forceinline__    
    __host__ __device__ 
    void set_prefetch_val(const size_t i, const size_t prefetch_val){
	array -> set_prefetching(i, 2);
    }
    
    __forceinline__    
    __host__ __device__ 
    void set_window_buffer_counter(const size_t i, const size_t prefetch_val){
	array -> set_window_buffer_counter(i, 1);

    }

    __forceinline__
    __host__ __device__
    T operator[](const size_t i) const {
	if ((i < start) || (i >= end)) {
	   	T* tmpaddr =  update_page(i);
        }
        return addr[i-start];
    }

    __forceinline__
    __host__ __device__
    T read(size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);  // no page->state.fetch_or
        }
        return addr[i - start];
    }

   
    __host__ __device__
    T* memref(size_t i) {
        T* ret_; 
        if ((i < start) || (i >= end)) {
           ret_ =  update_page(i);
        }
        return ret_;
    }


    __forceinline__
    __host__ __device__
    T& operator[](const size_t i) {
	if ((i < start) || (i >= end)) {
            update_page(i);
            page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        return addr[i-start];
    }
 };


template<typename T>
struct gids_ptr {
    data_page_t* page = nullptr;
    array_d_t<T>* array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    T* addr = nullptr;

    __forceinline__
    __host__ __device__
    gids_ptr(array_d_t<T>* a) { init(a); }

    __forceinline__
    __host__ __device__
    ~gids_ptr() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a) { array = a; }

    __forceinline__
    __host__ __device__
    void fini(void) {
        if (page) {
            array->release_page(page, range_id, start);
            page = nullptr;
        }
    }

    __forceinline__
    __host__ __device__
    T* update_page(const size_t i) {

    fini(); //destructor
    addr = (T*) array->acquire_page(i, page, start, end, range_id);

     return addr;
    }


    __forceinline__    
    __host__ __device__ 
    void set_prefetch_val(const size_t i, const size_t prefetch_val){
	array -> set_prefetching(i, 2);
    }
    
    __forceinline__    
    __host__ __device__ 
    void set_window_buffer_counter(const size_t i, const size_t prefetch_val){
	array -> set_window_buffer_counter(i, 1);

    }

    __forceinline__
    __host__ __device__
    T operator[](const size_t i) const {
	if ((i < start) || (i >= end)) {
	   	T* tmpaddr =  update_page(i);
        }
        return addr[i-start];
    }
    
    __host__ __device__
    T* memref(size_t i) {
        T* ret_; 
        if ((i < start) || (i >= end)) {
           ret_ =  update_page(i);
        }
        return ret_;
    }


    __forceinline__
    __host__ __device__
    T& operator[](const size_t i) {
	if ((i < start) || (i >= end)) {
            update_page(i);
            page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        return addr[i-start];
    }
};


template<typename T>
struct wb_bam_ptr {
    data_page_t* page = nullptr;
    array_d_t<T>* array = nullptr;
    size_t start = 0;
    size_t end = 0;
    int64_t range_id = -1;
    T* addr = nullptr;
    
    uint32_t* wb_queue_counter;
    uint32_t  wb_depth;
    T* queue_ptr;
    uint64_t* wb_id_array;
    uint32_t q_depth;

    uint8_t time_step;
    uint32_t head_ptr;

    __forceinline__
    __host__ __device__
    wb_bam_ptr(array_d_t<T>* a) { init(a); }

    __forceinline__
    __host__ __device__
    ~wb_bam_ptr() { fini(); }

    __forceinline__
    __host__ __device__
    void init(array_d_t<T>* a) { array = a; }

    __forceinline__
    __host__ __device__
    void fini(void) {
        if (page) {
            array->release_page(page, range_id, start);
            page = nullptr; 
        }
    }
    
    __forceinline__
    __host__ __device__
    void flush_wb_counter(uint64_t page ){
        array -> flush_wb_counter(page);
    }

     __forceinline__
    __host__ __device__
    void count_mask(uint64_t page, uint64_t* mask_counter){
        array -> count_mask(page,mask_counter);
    }

     __forceinline__
    __host__ __device__
    uint64_t get_page_id(uint64_t page ){
        return (array -> get_page_id(page));
    }



    __forceinline__
    __host__ __device__
    void set_wb(uint32_t* wb_q, uint32_t wb_d, T* pinned_ptr, uint64_t* id_array, uint32_t q_d){
        wb_queue_counter = wb_q;
        wb_depth = wb_d;
        queue_ptr = pinned_ptr;
        wb_id_array = id_array;
        q_depth = q_d;
    }

    __forceinline__
    __host__ __device__
    void set_time(uint8_t cur_time, uint32_t cur_head){
        time_step = cur_time;
        head_ptr = cur_head;
    }

    
    __forceinline__
    __host__ __device__
    void update_wb(uint64_t page_id, uint32_t reuse_time, uint64_t idx){
        array->wb_update_wb(page_id, reuse_time, idx);
        return;
    }

    __forceinline__
    __host__ __device__
    void update_wb_list(uint64_t page_id, uint64_t reuse_time, uint64_t idx){
        array->wb_update_wb_list(page_id, reuse_time, idx);
        return;
    }

     __forceinline__
    __host__ __device__
    void update_wb_test(uint64_t page_id, uint32_t reuse_time, uint64_t idx){
        //printf("page id: %lu reuse_time :%lu\n", (unsigned long) page_id, (unsigned long) reuse_time);
        array->wb_update_wb(page_id, reuse_time, idx);
        //printf("done page id: %lu reuse_time :%lu\n", (unsigned long) page_id, (unsigned long) reuse_time);

        return;
    }
    
    __forceinline__
    __host__ __device__
    uint32_t check_reuse_val(uint32_t page_id){
        auto val = array->wb_check_reuse_val(page_id);
        return val;
    }
    

    __forceinline__
    __host__ __device__
    T* update_page(const size_t i) {
    fini(); //destructor
    addr = (T*) array->wb_acquire_page(i, page, start, end, range_id, 
                                       wb_queue_counter, wb_depth, queue_ptr, wb_id_array, q_depth,
                                       time_step, head_ptr);
     return addr;
    }
   

    __forceinline__
    __host__ __device__
    T operator[](const size_t i) const {
        if ((i < start) || (i >= end)) {
            T* tmpaddr =  update_page(i);
        }

        return addr[i-start];
    }
    
    __host__ __device__
    T* memref(size_t i) {
        T* ret_; 
        if ((i < start) || (i >= end)) {
           ret_ =  update_page(i);
        }
        return ret_;
    }


    __forceinline__
    __host__ __device__
    T& operator[](const size_t i) {
        if ((i < start) || (i >= end)) {
            update_page(i);
            page->state.fetch_or(DIRTY, simt::memory_order_relaxed);
        }
        return addr[i-start];
    }
};


typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock; //state
    uint64_t page_translation;
    uint64_t next_reuse;
    //uint64_t reuse_mask;
    uint64_t reuse_chunk[8];
} __attribute__((aligned (32))) cache_page_t;

typedef struct __align__(32) {
    simt::atomic<uint32_t, simt::thread_scope_device>  page_take_lock; //state
    uint64_t page_translation;
    uint64_t next_reuse;
} __attribute__((aligned (32))) wb_cache_page_t;



struct page_cache_d_t {
    uint8_t* base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    cache_page_t* cache_pages;

    padded_struct_pc* page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    uint64_t    ctrl_page_size;
    uint64_t  range_cap;
    pages_t*   ranges;
    pages_t*   h_ranges;
    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;
    uint64_t n_cachelines_for_states;

    uint64_t* ranges_page_starts;
    data_dist_t* ranges_dists;
    simt::atomic<uint64_t, simt::thread_scope_device>* ctrl_counter;

    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;

    Controller** d_ctrls;
    uint64_t n_ctrls;
    bool prps;

    uint64_t n_blocks_per_page;

    //window buffer ptrs
    bool cpu_agg;

    uint64_t* cpu_agg_meta_queue;
    uint64_t* cpu_agg_loc_queue;
    uint32_t cpu_agg_queue_depth;
    uint32_t* cpu_agg_queue_counter;


    __forceinline__
    __device__
    cache_page_t* get_cache_page(const uint32_t page) const;

    __forceinline__
    __device__
    uint32_t find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_,  simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array);
    
    __forceinline__
    __device__
    uint32_t wb_find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_,  simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array, 
                          uint32_t* wb_queue_counter,  uint32_t  wb_depth,  uint64_t* queue_ptr,
                         int& evict_cpu, uint32_t& evicted_page_id, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                         uint8_t time_step, uint32_t head_ptr, uint32_t& evict_time);

    __forceinline__
    __device__
    uint32_t wb_find_slot_cpu_agg(uint64_t address, uint64_t range_id, const uint32_t queue_,  simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array, 
                          uint32_t* wb_queue_counter,  uint32_t  wb_depth,  uint64_t* queue_ptr,
                         int& evict_cpu, uint32_t& evicted_page_id, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                         uint8_t time_step, uint32_t head_ptr, uint32_t& evict_time);

};


__device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);

__forceinline__
__device__
uint64_t get_backing_page_(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t page = page_start;
    if (dist == STRIPE) {
        page += page_offset / n_ctrls;
    }
    else if (dist == REPLICATE) {
        page += page_offset;
    }

    return page;
}
__forceinline__
__device__
uint64_t get_backing_ctrl_(const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
    uint64_t ctrl;

    if (dist == STRIPE) {
        ctrl = page_offset % n_ctrls;
    }
    else if (dist == REPLICATE) {
        ctrl = ALL_CTRLS;
    }
    return ctrl;

}

__global__
void __flush(page_cache_d_t* pc) {
    uint64_t page = threadIdx.x + blockIdx.x * blockDim.x;

    if (page < pc->n_pages) {
        uint64_t previous_global_address = pc->cache_pages[page].page_translation;
        //uint8_t previous_range = this->cache_pages[page].range_id;
        uint64_t previous_range = previous_global_address & pc->n_ranges_mask;
        uint64_t previous_address = previous_global_address >> pc->n_ranges_bits;
        //uint32_t new_state = BUSY;

        uint32_t expected_state = pc->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

        uint32_t d = expected_state & DIRTY;
        uint32_t smid = get_smid();
        if (d) {

            uint64_t ctrl = get_backing_ctrl_(previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
            uint64_t index = get_backing_page_(pc->ranges_page_starts[previous_range], previous_address, pc->n_ctrls, pc->ranges_dists[previous_range]);
            // //printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
            //        (unsigned long long) previous_range, (unsigned long long)previous_address,
            //        (unsigned long long) ctrl, (unsigned long long) index);
            if (ctrl == ALL_CTRLS) {
                for (ctrl = 0; ctrl < pc->n_ctrls; ctrl++) {
                    Controller* c = pc->d_ctrls[ctrl];
                    uint32_t queue = smid % (c->n_qps);
                    //printf("ALL CTRIL flush page:%i\n", (int) page);

                    write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
                }
            }
            else {

                Controller* c = pc->d_ctrls[ctrl];
                uint32_t queue = smid % (c->n_qps);

                //index = ranges_page_starts[previous_range] + previous_address;
                //printf("flush page:%i\n", (int) page);

                write_data(pc, (c->d_qps)+queue, (index*pc->n_blocks_per_page), pc->n_blocks_per_page, page);
            }

            pc->ranges[previous_range][previous_address].state.fetch_and(~DIRTY);

        }
    }
}

struct page_cache_t {

    page_cache_d_t pdt;

    pages_t*   h_ranges;
    uint64_t* h_ranges_page_starts;
    data_dist_t* h_ranges_dists;
    page_cache_d_t* d_pc_ptr;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr cache_pages_buf;
    //BufferPtr page_translation_buf;
    //BufferPtr page_take_lock_buf;
    BufferPtr ranges_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;
    BufferPtr ranges_page_starts_buf;
    BufferPtr ranges_dists_buf;

    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;
    BufferPtr q_head_buf;
    BufferPtr q_tail_buf;
    BufferPtr q_lock_buf;
    BufferPtr extra_reads_buf;


    uint32_t wb_depth;
    uint64_t* h_cpu_agg_meta_queue;
    uint64_t* h_cpu_agg_loc_queue;
    //window buffer ptrs
    uint64_t* cpu_agg_meta_queue;
    //batch_ptr
    uint64_t* cpu_agg_loc_queue;
    uint32_t* cpu_agg_queue_counter;

    uint32_t cpu_agg_queue_depth;
    bool cpu_agg;

    void print_reset_stats(void) {
        uint64_t v = 0;
        cuda_err_chk(cudaMemcpy(&v, pdt.extra_reads, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaMemset(pdt.extra_reads, 0, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
    }

    void flush_cache() {
        size_t threads = 64;
        size_t n_blocks = (pdt.n_pages + threads - 1) / threads;

        __flush<<<n_blocks, threads>>>(d_pc_ptr);

    }

    template <typename T>
    void add_range(range_t<T>* range) {
        range->rdt.range_id  = pdt.n_ranges++;
        h_ranges[range->rdt.range_id] = range->rdt.pages;
        h_ranges_page_starts[range->rdt.range_id] = range->rdt.page_start;
        h_ranges_dists[range->rdt.range_id] = range->rdt.dist;
        cuda_err_chk(cudaMemcpy(pdt.ranges_page_starts, h_ranges_page_starts, pdt.n_ranges * sizeof(uint64_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges, h_ranges, pdt.n_ranges* sizeof(pages_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges_dists, h_ranges_dists, pdt.n_ranges* sizeof(data_dist_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));

    }

    page_cache_t(const uint64_t ps, const uint64_t np, const uint32_t cudaDevice, const Controller& ctrl, const uint64_t max_range, const std::vector<Controller*>& ctrls,
                uint32_t wb_depth = 256, bool cpu_agg_flag = false, uint32_t cpu_agg_q_depth = 0 ) {
        
        cpu_agg = cpu_agg_flag;
        if(cpu_agg_flag){

            cpu_agg_queue_depth = cpu_agg_q_depth;
            
            cuda_err_chk(cudaHostAlloc((uint64_t **)&h_cpu_agg_meta_queue, sizeof(uint64_t) * wb_depth * cpu_agg_q_depth , cudaHostAllocMapped));
            cudaHostGetDevicePointer((uint64_t **)&cpu_agg_meta_queue, (uint64_t *)h_cpu_agg_meta_queue, 0);

            cuda_err_chk(cudaHostAlloc((uint64_t **)&h_cpu_agg_loc_queue, sizeof(uint64_t) * wb_depth * cpu_agg_q_depth , cudaHostAllocMapped));
            cudaHostGetDevicePointer((uint64_t **)&cpu_agg_loc_queue, (uint64_t *)h_cpu_agg_loc_queue, 0);

            cuda_err_chk(cudaMalloc(&cpu_agg_queue_counter, sizeof(uint32_t) * wb_depth));
        }

        pdt.cpu_agg_loc_queue = cpu_agg_loc_queue;
        pdt.cpu_agg_meta_queue = cpu_agg_meta_queue;
        pdt.cpu_agg_queue_depth = cpu_agg_queue_depth;
        pdt.cpu_agg_queue_counter = cpu_agg_queue_counter;
        pdt.cpu_agg = cpu_agg;

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_head_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_tail_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_lock_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        extra_reads_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();
        pdt.page_size = ps;
        pdt.q_head = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_head_buf.get();
        pdt.q_tail = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_tail_buf.get();
        pdt.q_lock = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_lock_buf.get();
        pdt.extra_reads = (simt::atomic<uint64_t, simt::thread_scope_device>*)extra_reads_buf.get();
        pdt.page_size_minus_1 = ps - 1;
        pdt.n_pages = np;
        pdt.ctrl_page_size = ctrl.ctrl->page_size;
        pdt.n_pages_minus_1 = np - 1;
        pdt.n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(pdt.n_ctrls * sizeof(Controller*), cudaDevice);
        pdt.d_ctrls = (Controller**) d_ctrls_buff.get();
        pdt.n_blocks_per_page = (ps/ctrl.blk_size);
        pdt.n_cachelines_for_states = np/STATES_PER_CACHELINE;
        for (size_t k = 0; k < pdt.n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(pdt.d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        

        pdt.range_cap = max_range;
        pdt.n_ranges = 0;
        pdt.n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        pdt.n_ranges_mask = max_range-1;
        std::cout << "n_ranges_bits: " << std::dec << pdt.n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << pdt.n_ranges_mask << std::endl;

        pdt.page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(pages_t), cudaDevice);
        pdt.ranges = (pages_t*)ranges_buf.get();
        h_ranges = new pages_t[max_range];

        h_ranges_page_starts = new uint64_t[max_range];
        std::memset(h_ranges_page_starts, 0, max_range * sizeof(uint64_t));

        //pages_translation_buf = createBuffer(np * sizeof(uint32_t), cudaDevice);
        //pdt.page_translation = (uint32_t*)page_translation_buf.get();
        //page_translation_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //page_translation = (padded_struct_pc*)page_translation_buf.get();

        //page_take_lock_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //pdt.page_take_lock =  (padded_struct_pc*)page_take_lock_buf.get();

        cache_pages_buf = createBuffer(np * sizeof(cache_page_t), cudaDevice);
        pdt.cache_pages = (cache_page_t*)cache_pages_buf.get();

        ranges_page_starts_buf = createBuffer(max_range * sizeof(uint64_t), cudaDevice);
        pdt.ranges_page_starts = (uint64_t*) ranges_page_starts_buf.get();

        page_ticket_buf = createBuffer(1 * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_ticket =  (padded_struct_pc*)page_ticket_buf.get();
        //std::vector<padded_struct_pc> tps(np, FREE);
        cache_page_t* tps = new cache_page_t[np];
        for (size_t i = 0; i < np; i++){
            tps[i].page_take_lock = FREE;
            tps[i].next_reuse = (0xFFFF000000000000);
           // tps[i].reuse_mask = 0;
            
            for(int j = 0; j < 8; j++){
                tps[i].reuse_chunk[j] = no_reuse;
            }
        }
        cuda_err_chk(cudaMemcpy(pdt.cache_pages, tps, np*sizeof(cache_page_t), cudaMemcpyHostToDevice));
        delete tps;

        ranges_dists_buf = createBuffer(max_range * sizeof(data_dist_t), cudaDevice);
        pdt.ranges_dists = (data_dist_t*)ranges_dists_buf.get();
        h_ranges_dists = new data_dist_t[max_range];

        uint64_t cache_size = ps*np;
        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        pdt.base_addr = (uint8_t*) this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((pdt.page_size > (ctrl.ctrl->page_size * uints_per_page)) || (np == 0) || (pdt.page_size < ctrl.ns.lba_data_size))
            throw error(string("page_cache_t: Can't have such page size or number of pages"));
        if (ps <= this->pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/ps;
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();


            std::cout << np << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one *  this->pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  this->pages_dma.get()->n_ioaddrs);
            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps;
                    //std::cout << std::dec << "\ti: " << i << "\tj: " << j << "\tindex: "<< (i*how_many_in_one + j) << "\t" << std::hex << (((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps) << std::dec << std::endl;
                }
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete temp;
            //std::cout << "HERE1\n";
            //free(temp);
            //std::cout << "HERE2\n";
            pdt.prps = false;
        }

        else if ((ps > this->pages_dma.get()->page_size) && (ps <= (this->pages_dma.get()->page_size * 2))) {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            //uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            //uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            pdt.prps = true;
        }
        else {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
            uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t* temp3 = new uint64_t[prp_list_size];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            std::memset(temp3, 0, prp_list_size);
            uint32_t how_many_in_one = ps /  ctrl.ctrl->page_size ;
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one]);
                temp2[i] = ((uint64_t) this->prp_list_dma.get()->ioaddrs[i]);
                for(size_t j = 0; j < (how_many_in_one-1); j++) {
                    temp3[i*uints_per_page + j] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one + j + 1]);
                }
            }
            /*
              for (size_t i = 0; i < this->pages_dma.get()->n_ioaddrs; i+=how_many_in_one) {
              temp1[i/how_many_in_one] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]);
              temp2[i/how_many_in_one] = ((uint64_t)this->prp_list_dma.get()->ioaddrs[i]);
              for (size_t j = 0; j < (how_many_in_one-1); j++) {

              temp3[(i/how_many_in_one)*uints_per_page + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i+1+j]);
              }
              }
            */

            std::cout << "Done creating PRP\n";
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            delete temp3;
            pdt.prps = true;
        }


        pc_buff = createBuffer(sizeof(page_cache_d_t), cudaDevice);
        d_pc_ptr = (page_cache_d_t*)pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";

    }

    ~page_cache_t() {
        delete h_ranges;
        delete h_ranges_page_starts;
        delete h_ranges_dists;
    }





};



template <typename T>
struct range_d_t {
    uint64_t index_start;
    uint64_t count;
    uint64_t range_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_count;
    size_t n_elems_per_page;
    data_dist_t dist;
    uint8_t* src;

    simt::atomic<uint64_t, simt::thread_scope_device> access_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> read_io_cnt;

    simt::atomic<uint64_t, simt::thread_scope_device> debug_cnt;
    uint64_t* evicted_p_array;

    pages_t pages;
    //padded_struct_pc* page_addresses;
    //uint32_t* page_addresses;
    //padded_struct_pc* page_vals;  //len = num of pages for data
    //void* self_ptr;
    page_cache_d_t cache;
    //range_d_t(range_t<T>* rt);
    __forceinline__
    __device__
    uint64_t get_backing_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_backing_ctrl(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_sector_size() const;
    __forceinline__
    __device__
    uint64_t get_page(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_subindex(const size_t i) const;
    __forceinline__
    __device__
    uint64_t get_global_address(const size_t page) const;
    __forceinline__
    __device__
    void release_page(const size_t pg) const;
    __forceinline__
    __device__
    void release_page(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    uint64_t acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue);
    __forceinline__
    __device__
    uint64_t wb_acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue,
                             uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr,
                             int& evict_cpu, uint32_t& evicted_page_id, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                             uint8_t time_step, uint32_t head_ptr);
    __forceinline__
    __device__
    bool wb_check_page (const size_t pg, uint64_t & page_trans);
    
    __forceinline__
    __device__
    uint64_t wb_cont_acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl, const uint32_t queue,
                             uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr, uint32_t page_trans) ;
    __forceinline__
    __device__
    void write_done(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    T operator[](const size_t i) ;
    __forceinline__
    __device__
    void operator()(const size_t i, const T val);
    __forceinline__
    __device__
    cache_page_t* get_cache_page(const size_t pg) const;
    __forceinline__
    __device__
    uint64_t get_cache_page_addr(const uint32_t page_trans) const;
    __forceinline__
    __device__
    void mark_page_dirty(const size_t index);
    __forceinline__
    __device__
    void set_cpu_buffer(const size_t index, uint32_t offset);
    __forceinline__
    __device__
    uint32_t get_cpu_offset(const size_t index);


    __forceinline__
    __device__
    void set_prefetch_val(const size_t pg, const size_t count) const;
     __forceinline__
    __device__
    void set_window_buffer_counter(const size_t pg, const size_t count) const;
    

};

template <typename T>
struct range_t {
    range_d_t<T> rdt;

    range_d_t<T>* d_range_ptr;
    page_cache_d_t* cache;

    BufferPtr pages_buff;
    //BufferPtr page_addresses_buff;

    BufferPtr range_buff;



    range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE);



};

template <typename T>
range_t<T>::range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice, data_dist_t dist) {
    rdt.access_cnt = 0;
    rdt.miss_cnt = 0;
    rdt.hit_cnt = 0;
    rdt.read_io_cnt = 0;
    rdt.index_start = is;
    rdt.count = count;

    rdt.debug_cnt = 0;
    //range_id = (c_h->range_count)++;
    rdt.page_start = ps;
    rdt.page_count = pc;
    rdt.page_size = c_h->pdt.page_size;
    rdt.page_start_offset = pso;
    rdt.dist = dist;
    size_t s = pc;//(rdt.page_end-rdt.page_start);//*page_size / c_h->page_size;
    rdt.n_elems_per_page = rdt.page_size / sizeof(T);
    cache = (page_cache_d_t*) c_h->d_pc_ptr;
    pages_buff = createBuffer(s * sizeof(data_page_t), cudaDevice);
    rdt.pages = (pages_t) pages_buff.get();
    //std::vector<padded_struct_pc> ts(s, INVALID);
    data_page_t* ts = new data_page_t[s];
    for (size_t i = 0; i < s; i++) {
        ts[i].state = INVALID;
        ts[i].cpu_feature_offset = 0;

        ts[i].prefetch_count=0;
        ts[i].prefetch_counter=0;
    }
    ////printf("S value: %llu\n", (unsigned long long)s);
   uint64_t* evicted_p_array; 
//    cuda_err_chk(cudaMallocManaged(&evicted_p_array, sizeof(uint64_t) * 700000));
//    printf("e p array:%p\n", evicted_p_array);
    rdt.evicted_p_array=evicted_p_array;
    cuda_err_chk(cudaMemcpy(rdt.pages//_states
                            , ts, s * sizeof(data_page_t), cudaMemcpyHostToDevice));
    
    delete ts;


    //page_addresses_buff = createBuffer(s * sizeof(uint32_t), cudaDevice);
    //rdt.page_addresses = (uint32_t*) page_addresses_buff.get();
    //page_addresses_buff = createBuffer(s * sizeof(padded_struct_pc), cudaDevice);
    //page_addresses = (padded_struct_pc*) page_addresses_buff.get();

    range_buff = createBuffer(sizeof(range_d_t<T>), cudaDevice);
    d_range_ptr = (range_d_t<T>*)range_buff.get();
    //rdt.range_id  = c_h->pdt.n_ranges++;


    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

    c_h->add_range(this);

    rdt.cache = c_h->pdt;
    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_page(const size_t page_offset) const {
    return get_backing_page_(page_start, page_offset, cache.n_ctrls, dist);
}




template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_backing_ctrl(const size_t page_offset) const {
    return get_backing_ctrl_(page_offset, cache.n_ctrls, dist);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_sector_size() const {
    return page_size;
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_page(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) >> (cache.page_size_log);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_subindex(const size_t i) const {
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) & (cache.page_size_minus_1);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_global_address(const size_t page) const {
    return ((page << cache.n_ranges_bits) | range_id);
}
template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg) const {
 uint64_t index = pg;
    uint32_t tc = 0;

    uint8_t wb_st = pages[index].prefetch_counter.load(simt::memory_order_acquire);
    uint8_t window_count = (wb_st & 0x7F);
    if(window_count >= 1){
        if(window_count  == 1 && wb_st >> 7 == 1){
            tc=1;
            pages[index].prefetch_counter.store(0,simt::memory_order_release);
            }
        else{
            uint8_t wb_count_after = pages[index].prefetch_counter.fetch_sub(1, simt::memory_order_release);
        }
    }




    //printf("release2 idx:%llu p_count:%lu  count: %lu tc:%lu\n",(unsigned long long) pg, (unsigned long) window_count, (unsigned long) count,(unsigned long)tc);    
    uint64_t st = pages[index].state.fetch_sub(1+tc, simt::memory_order_release);
	uint32_t cnt = st & CNT_MASK;


}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg, const uint32_t count) const {
    uint64_t index = pg;
    uint32_t tc = 0;

    uint8_t p_st = pages[index].prefetch_count.load(simt::memory_order_acquire);
    uint8_t p_count = (p_st & 0x7F);
    if(p_count > 0){
	if(p_count  == 1 && p_st >> 7 == 1){
		tc=1;
		pages[index].prefetch_count.store(0,simt::memory_order_release);
       	}
	else{
		 uint8_t p_count_after = pages[index].prefetch_count.fetch_sub(1, simt::memory_order_release);
	}
    }

    uint8_t wb_st = pages[index].prefetch_counter.load(simt::memory_order_acquire);
    uint8_t window_count = (wb_st & 0x7F);
    if(window_count >= 1){
	    if(window_count  == 1 && wb_st >> 7 == 1){
		tc=1;
		pages[index].prefetch_counter.store(0,simt::memory_order_release);
       	}
	else{
		 uint8_t wb_count_after = pages[index].prefetch_counter.fetch_sub(1, simt::memory_order_release);
	}
    }


    //printf("release2 idx:%llu p_count:%lu  count: %lu tc:%lu\n",(unsigned long long) pg, (unsigned long) window_count, (unsigned long) count, (unsigned long)tc);


  //  printf("release2 idx:%llu p_count:%lu  count: %lu tc:%lu\n",(unsigned long long) pg, (unsigned long) p_count, (unsigned long) count,(unsigned long)tc);    
    uint64_t st = pages[index].state.fetch_sub(count+tc, simt::memory_order_release);
	uint32_t cnt = st & CNT_MASK;
//	printf("pg: %llu r2 cnt: %lu\n",(unsigned long long) pg,  (unsigned long)cnt);
}

template <typename T>
__forceinline__
__device__
cache_page_t* range_d_t<T>::get_cache_page(const size_t pg) const {
    uint32_t page_trans = pages[pg].offset;
    return cache.get_cache_page(page_trans);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_cache_page_addr(const uint32_t page_trans) const {
    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::mark_page_dirty(const size_t index) {
    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::set_cpu_buffer(uint64_t index, uint32_t offset) {
   // pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
    pages[index].cpu_feature_offset = (offset << 1) | (0x1);
}

template <typename T>
__forceinline__
__device__
uint32_t range_d_t<T>::get_cpu_offset(const size_t index) {
   // pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
    return pages[index].cpu_feature_offset;
}






template <typename T>
__forceinline__
__device__
void range_d_t<T>::set_prefetch_val(const size_t pg, const size_t count) const{
	 uint64_t index = pg;
	 uint8_t p_count = count;
	pages[index].prefetch_count.fetch_add(count, simt::memory_order_release); 
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::set_window_buffer_counter(const size_t pg, const size_t count) const{
	uint64_t index = pg;
	uint8_t p_count = count;
	pages[index].prefetch_counter.fetch_add(count, simt::memory_order_release); 
}


    
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue) {
    uint64_t index = pg;
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    
    bool fail = true;
    unsigned int ns = 8;


    uint8_t prefetch_count = pages[index].prefetch_count.load(simt::memory_order_acquire);  
    uint32_t p_count = 0;
    if(prefetch_count != 0 && (prefetch_count >>7 == 0) ){
        p_count = 1;
        pages[index].prefetch_count.store(prefetch_count | 0x80, simt::memory_order_release);
    }

    uint8_t window_count = pages[index].prefetch_counter.load(simt::memory_order_acquire); 
    
    if(window_count > 1 && (window_count >>7 == 0) ){    
    	p_count = 1;
	    pages[index].prefetch_counter.store(window_count | 0x80, simt::memory_order_release);
    }

    uint64_t read_state,st,st_new;
    read_state = pages[index].state.fetch_add(count+p_count, simt::memory_order_acquire);


    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;

        switch (st) {
            //invalid
        case NV_NB:
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);
            if ((st_new & BUSY) == 0) {
                
                uint32_t page_trans = cache.find_slot(index, range_id, queue, read_io_cnt, evicted_p_array);
                
	       	//fill in
                //uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                //uint32_t sm_id = get_smid();
                //uint32_t ctrl = (tid/32) % (cache.n_ctrls);
                //uint32_t ctrl = sm_id % (cache.n_ctrls);
                //uint32_t ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                uint64_t ctrl = get_backing_ctrl(index);
                if (ctrl == ALL_CTRLS)
                    ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                //ctrl = ctrl_;
                uint64_t b_page = get_backing_page(index);
                
		Controller* c = cache.d_ctrls[ctrl];
                c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                //uint32_t queue = (tid/32) % (c->n_qps);
                //uint32_t queue = c->queue_counter.fetch_add(1, simt::memory_order_relaxed) % (c->n_qps);
                //uint32_t queue = ((sm_id * 64) + warp_id()) % (c->n_qps);
                //read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                
		read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                //page_addresses[index].store(page_trans, simt::memory_order_release);
		pages[index].offset = page_trans;
                // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                //     __nanosleep(100);
                //miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                //new_state = VALID;
                if (write)
                    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                //new_state |= DIRTY;
                //pages[index].state.fetch_or(new_state, simt::memory_order_relaxed);
                pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                return page_trans;

                fail = false;
            }

            break;
            //valid
        case V_NB:
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
            //uint32_t page_trans = pages[index].offset.load(simt::memory_order_acquire);
            uint64_t page_trans = pages[index].offset;
            // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
            //     __nanosleep(100);
            //hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            return page_trans;

            //pages[index].fetch_sub(1, simt::memory_order_release);
            fail = false;

            break;
        case NV_B:
        case V_B:
        default:
            break;



         
        }
        if (fail) {
            //if ((++j % 1000000) == 0)
            //    printf("failed to acquire_page: j: %llu\tcnt_shift+1: %llu\tpage: %llu\tread_state: %llx\tst: %llx\tst_new: %llx\n", (unsigned long long)j, (unsigned long long) (CNT_SHIFT+1), (unsigned long long) index, (unsigned long long)read_state, (unsigned long long)st, (unsigned long long)st_new);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::wb_cont_acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue,
                                       uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr,
                                           uint32_t page_trans) {
    uint64_t index = pg;
    uint64_t ctrl = get_backing_ctrl(index);
    if (ctrl == ALL_CTRLS)
        ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                
    uint64_t b_page = get_backing_page(index);
                
    Controller* c = cache.d_ctrls[ctrl];
    c->access_counter.fetch_add(1, simt::memory_order_relaxed);
             
                
    read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
    pages[index].offset = page_trans;

    miss_cnt.fetch_add(count, simt::memory_order_relaxed);

    if (write)
        pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);

    pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
    return page_trans;    
}


template <typename T>
__forceinline__
__device__
bool range_d_t<T>::wb_check_page(const size_t pg, uint64_t & page_trans){
   
    uint64_t index = pg;
    uint64_t read_state,st,st_new;
    read_state = pages[index].state;
//        read_state = pages[index].state.load(simt::memory_order_relaxed);

    st = (read_state >> (CNT_SHIFT+1)) & 0x03;
    if(st == V_NB){
	   page_trans = pages[index].offset;
        return true;
    }
    return false;
	/*
	 uint64_t index = pg;
	uint64_t off =  pages[index].offset;
	if(off == 1)
	       	printf("off 1\n");
*/
}


template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::wb_acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue,
                                       uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr,
                                       int& evict_cpu, uint32_t& queue_idx, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                                       uint8_t time_step, uint32_t head_ptr) {
    uint64_t index = pg;
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    
    bool fail = true;
    unsigned int ns = 8;

    uint64_t read_state,st,st_new;
    read_state = pages[index].state.fetch_add(count, simt::memory_order_acquire);

    do {
        st = (read_state >> (CNT_SHIFT+1)) & 0x03;

        switch (st) {
            //invalid
        case NV_NB:
            st_new = pages[index].state.fetch_or(BUSY, simt::memory_order_acquire);

	    uint32_t evict_time = 65535;
            if ((st_new & BUSY) == 0) {
                
                evict_cpu = 0;
                uint32_t page_trans;
                if(cache.cpu_agg){
                    page_trans = cache.wb_find_slot_cpu_agg(index, range_id, queue, read_io_cnt, evicted_p_array,
                                                          wb_queue_counter,  wb_depth, (uint64_t*) queue_ptr,
                                                         evict_cpu, queue_idx, queue_reuse, id_array, q_depth,
                                                         time_step, head_ptr, evict_time);
                }
                 
                else{
                    page_trans = cache.wb_find_slot(index, range_id, queue, read_io_cnt, evicted_p_array,
                                                          wb_queue_counter,  wb_depth, (uint64_t*) queue_ptr,
                                                         evict_cpu, queue_idx, queue_reuse, id_array, q_depth,
                                                         time_step, head_ptr, evict_time);
                }
                //Need to write back to CPU
                if(evict_time < 8){
		    	debug_cnt.fetch_add(count, simt::memory_order_relaxed); 
		 }
		if(evict_cpu){
		    return page_trans;
                }

                uint64_t ctrl = get_backing_ctrl(index);
                if (ctrl == ALL_CTRLS)
                    ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                uint64_t b_page = get_backing_page(index);
                
                Controller* c = cache.d_ctrls[ctrl];
                c->access_counter.fetch_add(1, simt::memory_order_relaxed);
             
                
                read_data(&cache, (c->d_qps)+queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                pages[index].offset = page_trans;
              
                miss_cnt.fetch_add(count, simt::memory_order_relaxed);

                if (write)
                    pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);

                pages[index].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID, simt::memory_order_release);
                return page_trans;

                fail = false;
            }

            break;
            //valid
        case V_NB:
            if (write && ((read_state & DIRTY) == 0))
                pages[index].state.fetch_or(DIRTY, simt::memory_order_relaxed);
                uint64_t page_trans = pages[index].offset;
         
                hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                return page_trans;

            
                fail = false;

            break;
        case NV_B:
        case V_B:
        default:
            break;

        }
        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
            read_state = pages[index].state.load(simt::memory_order_acquire);
        }

    } while (fail);
    return 0;
}


template<typename T>
struct array_d_t {
    uint64_t n_elems;
    uint64_t start_offset;
    uint64_t n_ranges;
    uint8_t *src;

    range_d_t<T>* d_ranges;

    uint32_t* wb_queue_counter;
    uint32_t  wb_depth;
    uint32_t q_depth;

    T* queue_ptr;
    uint32_t* wb_id_array;
    
    T* cpu_cache; 
    
    

    __forceinline__
    __device__
    void get_page_gid(const uint64_t i, range_d_t<T>*& r_, size_t& pg, size_t& gid) const {
        int64_t r = find_range(i);
        r_ = d_ranges+r;

        if (r != -1) {
            r_ = d_ranges+r;
            pg = r_->get_page(i);
            gid = r_->get_global_address(pg);
        }
        else {
            r_ = nullptr;
            printf("here\n");
        }
    }
    __forceinline__
    __device__
    void memcpy(const uint64_t i, const uint64_t count, T* dest) {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = 0xffffffff;
#endif
            uint32_t leader = 0;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            //uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = 32;
            uint32_t eq_mask = mask;
            int master = 0;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = 1;
            if (master == lane) {
                //std::pair<uint64_t, bool> base_memcpyflag;
                base = r_->acquire_page(page, count, false, ctrl, queue);
                base_master = base;
//                //printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            //
            ulonglong4* src_ = (ulonglong4*) r_->get_cache_page_addr(base_master);
            ulonglong4* dst_ = (ulonglong4*) dest;
            warp_memcpy<ulonglong4>(dst_, src_, 512/32);

            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }

    }
    __forceinline__
    __device__
    int64_t find_range(const size_t i) const {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++) {
            if ((d_ranges[k].index_start <= i) && (d_ranges[k].count > i)) {
                range = k;
                break;
            }

        }
        return range;
    }

 

    __forceinline__
    __device__
    void coalesce_page(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const uint64_t gaddr, const bool write,
                       uint32_t& eq_mask, int& master, uint32_t& count, uint64_t& base_master) const {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        auto r_ = d_ranges+r;
        if (lane == leader) {
            page_cache_d_t* pc = &(r_->cache);
            ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[0]->n_qps);
        }

        ctrl = 0; //__shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);


        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        //bool memcpyflag_master;
        //bool memcpyflag;
        count = __popc(eq_mask);
        if (master == lane) {
            //std::pair<uint64_t, bool> base_memcpyflag;
            base = r_->acquire_page(page, count, dirty, ctrl, queue);
            base_master = base;
//                //printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
        }
        base_master = __shfl_sync(eq_mask,  base_master, master);
    }

    __forceinline__
    __device__
    void wb_coalesce_page(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const uint64_t gaddr, const bool write, uint32_t& eq_mask, int& master, uint32_t& count, uint64_t& base_master, 
                          uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr, uint64_t* id_array, uint32_t q_depth,
                          uint8_t time_step, uint32_t head_ptr) const {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        auto r_ = d_ranges+r;
        if (lane == leader) {
            page_cache_d_t* pc = &(r_->cache);
            ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[0]->n_qps);
        }

        ctrl = 0; //__shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);


        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        //bool memcpyflag_master;
        //bool memcpyflag;
        count = __popc(eq_mask);
        
        int evict_cpu = 0;
        uint32_t queue_idx = 0;
        uint32_t page_trans;
        uint32_t queue_reuse = 0;

        if (master == lane) {
            base = r_->wb_acquire_page(page, count, dirty, ctrl, queue, 
                                      wb_queue_counter, wb_depth, queue_ptr,
                                      evict_cpu, queue_idx, queue_reuse, id_array, q_depth, time_step, head_ptr);
            page_trans = base;
            base_master = base;
        }
        base_master = __shfl_sync(eq_mask,  base_master, master);

        evict_cpu =  __shfl_sync(eq_mask,  evict_cpu, master);
        if(evict_cpu){
            queue_idx = __shfl_sync(eq_mask,  queue_idx, master);
            queue_reuse = __shfl_sync(eq_mask,  queue_reuse, master);

            //To DO write
            
            //uint8_t* write_adder = this (cache_d_t*) -> base_addr + (page * this->page_size);
            void* cache_ptr = (void*)r_->get_cache_page_addr(base_master);
            write_to_queue<uint64_t>(cache_ptr, (void*)(queue_ptr) + (q_depth * queue_reuse *(r_->cache.page_size) ) + (queue_idx * ( r_->cache.page_size)), (r_->cache.page_size), eq_mask);
            
            __syncwarp(eq_mask);
            if (master == lane) {
                base = r_->wb_cont_acquire_page(page, count, dirty, ctrl, queue, 
                                          wb_queue_counter, wb_depth, queue_ptr,
                                          page_trans);
               base_master = base;
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);
        }
         
    }
    
    __forceinline__
    __device__
    returned_cache_page_t<T> get_raw(const size_t i) const {
        returned_cache_page_t<T> ret;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);



            ret.addr = (T*) r_->get_cache_page_addr(base_master);
            ret.size = r_->get_sector_size()/sizeof(T);
            ret.offset = subindex/sizeof(T);
            //ret.page = page;
            __syncwarp(mask);


        }
        return ret;
    }
    __forceinline__
    __device__
    void release_raw(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    void* acquire_page_(const size_t i, data_page_t*& page_, size_t& start, size_t& end, range_d_t<T>* r_, const size_t page) const {
        //uint32_t lane = lane_id();



        void* ret = nullptr;
        page_ = nullptr;
        if (r_) {
            //uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);
            page_cache_d_t* pc = &(r_->cache);
            uint32_t ctrl = 0;//pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            uint32_t queue = get_smid() % (pc->d_ctrls[0]->n_qps);
            uint64_t base_master = r_->acquire_page(page, 1, false, ctrl, queue);
            //coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;

        }
        return ret;
    }
    __forceinline__
    __device__
    void* acquire_page(const size_t i, data_page_t*& page_, size_t& start, size_t& end, int64_t& r) const {
        uint32_t lane = lane_id();
        r = find_range(i);
        auto r_ = d_ranges+r;

        void* ret = nullptr;
        page_ = nullptr;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);
            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;
            __syncwarp(mask);
        }
        return ret;
    }
    
        __forceinline__
    __device__
    void* wb_acquire_page(const size_t i, data_page_t*& page_, size_t& start, size_t& end, int64_t& r,  uint32_t* wb_queue_counter,  uint32_t  wb_depth,  T* queue_ptr, uint64_t* id_array, uint32_t q_depth,
                          uint8_t time_step, uint32_t head_ptr) const {
        uint32_t lane = lane_id();
        r = find_range(i);
        auto r_ = d_ranges+r;

        void* ret = nullptr;
        page_ = nullptr;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            wb_coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master,
                                wb_queue_counter, wb_depth, queue_ptr, id_array, q_depth, time_step, head_ptr);
            page_ = &r_->pages[base_master];


            ret = (void*)r_->get_cache_page_addr(base_master);
            start = r_->n_elems_per_page * page;
            end = start +r_->n_elems_per_page;// * (page+1);
            //ret.page = page;
            __syncwarp(mask);
        }
        return ret;
    }
    
    __forceinline__
    __device__
    void wb_update_wb(uint64_t page, uint32_t reuse_time, uint64_t idx) {
       // TO DO Fix ranges
        auto r_ = d_ranges;
        uint64_t page_trans;
        bool cl_find = r_->wb_check_page(page, page_trans);
        if(cl_find){
            uint64_t update_val = reuse_time;
            update_val = update_val << 48;
            update_val = (update_val | idx);
            atomicMin((unsigned long long int*) &(r_->cache.cache_pages[page_trans].next_reuse), (unsigned long long int) update_val);
        }
            
        return;
    }

    __forceinline__
    __device__
    void wb_update_wb_list(uint64_t page, uint64_t reuse_time, uint64_t idx) {
        auto r_ = d_ranges;
        uint64_t page_trans;
        bool cl_find = r_->wb_check_page(page, page_trans);
        if(cl_find){
            //printf("found\n");
            uint64_t update_val = reuse_time;
            update_val = update_val << 48;
            update_val = (update_val | idx);
            atomicMin((unsigned long long int*) &(r_->cache.cache_pages[page_trans].next_reuse), (unsigned long long int) update_val);
            if(reuse_time < 128){
                //unsigned long long int temp = atomicOr((unsigned long long int*) &(r_->cache.cache_pages[page_trans].reuse_mask), ((uint64_t) 1 << reuse_time));
                unsigned int reuse_sec = reuse_time / 16;
                atomicMin((unsigned long long int*) &(r_->cache.cache_pages[page_trans].reuse_chunk[reuse_sec]), (unsigned long long int) update_val);

                //printf("temp: %llx\n", temp);
                //r_->cache.cache_pages[page_trans].reuse_mask | ((uint64_t) 1 << reuse_time);
            }
        }
    
        return;
    }
     
    __forceinline__
    __device__
    void wb_update_wb_test(uint64_t page, uint32_t reuse_time, uint64_t idx) {
       // TO DO Fix ranges
        auto r_ = d_ranges;
        uint64_t page_trans;
        bool cl_find = r_->wb_check_page(page, page_trans);
	if(cl_find){
        //     printf("update wb page: %llu reuse:%lu\n", (unsigned long long) page, (unsigned long) reuse_time);
		uint64_t update_val = reuse_time;
		update_val = update_val << 48;
		
		update_val = (update_val | idx);

      		atomicMin((unsigned long long int*) &(r_->cache.cache_pages[page_trans].next_reuse), (unsigned long long int) update_val);
 //      	r_->cache.cache_pages[page_trans].next_reuse = reuse_time;
	}
        
        return;
    }
    
    __forceinline__
    __device__
    void flush_wb_counter(uint64_t page) {
       // TO DO Fix ranges
        auto r_ = d_ranges;
        r_->cache.cache_pages[page].next_reuse = 0xFFFF000000000000;
      //  r_->cache.cache_pages[page].reuse_mask = 0;

        for(int i = 0; i < 8; i++){
            r_->cache.cache_pages[page].reuse_chunk[i] = no_reuse;
        }
        
        return;
    }

    __forceinline__
    __device__
    void count_mask(uint64_t page, uint64_t* mask_counter) {
       // TO DO Fix ranges
        auto r_ = d_ranges;
        uint64_t mask_val = r_->cache.cache_pages[page].reuse_mask;
        
        int reuse_val = __popcll((unsigned long long) mask_val);
        if(reuse_val == 0){
            atomicAdd((unsigned long long * )mask_counter, 1);
        }
        return;
    }

    __forceinline__
    __device__
    uint64_t get_page_id(uint64_t page) {
       // TO DO Fix ranges
        auto r_ = d_ranges;
    	uint64_t previous_global_address = (r_->cache.cache_pages[page].page_translation);
       	uint64_t previous_range = previous_global_address & (r_->cache.n_ranges_mask);
        uint64_t previous_address = previous_global_address >> (r_->cache.n_ranges_bits);
	return previous_address;


    }


    __forceinline__
    __device__
    uint32_t wb_check_reuse_val(uint32_t page) {
       
       // TO DO Fix ranges
        auto r_ = d_ranges;
        uint32_t reuse_val = r_->cache.cache_pages[page].next_reuse;
        return reuse_val;
    }

    __forceinline__
    __device__
    void set_prefetching(const size_t i, uint8_t p_val){
	auto r = find_range(i);
	auto r_ = d_ranges+r;

	if (r != -1) {
#ifndef __CUDACC__
	    uint32_t mask = 1;
#else 
	    uint32_t mask = __activemask();
#endif
	    uint64_t page = r_->get_page(i);
	    size_t lane = lane_id();
	    uint32_t leader = __ffs(mask) - 1;
       
	    if(lane == leader){
		r_ -> set_prefetch_val(page, p_val);
	    }
	}
	return;
    }
	

    __forceinline__
    __device__
    void set_window_buffer_counter(const size_t i, uint8_t p_val){
	auto r = find_range(i);
	auto r_ = d_ranges+r;

	if (r != -1) {
#ifndef __CUDACC__
	    uint32_t mask = 1;
#else 
	    uint32_t mask = __activemask();
#endif
	    uint64_t page = r_->get_page(i);
	    size_t lane = lane_id();
	    uint32_t leader = __ffs(mask) - 1;
       
	    if(lane == leader){
		r_ -> set_window_buffer_counter(page, p_val);
	    }
	}
	return;
    }

    __forceinline__
    __device__
    void pin_page(const size_t i) const {
	auto r = find_range(i);
	auto r_ = d_ranges+r;

	if (r != -1) {
#ifndef __CUDACC__
	    uint32_t mask = 1;
#else 
	     uint32_t mask = __activemask();
#endif
	    uint64_t page = r_->get_page(i);
	    r_->pin_page(i, page);
	}
	return;
    }


    __forceinline__
    __device__
    void release_page(data_page_t* page_, const int64_t r, const size_t i) const {
        uint32_t lane = lane_id();
        auto r_ = d_ranges+r;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t gaddr = r_->get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);



        }
    }

    __forceinline__
    __device__
    T seq_read(const size_t i) const {
	    uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T ret;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0];
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
        return ret;
    }

    __forceinline__
    __device__
    void pin_memory(const size_t i) const {
	   uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T ret;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0];
            __syncwarp(eq_mask);

        }
        return;
    }



    __forceinline__
    __device__
    void unpin_memory(const size_t i) const {
     	uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;
        T ret;
        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count = 32;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

	    if (master == lane)  
		 r_->release_page(page, count); 

	    __syncwarp(mask);
	}
	return;
     }

    __forceinline__
    __device__
    void seq_write(const size_t i, const T val) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;


        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);
            uint64_t gaddr = r_->get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, true, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ((T*)(r_->get_cache_page_addr(base_master)+subindex))[0] = val;
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);

        }
    }
    __forceinline__
    __device__
    T operator[](size_t i) const {
    	return seq_read(i);
        // size_t k = 0;
        // bool found = false;
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }

        // }
        // if (found)
        //     return (((d_ranges[k]))[i-d_ranges[k].index_start]);
    }
    __forceinline__
    __device__
    void operator()(size_t i, T val) const {
        seq_write(i, val);
        // size_t k = 0;
        // bool found = false;
        // uint32_t mask = __activemask();
        // for (; k < n_ranges; k++) {
        //     if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {
        //         found = true;
        //         break;
        //     }
        // }
        // __syncwarp(mask);
        // if (found)
        //     ((d_ranges[k]))(i-d_ranges[k].index_start, val);
    }


    __forceinline__
    __device__
    T AtomicAdd(const size_t i, const T val) const {
        //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        auto r_ = d_ranges+r;

        T old_val = 0;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1) {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t leader = __ffs(mask) - 1;
            if (lane == leader) {
                page_cache_d_t* pc = &(r_->cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = r_->get_page(i);
            uint64_t subindex = r_->get_subindex(i);


            uint64_t gaddr = r_->get_global_address(page);
            //uint64_t p_s = r_->page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane) {
                base = r_->acquire_page(page, count, true, ctrl, queue);
                base_master = base;
                //    //printf("++tid: %llu\tbase: %llu  memcpyflag_master:%llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master, (unsigned long long) memcpyflag_master);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            ////printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            // ((T*)(base_master+subindex))[0] = val;
            old_val = atomicAdd((T*)(r_->get_cache_page_addr(base_master)+subindex), val);
            // //printf("AtomicAdd: tid: %llu\tpage: %llu\tsubindex: %llu\tval: %llu\told_val: %llu\tbase_master: %llx\n",
            //        (unsigned long long) tid, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) val,
            //     (unsigned long long) old_val, (unsigned long long) base_master);
            __syncwarp(eq_mask);
            if (master == lane)
                r_->release_page(page, count);
            __syncwarp(mask);
        }

        return old_val;
    }




};

template<typename T>
struct array_t {
    array_d_t<T> adt;

    //range_t<T>** d_ranges;
    array_d_t<T>* d_array_ptr;



    BufferPtr d_array_buff;
    BufferPtr d_ranges_buff;
    BufferPtr d_d_ranges_buff;

    void print_reset_stats(void) {
        std::vector<range_d_t<T>> rdt(adt.n_ranges);
        //range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
        cuda_err_chk(cudaMemcpy(rdt.data(), adt.d_ranges, adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < adt.n_ranges; i++) {

            std::cout << std::dec << "#READ IOs: "  << rdt[i].read_io_cnt 
                                  << "\t#Accesses:" << rdt[i].access_cnt
                                  << "\t#Misses:"   << rdt[i].miss_cnt 
                                  << "\tMiss Rate:" << ((float)rdt[i].miss_cnt/rdt[i].access_cnt)
                                  << "\t#Hits: "    << rdt[i].hit_cnt 
                                  << "\tHit Rate:"  << ((float)rdt[i].hit_cnt/rdt[i].access_cnt) 
                                  << "\tCLSize:"    << rdt[i].page_size 
				  << "\tDebug Cnt: " << rdt[i].debug_cnt
                                  << std::endl;
            std::cout << "*********************************" << std::endl;
//	    for (size_t j = 0; j < rdt[i].read_io_cnt; j++){
//		std::cout << "evicted: " << rdt[i].evicted_p_array[j] << std::endl;

//	    }
	
            rdt[i].read_io_cnt = 0;
            rdt[i].access_cnt = 0;
            rdt[i].miss_cnt = 0;
            rdt[i].hit_cnt = 0;
	    rdt[i].debug_cnt = 0;
        }
        cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt.data(), adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    }

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, uint32_t cudaDevice, uint32_t* q_counter = nullptr, uint32_t wb_dep = 128, uint32_t q_dep = 128, uint32_t* wb_id = nullptr, T* q_ptr=nullptr) {
        adt.n_elems = num_elems;
        adt.start_offset = disk_start_offset;
        adt.n_ranges = ranges.size();
        
        //window buffers
        adt.wb_queue_counter =q_counter;
        adt.wb_depth = wb_dep;
        adt.queue_ptr = q_ptr;
        adt.wb_id_array = wb_id;
        adt.q_depth =q_dep;
        
        
        d_array_buff = createBuffer(sizeof(array_d_t<T>), cudaDevice);
        d_array_ptr = (array_d_t<T>*) d_array_buff.get();

        //d_ranges_buff = createBuffer(n_ranges * sizeof(range_t<T>*), cudaDevice);
        d_d_ranges_buff = createBuffer(adt.n_ranges * sizeof(range_d_t<T>), cudaDevice);
        adt.d_ranges = (range_d_t<T>*)d_d_ranges_buff.get();
        //d_ranges = (range_t<T>**) d_ranges_buff.get();
        for (size_t k = 0; k < adt.n_ranges; k++) {
            //cuda_err_chk(cudaMemcpy(d_ranges+k, &(ranges[k]->d_range_ptr), sizeof(range_t<T>*), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(adt.d_ranges+k, (ranges[k]->d_range_ptr), sizeof(range_d_t<T>), cudaMemcpyDeviceToDevice));
        }

        cuda_err_chk(cudaMemcpy(d_array_ptr, &adt, sizeof(array_d_t<T>), cudaMemcpyHostToDevice));
    }
   
    

};

__forceinline__
__device__
cache_page_t* page_cache_d_t::get_cache_page(const uint32_t page) const {
    return &this->cache_pages[page];
}

__forceinline__
__device__
        uint32_t page_cache_d_t::wb_find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_, simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array,
                                      uint32_t* wb_queue_counter,  uint32_t  wb_depth,  uint64_t* queue_ptr,
                                      int& evict_cpu, uint32_t& queue_idx, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                                      uint8_t time_step, uint32_t head_ptr, uint32_t& evict_time) {
    bool fail = true;
    uint64_t count = 0;
    uint64_t global_address =(uint64_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;

    uint32_t iter_count = 0;
    uint64_t evict_th = (this->n_pages) / 4;
    bool evict_prior = true;

    do {
	if(iter_count >= evict_th)
        	evict_prior = false;
	page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
     	iter_count++; 

        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);
        if ( v == FREE ) {
    		lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                
                this->cache_pages[page].next_reuse = 0xFFFF000000000000;
              //  this->cache_pages[page].reuse_mask = 0;
                this->cache_pages[page].page_translation = global_address;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {

            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                //TODO check page id
                //previous address is a page id
                uint64_t previous_global_address = this->cache_pages[page].page_translation;
                uint64_t previous_range = previous_global_address & n_ranges_mask;
                uint64_t previous_address = previous_global_address >> n_ranges_bits;
          
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
                if ((cnt == 0) && (b == 0) ) {

                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                           
			            if (((new_expected_state & CNT_MASK ) == 0) ) {                         
				             //Write Back
                           	 if ((new_expected_state & DIRTY)) {
                                	uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);  
                                	uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                                	if (ctrl == ALL_CTRLS) {
                                    	for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                       	 	Controller* c = this->d_ctrls[ctrl];
                                        	uint32_t queue = queue_ % (c->n_qps);
                                      	 	 write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                    	}
                                	}
                                else {

                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);
                                    write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                }
                             }
                            //Writeback done
                           // //printf("CHECK REUSE prev_ page: %lu reuse val: %u \n", (unsigned long) prev_page, (unsigned) reuse_val );
			                uint64_t prev_page = previous_address;
                            uint64_t reuse_full = this->cache_pages[page].next_reuse;
                            uint64_t reuse_val = reuse_full >> 48;
                          
                            if(reuse_val < 16 && evict_prior){
                            //if(false){
                                this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                                continue;
                            }

                       
                            if(reuse_val < 65535){
                                //wb_depth
                                uint32_t evict_t = ((reuse_val) + wb_depth - time_step);
                                uint32_t cur_head = (evict_t + head_ptr) % wb_depth;

                                uint32_t queue_counter = atomicAdd(wb_queue_counter+cur_head, 1);
                                evict_time = reuse_val;
                                if(queue_counter < q_depth){

                                    
                                        evict_cpu = 1;
                                        queue_idx = queue_counter;
                                        queue_reuse = cur_head;
                                                // if(queue_reuse >= 4 || queue_idx >= 32)
                                            // printf("queue_idx: %lu head: %lu id:%llu\n", (unsigned long) queue_idx, (unsigned long) queue_reuse, (unsigned long long) prev_page);
                                        id_array[queue_idx + q_depth * queue_reuse] = (reuse_full & (0x0000FFFFFFFFFFFF));
                                    
                                            //printf("prev_ page: %lu reuse val: %u queue idx: %u\n", prev_page, queue_reuse , queue_idx);
                                }
                            } 
        
                            fail = false;

                            this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                        }
                        else { 
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                        }
                    }
                }

                if (!fail) {
                    this->cache_pages[page].next_reuse = (0xFFFF000000000000);
                   // this->cache_pages[page].reuse_mask = (0);
                    this->cache_pages[page].page_translation = global_address;
                   
                }
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }


        }

        count++;
        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
//             __nanosleep(ns);
//             if (ns < 256) {
//                 ns *= 2;
//             }
#endif

        }

    } while(fail);
    return page;

}


__forceinline__
__device__
        uint32_t page_cache_d_t::wb_find_slot_cpu_agg(uint64_t address, uint64_t range_id, const uint32_t queue_, simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array,
                                      uint32_t* wb_queue_counter,  uint32_t  wb_depth,  uint64_t* queue_ptr,
                                      int& evict_cpu, uint32_t& queue_idx, uint32_t& queue_reuse, uint64_t* id_array, uint32_t q_depth,
                                      uint8_t time_step, uint32_t head_ptr, uint32_t& evict_time) {
    bool fail = true;
    uint64_t count = 0;
    uint64_t global_address =(uint64_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;

    uint32_t iter_count = 0;
    uint64_t evict_th = (this->n_pages) / 4;
    bool evict_prior = true;

    do {
	    if(iter_count >= evict_th)
        	evict_prior = false;
	    page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
     	iter_count++; 

        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);
        if ( v == FREE ) {
    		lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                for(int j  = 0; j < 8; j++){
                    this->cache_pages[page].reuse_chunk[j] = no_reuse;
                }
                this->cache_pages[page].next_reuse = 0xFFFF000000000000;
                //this->cache_pages[page].reuse_mask = 0;
                this->cache_pages[page].page_translation = global_address;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {

            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                //TODO check page id
                //previous address is a page id
                uint64_t previous_global_address = this->cache_pages[page].page_translation;
                uint64_t previous_range = previous_global_address & n_ranges_mask;
                uint64_t previous_address = previous_global_address >> n_ranges_bits;
          
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
                if ((cnt == 0) && (b == 0) ) {

                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                           
			            if (((new_expected_state & CNT_MASK ) == 0) ) {                         
				             //Write Back
                           	 if ((new_expected_state & DIRTY)) {
                                	uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);  
                                	uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                                	if (ctrl == ALL_CTRLS) {
                                    	for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                       	 	Controller* c = this->d_ctrls[ctrl];
                                        	uint32_t queue = queue_ % (c->n_qps);
                                      	 	 write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                    	}
                                	}
                                else {

                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);
                                    write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                }
                             }

                            //Writeback done
			                uint64_t prev_page = previous_address;
                            uint64_t reuse_full = this->cache_pages[page].next_reuse;
                            uint64_t reuse_val = reuse_full >> 48;
                          
                            if(reuse_val < 16 && evict_prior){
                            //if(false){
                                this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                                continue;
                            }
                       
                            if(reuse_val < 65535){
                                //wb_depth
                                uint32_t evict_t = ((reuse_val) + wb_depth - time_step);
                                uint32_t cur_head = (evict_t + head_ptr) % wb_depth;

                                uint32_t queue_counter = atomicAdd(wb_queue_counter+cur_head, 1);
                                evict_time = reuse_val;
                                if(queue_counter < q_depth){

                                    evict_cpu = 1;
                                    queue_idx = queue_counter;
                                    queue_reuse = cur_head;
                                    id_array[queue_idx + q_depth * queue_reuse] = (reuse_full & (0x0000FFFFFFFFFFFF));
                                    

                                    for(int sec = 0; sec < 8; sec++){
                                        uint64_t reuse_sec = this->cache_pages[page].reuse_chunk[sec];
                                        uint64_t reuse_sec_val = reuse_sec >> 48;
                                        //if it is not a source pointer
                                        if(reuse_sec_val != reuse_val){
                                            uint32_t sec_head_ptr = (reuse_sec_val + head_ptr) % wb_depth;
                                            if(reuse_sec_val < 65535){
                                                uint32_t cpu_queue_counter = atomicAdd(cpu_agg_queue_counter+sec_head_ptr, 1);
                                                if(cpu_queue_counter < cpu_agg_queue_depth){

                                                    // if(cpu_queue_counter == 1){
                                                    //     printf("memcpy ptr: reuse: %llu idx: %llu\n", (unsigned long long) (reuse_sec & (0x0000FFFFFFFFFFFF)), (unsigned long long) reuse_sec_val);
                                                    // }
                                                    uint64_t memcpy_ptr = ((reuse_val - reuse_sec_val) << 48) | (queue_idx & 0x0000FFFFFFFFFFFF);

                                                    cpu_agg_meta_queue[cpu_queue_counter + sec_head_ptr * cpu_agg_queue_depth] = memcpy_ptr;
                                                    cpu_agg_loc_queue[cpu_queue_counter + sec_head_ptr * cpu_agg_queue_depth] = (reuse_sec & (0x0000FFFFFFFFFFFF));
                                                }
                                            }
                                        }
                                    }                                     
                                }
                            } 
            
                            fail = false;

                            this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                        }
                        else { 
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
                        }
                    }
                }

                if (!fail) {
                    this->cache_pages[page].next_reuse = (0xFFFF000000000000);
                    //this->cache_pages[page].reuse_mask = (0);
                    this->cache_pages[page].page_translation = global_address;
                    for(int j = 0; j < 8; j++){
                        this->cache_pages[page].reuse_chunk[j] = no_reuse;
                    }
                }
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }


        }

        count++;
        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
//             __nanosleep(ns);
//             if (ns < 256) {
//                 ns *= 2;
//             }
#endif

        }

    } while(fail);
    return page;

}


__forceinline__
__device__
uint32_t page_cache_d_t::find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_, simt::atomic<uint64_t, simt::thread_scope_device>& access_cnt, uint64_t* evicted_p_array) {
    bool fail = true;
    uint64_t count = 0;
    uint64_t global_address =(uint64_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    unsigned int ns = 8;
	uint64_t j = 0;
    uint64_t expected_state = VALID;
    uint64_t new_expected_state = 0;

    do {

//	if (++count %100000 == 0)
//		printf("here\tc: %llu\n", (unsigned long long) count);

        //if (count < this->n_pages)
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_relaxed);
        //this->page_take_lock[page].compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
        //not assigned to anyone yet
        if ( v == FREE ) {
    		lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                this->cache_pages[page].page_translation = global_address;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {

            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                //uint32_t previous_address = this->cache_pages[page].page_translation;
                uint64_t previous_global_address = this->cache_pages[page].page_translation;
                //uint8_t previous_range = this->cache_pages[page].range_id;
                uint64_t previous_range = previous_global_address & n_ranges_mask;
                uint64_t previous_address = previous_global_address >> n_ranges_bits;
                //uint32_t new_state = BUSY;
                //if ((previous_range >= range_cap) || (previous_address >= n_pages))
 		//printf("prev add:%llu\n",(unsigned long long) previous_address);
 		//printf("prev_ga: %llu\tprev_range: %llu\tprev_add: %llu\trange_cap: %llu\tn_pages: %llu\n", (unsigned long long) previous_global_address, (unsigned long long) previous_range, (unsigned long long) previous_address,           (unsigned long long) range_cap, (unsigned long long) n_pages);
                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_relaxed);

                uint32_t cnt = expected_state & CNT_MASK;
                uint32_t b = expected_state & BUSY;
 		//printf("cnt: %lu b: %lu\n", (unsigned long)cnt, (unsigned long)  b);
 		if ((cnt == 0) && (b == 0) ) {
                    new_expected_state = this->ranges[previous_range][previous_address].state.fetch_or(BUSY, simt::memory_order_acquire);
                    if (((new_expected_state & BUSY ) == 0) ) {
                        //while ((new_expected_state & CNT_MASK ) != 0) new_expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_acquire);
                        if (((new_expected_state & CNT_MASK ) == 0) ) {
                            if ((new_expected_state & DIRTY)) {
                                uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);
                                //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
                                uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                   //            printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
                     //                   (unsigned long long) previous_range, (unsigned long long)previous_address,
                       //                (unsigned long long) ctrl, (unsigned long long) index);
                                if (ctrl == ALL_CTRLS) {
                                    for (ctrl = 0; ctrl < n_ctrls; ctrl++) {
                                        Controller* c = this->d_ctrls[ctrl];
                                        uint32_t queue = queue_ % (c->n_qps);
                                        write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                    }
                                }
                                else {

                                    Controller* c = this->d_ctrls[ctrl];
                                    uint32_t queue = queue_ % (c->n_qps);

                                    //index = ranges_page_starts[previous_range] + previous_address;


                                    write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                                }
                            }

                            fail = false;
//               	printf("prev add:%llu\n",(unsigned long long) previous_address);
		// uint64_t e_idx = access_cnt.fetch_add(1, simt::memory_order_relaxed);
	//	evicted_p_array[e_idx]=previous_address;

	       		    this->ranges[previous_range][previous_address].state.fetch_and(CNT_MASK, simt::memory_order_release);
                        }
                        else { 
                            this->ranges[previous_range][previous_address].state.fetch_and(DISABLE_BUSY_MASK, simt::memory_order_release);
//if ((j % 1000000) == 0) {
//                printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//}
                        }
                    }
                }

                //this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                if (!fail) {
                    //this->cache_pages[page].page_translation = address;
                    //this->cache_pages[page].range_id = range_id;
//                    this->page_translation[page] = global_address;
                    this->cache_pages[page].page_translation = global_address;
                }
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }


        }

        count++;
/*if (fail) {
  if ((++j % 1000000) == 0) {
  printf("failed to find slot j: %llu\n", (unsigned long long) j);
  }
  }*/
        if (fail) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
//             __nanosleep(ns);
//             if (ns < 256) {
//                 ns *= 2;
//             }
#endif
            //   if ((j % 10000000) == 0) {
            //     printf("failed to find slot j: %llu\taddr: %llx\tpage: %llx\texpected_state: %llx\tnew_expected_date: %llx\n", (unsigned long long) j, (unsigned long long) address, (unsigned long long)page, (unsigned long long) expected_state, (unsigned long long) new_expected_state);
//            }
//	   expected_state = 0;
//	   new_expected_state = 0;


        }

    } while(fail);
    return page;

}


inline __device__ void poll_async(QueuePair* qp, uint16_t cid, uint16_t sq_pos) {
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    //sq_dequeue(&qp->sq, sq_pos);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq);



    put_cid(&qp->sq, cid);
}

inline __device__ void access_data_async(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode, uint16_t * cid, uint16_t* sq_pos) {
    nvm_cmd_t cmd;
    *cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    *sq_pos = sq_enqueue(&qp->sq, &cmd);



}

inline __device__ void enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
    nvm_cmd_rw_blks(cmd, starting_lba, 1);
    unsigned int ns = 8;
    do {
        //check if new head past pc_pos
        //cur_pc_head == new head
        //prev_pc_head == old head
        //pc_pos == position i wanna move the head past
        uint64_t cur_pc_head = pc->q_head->load(simt::memory_order_relaxed);
        //sec == true when cur_pc_head past pc_pos
        bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
            ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
            ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

        if (sec) break;

        //if not
        uint64_t qlv = pc->q_lock->load(simt::memory_order_relaxed);
        //got lock
        if (qlv == 0) {
            qlv = pc->q_lock->fetch_or(1, simt::memory_order_acquire);
            if (qlv == 0) {
                uint64_t cur_pc_tail;// = pc->q_tail.load(simt::memory_order_acquire);

                uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, pc->q_tail, &cur_pc_tail);
                uint32_t head, head_;
                uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

                pc->q_head->store(cur_pc_tail, simt::memory_order_release);
                pc->q_lock->store(0, simt::memory_order_release);
                pc->extra_reads->fetch_add(1, simt::memory_order_relaxed);
                cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);



                break;
            }
        }
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
    } while(true);

}

inline __device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);
   // printf("read_data: starting_lba: %llu, n_blocks: %llu, pc_entry: %llu\n", starting_lba, n_blocks, pc_entry);


    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);


    //enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head)
    enqueue_second(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);



    put_cid(&qp->sq, cid);


}


inline __device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);

    //printf("write_data: starting_lba: %llu, n_blocks: %llu, pc_entry: %llu\n", starting_lba, n_blocks, pc_entry);

    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
    uint32_t head, head_;
    uint64_t pc_pos;
    uint64_t pc_prev_head;

    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);
    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
    pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);

}

inline __device__ void access_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    ////printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    ////printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    //sq_dequeue(&qp->sq, sq_pos);




    put_cid(&qp->sq, cid);


}



//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif


#endif // __PAGE_CACHE_H__
