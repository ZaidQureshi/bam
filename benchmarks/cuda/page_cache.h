#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#include "util.h"
#include <nvm_types.h>
#include <nvm_util.h>
#include "buffer.h"
#include "ctrl.h"
#include "settings.h"
#include <iostream>

#define FREE 2
// enum locality {HIGH_SPATIAL, LOW_SPATIAL, MEDIUM_SPATIAL};
// template <typname T>
// stryct array_t {
//   range_t<T>* ranges;
//   uint32_t n_ranges;

//   void add_range(start_idx, end_idx, locality l )
//   {
//   ranges.push_back(new range(star))
//   }
// }

enum page_state {USE = 1ULL, USE_DIRTY = ((1ULL << 63) | 1), VALID_DIRTY = (1ULL << 63), VALID = 0ULL, INVALID = (ULLONG_MAX & 0x7fffffffffffffff), BUSY = ((ULLONG_MAX & 0x7fffffffffffffff)-1)};

#define USE (1ULL)
#define USE_DIRTY ((1ULL << 63) | 1)
#define VALID_DIRTY (1ULL << 63)
#define VALID (0ULL)
#define INVALID (ULLONG_MAX & 0x7fffffffffffffff)
#define BUSY ((ULLONG_MAX & 0x7fffffffffffffff)-1)

struct page_cache_t;

typedef padded_struct* page_states_t;



__device__ void read_data(page_cache_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);

template <typename T>
struct range_t {
    uint64_t index_start;
    uint64_t index_end;
    uint64_t range_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_end;
    page_cache_t* cache;
    page_states_t page_states;
    padded_struct* page_addresses;
    //padded_struct* page_vals;  //len = num of pages for data
    //

    BufferPtr page_states_buff;
    BufferPtr page_addresses_buff;

    BufferPtr range_buff;

    void* d_range_ptr;


    range_t(uint64_t is, uint64_t ie, uint64_t ps, uint64_t pe, uint64_t pso, uint64_t p_size, page_cache_t* c_h, const Settings& settings) {
        index_start = is;
        index_end = ie;
        //range_id = (c_h->range_count)++;
        page_start = ps;
        page_end = pe;
        page_size = p_size;
        page_start_offset = pso;
        size_t s = (page_end-page_start);//*page_size / c_h->page_size;

        cache = (page_cache_t*) c_h->d_pc_ptr;
        page_states_buff = createBuffer(s * sizeof(padded_struct), settings.cudaDevice);
        page_states = (page_states_t) page_states_buff.get();

        padded_struct* ts = new padded_struct[s];
        for (size_t i = 0; i < s; i++)
            ts[i].val = INVALID;
        cuda_err_chk(cudaMemcpy(page_states, ts, s * sizeof(padded_struct), cudaMemcpyHostToDevice));
        delete ts;

        page_addresses_buff = createBuffer(s * sizeof(padded_struct), settings.cudaDevice);
        page_addresses = (padded_struct*) page_addresses_buff.get();

        range_buff = createBuffer(sizeof(range_t<T>), settings.cudaDevice);
        d_range_ptr = range_buff.get();
        range_id  = c_h->n_ranges++;

        cuda_err_chk(cudaMemcpy(d_range_ptr, this, sizeof(range_t<T>), cudaMemcpyHostToDevice));

        c_h->add_range(this);

    }
    __device__
    uint64_t get_page(const size_t i) const {
        uint64_t index = ((index_start + i) * sizeof(T) + page_start_offset) >> (cache->page_size_log);
        return index;
    }
    __device__
    uint64_t get_subindex(const size_t i) const {
        uint64_t index = ((index_start + i) * sizeof(T) + page_start_offset) & (cache->page_size_minus_1);
        return index;
    }
    __device__
    uint64_t get_global_address(const size_t page) const {
        return ((page << cache->n_ranges_bits) | range_id);
    }
    __device__
    void release_page(const size_t pg) const {
        uint64_t index = pg;
        page_states[index].val.fetch_sub(1, simt::memory_order_release);
    }
    __device__
    uint64_t acquire_page(const size_t pg, const uint32_t count, const bool write) const {
        uint64_t index = pg;
        uint64_t expected_state = VALID;
        uint64_t new_state = USE;
        uint64_t global_address = (index << cache->n_ranges_bits) | range_id;
        bool fail = true;
        T ret;
        do {
            bool pass = false;
            expected_state = page_states[index].val.load(simt::memory_order_acquire);
            switch (expected_state) {
                case VALID:
                    new_state = ((write) ? USE_DIRTY : USE) + count - 1;
                    pass = page_states[index].val.compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        return ((uint64_t)((cache->base_addr+(page_trans * cache->page_size))));
                        //page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
                case BUSY:
                    //expected_state = VALID;
                    //new_state = USE;
                    break;
                case INVALID:
                    pass = page_states[index].val.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = cache->find_slot(index, range_id);
                        //fill in
                        uint32_t bid = blockIdx.x;
                        uint32_t smid = get_smid();

                        uint32_t ctrl = bid & ((cache->n_ctrls)-1);
                        Controller* c = cache->d_ctrls[ctrl];
                        uint32_t queue = smid & (c->n_qps-1);

                        read_data(cache, (c->d_qps)+queue, index, cache->page_size >> c->blk_size_log, page_trans);
                        page_addresses[index].val.store(page_trans, simt::memory_order_release);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        new_state = ((write) ? USE_DIRTY : USE) + count - 1;
                        page_states[index].val.store(new_state, simt::memory_order_release);
                        return ((uint64_t)((cache->base_addr+(page_trans * cache->page_size))));

                        fail = false;
                    }
                    //else {
                    //    expected_state = INVALID;
                    //    new_state = BUSY;
                    //}


                    break;
                default:
                    new_state = expected_state + count;
                    if (write)
                        new_state |= VALID_DIRTY;
                    pass = page_states[index].val.compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        return ((uint64_t)((cache->base_addr+(page_trans * cache->page_size))));
                        //page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
            }

        } while (fail);
        //return ret;
    }
    __device__
    T operator[](const size_t i) const {
        uint64_t index = get_page(i);
        uint64_t subindex = get_subindex(i);
        //printf("tid: %llu\ti: %llu\tindex: %llu\tsubindex: %llu\tpage_size_log: %llu\tpage_size_minus_1: %llu\n", (unsigned long long) (blockIdx.x * blockDim.x + threadIdx.x), (unsigned long long) i,
        //       (unsigned long long) index, (unsigned long long) subindex, (unsigned long long) cache->page_size_log, (unsigned long long) cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE;
        uint64_t global_address = (index << cache->n_ranges_bits) | range_id;
        //if (threadIdx.x == 63) {
        //printf("page: %llu\tsubindex: %llu\n", (unsigned long long) index, (unsigned long long) subindex);
        //}
        bool fail = true;
        T ret;

        do {
            bool pass = false;
            expected_state = page_states[index].val.load(simt::memory_order_acquire);
            switch (expected_state) {
                case VALID:
                    pass = page_states[index].val.compare_exchange_weak(expected_state, USE, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ret = ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0];
                        page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
                case BUSY:
                    //expected_state = VALID;
                    //new_state = USE;
                    break;
                case INVALID:
                    pass = page_states[index].val.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = cache->find_slot(index, range_id);
                        //fill in
                        uint32_t bid = blockIdx.x;
                        uint32_t smid = get_smid();

                        uint32_t ctrl = bid & ((cache->n_ctrls)-1);
                        Controller* c = cache->d_ctrls[ctrl];
                        uint32_t queue = smid & (c->n_qps-1);

                        read_data(cache, (c->d_qps)+queue, index, cache->page_size >> c->blk_size_log, page_trans);
                        page_addresses[index].val.store(page_trans, simt::memory_order_release);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ret = ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0];
                        page_states[index].val.store(VALID, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = INVALID;
                    //    new_state = BUSY;
                    //}


                    break;
                default:
                    new_state = expected_state + 1;
                    pass = page_states[index].val.compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ret = ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0];
                        page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
            }

        } while (fail);
        return ret;
    }

    __device__
    void operator()(const size_t i, const T val) {
        uint64_t index = get_page(i);
        uint64_t subindex = get_subindex(i);
        //printf("tid: %llu\ti: %llu\tindex: %llu\tsubindex: %llu\tpage_size_log: %llu\tpage_size_minus_1: %llu\n", (unsigned long long) (blockIdx.x * blockDim.x + threadIdx.x), (unsigned long long) i,
        //       (unsigned long long) index, (unsigned long long) subindex, (unsigned long long) cache->page_size_log, (unsigned long long) cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE_DIRTY;
        uint64_t global_address = (index << cache->n_ranges_bits) | range_id;
        bool fail = true;
        T ret;
        do {
            bool pass = false;
            expected_state = page_states[index].val.load(simt::memory_order_acquire);
            switch (expected_state) {
                case VALID:
                    pass = page_states[index].val.compare_exchange_weak(expected_state, USE_DIRTY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0] = val;
                        page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
                case BUSY:
                    //expected_state = VALID;
                    //new_state = USE;
                    break;
                case INVALID:
                    pass = page_states[index].val.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = cache->find_slot(index, range_id);
                        //fill in
                        page_addresses[index].val.store(page_trans, simt::memory_order_release);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0] = val;
                        page_states[index].val.store(VALID_DIRTY, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = INVALID;
                    //    new_state = BUSY;
                    //}


                    break;
                default:
                    new_state = (expected_state + 1) | VALID_DIRTY;
                    pass = page_states[index].val.compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ((T*)((cache->base_addr+(page_trans * cache->page_size)) + subindex))[0] = val;
                        page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    //else {
                    //    expected_state = VALID;
                    //    new_state = USE;
                    //}
                    break;
            }

        } while (fail);


    }
};


template<typename T>
struct array_t {
    uint64_t n_elems;
    uint64_t start_offset;
    array_t<T>* d_array_ptr;
    range_t<T>** d_ranges;
    uint64_t n_ranges;

    BufferPtr d_array_buff;
    BufferPtr d_ranges_buff;

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings) {
        n_elems = num_elems;
        start_offset = disk_start_offset;

        n_ranges = ranges.size();
        d_array_buff = createBuffer(sizeof(array_t<T>), settings.cudaDevice);
        d_array_ptr = (array_t<T>*) d_array_buff.get();

        d_ranges_buff = createBuffer(n_ranges * sizeof(range_t<T>*), settings.cudaDevice);
        d_ranges = (range_t<T>**) d_ranges_buff.get();
        for (size_t k = 0; k < n_ranges; k++)
            cuda_err_chk(cudaMemcpy(d_ranges+k, &(ranges[k]->d_range_ptr), sizeof(range_t<T>*), cudaMemcpyHostToDevice));

        cuda_err_chk(cudaMemcpy(d_array_ptr, this, sizeof(array_t<T>), cudaMemcpyHostToDevice));
    }
    __device__
    int64_t find_range(const size_t i) const {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++) {
            if ((d_ranges[k]->index_start <= i) && (d_ranges[k]->index_end > i)) {;
                range = k;
                break;
            }

        }
        return range;
    }
    __device__
    T seq_read(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        T ret;

        if (r != -1) {
            uint32_t mask = __activemask();
            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = 1 << lane;
            uint32_t bef = 0;
            uint32_t aft = 0;
            uint32_t mask_copy = mask;
            uint64_t page = d_ranges[r]->get_page(i);
            uint64_t subindex = d_ranges[r]->get_subindex(i);
            uint64_t gaddr = d_ranges[r]->get_global_address(page);

            uint64_t master = lane;
            //#pragma unroll
            for (size_t k = 0; k < 32; k++) {
                uint32_t b_m = 1 << k;
                if ((mask & b_m) != 0) {
                    uint64_t o_ga = __shfl_sync(mask, gaddr, k);
                    if ((o_ga == gaddr) && (k < lane)) {
                        bef++;
                        eq_mask |= 1 << k;
                        if (master > k)
                            master = k;
                    }
                    else if ((o_ga == gaddr) && (k > lane)) {
                        aft++;
                        eq_mask |= 1 << k;
                        if (master > k)
                            master = k;
                    }
                }
            }
            
            uint64_t base_master;
            uint64_t base;
            uint32_t count = __popc(eq_mask);
            if (bef == 0) {
                base = d_ranges[r]->acquire_page(page, count, false);
                base_master = base;
                //printf("++tid: %llu\tbase: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master);

            }
            base_master = __shfl_sync(eq_mask,  base_master, master);
            //if (threadIdx.x == 63) {
                //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T*)(base_master+subindex))[0];
            //__syncwarp(eq_mask);
            //if (bef == 0) {
                d_ranges[r]->release_page(page);
            //}



        }
        return ret;

    }
    __device__
    void seq_write(const size_t i, const T val) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        if (r != -1) {
            uint32_t mask = __activemask();
            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = 1 << lane;
            uint32_t bef = 0;
            uint32_t aft = 0;
            uint32_t mask_copy = mask;
            uint64_t page = d_ranges[r]->get_page(i);
            uint64_t subindex = d_ranges[r]->get_subindex(i);
            uint64_t gaddr = d_ranges[r]->get_global_address(page);
            uint64_t master = lane;
            #pragma unroll
            for (size_t k = 0; k < 32; k++) {
                uint32_t b_m = 1 << k;
                if ((mask & b_m) != 0) {
                    uint64_t o_ga = __shfl_sync(mask, gaddr, k);
                    if ((o_ga == gaddr) && (k < lane)) {
                        bef++;
                        eq_mask |= 1 << k;
                        if (master < k)
                            master = k;
                    }
                    else if ((o_ga == gaddr) && (k > lane)) {
                        aft++;
                        eq_mask |= 1 << k;
                        if (master < k)
                            master = k;
                    }
                }
            }

            uint64_t base_master;
            uint64_t base;
            uint32_t count = __popc(eq_mask);
            if (bef == 0) {
                base = d_ranges[r]->acquire_page(page, count, true);
                base_master = base;

            }
            base_master = __shfl_sync(eq_mask, base, master);
            ((T*)(base_master+subindex))[0] = val;
            //__syncwarp(eq_mask);
            //if (bef == 0) {
                d_ranges[r]->release_page(page);
            //}



        }
    }
    __device__
    T operator[](size_t i) const {
        size_t k = 0;
        bool found = false;
        for (; k < n_ranges; k++) {
            if ((d_ranges[k]->index_start <= i) && (d_ranges[k]->index_end > i)) {
                found = true;
                break;
            }

        }
        if (found)
            return ((*(d_ranges[k]))[i-d_ranges[k]->index_start]);
    }
    __device__
    void operator()(size_t i, T val) const {
        size_t k = 0;
        bool found = false;
        uint32_t mask = __activemask();
        for (; k < n_ranges; k++) {
            if ((d_ranges[k]->index_start <= i) && (d_ranges[k]->index_end > i)) {
                found = true;
                break;
            }

        }
        __syncwarp(mask);
        if (found)
            (*(d_ranges[k]))(i-d_ranges[k]->index_start, val);
    }

};


struct page_cache_t {
    uint8_t* base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    padded_struct* page_translation;         //len = num of pages in cache
    padded_struct* page_take_lock;      //len = num of pages in cache
    padded_struct page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t* prp_list;              //len = num of pages in cache if page_size > ctrl.page_size *2
    uint64_t    ctrl_page_size;
    uint64_t  range_cap;
    //uint64_t  range_count;
    page_states_t*   ranges;
    page_states_t*   h_ranges;
    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;

    //void* d_pc;

    //BufferPtr prp2_list_buf;
    bool prps;
    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr page_translation_buf;
    BufferPtr page_take_lock_buf;
    BufferPtr ranges_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;


    void* d_pc_ptr;

    Controller** d_ctrls;
    uint64_t n_ctrls;

    template <typename T>
    void add_range(range_t<T>* range) {

        h_ranges[range->range_id] = range->page_states;
        cuda_err_chk(cudaMemcpy(ranges, h_ranges, n_ranges* sizeof(page_states_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, this, sizeof(page_cache_t), cudaMemcpyHostToDevice));

    }

    page_cache_t(const uint64_t ps, const uint64_t np, const Settings& settings, const Controller& ctrl, const uint64_t max_range, const std::vector<Controller*>& ctrls)
        : page_size(ps), page_size_minus_1(ps-1), n_pages(np), n_pages_minus_1(np-1), ctrl_page_size(ctrl.ctrl->page_size) {
        n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), settings.cudaDevice);
        d_ctrls = (Controller**) d_ctrls_buff.get();
        for (size_t k = 0; k < n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));

        range_cap = max_range;
        n_ranges = 0;
        n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        n_ranges_mask = max_range-1;
        std::cout << "n_ranges_bits: " << std::dec << n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << n_ranges_mask << std::endl;
        page_ticket.val = 0;
        page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(page_states_t), settings.cudaDevice);
        ranges = (page_states_t*)ranges_buf.get();
        h_ranges = new page_states_t[max_range];

        page_translation_buf = createBuffer(np * sizeof(padded_struct), settings.cudaDevice);
        page_translation = (padded_struct*)page_translation_buf.get();

        page_take_lock_buf = createBuffer(np * sizeof(padded_struct), settings.cudaDevice);
        page_take_lock =  (padded_struct*)page_take_lock_buf.get();

        padded_struct* tps = new padded_struct[np];
        for (size_t i = 0; i < np; i++)
            tps[i].val = FREE;
        cuda_err_chk(cudaMemcpy(page_take_lock, tps, np*sizeof(padded_struct), cudaMemcpyHostToDevice));
        delete tps;



        uint64_t cache_size = ps*np;
        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);
        base_addr = (uint8_t*) this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((page_size > (ctrl.ns.lba_data_size * uints_per_page)) || (np == 0))
            throw error(string("page_cache_t: Can't have such big io reqs"));
        if (ps <= this->pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/ps;
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) this->prp1_buf.get();


            std::cout << np << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one *  this->pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  this->pages_dma.get()->n_ioaddrs);
            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps;
                    //std::cout << std::dec << (i*how_many_in_one + j) << "\t" << std::hex << (((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*ps) << std::endl;
                }
            }
            cuda_err_chk(cudaMemcpy(prp1, temp, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete temp;
            //std::cout << "HERE1\n";
            //free(temp);
            //std::cout << "HERE2\n";
            prps = false;
        }

        else if ((ps > this->pages_dma.get()->page_size) && (ps <= (this->pages_dma.get()->page_size * 2))) {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) this->prp1_buf.get();
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp2 = (uint64_t*) this->prp2_buf.get();
            uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            std::memset(temp1, 0, np * sizeof(uint64_t));
            uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
            }
            cuda_err_chk(cudaMemcpy(prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            prps = true;
        }
        else {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp1 = (uint64_t*) this->prp1_buf.get();
            uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), settings.cudaDevice);
            prp2 = (uint64_t*) this->prp2_buf.get();
            uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t* temp3 = (uint64_t*) malloc(prp_list_size);
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
            cuda_err_chk(cudaMemcpy(prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            free(temp3);
            prps = true;
        }


        pc_buff = createBuffer(sizeof(page_cache_t), settings.cudaDevice);
        d_pc_ptr = pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, this, sizeof(page_cache_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";

    }


    __device__
    uint64_t find_slot(uint64_t address, uint64_t range_id) {
        bool fail = true;
        uint64_t count = 0;
        uint64_t global_address = (address << n_ranges_bits) | range_id;
        uint64_t page = 0;
        do {
            //if (count < this->n_pages)
                page = this->page_ticket.val.fetch_add(1, simt::memory_order_acquire)  & (this->n_pages_minus_1);
            uint64_t unlocked = UNLOCKED;
            bool lock = false;
            uint64_t v = this->page_take_lock[page].val.load(simt::memory_order_acquire);
            //this->page_take_lock[page].val.compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            //not assigned to anyone yet
            if ( v == FREE ) {
                lock = this->page_take_lock[page].val.compare_exchange_strong(v, LOCKED, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                if ( lock ) {
                    this->page_translation[page].val.store(global_address, simt::memory_order_release);
                    this->page_take_lock[page].val.store(UNLOCKED, simt::memory_order_release);
                    fail = false;
                }
            }
            //assigned to someone and was able to take lock
            else if ( v == UNLOCKED ) {

                lock = this->page_take_lock[page].val.compare_exchange_strong(v, LOCKED, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                if (lock) {
                    uint64_t previous_global_address = this->page_translation[page].val.load(simt::memory_order_acquire);
                    uint64_t previous_range = previous_global_address & n_ranges_mask;
                    uint64_t previous_address = previous_global_address >> n_ranges_bits;
                    uint64_t expected_state = VALID;
                    uint64_t new_state = BUSY;
                    bool pass = false;
                    //if ((previous_range >= range_cap) || (previous_address >= n_pages))
                    //    printf("prev_ga: %llu\tprev_range: %llu\tprev_add: %llu\trange_cap: %llu\tn_pages: %llu\n", (unsigned long long) previous_global_address, (unsigned long long) previous_range, (unsigned long long) previous_address,
                    //           (unsigned long long) range_cap, (unsigned long long) n_pages);
                    expected_state = this->ranges[previous_range][previous_address].val.load(simt::memory_order_acquire);

                    //this->ranges[previous_range][previous_address].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                    switch(expected_state) {
                        case VALID:
                            pass = this->ranges[previous_range][previous_address].val.compare_exchange_strong(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                            if (pass) {
                                this->ranges[previous_range][previous_address].val.store(INVALID, simt::memory_order_release);
                                fail = false;
                            }
                            break;
                        case INVALID:
                            pass =  this->ranges[previous_range][previous_address].val.compare_exchange_strong(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                            if (pass) {
                                this->ranges[previous_range][previous_address].val.store(INVALID, simt::memory_order_release);
                                fail = false;
                            }
                            break;
                        case VALID_DIRTY:


                            //if ((count > this->n_pages)) {
                            pass =  this->ranges[previous_range][previous_address].val.compare_exchange_strong(expected_state, BUSY, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                            if  (pass) {
                                //if ((this->page_dirty_start[page].load(simt::memory_order_acquire) == this->page_dirty_end[page].load(simt::memory_order_acquire))) {

                                //writeback
                                this->ranges[previous_range][previous_address].val.store(INVALID, simt::memory_order_release);

                                fail = false;
                                //}
                                //else {
                                //    page_states[previous_address].store(expected_state, simt::memory_order_release);
                                //}
                            }
                            //}
                            break;
                        default:
                            //printf("here\n");
                            break;

                    }
                    if (!fail)
                        this->page_translation[page].val.store(global_address, simt::memory_order_release);
                    this->page_take_lock[page].val.store(UNLOCKED, simt::memory_order_release);
                }


            }

            count++;


        } while(fail);
        return page;
    }




};

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
    cq_dequeue(&qp->cq, cq_pos);
    sq_dequeue(&qp->sq, sq_pos);



    put_cid(&qp->sq, cid);


}





#endif // __PAGE_CACHE_H__
