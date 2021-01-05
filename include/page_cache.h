#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#include "util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
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

enum page_state {USE = 1U, USE_DIRTY = ((1U << 31) | 1), VALID_DIRTY = (1U << 31), VALID = 0U, INVALID = (UINT_MAX & 0x7fffffff), BUSY = ((UINT_MAX & 0x7fffffff)-1)};

#define USE (1U)
#define USE_DIRTY ((1U << 31) | 1)
#define VALID_DIRTY (1U << 31)
#define VALID (0U)
#define INVALID (UINT_MAX & 0x7fffffff)
#define BUSY ((UINT_MAX & 0x7fffffff)-1)

struct page_cache_t;

struct page_cache_d_t;

typedef padded_struct_pc* page_states_t;

template <typename T>
struct range_t;


struct page_cache_d_t {
    uint8_t* base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    uint32_t* page_translation;         //len = num of pages in cache
    //padded_struct_pc* page_translation;         //len = num of pages in cache
    padded_struct_pc* page_take_lock;      //len = num of pages in cache
    padded_struct_pc* page_ticket;
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

    uint64_t* ranges_page_starts;
    simt::atomic<uint64_t, simt::thread_scope_device>* ctrl_counter;


    Controller** d_ctrls;
    uint64_t n_ctrls;
    bool prps;
    
    uint64_t n_blocks_per_page; 

    __forceinline__
    __device__
    uint32_t find_slot(uint64_t address, uint64_t range_id);
};


template <typename T>
struct range_d_t {
    uint64_t index_start;
    uint64_t index_end;
    uint64_t range_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_end;
    uint8_t* src;

    simt::atomic<uint64_t, simt::thread_scope_device> access_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> read_io_cnt;


    page_states_t page_states;
    //padded_struct_pc* page_addresses;
    uint32_t* page_addresses;
    //padded_struct_pc* page_vals;  //len = num of pages for data
    //void* self_ptr;
    page_cache_d_t cache;
    //range_d_t(range_t<T>* rt);
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
    uint64_t acquire_page(const size_t pg, const uint32_t count, const bool write) ;
    __forceinline__
    __device__
    void write_done(const size_t pg, const uint32_t count) const;
    __forceinline__
    __device__
    T operator[](const size_t i) ;
    __forceinline__
    __device__
    void operator()(const size_t i, const T val);
};

__device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);


template <typename T>
struct range_t {
    range_d_t<T> rdt;

    range_d_t<T>* d_range_ptr;
    page_cache_d_t* cache;

    BufferPtr page_states_buff;
    BufferPtr page_addresses_buff;

    BufferPtr range_buff;



    range_t(uint64_t is, uint64_t ie, uint64_t ps, uint64_t pe, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice);



};

template <typename T>
range_t<T>::range_t(uint64_t is, uint64_t ie, uint64_t ps, uint64_t pe, uint64_t pso, uint64_t p_size, page_cache_t* c_h, uint32_t cudaDevice) {
    rdt.access_cnt = 0;
    rdt.miss_cnt = 0;
    rdt.hit_cnt = 0;
    rdt.read_io_cnt = 0;
    rdt.index_start = is;
    rdt.index_end = ie;
    //range_id = (c_h->range_count)++;
    rdt.page_start = ps;
    rdt.page_end = pe;
    rdt.page_size = p_size;
    rdt.page_start_offset = pso;
    size_t s = pe;//(rdt.page_end-rdt.page_start);//*page_size / c_h->page_size;

    cache = (page_cache_d_t*) c_h->d_pc_ptr;
    page_states_buff = createBuffer(s * sizeof(padded_struct_pc), cudaDevice);
    rdt.page_states = (page_states_t) page_states_buff.get();

    padded_struct_pc* ts = new padded_struct_pc[s];
    for (size_t i = 0; i < s; i++)
        ts[i] = INVALID;
    printf("S value: %llu\n", (unsigned long long)s);
    cuda_err_chk(cudaMemcpy(rdt.page_states, ts, s * sizeof(padded_struct_pc), cudaMemcpyHostToDevice));
    delete ts;

    page_addresses_buff = createBuffer(s * sizeof(uint32_t), cudaDevice);
    rdt.page_addresses = (uint32_t*) page_addresses_buff.get();
    //page_addresses_buff = createBuffer(s * sizeof(padded_struct_pc), cudaDevice);
    //page_addresses = (padded_struct_pc*) page_addresses_buff.get();

    range_buff = createBuffer(sizeof(range_d_t<T>), cudaDevice);
    d_range_ptr = (range_d_t<T>*)range_buff.get();
    rdt.range_id  = c_h->pdt.n_ranges++;


    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

    c_h->add_range(this);

    rdt.cache = c_h->pdt;
    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_page(const size_t i) const {
    uint64_t index = ((index_start + i) * sizeof(T) + page_start_offset) >> (cache.page_size_log);
    return index;
}
template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::get_subindex(const size_t i) const {
    uint64_t index = ((index_start + i) * sizeof(T) + page_start_offset) & (cache.page_size_minus_1);
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
    page_states[index].fetch_sub(1, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
void range_d_t<T>::release_page(const size_t pg, const uint32_t count) const {
    uint64_t index = pg;
    page_states[index].fetch_sub(count, simt::memory_order_release);
}

template <typename T>
__forceinline__
__device__
uint64_t range_d_t<T>::acquire_page(const size_t pg, const uint32_t count, const bool write) {
    uint64_t index = pg;
    uint32_t expected_state = VALID;
    uint32_t new_state = USE;
    uint32_t global_address = (index << cache.n_ranges_bits) | range_id;
    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    bool miss = false;
    T ret;
    do {
        bool pass = false;
        expected_state = page_states[index].load(simt::memory_order_acquire);
        switch (expected_state) {
            case VALID:
                new_state = ((write) ? USE_DIRTY : USE) + count - 1;
                pass = page_states[index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                if (pass) {
                    //uint32_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                    uint32_t page_trans = page_addresses[index];
                    // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));

                    //page_states[index].fetch_sub(1, simt::memory_order_release);
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
                pass = page_states[index].compare_exchange_weak(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                if (pass) {
                    uint32_t page_trans = cache.find_slot(index, range_id);
                    //fill in
                    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                    //uint32_t ctrl = (tid/32) % (n_ctrls);
                    //uint32_t ctrl = get_smid() % (cache.n_ctrls);
                    uint32_t ctrl = cache.ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (cache.n_ctrls);
                    Controller* c = cache.d_ctrls[ctrl];
                    uint32_t queue = (tid/32) % (c->n_qps);
                    //uint32_t queue = c->queue_counter.fetch_add(1, simt::memory_order_relaxed) % (c->n_qps);
                    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                    read_data(&cache, (c->d_qps)+queue, ((index+page_start)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                    //page_addresses[index].store(page_trans, simt::memory_order_release);
                    page_addresses[index] = page_trans;
                    // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                    new_state = ((write) ? USE_DIRTY : USE) + count - 1;
                    page_states[index].store(new_state, simt::memory_order_release);
                    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));

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
                pass = page_states[index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                if (pass) {
                    //uint32_t page_trans = page_addresses[index].load(simt::memory_order_acquire);
                    uint32_t page_trans = page_addresses[index];
                    // while (cache.page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                    return ((uint64_t)((cache.base_addr+(page_trans * cache.page_size))));
                    //page_states[index].fetch_sub(1, simt::memory_order_release);
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




template<typename T>
struct array_d_t {
    uint64_t n_elems;
    uint64_t start_offset;
    uint64_t n_ranges;
    uint8_t *src;

    range_d_t<T>* d_ranges;
    __forceinline__
    __device__
    int64_t find_range(const size_t i) const {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++) {
            if ((d_ranges[k].index_start <= i) && (d_ranges[k].index_end > i)) {;
                range = k;
                break;
            }

        }
        return range;
    }
    __forceinline__
    __device__
    T seq_read(const size_t i) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        T ret;

        if (r != -1) {
            uint32_t mask = __activemask();
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);
            uint64_t p_s = d_ranges[r].page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(__activemask(), gaddr);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            bool memcpyflag_master;
            bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane) {
                //std::pair<uint64_t, bool> base_memcpyflag;
                base = d_ranges[r].acquire_page(page, count, false);
                base_master = base;
//                printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T*)(base_master+subindex))[0];
            __syncwarp(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);

        }
        return ret;
    }
    __forceinline__
    __device__
    void seq_write(const size_t i, const T val) const {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        if (r != -1) {
            uint32_t mask = __activemask();
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);
            uint64_t p_s = d_ranges[r].page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(__activemask(), gaddr);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            bool memcpyflag_master;
            bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane) {
                base = d_ranges[r].acquire_page(page, count, true);
                base_master = base;
//                printf("++tid: %llu\tbase: %llu  memcpyflag_master:%llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master, (unsigned long long) memcpyflag_master);
            }
            base_master = __shfl_sync(eq_mask,  base_master, master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ((T*)(base_master+subindex))[0] = val;
            __syncwarp(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
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
        range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
        cuda_err_chk(cudaMemcpy(rdt, adt.d_ranges, adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < adt.n_ranges; i++) {

            std::cout << "*********************************" << std::endl;
            std::cout << std::dec << "# READ IOs:\t" << rdt[i].read_io_cnt << std::endl;
            std::cout << std::dec << "# Accesses:\t" << rdt[i].access_cnt << std::endl;
            std::cout << std::dec << "# Misses:\t" << rdt[i].miss_cnt << std::endl << "Miss Rate:\t" << ((float)rdt[i].miss_cnt/rdt[i].access_cnt) << std::endl;
            std::cout << std::dec << "# Hits:\t" << rdt[i].hit_cnt << std::endl << "Hit Rate:\t" << ((float)rdt[i].hit_cnt/rdt[i].access_cnt) << std::endl;
            rdt[i].read_io_cnt = 0;
            rdt[i].access_cnt = 0;
            rdt[i].miss_cnt = 0;
            rdt[i].hit_cnt = 0;
        }
         cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt, adt.n_ranges*sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
    }

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, uint32_t cudaDevice) {
        adt.n_elems = num_elems;
        adt.start_offset = disk_start_offset;

        adt.n_ranges = ranges.size();
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


struct page_cache_t {


    //void* d_pc;

    //BufferPtr prp2_list_buf;
    page_cache_d_t pdt;
    //void* d_pc;
    //BufferPtr prp2_list_buf;
    //bool prps;
    page_states_t*   h_ranges;
    uint64_t* h_ranges_page_starts;
    page_cache_d_t* d_pc_ptr;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr page_translation_buf;
    BufferPtr page_take_lock_buf;
    BufferPtr ranges_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;
    BufferPtr ranges_page_starts_buf;

    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;




    template <typename T>
    void page_cache_t::add_range(range_t<T>* range) {

        h_ranges[range->rdt.range_id] = range->rdt.page_states;
        h_ranges_page_starts[range->rdt.range_id] = range->rdt.page_start;
        cuda_err_chk(cudaMemcpy(pdt.ranges_page_starts, h_ranges_page_starts, pdt.n_ranges * sizeof(uint64_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges, h_ranges, pdt.n_ranges* sizeof(page_states_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));

    }

    page_cache_t(const uint64_t ps, const uint64_t np, const uint32_t cudaDevice, const Controller& ctrl, const uint64_t max_range, const std::vector<Controller*>& ctrls) {

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();
        pdt.page_size = ps;
        pdt.page_size_minus_1 = ps - 1;
        pdt.n_pages = np;
        pdt.ctrl_page_size = ctrl.ctrl->page_size;
        pdt.n_pages_minus_1 = np - 1;
        pdt.n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(pdt.n_ctrls * sizeof(Controller*), cudaDevice);
        pdt.d_ctrls = (Controller**) d_ctrls_buff.get();
        pdt.n_blocks_per_page = (ps/ctrl.blk_size);

        for (size_t k = 0; k < pdt.n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(pdt.d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        //n_ctrls = ctrls.size();
        //d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), cudaDevice);
        //d_ctrls = (Controller**) d_ctrls_buff.get();
        //for (size_t k = 0; k < n_ctrls; k++)
        //    cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));

        pdt.range_cap = max_range;
        pdt.n_ranges = 0;
        pdt.n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        pdt.n_ranges_mask = max_range-1;
        std::cout << "n_ranges_bits: " << std::dec << pdt.n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << pdt.n_ranges_mask << std::endl;

        pdt.page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(page_states_t), cudaDevice);
        pdt.ranges = (page_states_t*)ranges_buf.get();
        h_ranges = new page_states_t[max_range];

        h_ranges_page_starts = new uint64_t[max_range];
        std::memset(h_ranges_page_starts, 0, max_range * sizeof(uint64_t));

        page_translation_buf = createBuffer(np * sizeof(uint32_t), cudaDevice);
        pdt.page_translation = (uint32_t*)page_translation_buf.get();
        //page_translation_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        //page_translation = (padded_struct_pc*)page_translation_buf.get();

        page_take_lock_buf = createBuffer(np * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_take_lock =  (padded_struct_pc*)page_take_lock_buf.get();


        ranges_page_starts_buf = createBuffer(max_range * sizeof(uint64_t), cudaDevice);
        pdt.ranges_page_starts = (uint64_t*) ranges_page_starts_buf.get();

        page_ticket_buf = createBuffer(1 * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_ticket =  (padded_struct_pc*)page_ticket_buf.get();

        padded_struct_pc* tps = new padded_struct_pc[np];
        for (size_t i = 0; i < np; i++)
            tps[i] = FREE;
        cuda_err_chk(cudaMemcpy(pdt.page_take_lock, tps, np*sizeof(padded_struct_pc), cudaMemcpyHostToDevice));
        delete tps;



        uint64_t cache_size = ps*np;
        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        pdt.base_addr = (uint8_t*) this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((pdt.page_size > (ctrl.ns.lba_data_size * uints_per_page)) || (np == 0) || (pdt.page_size < ctrl.ns.lba_data_size))
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
            uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
            std::memset(temp1, 0, np * sizeof(uint64_t));
            uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++) {
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            pdt.prps = true;
        }
        else {
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t*) this->prp1_buf.get();
            uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t*) this->prp2_buf.get();
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
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            free(temp1);
            free(temp2);
            free(temp3);
            pdt.prps = true;
        }


        pc_buff = createBuffer(sizeof(page_cache_d_t), cudaDevice);
        d_pc_ptr = (page_cache_d_t*)pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";

    }







};

__forceinline__
__device__
uint32_t page_cache_d_t::find_slot(uint64_t address, uint64_t range_id) {
    bool fail = true;
    uint64_t count = 0;
    uint32_t global_address =(uint32_t) ((address << n_ranges_bits) | range_id); //not elegant. but hack
    uint32_t page = 0;
    do {

        //if (count < this->n_pages)
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed)  % (this->n_pages);
        //uint64_t unlocked = UNLOCKED;

        // uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        //printf("tid: %llu page: %llu\n", tid, page);

        bool lock = false;
        uint32_t v = this->page_take_lock[page].load(simt::memory_order_acquire);
        //this->page_take_lock[page].compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
        //not assigned to anyone yet
        if ( v == FREE ) {
            lock = this->page_take_lock[page].compare_exchange_strong(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                this->page_translation[page] = global_address;
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->page_take_lock[page].store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if ( v == UNLOCKED ) {

            lock = this->page_take_lock[page].compare_exchange_strong(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock) {
                uint32_t previous_global_address = this->page_translation[page];
                //uint32_t previous_global_address = this->page_translation[page].load(simt::memory_order_acquire);
                uint32_t previous_range = previous_global_address & n_ranges_mask;
                uint32_t previous_address = previous_global_address >> n_ranges_bits;
                uint32_t expected_state = VALID;
                //uint32_t new_state = BUSY;
                bool pass = false;
                //if ((previous_range >= range_cap) || (previous_address >= n_pages))
                //    printf("prev_ga: %llu\tprev_range: %llu\tprev_add: %llu\trange_cap: %llu\tn_pages: %llu\n", (unsigned long long) previous_global_address, (unsigned long long) previous_range, (unsigned long long) previous_address,
                //           (unsigned long long) range_cap, (unsigned long long) n_pages);
                expected_state = this->ranges[previous_range][previous_address].load(simt::memory_order_acquire);

                //this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                switch(expected_state) {
                    case VALID:
                        pass = this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if (pass) {
                            this->ranges[previous_range][previous_address].store(INVALID, simt::memory_order_release);
                            fail = false;
                        }
                        break;
                    case INVALID:
                        pass =  this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if (pass) {
                            this->ranges[previous_range][previous_address].store(INVALID, simt::memory_order_release);
                            fail = false;
                        }
                        break;
                    case VALID_DIRTY:


                        //if ((count > this->n_pages)) {
                        pass =  this->ranges[previous_range][previous_address].compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if  (pass) {
                            //if ((this->page_dirty_start[page].load(simt::memory_order_acquire) == this->page_dirty_end[page].load(simt::memory_order_acquire))) {

                            //writeback
                            uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
                            //uint32_t ctrl = (tid/32) % (n_ctrls);
                            //uint32_t ctrl = get_smid() % (n_ctrls);
                            uint32_t ctrl = ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (n_ctrls);
                            Controller* c = this->d_ctrls[ctrl];
                            uint32_t queue = (tid/32) % (c->n_qps);

                            uint64_t index = ranges_page_starts[previous_range] + previous_global_address;

                            write_data(this, (c->d_qps)+queue, (index*this->n_blocks_per_page), this->n_blocks_per_page, page);
                            this->ranges[previous_range][previous_address].store(INVALID, simt::memory_order_release);

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
                    this->page_translation[page] = global_address;
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->page_take_lock[page].store(UNLOCKED, simt::memory_order_release);
            }


        }

        count++;


    } while(fail);
    return page;

}

__device__ void read_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
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
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos);
    sq_dequeue(&qp->sq, sq_pos);



    put_cid(&qp->sq, cid);


}


__device__ void write_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos);
    sq_dequeue(&qp->sq, sq_pos);



    put_cid(&qp->sq, cid);


}


__device__ void access_data(page_cache_d_t* pc, QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode) {
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);



    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);


    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos);
    sq_dequeue(&qp->sq, sq_pos);



    put_cid(&qp->sq, cid);


}




#endif // __PAGE_CACHE_H__
