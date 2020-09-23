#ifndef __PAGE_CACHE_H__
#define __PAGE_CACHE_H__

#include "util.h"
#include <nvm_types.h>
#include "buffer.h"
#include "ctrl.h"
#include "settings.h"
#include <iostream>


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

struct page_cache_t;

struct range_states_t {
    padded_struct* page_states;
}

template <typename T>
struct range_t {
    uint64_t index_start;
    uint64_t index_end;
    uint64_t range_id;
    uint64_t blk_start;
    uint64_t blk_end;
    page_cache_t* cache;
    range_states_t*  page_states;
    padded_struct* page_addresses;
    //padded_struct* page_vals;  //len = num of pages for data
    //


    __device__
    T operator[](size_t i) const {
        uint64_t index = (i * sizeof(T)) >> (cache->page_size_log);
        uint64_t subindex = (i * sizeof(T)) & (cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE;
        uint64_t global_address = (index << cache->n_ranges_bits) | range_id;
        bool fail = true;
        T ret;
        do {
            bool pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            switch (expected_state) {
                case VALID:
                    uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                    // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                    page_states->page_states[index].val.fetch_sub(1, simt::memory_order_release);
                    fail = false;
                    break;
                case BUSY:
                    expected_state = VALID;
                    new_state = USE;
                    break;
                case INVALID:
                    pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        cache->find_slot(index, range_id);
                        //fill in
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                        page_states->page_states[index].val.store(VALID, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = INVALID;
                        new_state = BUSY;
                    }


                    break;
                default:
                    new_state = expected_state + 1;
                    pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                        page_states->page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = VALID;
                        new_state = USE;
                    }
                    break;
            }

        } while (fail);
        return ret;
    }

    __device__
    void operator()(size_t i, T val) {
        uint64_t index = (i * sizeof(T)) >> (cache->page_size_log);
        uint64_t subindex = (i * sizeof(T)) & (cache->page_size_minus_1);
        uint64_t expected_state = VALID;
        uint64_t new_state = USE_DIRTY;
        uint64_t global_address = (index << cache->n_ranges_bits) | range_id;
        bool fail = true;
        T ret;
        do {
            bool pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            switch (expected_state) {
                case VALID:
                    uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                    // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                    //     __nanosleep(100);
                    ret = ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex];
                    page_states->page_states[index].val.fetch_sub(1, simt::memory_order_release);
                    fail = false;
                    break;
                case BUSY:
                    expected_state = VALID;
                    new_state = USE;
                    break;
                case INVALID:
                    pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        cache->find_slot(index, range_id);
                        //fill in
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex] = val;
                        page_states->page_states[index].val.store(VALID, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = INVALID;
                        new_state = BUSY;
                    }


                    break;
                default:
                    new_state = expected_state + 1;
                    pass = page_states->page_states[index].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t page_trans = page_addresses[index].val.load(simt::memory_order_acquire);
                        // while (cache->page_translation[global_page].load(simt::memory_order_acquire) != page_trans)
                        //     __nanosleep(100);
                        ((T*)(cache->base_addr+(page_trans * cache->page_size)))[subindex] = val;
                        page_states->page_states[index].val.fetch_sub(1, simt::memory_order_release);
                        fail = false;
                    }
                    else {
                        expected_state = VALID;
                        new_state = USE;
                    }
                    break;
            }

        } while (fail);
    }
};




struct page_cache_t {
    uint8_t* base_addr;
    uint32_t page_size;
    uint32_t page_size_log;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    padded_struct* page_translation;         //len = num of pages in cache
    padded_struct* page_take_lock;      //len = num of pages in cache
    padded_struct* page_use_end;      //len = num of pages in cache
    padded_struct page_ticket;
    uint64_t* prp1;                  //len = num of pages in cache
    uint64_t* prp2;                  //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t* prp_list;              //len = num of pages in cache if page_size > ctrl.page_size *2
    uint64_t    ctrl_page_size;
    range_states_t**   ranges;
    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;

    //BufferPtr prp2_list_buf;
    bool prps;
    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;




    page_cache_t(const uint32_t ps, const uint64_t np, const Settings& settings, const Controller& ctrl)
        : page_size(ps), n_pages(np), ctrl_page_size(ctrl.ctrl->page_size) {

        page_ticket.val = 0;
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
        std::cout << "Finish Making Page Cache\n";

    }


    __device__
    uint64_t find_slot(uint64_t address, uint64_t range_id) {
        bool fail = true;
        uint64_t count = 0;
        uint64_t global_address = (address << n_ranges_bits) | range_id;
        do {
            uint64_t page = this->page_ticket.val.fetch_add(1, simt::memory_order_acquire)  & (this->n_pages_minus_1);
            uint64_t unlocked = UNLOCKED;
            bool lock = this->page_take_lock[page].val.compare_exchange_strong(unlocked, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if ( lock ) {
                uint64_t previous_global_address = this->page_translation[page].val.load(simt::memory_order_acquire);
                uint64_t previous_range = previous_global_address & n_ranges_mask;
                uint64_t previous_address = previous_global_address >> n_ranges_bits;
                uint64_t expected_state = VALID;
                uint64_t new_state = BUSY;
                bool pass = this->ranges[previous_range]->page_states[previous_address].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);

                switch(expected_state) {
                    case VALID:
                        this->ranges[previous_range]->page_states[previous_address].val.store(INVALID, simt::memory_order_release);
                        fail = false;
                        break;
                    case INVALID:
                        pass =  this->ranges[previous_range]->page_states[previous_address].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if (pass) {
                            this->ranges[previous_range]->page_states[previous_address].val.store(INVALID, simt::memory_order_release);
                            fail = false;
                        }
                        break;
                    case VALID_DIRTY:


                        //if ((count > this->n_pages)) {
                        pass =  this->ranges[previous_range]->page_states[previous_address].val.compare_exchange_strong(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                        if  (pass) {
                            //if ((this->page_dirty_start[page].load(simt::memory_order_acquire) == this->page_dirty_end[page].load(simt::memory_order_acquire))) {

                            //writeback
                            this->ranges[previous_range]->page_states[previous_address].val.store(INVALID, simt::memory_order_release);

                            fail = false;
                            //}
                            //else {
                            //    page_states[previous_address].store(expected_state, simt::memory_order_release);
                            //}
                        }
                        //}
                        break;
                    default:

                        break;

                }
                if (!fail)
                    this->page_translation[page].val.store(global_address, simt::memory_order_release);
                this->page_take_lock[page].val.store(UNLOCKED, simt::memory_order_release);

            }

            count++;


        } while(fail);
    }




};




#endif // __PAGE_CACHE_H__
