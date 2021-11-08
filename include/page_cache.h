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

// enum page_state {USE = 1U, USE_DIRTY = ((1U << 31) | 1), VALID_DIRTY = (1U << 31),
//     VALID = 0U, INVALID = (UINT_MAX & 0x7fffffff),
//     BUSY = ((UINT_MAX & 0x7fffffff)-1)};

enum data_dist_t
{
    REPLICATE = 0,
    STRIPE = 1
};

#define USE (1ULL)
#define VALID_DIRTY (1ULL << 31)
#define USE_DIRTY (VALID_DIRTY | USE)
#define VALID (0ULL)
#define INVALID (0xffffffffULL)
#define BUSY ((0xffffffffULL) - 1)

#define SECTOR_VALID 1
#define SECTOR_INVALID 0
#define SECTOR_BUSY 2
#define SECTOR_DIRTY 4
#define SECTOR_COALESCE 8

#define ALL_CTRLS 0xffffffffffffffff

struct page_cache_t;

struct page_cache_d_t;

//typedef padded_struct_pc* page_states_t;

template <typename T>
struct range_t;
#define N_SECTORS_PER_PAGE 8
//template <size_t n_sectors_per_page = N_SECTORS_PER_PAGE>
struct data_page_t
{
    simt::atomic<uint64_t, simt::thread_scope_device> state; //state
    //simt::atomic<uint8_t, simt::thread_scope_device> sector_states[(N_SECTORS_PER_PAGE+2-1) / 2];
    simt::atomic<uint32_t, simt::thread_scope_device> sector_states[(N_SECTORS_PER_PAGE+8-1) / 8];

    //constexpr size_t get_sector_size() const { return page_size / n_sectors_per_page; }
};

typedef data_page_t *page_states_t;

template <typename T>
struct returned_cache_page_t
{
    T *addr;
    uint32_t size;
    uint32_t offset;

    T operator[](size_t i) const
    {
        if (i < size)
            return addr[i];
        else
            return 0;
    }

    T &operator[](size_t i)
    {
        if (i < size)
            return addr[i];
        else
            return addr[0];
    }
};

struct cache_page_t
{
    simt::atomic<uint32_t, simt::thread_scope_device> page_take_lock;
    uint32_t page_translation;
    uint8_t range_id;
};


struct page_cache_d_t
{
    uint8_t *base_addr;
    uint64_t page_size;
    uint64_t page_size_minus_1;
    uint64_t page_size_log;
    size_t sector_size;
    size_t sector_size_log;
    size_t sector_size_minus_1;
    uint64_t n_pages;
    uint64_t n_pages_minus_1;
    cache_page_t *cache_pages;
    padded_struct_pc *page_ticket;
    uint64_t *prp1; //len = num of pages in cache
    uint64_t *prp2; //len = num of pages in cache if page_size = ctrl.page_size *2
    //uint64_t *prplist;
    uint64_t ctrl_page_size;
    uint64_t range_cap;
    page_states_t *ranges;
    page_states_t *h_ranges;
    uint64_t n_ranges;
    uint64_t n_ranges_bits;
    uint64_t n_ranges_mask;
    uint64_t n_cachelines_for_states;

    size_t n_sectors_per_page;
    size_t n_sectors_per_page_minus_1;
    size_t n_sectors_per_page_log;

    uint64_t *ranges_page_starts;
    data_dist_t *ranges_dists;
    simt::atomic<uint64_t, simt::thread_scope_device> *ctrl_counter;

    Controller **d_ctrls;
    uint64_t n_ctrls;
    bool prps;

    uint64_t n_blocks_per_page;
    uint64_t n_blocks_per_sector;

    __forceinline__
        __device__
            cache_page_t *
            get_cache_page(const uint32_t page) const;

    __forceinline__
        __device__
            uint32_t
            find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_);
};

struct page_cache_t
{

    page_cache_d_t pdt;
    page_states_t *h_ranges;
    uint64_t *h_ranges_page_starts;
    data_dist_t *h_ranges_dists;
    page_cache_d_t *d_pc_ptr;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf;
    BufferPtr cache_pages_buf;
    BufferPtr ranges_buf;
    BufferPtr pc_buff;
    BufferPtr d_ctrls_buff;
    BufferPtr ranges_page_starts_buf;
    BufferPtr ranges_dists_buf;

    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;

    template <typename T>
    void add_range(range_t<T> *range)
    {
        range->rdt.range_id = pdt.n_ranges++;
        h_ranges[range->rdt.range_id] = range->rdt.page_states;
        h_ranges_page_starts[range->rdt.range_id] = range->rdt.page_start;
        h_ranges_dists[range->rdt.range_id] = range->rdt.dist;
        cuda_err_chk(cudaMemcpy(pdt.ranges_page_starts, h_ranges_page_starts, pdt.n_ranges * sizeof(uint64_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges, h_ranges, pdt.n_ranges * sizeof(page_states_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(pdt.ranges_dists, h_ranges_dists, pdt.n_ranges * sizeof(data_dist_t), cudaMemcpyHostToDevice));
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
    }

    page_cache_t(const uint64_t ps, const uint64_t np, const uint32_t cudaDevice, const Controller &ctrl, const uint64_t max_range, const std::vector<Controller *> &ctrls)
    {

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        pdt.ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device> *)ctrl_counter_buf.get();
        pdt.page_size = ps;
        pdt.page_size_minus_1 = ps - 1;
        pdt.n_pages = np;
        pdt.ctrl_page_size = ctrl.ctrl->page_size;
        std::cout << "pdt.ctrl_page_size = " << pdt.ctrl_page_size <<"\n";
        pdt.n_pages_minus_1 = np - 1;
        pdt.n_ctrls = ctrls.size();
        std::cout << "pdt.n_ctrls = "<< pdt.n_ctrls << "\n";
        d_ctrls_buff = createBuffer(pdt.n_ctrls * sizeof(Controller *), cudaDevice);
        pdt.d_ctrls = (Controller **)d_ctrls_buff.get();
        pdt.n_blocks_per_page = (ps / ctrl.blk_size);
        std::cout << "ctrl.blk_size = " << ctrl.blk_size << "\n";
        std::cout << "pdt.n_blocks_per_page = " << pdt.n_blocks_per_page << "\n";
        pdt.n_sectors_per_page = N_SECTORS_PER_PAGE;
        pdt.n_sectors_per_page_minus_1 = N_SECTORS_PER_PAGE - 1;
        pdt.n_sectors_per_page_log = std::log2(N_SECTORS_PER_PAGE);
        for (size_t k = 0; k < pdt.n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(pdt.d_ctrls + k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller *), cudaMemcpyHostToDevice));

        pdt.range_cap = max_range;
        pdt.n_ranges = 0;
        pdt.n_ranges_bits = (max_range == 1) ? 1 : std::log2(max_range);
        pdt.n_ranges_mask = max_range - 1;
        std::cout << "n_ranges_bits: " << std::dec << pdt.n_ranges_bits << std::endl;
        std::cout << "n_ranges_mask: " << std::dec << pdt.n_ranges_mask << std::endl;

        pdt.page_size_log = std::log2(ps);
        ranges_buf = createBuffer(max_range * sizeof(page_states_t), cudaDevice);
        pdt.ranges = (page_states_t *)ranges_buf.get();
        h_ranges = new page_states_t[max_range];
        pdt.sector_size = ceil(ps/N_SECTORS_PER_PAGE);
        std::cout << "pdt.sector_size = " << pdt.sector_size << "\n";
        pdt.sector_size_log = std::log2(pdt.sector_size);
        pdt.sector_size_minus_1 = pdt.sector_size -1;
        pdt.n_blocks_per_sector = pdt.sector_size / ctrl.blk_size;
        std::cout << "pdt.n_blocks_per_sector = " << pdt.n_blocks_per_sector << "\n";

        h_ranges_page_starts = new uint64_t[max_range];
        std::memset(h_ranges_page_starts, 0, max_range * sizeof(uint64_t));

        cache_pages_buf = createBuffer(np * sizeof(cache_page_t), cudaDevice);
        pdt.cache_pages = (cache_page_t *)cache_pages_buf.get();

        ranges_page_starts_buf = createBuffer(max_range * sizeof(uint64_t), cudaDevice);
        pdt.ranges_page_starts = (uint64_t *)ranges_page_starts_buf.get();

        page_ticket_buf = createBuffer(1 * sizeof(padded_struct_pc), cudaDevice);
        pdt.page_ticket = (padded_struct_pc *)page_ticket_buf.get();
        //std::vector<padded_struct_pc> tps(np, FREE);
        cache_page_t *tps = new cache_page_t[np];
        for (size_t i = 0; i < np; i++)
            tps[i].page_take_lock = FREE;
        cuda_err_chk(cudaMemcpy(pdt.cache_pages, tps, np * sizeof(cache_page_t), cudaMemcpyHostToDevice));
        delete tps;

        ranges_dists_buf = createBuffer(max_range * sizeof(data_dist_t), cudaDevice);
        pdt.ranges_dists = (data_dist_t *)ranges_dists_buf.get();
        h_ranges_dists = new data_dist_t[max_range];

        uint64_t cache_size = ps * np;
        std::cout << "cache_size = " << cache_size << "\n";

        this->pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        pdt.base_addr = (uint8_t *)this->pages_dma.get()->vaddr;
        std::cout << "pages_dma: " << std::hex << this->pages_dma.get()->vaddr << "\t" << this->pages_dma.get()->ioaddrs[0] << std::endl;
        std::cout << "HEREN\n";
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);
        if ((pdt.page_size > (ctrl.ns.lba_data_size * uints_per_page)) || (np == 0) || (pdt.sector_size < ctrl.ns.lba_data_size))
            throw error(string("page_cache_t: Can't have such page size or number of pages"));
        std::cout<<"pages_dma.page_size = " << this->pages_dma.get()->page_size << "\n";
        if (pdt.page_size <= this->pages_dma.get()->page_size)
        {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size / pdt.page_size;
            std::cout << "ctrl.page_size = " << ctrl.ctrl->page_size << "\n";
            std::cout << "how_many_in_one = " << how_many_in_one << "\n";
            this->prp1_buf = createBuffer(np*sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t *)this->prp1_buf.get();

            std::cout << np << "  " << N_SECTORS_PER_PAGE << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs << std::endl;
            uint64_t *temp = new uint64_t[how_many_in_one * this->pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one * this->pages_dma.get()->n_ioaddrs);
            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs); i++)
            {
                std::cout << std::dec << "\ti: " << i << "\t" << std::hex << ((uint64_t)this->pages_dma.get()->ioaddrs[i]) << std::dec << std::endl;
                for (size_t j = 0; (j < how_many_in_one); j++)
                {
                    temp[i * how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j * pdt.page_size;
                    std::cout << std::dec << "\ti: " << i << "\tj: " << j << "\tindex: "<< (i*how_many_in_one + j) << "\t" << std::hex << (((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*pdt.page_size) << std::dec << std::endl;
                }
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp, np*sizeof(uint64_t), cudaMemcpyHostToDevice));
            delete temp;
            //std::cout << "HERE1\n";
            //std::cout << "HERE2\n";
            pdt.prps = false;
        }

        else if ((pdt.page_size > this->pages_dma.get()->page_size) && (pdt.page_size <= (this->pages_dma.get()->page_size * 2)))
        {
            std::cout << "Cond2\n";
            this->prp1_buf = createBuffer(np*sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t *)this->prp1_buf.get();
            this->prp2_buf = createBuffer(np* sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t *)this->prp2_buf.get();
            uint64_t *temp1 = new uint64_t[np* sizeof(uint64_t)];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            //uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
            uint64_t *temp2 = new uint64_t[np * sizeof(uint64_t)];
            std::memset(temp2, 0, np * sizeof(uint64_t));
            for (size_t i = 0; i < np; i++)
            {
                std::cout << std::dec << "\ti: " << i << "\t" << std::hex << ((uint64_t)this->pages_dma.get()->ioaddrs[i]) << std::dec << std::endl;
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i * 2]);
                temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i * 2 + 1]);
            }
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            pdt.prps = true;
        }
        else
        {
            std::cout << "Cond3\n";
            this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp1 = (uint64_t *)this->prp1_buf.get();
            uint32_t prp_list_size = ctrl.ctrl->page_size * np;
            std::cout << "ctrl.page_size = "<< ctrl.ctrl->page_size << "\n";
            std::cout << "np = " << np << "\n";
            std::cout << "prp_list_size = " << prp_list_size << "\n";
            this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
            //pdt.prplist = (uint64_t *)this->prp_list_dma.get();
            this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
            pdt.prp2 = (uint64_t *)this->prp2_buf.get();
            uint64_t *temp1 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t *temp2 = new uint64_t[np * sizeof(uint64_t)];
            uint64_t *temp3 = new uint64_t[prp_list_size];
            std::memset(temp1, 0, np * sizeof(uint64_t));
            std::memset(temp2, 0, np * sizeof(uint64_t));
            std::memset(temp3, 0, prp_list_size);
            uint32_t how_many_in_one = pdt.page_size / ctrl.ctrl->page_size;
            std::cout << "pdt.page_size = " << pdt.page_size << "\n";
            std::cout << "how_many_in_one = " << how_many_in_one << "\n";
            for (size_t i = 0; i < np; i++)
            {
                std::cout << std::dec << "\ti: " << i << "\t" << std::hex << ((uint64_t)this->pages_dma.get()->ioaddrs[i*how_many_in_one]) << std::dec << std::endl;
                temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i * how_many_in_one]);
                temp2[i] = ((uint64_t)this->prp_list_dma.get()->ioaddrs[i]);
                for (size_t j = 0; j < (how_many_in_one - 1); j++)
                {
                    temp3[i * uints_per_page + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i * how_many_in_one + j + 1]);
                }
            }

            std::cout << "Done creating PRP\n";
            cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np  * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np  * sizeof(uint64_t), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

            delete temp1;
            delete temp2;
            delete temp3;
            pdt.prps = true;
        }

        pc_buff = createBuffer(sizeof(page_cache_d_t), cudaDevice);
        d_pc_ptr = (page_cache_d_t *)pc_buff.get();
        cuda_err_chk(cudaMemcpy(d_pc_ptr, &pdt, sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
        std::cout << "Finish Making Page Cache\n";
    }

    ~page_cache_t()
    {
        delete h_ranges;
        delete h_ranges_page_starts;
        delete h_ranges_dists;
    }
};

template <typename T>
struct range_d_t
{
    uint64_t index_start;
    uint64_t count;
    uint64_t range_id;
    uint64_t page_start_offset;
    uint64_t page_size;
    uint64_t page_start;
    uint64_t page_count;
    data_dist_t dist;
    uint8_t *src;

    simt::atomic<uint64_t, simt::thread_scope_device> access_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> read_io_cnt;

    page_states_t page_states;
    uint32_t *page_addresses;
    page_cache_d_t cache;

    __forceinline__
        __device__
            uint64_t
            get_backing_page(const size_t i) const;
    __forceinline__
        __device__
            uint64_t
            get_backing_ctrl(const size_t i) const;
    /*__forceinline__
        __device__
            uint64_t
            get_sector_size() const;*/
    __forceinline__
        __device__
            uint64_t
            get_page(const size_t i) const;
    __forceinline__
        __device__
            uint64_t
            get_subindex(const size_t i) const;
    __forceinline__
        __device__
            uint64_t
            get_sectorindex(const size_t i) const;        
    __forceinline__
        __device__
            uint64_t
            get_global_address(const size_t page) const;
    __forceinline__
        __device__ void
        release_page(const size_t pg) const;
    __forceinline__
        __device__ void
        release_page(const size_t pg, const uint32_t count) const;
    __forceinline__
        __device__
            uint64_t
            acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t queue);
    __forceinline__
        __device__
            bool
            acquire_sector(const uint64_t page_index, const size_t sector_index, uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue);
    __forceinline__
        __device__ void
        write_done(const size_t pg, const uint32_t count) const;
    __forceinline__
        __device__
            T
            operator[](const size_t i);
    __forceinline__
        __device__ void
        operator()(const size_t i, const T val);
    __forceinline__
        __device__
            cache_page_t *
            get_cache_page(const size_t pg) const;
    __forceinline__
        __device__
            uint64_t
            get_cache_page_addr(const uint32_t page_trans) const;
    __forceinline__
        __device__
            uint64_t
            get_cache_sector_size() const;
    __forceinline__
        __device__
            uint64_t
            get_cache_sector_size_log() const;
    __forceinline__
        __device__
            uint64_t
            get_sectorsubindex(const size_t i) const;
};

__device__ void read_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);
__device__ void write_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry);

template <typename T>
struct range_t
{
    range_d_t<T> rdt;

    range_d_t<T> *d_range_ptr;
    page_cache_d_t *cache;

    BufferPtr page_states_buff;
    BufferPtr page_addresses_buff;

    BufferPtr range_buff;

    range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t *c_h, uint32_t cudaDevice, data_dist_t dist = REPLICATE);
};

template <typename T>
range_t<T>::range_t(uint64_t is, uint64_t count, uint64_t ps, uint64_t pc, uint64_t pso, uint64_t p_size, page_cache_t *c_h, uint32_t cudaDevice, data_dist_t dist)
{
    rdt.access_cnt = 0;
    rdt.miss_cnt = 0;
    rdt.hit_cnt = 0;
    rdt.read_io_cnt = 0;
    rdt.index_start = is;
    rdt.count = count;
    rdt.page_start = ps;
    rdt.page_count = pc;
    rdt.page_size = c_h->pdt.page_size;
    rdt.page_start_offset = pso;
    rdt.dist = dist;
    size_t s = pc; //(rdt.page_end-rdt.page_start);//*page_size / c_h->page_size;
    std::cout << "creating range\n";
    cache = (page_cache_d_t *)c_h->d_pc_ptr; //why d_pc_ptr, why not pdt
    std::cout << "n_sectors_per_page = " << c_h->pdt.n_sectors_per_page << "\n";
    page_states_buff = createBuffer(s * sizeof(data_page_t), cudaDevice);
    rdt.page_states = (page_states_t)page_states_buff.get();
    data_page_t *ts = new data_page_t[s];
    for (size_t i = 0; i < s; i++) {
        ts[i].state = INVALID;
        for (size_t j=0; j<(c_h->pdt.n_sectors_per_page+7)/8; j++) {
            ts[i].sector_states[j] = SECTOR_INVALID;
        }  
    }
    printf("S value: %llu\n", (unsigned long long)s);
    cuda_err_chk(cudaMemcpy(rdt.page_states, ts, s * sizeof(data_page_t), cudaMemcpyHostToDevice));
    delete ts;

    page_addresses_buff = createBuffer(s * sizeof(uint32_t), cudaDevice);
    rdt.page_addresses = (uint32_t *)page_addresses_buff.get();

    range_buff = createBuffer(sizeof(range_d_t<T>), cudaDevice);
    d_range_ptr = (range_d_t<T> *)range_buff.get();

    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice));

    c_h->add_range(this);

    rdt.cache = c_h->pdt;
    cuda_err_chk(cudaMemcpy(d_range_ptr, &rdt, sizeof(range_d_t<T>), cudaMemcpyHostToDevice)); //why double copy
}

__forceinline__
    __device__
        uint64_t
        get_backing_page_(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist)
{
    uint64_t page = page_start;
    if (dist == STRIPE)
    {
        page += page_offset / n_ctrls;
    }
    else if (dist == REPLICATE)
    {
        page += page_offset;
    }

    return page;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_backing_page(const size_t page_offset) const
{
    return get_backing_page_(page_start, page_offset, cache.n_ctrls, dist);
}

__forceinline__
    __device__
        uint64_t
        get_backing_ctrl_(const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist)
{
    uint64_t ctrl;

    if (dist == STRIPE)
    {
        ctrl = page_offset % n_ctrls;
    }
    else if (dist == REPLICATE)
    {
        ctrl = ALL_CTRLS;
    }
    return ctrl;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_backing_ctrl(const size_t page_offset) const
{
    return get_backing_ctrl_(page_offset, cache.n_ctrls, dist);
}

/*template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_sector_size() const
{
    return page_size;
}*/

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_page(const size_t i) const
{
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) >> (cache.page_size_log);
    return index;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_subindex(const size_t i) const
{
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) & (cache.page_size_minus_1);
    return index;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_sectorindex(const size_t i) const
{
    printf("tid %d\tindex_calc %llu\n", (blockIdx.x*blockDim.x+threadIdx.x), (unsigned long long)((i - index_start) * sizeof(T) + page_start_offset));
    uint64_t index = (((i - index_start) * sizeof(T) + page_start_offset) >> (cache.sector_size_log)) & (cache.n_sectors_per_page_minus_1);
    return index;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_sectorsubindex(const size_t i) const
{
    uint64_t index = ((i - index_start) * sizeof(T) + page_start_offset) & (cache.sector_size_minus_1);
    return index;
}

//what does this function do?
template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_global_address(const size_t page) const
{
    return ((page << cache.n_ranges_bits) | range_id);
}

template <typename T>
__forceinline__
    __device__ void
    range_d_t<T>::release_page(const size_t pg) const
{
    uint64_t index = pg;
    page_states[index].state.fetch_sub(1, simt::memory_order_release);
}

template <typename T>
__forceinline__
    __device__ void
    range_d_t<T>::release_page(const size_t pg, const uint32_t count) const
{
    uint64_t index = pg;
    page_states[index].state.fetch_sub(count, simt::memory_order_release);
}

template <typename T>
__forceinline__
    __device__
        cache_page_t *
        range_d_t<T>::get_cache_page(const size_t pg) const
{
    uint32_t page_trans = page_addresses[pg];
    return cache.get_cache_page(page_trans);
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_cache_page_addr(const uint32_t page_trans) const
{
    return ((uint64_t)((cache.base_addr + (page_trans * cache.page_size))));
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_cache_sector_size() const
{
    return cache.sector_size;
}

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::get_cache_sector_size_log() const
{
    return cache.sector_size_log;
}

template <typename T>
__forceinline__
    __device__
        bool
        range_d_t<T>::acquire_sector(const uint64_t page_index, const size_t sector_index, uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue)
{
    printf("tid %d\t count %d\tpage_index %llu\tsector_index%d\tin acquire_sector\n", (blockIdx.x*blockDim.x+threadIdx.x), count, (unsigned long long)page_index,sector_index);
    bool fail = true;
    uint8_t sector_number = (sector_index) && (cache.n_sectors_per_page_minus_1);
    sector_index = (sector_index) >> (cache.n_sectors_per_page_log);
    uint32_t original_state;
    uint32_t expected_state = SECTOR_VALID;
    uint32_t new_state = SECTOR_VALID;
    uint64_t page_trans = (page_addresses[page_index]<< cache.n_sectors_per_page_log) | sector_index;
    printf("tid %d\t page_address %d\tpage_trans %llu\n", (blockIdx.x*blockDim.x+threadIdx.x), page_addresses[page_index], (unsigned long long)page_trans);

    uint64_t temp_mask = 0xFFFFFFF0FFFFFFFF << (4*sector_number);
    uint32_t mask = ((uint32_t*)&temp_mask)[1];
    access_cnt.fetch_add(count, simt::memory_order_relaxed);

    do {
        bool pass = false;
        original_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
        expected_state = (original_state & (0x0000000F << 4*sector_number)) >> 4*sector_number;
        printf("tid %d\toriginal state %08x\tpage_index %llu\tsector_index %d\tsector_number %d\texpected_state %08x\n", (blockIdx.x*blockDim.x+threadIdx.x), original_state,(unsigned long long)page_index, sector_index, sector_number, expected_state);
        switch(expected_state){
            case SECTOR_BUSY:
               //do nothing
            break;
            case SECTOR_INVALID:
               new_state = (original_state & mask) | (SECTOR_BUSY << 4*sector_number);
               pass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(original_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
               printf("tid %d\t original_state %08x\t new_state %08x\t in SECTOR_INVALID passed %d\n",(blockIdx.x*blockDim.x+threadIdx.x), original_state, new_state, pass);
               if (pass) {
                    uint64_t ctrl = get_backing_ctrl(page_index);
                    if (ctrl == ALL_CTRLS)
                        ctrl = ctrl_;
                    uint64_t b_page = get_backing_page(page_index);
                    Controller *c = cache.d_ctrls[ctrl];
                    c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                    read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                    printf("tid %d\t in acquire_sector reading data\n", (blockIdx.x*blockDim.x+threadIdx.x));
                    read_data(&cache, (c->d_qps) + queue, ((b_page)*cache.n_blocks_per_page) + (sector_index*cache.n_blocks_per_sector), cache.n_blocks_per_sector, page_trans);
                    bool caspass = false;
                    while (!caspass) {
                        original_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
                        new_state = (original_state & mask) | (SECTOR_VALID << 4*sector_number);
                        caspass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(original_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                    }
                    fail = false;
               }
            break;
            default:
                printf("tid %d\toriginal_state %08x\texpected_state %08x\tpage_index %llu\tsector_index %d\thit\n", (blockIdx.x*blockDim.x+threadIdx.x),original_state, expected_state,page_index,sector_index);
                fail = false;
                hit_cnt.fetch_add(count, simt::memory_order_relaxed);
            break;
        }

    }while (fail);
    return !fail;
    
}

/*template <typename T>
__forceinline__
    __device__
        bool
        range_d_t<T>::acquire_sector(const uint64_t page_index, const size_t sector_index, uint32_t count, const bool write, const uint32_t ctrl_, const uint32_t queue)
{
    bool fail = true;
    bool odd_sector = sector_index & (1UL);
    uint8_t expected_state;
    uint8_t new_state;
    uint64_t page_trans = (page_addresses[page_index]<< cache.n_sectors_per_page_log) | sector_index;

    access_cnt.fetch_add(count, simt::memory_order_relaxed);
    //expected_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
    //if (odd_sector) {
    //    expected_state = (expected_state & 0xF0) | SECTOR1_VALID;
    //}
    //else {
    //    expected_state = (expected_state & 0x0F) | SECTOR0_VALID;
    //}
    //new_state = expected_state;

    do
    {
        bool pass = false;
        if (odd_sector) {
            expected_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
            switch (expected_state & 0xF0) {
                case SECTOR1_BUSY:
                   //do nothing
                break;
                case SECTOR1_INVALID:
                    new_state = (expected_state & 0x0F) | SECTOR1_BUSY;
                    pass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t ctrl = get_backing_ctrl(page_index);
                        if (ctrl == ALL_CTRLS)
                            ctrl = ctrl_;
                        uint64_t b_page = get_backing_page((page_index<<get_cache_sector_size_log()) | sector_index);
                        Controller *c = cache.d_ctrls[ctrl];
                        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                        read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                        read_data(&cache, (c->d_qps) + queue, ((b_page)*cache.n_blocks_per_sector), cache.n_blocks_per_sector, page_trans);
                        //have a loop here for CAS to change the state
                        bool caspass = false;
                        while (!caspass) {
                            expected_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
                            new_state = (expected_state & 0x0F) | SECTOR1_VALID;
                            if (write) new_state |= SECTOR1_DIRTY;
                            caspass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                        }
                        fail = false;
                    }
                    
                break;
                default:
                    new_state = expected_state;
                    if (write && !(new_state & SECTOR1_DIRTY)) {
                        new_state |= SECTOR1_DIRTY;
                        pass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    }
                    else {
                        pass = true;
                    }                        
                    if (pass) {
                        fail = false;
                        hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                    }
                
                break;   
            }

        }
        else {
            expected_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
            switch (expected_state & 0x0F) {
                case SECTOR0_BUSY:
                   //do nothing
                break;
                case SECTOR0_INVALID:
                    new_state = (expected_state & 0xF0) | SECTOR0_BUSY;
                    pass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass) {
                        uint64_t ctrl = get_backing_ctrl(page_index);
                        if (ctrl == ALL_CTRLS)
                            ctrl = ctrl_;
                        uint64_t b_page = get_backing_page((page_index<<get_cache_sector_size_log()) | sector_index);
                        Controller *c = cache.d_ctrls[ctrl];
                        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                        read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                        read_data(&cache, (c->d_qps) + queue, ((b_page)*cache.n_blocks_per_sector), cache.n_blocks_per_sector, page_trans);
                        bool caspass= false;
                        while (!caspass) {
                            expected_state = page_states[page_index].sector_states[sector_index].load(simt::memory_order_acquire);
                            new_state = (expected_state & 0xF0) | SECTOR0_VALID;
                            if (write) new_state |= SECTOR0_DIRTY;
                            caspass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acq_rel, simt::memory_order_relaxed);
                        }
                        fail = false;
                    }
                    
                break; 
                default:
                    new_state = expected_state;
                    if (write && !(new_state & SECTOR0_DIRTY)) {
                        new_state |= SECTOR0_DIRTY;
                        pass = page_states[page_index].sector_states[sector_index].compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
                    }
                    else {
                        pass = true;
                    }                        
                    if (pass) {
                        fail = false;
                        hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                    }
                break; 
            }

        }

    } while (fail);
    return !fail;
}*/

template <typename T>
__forceinline__
    __device__
        uint64_t
        range_d_t<T>::acquire_page(const size_t pg, const uint32_t count, const bool write, const uint32_t queue)
{
    printf("tid %d\t count %d\tin acquire_page\n", (blockIdx.x*blockDim.x+threadIdx.x), count);
    uint64_t index = pg;
    //size_t sector_index = sector;
    uint64_t expected_state = VALID;
    uint32_t new_state = USE;
    //access_cnt.fetch_add(count, simt::memory_order_relaxed);
    bool fail = true;
    do
    {
        bool pass = false;
        expected_state = page_states[index].state.load(simt::memory_order_acquire);
        switch (expected_state)
        {
        case VALID:
            new_state = count;
            if (write)
                new_state |= VALID_DIRTY;
            pass = page_states[index].state.compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (pass)
            {
                //acquire_sector(index, sector_index, count, write);

                uint32_t page_trans = page_addresses[index];

                //hit_cnt.fetch_add(count, simt::memory_order_relaxed);

                return get_cache_page_addr(page_trans);
                fail = false;
            }
            break;
        case BUSY:
            //expected_state = VALID;
            //new_state = USE;
            break;
        case INVALID:
            pass = page_states[index].state.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (pass)
            {
                uint32_t page_trans = cache.find_slot(index, range_id, queue);
                //uint64_t ctrl = get_backing_ctrl(index);
                //if (ctrl == ALL_CTRLS)
                //    ctrl = ctrl_;
                //uint64_t b_page = get_backing_page(index);
                //Controller *c = cache.d_ctrls[ctrl];
                //c->access_counter.fetch_add(1, simt::memory_order_relaxed);
                //read_io_cnt.fetch_add(1, simt::memory_order_relaxed);
                //read_data(&cache, (c->d_qps) + queue, ((b_page)*cache.n_blocks_per_page), cache.n_blocks_per_page, page_trans);
                page_addresses[index] = page_trans;
                printf("tid %d\t index %llu\tfound slot page_trans %d\n",(blockIdx.x*blockDim.x+threadIdx.x),(unsigned long long)index, page_trans );
                //miss_cnt.fetch_add(count, simt::memory_order_relaxed);
                new_state = count;
                if (write)
                    new_state |= VALID_DIRTY;
                page_states[index].state.store(new_state, simt::memory_order_release);
                //acquire_sector(index, sector_index, count, write);
                return get_cache_page_addr(page_trans);

                fail = false;
            }

            break;
        default:
            new_state = expected_state + count;
            if (write)
                new_state |= VALID_DIRTY;
            pass = page_states[index].state.compare_exchange_weak(expected_state, new_state, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (pass)
            {
                uint32_t page_trans = page_addresses[index];
                //hit_cnt.fetch_add(count, simt::memory_order_relaxed);
                //acquire_sector(index, sector_index, count, write);
                return get_cache_page_addr(page_trans);
                fail = false;
            }
            break;
        }

    } while (fail);
    return 0;
}

template <typename T>
struct array_d_t
{
    uint64_t n_elems;
    uint64_t start_offset;
    uint64_t n_ranges;
    uint8_t *src;

    range_d_t<T> *d_ranges;

    __forceinline__
        __device__ void
        memcpy(const uint64_t i, const uint64_t count, T *dest)
    {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = 0xffffffff;
#endif
            uint32_t leader = 0;
            if (lane == leader)
            {
                page_cache_d_t *pc = &(d_ranges[r].cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = d_ranges[r].get_page(i);
            //uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);
            //uint64_t p_s = d_ranges[r].page_size;

            uint32_t active_cnt = 32;
            uint32_t eq_mask = mask;
            int master = 0;
            uint64_t base_master;
            uint64_t base;

            uint32_t count = 1;
            if (master == lane)
            {
                base = d_ranges[r].acquire_page(page, count, false, ctrl, queue);
                base_master = base;
            }
            base_master = __shfl_sync(eq_mask, base_master, master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            //
            ulonglong4 *src_ = (ulonglong4 *)base_master;
            ulonglong4 *dst_ = (ulonglong4 *)dest;
            warp_memcpy<ulonglong4>(dst_, src_, 512 / 32);

            __syncwarp(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);
        }
    }
    __forceinline__
        __device__
            int64_t
            find_range(const size_t i) const
    {
        int64_t range = -1;
        int64_t k = 0;
        for (; k < n_ranges; k++)
        {
            if ((d_ranges[k].index_start <= i) && (d_ranges[k].count > i))
            {
                range = k;
                break;
            }
        }
        return range;
    }

    __forceinline__
        __device__ void
        coalesce_page(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const size_t sector, const uint64_t gaddr, const bool write,
                      uint32_t &eq_mask, int &master, uint32_t &count, uint64_t &base_master, bool &sector_acquired_master) const
    {
        printf("tid %d\t in coalesce_page\n", (blockDim.x*blockIdx.x+threadIdx.x));
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        if (lane == leader)
        {
            page_cache_d_t *pc = &(d_ranges[r].cache);
            ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
        }

        ctrl = __shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);

        uint32_t active_cnt = __popc(mask);
        eq_mask = __match_any_sync(mask, gaddr);
        eq_mask &= __match_any_sync(mask, (uint64_t)this);
        //eq_mask &= __match_any_sync(mask, sector_index); // not sure if correct
        master = __ffs(eq_mask) - 1;
        printf("tid %d\teq_mask for page %d\n", (blockIdx.x*blockDim.x+threadIdx.x), eq_mask);

        uint32_t dirty = __any_sync(eq_mask, write);

        uint64_t base;
        count = __popc(eq_mask);
        if (master == lane)
        {
            base = d_ranges[r].acquire_page(page, count, dirty, queue);
            base_master = base;
            printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
        }
        base_master = __shfl_sync(eq_mask, base_master, master);

        eq_mask &= __match_any_sync(mask, sector);
        printf("tid %d\teq_mask for sector %d\n", (blockIdx.x*blockDim.x+threadIdx.x), eq_mask);
        master = __ffs(eq_mask) - 1;
        dirty = __any_sync(eq_mask, write);

        bool sector_acquired;
        //count = __popc(eq_mask);
        if (master == lane)
        {
            sector_acquired = d_ranges[r].acquire_sector(page, sector, __popc(eq_mask), dirty, ctrl, queue);
            sector_acquired_master = sector_acquired;
            //                printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
        }
        sector_acquired_master = __shfl_sync(eq_mask, sector_acquired_master, master);
    }

    /*__forceinline__
        __device__ void
        coalesce_sector(const uint32_t lane, const uint32_t mask, const int64_t r, const uint64_t page, const size_t sector, const bool write,
                      uint32_t &eq_mask, int &master, bool &sector_acquired_master) const
    {
        uint32_t ctrl;
        uint32_t queue;
        uint32_t leader = __ffs(mask) - 1;
        if (lane == leader)
        {
            page_cache_d_t *pc = &(d_ranges[r].cache);
            ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
            queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
        }

        ctrl = __shfl_sync(mask, ctrl, leader);
        queue = __shfl_sync(mask, queue, leader);

        //uint32_t active_cnt = __popc(mask);
        //eq_mask = __match_any_sync(mask, gaddr);
        //eq_mask &= __match_any_sync(mask, (uint64_t)this);
        eq_mask &= __match_any_sync(mask, sector);
        master = __ffs(eq_mask) - 1;

        uint32_t dirty = __any_sync(eq_mask, write);

        bool sector_acquired;
        //count = __popc(eq_mask);
        if (master == lane)
        {
            sector_acquired = d_ranges[r].acquire_sector(page, sector, __popc(eq_mask), dirty, ctrl, queue);
            sector_acquired_master = sector_acquired;
            //                printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
        }
        sector_acquired_master = __shfl_sync(eq_mask, sector_acquired_master, master);
    }*/


    __forceinline__
        __device__
            returned_cache_page_t<T>
            get_raw(const size_t i) const
    {
        returned_cache_page_t<T> ret;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);

            coalesce_page(lane, mask, r, page, gaddr, false, eq_mask, master, count, base_master);

            ret.addr = (T *)base_master;
            ret.size = d_ranges[r].get_sector_size() / sizeof(T);
            ret.offset = subindex / sizeof(T);
            //ret.page = page;
            __syncwarp(mask);
        }
        return ret;
    }
    __forceinline__
        __device__ void
        release_raw(const size_t i) const
    {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            uint32_t count;
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);

            uint32_t active_cnt = __popc(mask);
            eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            master = __ffs(eq_mask) - 1;
            count = __popc(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);
        }
    }

    __forceinline__
        __device__
            T
            seq_read(const size_t i) const
    {
        printf("tid %d\ti %d\t in seq read\n", (blockDim.x*blockIdx.x+threadIdx.x),i);
        uint32_t lane = lane_id();
        int64_t r = find_range(i);
        T ret;

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            bool sector_acquired_master;
            uint32_t count;
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            size_t sector_index = d_ranges[r].get_sectorindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);
            printf("tid %d\tpage %llu\tsubindex %llu\tsector_index %d\tgaddr %llu\n", (blockIdx.x*blockDim.x+threadIdx.x), (unsigned long long)page, (unsigned long long)subindex, sector_index, (unsigned long long)gaddr);

            coalesce_page(lane, mask, r, page, sector_index, gaddr, false, eq_mask, master, count, base_master, sector_acquired_master);
            __syncwarp(eq_mask);
            /*uint32_t temp_eq_mask = eq_mask;
            coalesce_sector(lane, temp_eq_mask, r, page, sector_index, false, eq_mask, master, sector_acquired_master);*/

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ret = ((T *)(base_master + subindex))[0];
            
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);
        }
        return ret;
    }
    __forceinline__
        __device__ void
        seq_write(const size_t i, const T val) const
    {
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t eq_mask;
            int master;
            uint64_t base_master;
            bool sector_acquired_master;
            uint32_t count;
            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            uint64_t gaddr = d_ranges[r].get_global_address(page);
            size_t sector_index = d_ranges[r].get_sectorindex(i);

            coalesce_page(lane, mask, r, page, sector_index, gaddr, true, eq_mask, master, count, base_master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            ((T *)(base_master + subindex))[0] = val;
            __syncwarp(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);
        }
    }
    __forceinline__
        __device__
            T
            operator[](size_t i) const
    {
        printf("tid: %d\ti %d\t in operator[]\n", (blockIdx.x*blockDim.x+threadIdx.x),i);
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
        __device__ void
        operator()(size_t i, T val) const
    {
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
            T
            AtomicAdd(const size_t i, const T val) const
    {
        //uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t lane = lane_id();
        int64_t r = find_range(i);

        T old_val = 0;

        uint32_t ctrl;
        uint32_t queue;

        if (r != -1)
        {
#ifndef __CUDACC__
            uint32_t mask = 1;
#else
            uint32_t mask = __activemask();
#endif
            uint32_t leader = __ffs(mask) - 1;
            if (lane == leader)
            {
                page_cache_d_t *pc = &(d_ranges[r].cache);
                ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) % (pc->n_ctrls);
                queue = get_smid() % (pc->d_ctrls[ctrl]->n_qps);
            }
            ctrl = __shfl_sync(mask, ctrl, leader);
            queue = __shfl_sync(mask, queue, leader);

            uint64_t page = d_ranges[r].get_page(i);
            uint64_t subindex = d_ranges[r].get_subindex(i);
            size_t sector = d_ranges[r].get_sectorindex(i);

            uint64_t gaddr = d_ranges[r].get_global_address(page);
            //uint64_t p_s = d_ranges[r].page_size;

            uint32_t active_cnt = __popc(mask);
            uint32_t eq_mask = __match_any_sync(mask, gaddr);
            eq_mask &= __match_any_sync(mask, (uint64_t)this);
            int master = __ffs(eq_mask) - 1;
            uint64_t base_master;
            uint64_t base;
            //bool memcpyflag_master;
            //bool memcpyflag;
            uint32_t count = __popc(eq_mask);
            if (master == lane)
            {
                base = d_ranges[r].acquire_page(page, count, true, queue);
                base_master = base;
                //    printf("++tid: %llu\tbase: %llu  memcpyflag_master:%llu\n", (unsigned long long) threadIdx.x, (unsigned long long) base_master, (unsigned long long) memcpyflag_master);
            }
            base_master = __shfl_sync(eq_mask, base_master, master);
            __syncwarp(eq_mask);

            eq_mask &= __match_any_sync(mask, sector);
            master = __ffs(eq_mask) - 1;

            bool sector_acquired;
            bool sector_acquired_master;
            //count = __popc(eq_mask);
            if (master == lane)
            {
                sector_acquired = d_ranges[r].acquire_sector(page, sector, __popc(eq_mask), true, ctrl, queue);
                sector_acquired_master = sector_acquired;
            //                printf("++tid: %llu\tbase: %p  page:%llu\n", (unsigned long long) threadIdx.x, base_master, (unsigned long long) page);
            }
            sector_acquired_master = __shfl_sync(eq_mask, sector_acquired_master, master);

            //if (threadIdx.x == 63) {
            //printf("--tid: %llu\tpage: %llu\tsubindex: %llu\tbase_master: %llu\teq_mask: %x\tmaster: %llu\n", (unsigned long long) threadIdx.x, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) base_master, (unsigned) eq_mask, (unsigned long long) master);
            //}
            // ((T*)(base_master+subindex))[0] = val;
            old_val = atomicAdd((T *)(base_master + subindex), val);
            // printf("AtomicAdd: tid: %llu\tpage: %llu\tsubindex: %llu\tval: %llu\told_val: %llu\tbase_master: %llx\n",
            //        (unsigned long long) tid, (unsigned long long) page, (unsigned long long) subindex, (unsigned long long) val,
            //     (unsigned long long) old_val, (unsigned long long) base_master);
            __syncwarp(eq_mask);
            if (master == lane)
                d_ranges[r].release_page(page, count);
            __syncwarp(mask);
        }

        return old_val;
    }
};

template <typename T>
struct array_t
{
    array_d_t<T> adt;

    //range_t<T>** d_ranges;
    array_d_t<T> *d_array_ptr;

    BufferPtr d_array_buff;
    BufferPtr d_ranges_buff;
    BufferPtr d_d_ranges_buff;

    void print_reset_stats(void)
    {
        std::vector<range_d_t<T>> rdt(adt.n_ranges);
        //range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
        cuda_err_chk(cudaMemcpy(rdt.data(), adt.d_ranges, adt.n_ranges * sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < adt.n_ranges; i++)
        {

            std::cout << "*********************************" << std::endl;
            std::cout << std::dec << "# READ IOs:\t" << rdt[i].read_io_cnt << std::endl;
            std::cout << std::dec << "# Accesses:\t" << rdt[i].access_cnt << std::endl;
            std::cout << std::dec << "# Misses:\t" << rdt[i].miss_cnt << std::endl
                      << "Miss Rate:\t" << ((float)rdt[i].miss_cnt / rdt[i].access_cnt) << std::endl;
            std::cout << std::dec << "# Hits:\t" << rdt[i].hit_cnt << std::endl
                      << "Hit Rate:\t" << ((float)rdt[i].hit_cnt / rdt[i].access_cnt) << std::endl;
            rdt[i].read_io_cnt = 0;
            rdt[i].access_cnt = 0;
            rdt[i].miss_cnt = 0;
            rdt[i].hit_cnt = 0;
        }
        cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt.data(), adt.n_ranges * sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
    }

    array_t(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T> *> &ranges, uint32_t cudaDevice)
    {
        adt.n_elems = num_elems;
        adt.start_offset = disk_start_offset;

        adt.n_ranges = ranges.size();
        d_array_buff = createBuffer(sizeof(array_d_t<T>), cudaDevice);
        d_array_ptr = (array_d_t<T> *)d_array_buff.get();

        //d_ranges_buff = createBuffer(n_ranges * sizeof(range_t<T>*), cudaDevice);
        d_d_ranges_buff = createBuffer(adt.n_ranges * sizeof(range_d_t<T>), cudaDevice);
        adt.d_ranges = (range_d_t<T> *)d_d_ranges_buff.get();
        //d_ranges = (range_t<T>**) d_ranges_buff.get();
        for (size_t k = 0; k < adt.n_ranges; k++)
        {
            //cuda_err_chk(cudaMemcpy(d_ranges+k, &(ranges[k]->d_range_ptr), sizeof(range_t<T>*), cudaMemcpyHostToDevice));
            cuda_err_chk(cudaMemcpy(adt.d_ranges + k, (ranges[k]->d_range_ptr), sizeof(range_d_t<T>), cudaMemcpyDeviceToDevice));
        }

        cuda_err_chk(cudaMemcpy(d_array_ptr, &adt, sizeof(array_d_t<T>), cudaMemcpyHostToDevice));
    }
};

__forceinline__
    __device__
        cache_page_t *
        page_cache_d_t::get_cache_page(const uint32_t page) const
{
    return &this->cache_pages[page];
}

__forceinline__
    __device__
        uint32_t
        page_cache_d_t::find_slot(uint64_t address, uint64_t range_id, const uint32_t queue_)
{
    bool fail = true;
    uint64_t count = 0;
    uint32_t page = 0;
    do
    {
        page = page_ticket->fetch_add(1, simt::memory_order_relaxed) % (this->n_pages);

        bool lock = false;
        uint32_t v = this->cache_pages[page].page_take_lock.load(simt::memory_order_acquire);

        if (v == FREE)
        {
            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock)
            {
                this->cache_pages[page].page_translation = address;
                this->cache_pages[page].range_id = range_id;
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
                fail = false;
            }
        }
        //assigned to someone and was able to take lock
        else if (v == UNLOCKED)
        {

            lock = this->cache_pages[page].page_take_lock.compare_exchange_weak(v, LOCKED, simt::memory_order_acquire, simt::memory_order_relaxed);
            if (lock)
            {
                uint32_t previous_address = this->cache_pages[page].page_translation;
                uint8_t previous_range = this->cache_pages[page].range_id;
                uint64_t expected_state = VALID;
                bool pass = false;

                expected_state = this->ranges[previous_range][previous_address].state.load(simt::memory_order_acquire);

                switch (expected_state)
                {
                case VALID:
                    pass = this->ranges[previous_range][previous_address].state.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass)
                    {
                        this->ranges[previous_range][previous_address].state.store(INVALID, simt::memory_order_release);
                        for (int i = 0; i< (this->n_sectors_per_page+7)/8; i++) {
                            this->ranges[previous_range][previous_address].sector_states[i].store(SECTOR_INVALID, simt::memory_order_release);
                        }
                        fail = false;
                    }
                    break;
                case INVALID:
                    pass = this->ranges[previous_range][previous_address].state.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass)
                    {
                        this->ranges[previous_range][previous_address].state.store(INVALID, simt::memory_order_release);
                        for (int i = 0; i< (this->n_sectors_per_page+7)/8; i++) {
                            this->ranges[previous_range][previous_address].sector_states[i].store(SECTOR_INVALID, simt::memory_order_release);
                        }
                        fail = false;
                    }
                    break;
                case VALID_DIRTY:
                    pass = this->ranges[previous_range][previous_address].state.compare_exchange_weak(expected_state, BUSY, simt::memory_order_acquire, simt::memory_order_relaxed);
                    if (pass)
                    {
                        uint64_t ctrl = get_backing_ctrl_(previous_address, n_ctrls, ranges_dists[previous_range]);
                        uint64_t index = get_backing_page_(ranges_page_starts[previous_range], previous_address, n_ctrls, ranges_dists[previous_range]);
                        // printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
                        //        (unsigned long long) previous_range, (unsigned long long)previous_address,
                        //        (unsigned long long) ctrl, (unsigned long long) index);
                        if (ctrl == ALL_CTRLS)
                        {
                            for (ctrl = 0; ctrl < n_ctrls; ctrl++)
                            {
                                Controller *c = this->d_ctrls[ctrl];
                                uint32_t queue = queue_ % (c->n_qps);
                                write_data(this, (c->d_qps) + queue, (index * this->n_blocks_per_page), this->n_blocks_per_page, page<<(this->n_sectors_per_page_log));
                            }
                        }
                        else
                        {

                            Controller *c = this->d_ctrls[ctrl];
                            uint32_t queue = queue_ % (c->n_qps);

                            //index = ranges_page_starts[previous_range] + previous_address;

                            write_data(this, (c->d_qps) + queue, (index * this->n_blocks_per_page), this->n_blocks_per_page, page<<(this->n_sectors_per_page_log));
                        }
                        this->ranges[previous_range][previous_address].state.store(INVALID, simt::memory_order_release);
                        for (int i = 0; i< (this->n_sectors_per_page+7)/8; i++) {
                            this->ranges[previous_range][previous_address].sector_states[i].store(SECTOR_INVALID, simt::memory_order_release);
                        }
                        fail = false;
                    }
                    break;
                default:
                    //printf("here\n");
                    break;
                }
                if (!fail)
                {
                    this->cache_pages[page].page_translation = address;
                    this->cache_pages[page].range_id = range_id;
                }
                //this->page_translation[page].store(global_address, simt::memory_order_release);
                this->cache_pages[page].page_take_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }

        count++;

    } while (fail);
    return page;
}

inline __device__ void poll_async(QueuePair *qp, uint16_t cid, uint16_t sq_pos)
{
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    sq_dequeue(&qp->sq, sq_pos);

    cq_dequeue(&qp->cq, cq_pos, &qp->sq);

    put_cid(&qp->sq, cid);
}

inline __device__ void access_data_async(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode, uint16_t *cid, uint16_t *sq_pos)
{
    nvm_cmd_t cmd;
    *cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);

    nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
    /*uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];*/
    uint64_t prp_entry = pc_entry >> (pc->n_sectors_per_page_log);
    uint64_t prp1 = pc->prp1[prp_entry];
    prp1 = (prp1) | ((pc_entry & (pc->n_sectors_per_page_minus_1))<<pc->sector_size_log);
    uint64_t prp2 = 0; //TODO: multiple prp1 lists
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    *sq_pos = sq_enqueue(&qp->sq, &cmd);
}

inline __device__ void read_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry)
{

    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
    /*uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];*/
    uint64_t prp_entry = pc_entry >> (pc->n_sectors_per_page_log);
    uint64_t prp1 = pc->prp1[prp_entry];
    printf("tid: %llu\tprp_entry: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) (prp_entry), (void*)prp1);
    prp1 = (prp1) | ((pc_entry & (pc->n_sectors_per_page_minus_1))<<pc->sector_size_log);
    uint64_t prp2 = 0; //TODO: multiple prp1 lists
    printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    sq_dequeue(&qp->sq, sq_pos);

    put_cid(&qp->sq, cid);
}

inline __device__ void write_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry)
{
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);

    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);
    // printf("write_data startinglba: %llu\tn_blocks: %llu\tpc_entry: %llu\tdata[0]: %llu\n", (unsigned long long) starting_lba, (unsigned long long) n_blocks, pc_entry,
    //        (unsigned long long) (((unsigned*)(pc->base_addr + (pc_entry*pc->page_size)))[0]));

    nvm_cmd_header(&cmd, cid, NVM_IO_WRITE, qp->nvmNamespace);
    /*uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];*/
    uint64_t prp_entry = pc_entry >> (pc->n_sectors_per_page_log);
    uint64_t prp1 = pc->prp1[prp_entry];
    prp1 = (prp1) | ((pc_entry & (pc->n_sectors_per_page_minus_1))<<pc->sector_size_log);
    uint64_t prp2 = 0; //TODO: multiple prp1 lists
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    sq_dequeue(&qp->sq, sq_pos);

    put_cid(&qp->sq, cid);
}

inline __device__ void access_data(page_cache_d_t *pc, QueuePair *qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, const uint8_t opcode)
{
    //uint64_t starting_lba = starting_byte >> qp->block_size_log;
    //uint64_t rem_bytes = starting_byte & qp->block_size_minus_1;
    //uint64_t end_lba = CEIL((starting_byte+num_bytes), qp->block_size);

    //uint16_t n_blocks = CEIL(num_bytes, qp->block_size, qp->block_size_log);

    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    //printf("cid: %u\n", (unsigned int) cid);

    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    /*uint64_t prp1 = pc->prp1[pc_entry];
    uint64_t prp2 = 0;
    if (pc->prps)
        prp2 = pc->prp2[pc_entry];*/
    uint64_t prp_entry = pc_entry >> (pc->n_sectors_per_page_log);
    uint64_t prp1 = pc->prp1[prp_entry];
    prp1 = (prp1) | ((pc_entry & (pc->n_sectors_per_page_minus_1))<<pc->sector_size_log);
    uint64_t prp2 = 0; //TODO: multiple prp1 lists
    //printf("tid: %llu\tstart_lba: %llu\tn_blocks: %llu\tprp1: %p\n", (unsigned long long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long) starting_lba, (unsigned long long) n_blocks, (void*) prp1);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
    uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    sq_dequeue(&qp->sq, sq_pos);

    put_cid(&qp->sq, cid);
}

//#ifndef __CUDACC__
//#undef __device__
//#undef __host__
//#undef __forceinline__
//#endif

#endif // __PAGE_CACHE_H__
