#ifndef __NVM_TYPES_H__
#define __NVM_TYPES_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <stddef.h>
#include <stdint.h>
#include <simt/atomic>

#ifndef __align__ 
#define __align__(x)
#endif



/* 
 * NVM controller handle.
 *
 * Note: This structure will be allocated by the API and needs to be
 *       released by the API.
 */
typedef struct
{
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    uint8_t                 dstrd;          // Doorbell stride (in encoded form)
    bool                    cqr;            // Contiguous queue required
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    uint32_t                max_qs;         // Maximum queue entries supported (MQES)
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to BAR0 of the physical device
} nvm_ctrl_t;




/*
 * NVM admin queue-pair reference handle.
 *
 * As only a single process can be responsible of resetting the controller and
 * setting administration queues, this structure represents a remote handle to
 * that process. It is used as a descriptor for executing RPC calls to the 
 * remote process owning the admin queues.
 *
 * Note: This structure will be allocated by the API and needs to be released
 *       by the API.
 */
struct nvm_admin_reference;
typedef struct nvm_admin_reference* nvm_aq_ref;



/*
 * DMA mapping descriptor.
 *
 * This structure describes a region of memory that is accessible for the
 * NVM controller using DMA. The API assumes a continuous virtual memory
 * address, but the physical pages do not need to be contiguous.
 *
 * The structure contains a variably sized array of bus addresses that maps
 * to the physical memory pages. The user should therefore not create a local
 * instance of this descriptor, but rather rely on the API to allocate and
 * instantiate members.
 *
 * Note: Only page-aligned addresses are supported in NVM Express
 *
 * Note: This structure will be allocated by the API and needs to be released
 *       by the API.
 */
typedef struct
{
    void*                   vaddr;          // Virtual address to start of region (NB! can be NULL)
    int8_t                  local;          // Is this local memory
    int8_t                  contiguous;     // Is memory contiguous
    size_t                  page_size;      // Controller's page size (MPS)
    size_t                  n_ioaddrs;      // Number of MPS-sized pages
    uint64_t                ioaddrs[];      // Physical/IO addresses of the memory pages
}  nvm_dma_t;



typedef simt::atomic<uint32_t, simt::thread_scope_device> padded_struct_pc;

#define CACHELINE_SIZE (128)

#define STATES_PER_CACHELINE (CACHELINE_SIZE/sizeof(padded_struct_pc))
/* typedef struct __align__(32) */
/* { */
/*     simt::atomic<uint32_t, simt::thread_scope_device>  val; */
/* //    uint8_t pad[32-4]; */
/* } __attribute__((aligned (32))) padded_struct_pc; */


typedef struct __align__(32)
{
    simt::atomic<uint32_t, simt::thread_scope_device>  val;
    //uint8_t pad[32-8];
} __attribute__((aligned (32))) padded_struct;

/* typedef struct __align__(32) */
/* { */
/*     simt::atomic<uint32_t, simt::thread_scope_system>  val; */
/*     //uint8_t pad[32-8]; */
/* } __attribute__((aligned (32))) padded_struct_pc; */


/* typedef struct __align__(32) */
/* { */
/*     simt::atomic<uint64_t, simt::thread_scope_system>  val; */
/*     uint8_t pad[32-8]; */
/* } __attribute__((aligned (32))) padded_struct; */

/* 
 * NVM queue descriptor.
 *
 * This structure represents an NVM IO queue and holds information 
 * about memory addresses, queue entries as well as a memory mapped pointer to 
 * the device doorbell register. Maximum queue size is limited to a single 
 * page.
 *
 * Note: This descriptor represents both completion and submission queues.
 */
typedef struct
{
    simt::atomic<uint32_t, simt::thread_scope_device> head_lock;
    uint8_t pad0[28];
    simt::atomic<uint32_t, simt::thread_scope_device> tail_lock;
    uint8_t pad1[28];
    simt::atomic<uint32_t, simt::thread_scope_device> head;
    uint8_t pad2[28];
    simt::atomic<uint32_t, simt::thread_scope_device> tail;
    uint8_t pad3[28];
    simt::atomic<uint32_t, simt::thread_scope_system> tail_copy;
    uint8_t pad4[28];
    simt::atomic<uint32_t, simt::thread_scope_system> head_copy;
    uint8_t pad5[28];

    /* padded_struct<simt::atomic<uint32_t, simt::thread_scope_system>> head; */
    /* padded_struct<simt::atomic<uint32_t, simt::thread_scope_system>> tail; */
    simt::atomic<uint32_t, simt::thread_scope_device> in_ticket;
    uint8_t pad6[28];
    simt::atomic<uint32_t, simt::thread_scope_device> cid_ticket;
    //uint8_t pad7[28];
    padded_struct* tickets;

    padded_struct* head_mark;
    padded_struct* tail_mark;
    padded_struct* cid;
    padded_struct* pos_locks;

    uint16_t* clean_cid;
    uint32_t qs_minus_1;
    uint32_t qs_log2;
    uint16_t                no;             // Queue number (must be unique per SQ/CQ pair)
    uint16_t                es;             // Queue entry size
    uint32_t                qs;             // Queue size (number of entries)
    //uint16_t                head;           // Queue's head pointer
    //uint16_t                tail;           // Queue's tail pointer
    int8_t                  phase;          // Current phase tag
    int8_t                  local;          // Is the queue allocated in local memory
    uint32_t                last;           // Used internally to check db writes
    volatile uint32_t*      db;             // Pointer to doorbell register (NB! write only)
    volatile void*          vaddr;          // Virtual address to start of queue memory
    uint64_t                ioaddr;         // Physical/IO address to start of queue memory
} nvm_queue_t;



/*
 * Convenience type for representing a single-page PRP list.
 */
typedef struct __align__(32)
{
    volatile void*          vaddr;          // Virtual address to memory page
    int16_t                 local;          // Indicates if the page is local memory
    size_t                  page_size;      // Page size
    uint64_t                ioaddr;         // Physical/IO address of memory page
} __attribute__((aligned (32))) nvm_prp_list_t;



/* 
 * NVM completion queue entry type (16 bytes) 
 */
typedef struct __align__(16) 
{
    volatile uint32_t                dword[4];       // The name DWORD is chosen to reflect the specification
} __attribute__((aligned (16))) nvm_cpl_t;



/* 
 * NVM command queue entry type (64 bytes) 
 */
typedef struct __align__(64) 
{
    uint32_t                dword[16];
} __attribute__((aligned (64))) nvm_cmd_t;



/*
 * Controller information structure.
 *
 * Holds information about an NVM controller retrieved from reading on-board
 * registers and running an IDENTIFY CONTROLLER admin command.
 */
struct nvm_ctrl_info
{
    uint32_t                nvme_version;   // NVM Express version number
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    size_t                  db_stride;      // Doorbell stride (DSTRD)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    int                     contiguous;     // Contiguous queues required (CQR)
    uint16_t                max_entries;    // Maximum queue entries supported (MQES)
    uint8_t                 pci_vendor[4];  // PCI vendor and subsystem vendor identifier
    char                    serial_no[20];  // Serial number (NB! not null terminated)
    char                    model_no[40];   // Model number (NB! not null terminated)
    char                    firmware[8];    // Firmware revision
    size_t                  max_data_size;  // Maximum data transfer size (MDTS)
    size_t                  max_data_pages; // Maximum data transfer size (in controller pages)
    size_t                  cq_entry_size;  // CQ entry size (CQES)
    size_t                  sq_entry_size;  // SQ entry size (SQES)
    size_t                  max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t                  max_n_ns;       // Maximum number of namespaces (NN)
};



/*
 * Namespace information structure.
 *
 * Holds informaiton about an NVM namespace.
 */
struct nvm_ns_info
{
    uint32_t                ns_id;          // Namespace identifier
    size_t                  size;           // Size in logical blocks (NSZE)
    size_t                  capacity;       // Capacity in logical blocks (NCAP)
    size_t                  utilization;    // Utilization in logical blocks (NUSE)
    size_t                  lba_data_size;  // Logical block size (LBADS)
    size_t                  metadata_size;  // Metadata size (MS)
};



//#ifndef __CUDACC__
//#undef __align__
//#endif

#endif /* __NVM_TYPES_H__ */
