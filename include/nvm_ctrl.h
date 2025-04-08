#ifndef __NVM_CTRL_H__
#define __NVM_CTRL_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __DIS_CLUSTER__
#include <sisci_types.h>
#endif



/* 
 * Minimum size of mapped controller memory.
 */
#define NVM_CTRL_MEM_MINSIZE                        0x2000



#if defined (__unix__)
/*
 * Initialize NVM controller handle.
 *
 * Read from controller registers and initialize controller handle. 
 * This function should be used when using the kernel module or to manually
 * read from sysfs.
 *
 * Note: fd must be opened with O_RDWR and O_NONBLOCK
 */
int nvm_ctrl_init(nvm_ctrl_t** ctrl, int fd);
#endif



/* 
 * Initialize NVM controller handle.
 *
 * Read from controller registers and initialize the controller handle using
 * a memory-mapped pointer to the PCI device BAR.
 *
 * This function should be used when neither SmartIO nor the disnvme kernel
 * module are used.
 *
 * Note: ctrl_mem must be at least NVM_CTRL_MEM_MINSIZE large and mapped
 *       as IO memory. See arguments for mmap() for more info.
 */
int nvm_raw_ctrl_init(nvm_ctrl_t** ctrl, volatile void* mm_ptr, size_t mm_size);



/*
 * Release controller handle.
 */
void nvm_ctrl_free(nvm_ctrl_t* ctrl);



/* 
 * Reset NVM controller.
 *
 * The queue memory must be memset to zero and be exactly one page size large.
 * IO addresses must align to the controller page size. 
 *
 * Note: The controller must be unbound from any driver before attempting to
 *       reset the controller.
 *
 * Note: This function is implicitly called by the controller manager, so it
 *       should not be necessary to call it directly.
 */
int nvm_raw_ctrl_reset(const nvm_ctrl_t* ctrl, uint64_t acq_ioaddr, uint64_t asq_ioaddr);



#ifdef __DIS_CLUSTER__
/* 
 * Initialize NVM controller handle.
 *
 * Read from device registers and initialize controller handle. 
 * This function should be used when SmartIO is being used.
 */
int nvm_dis_ctrl_init(nvm_ctrl_t** ctrl, uint32_t smartio_fdid);
#endif



#ifdef __DIS_CLUSTER__
int nvm_dis_ctrl_map_p2p_device(const nvm_ctrl_t* ctrl, sci_smartio_device_t dev, uint64_t* ioaddr);
#endif



#ifdef __DIS_CLUSTER__
void nvm_dis_ctrl_unmap_p2p_device(const nvm_ctrl_t* ctrl, sci_smartio_device_t dev);
#endif


#ifdef __GRAID__
int graid_ctrl_init(nvm_ctrl_t** ctrl, struct nvm_ctrl_info *info, struct nvm_ns_info *ns,
		    uint16_t *n_sqs, uint16_t *n_cqs, const int dg_id, const int vd_id);
void* graid_ctrl_sq_vaddr(const nvm_ctrl_t* ctrl, const int qp_id);
void* graid_ctrl_cq_vaddr(const nvm_ctrl_t* ctrl, const int qp_id);
uint64_t graid_ctrl_sq_ioaddr(const nvm_ctrl_t* ctrl, const int qp_id);
uint64_t graid_ctrl_cq_ioaddr(const nvm_ctrl_t* ctrl, const int qp_id);
#endif


#endif /* __NVM_CTRL_H__ */
