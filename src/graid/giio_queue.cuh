#ifndef __GIIO_QUEUE_CUH__
#define __GIIO_QUEUE_CUH__

#ifndef __GRAID__
#error "Must compile with GRAID support"
#endif

#include <stdio.h>
#include "dmapool.h"

/**************************************************************************
 *                         GIIO Command Allocation
 * A pool of commands is defined with: GIIOCmdPool
 * Each command's location of the pool is it's command id (cid): 0 ~ (GIIOCmdPoolSize - 1)
 * All available cids are stored in cmdids array.
 * Note that for performance reasons, the size of cmdids array is: (2 * GIIOCmdPoolSize)
 * 
 * The integer stored in cmdids have 3 parts: version, phase and GIIO_CMDID_ARRSZ_SHIFT-bits cid.
 * If the cmdids[i]:cid == EMPTY_CMDID, it means the cid is empty (Empty-CID)
 * 
 * If the phase extracted from tail does not equal to cmdids[i]:phase, it means that the
 * allocator has not finished updating this entry. Phase is for command queue to distinguish
 * if the cid is new, and for cmdids pool to see if the cid is valid for this phase.
 * 
 * Version is a counter that indicates the corresponding cid have been used how many times.
 * Version is for IO-working GPU to fill in complete entry along with cid, so that the IO-submitting
 * GPU can check if the latest command has been completed.
 * 
 * When allocate a command id:
 *   1): Atomic incr cmdid_head by 1 and get the old value for idx of cmdids array and the phase
 *   2): Wait for a valid cid:
 *       a) cmdids[idx]:phase == head:phase
 *       b) cmdids[idx]:cid != Empty-CID
 *   3): Update cmdids[idx] = (head:phase | Empty-CID)
 * 
 * When free a command id:
 *   1): Atomic incr cmdid_tail by 1 and get the old value for idx of cmdids array and the phase
 *   2): Wait cmdids[idx] == (prev-phase | Empty-CID)
 *   3): Use version, phase, and freeing cid to update cmdids[idx]
 **************************************************************************/
#define EMPTY_CMDID		(1U << GIIOCmdPoolSizeShift)
#define GIIO_CMDID_ARRSZ_SHIFT	(GIIOCmdPoolSizeShift + 1U)
#define GIIO_CMDID_ARRSZ	(1U << GIIO_CMDID_ARRSZ_SHIFT)
#define GIIO_CMDID_ARRSZ_MASK	(GIIO_CMDID_ARRSZ - 1U)
#define GIIO_CMDID_PHASE_MASK	(0xFU << GIIO_CMDID_ARRSZ_SHIFT)
#define GIIO_CMDID_VER_SHIFT	(GIIO_CMDID_ARRSZ_SHIFT + 4)
#define GIIO_CMDID_VER_MASK	(0xFU << GIIO_CMDID_VER_SHIFT)

#define GIIO_CMDID_IDX(val)		(val & GIIO_CMDID_ARRSZ_MASK)
#define GIIO_CMDID_CID(val)		GIIO_CMDID_IDX(val)
#define GIIO_CMDID_PHASE(val)		(val & GIIO_CMDID_PHASE_MASK)
#define GIIO_CMDID_PREV_PHASE(val)	((val - GIIO_CMDID_ARRSZ) & GIIO_CMDID_PHASE_MASK)
#define GIIO_CMDID_IS_EMPTY(val)	(GIIO_CMDID_CID(val) == EMPTY_CMDID)
#define GIIO_CMDID_VER(val)		((val >> GIIO_CMDID_VER_SHIFT) & 0xFU)

#define GIIO_CMDID_SET(ver, phase, cid) \
		/* ver is raw number, shift to target bit offset */ \
		(((ver) & 0xFU) << GIIO_CMDID_VER_SHIFT) | \
		(phase) | /* phase is already on target bit position */ \
		(cid)

struct GIIOQueueState {
	uint32_t qid;
	uint32_t pad;
	GIIOQueue *queue_devp;
	GIIOCpltPool *cplt_devp;
	GIIOCmdPool *cmdpool_devp;
	uint32_t cmdids[GIIO_CMDID_ARRSZ];
	uint32_t cmdid_head;
	uint32_t cmdid_tail;
	uint32_t queue_head;
	uint32_t queue_tail;
	uint64_t wait_alloc_counter;
	uint64_t wait_free_counter;
	uint64_t wait_ent_phase_counter;
	uint64_t wait_ent_ack_counter;
	uint64_t wait_ack_counter;
	uint64_t wait_complete_counter;
};

#ifdef __CUDACC__
extern __device__
int giioq_access_data(GIIOQueueState *giio_qs,
		      const uint64_t slba, const uint32_t nlb, uint8_t op,
		      const uint64_t prp1, const uint64_t prp2);
extern __global__
void giioq_print_counters(GIIOQueueState * const qqs);
#endif

#endif
