#ifndef __GRAID__
#error "Must compile with GRAID support"
#endif

#include <atomic>
#include <mutex>
#include <stdexcept>
#include "cuda.h"
#include "dprintf.h"
#include "lib_ctrl.h"
#include "nvm_ctrl.h"
#include "regs.h"
#include "ctrl.h"
#include "giio_queue.cuh"
#include "linux/map.h"

#define unlikely(x) __builtin_expect(!!(x), 0)
typedef unsigned char uuid_t[16];

#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "ioctl.h"
#include "ioctl_types.h"
#include "dmapool.h"
#include "linux-nvme.h"
#include <fcntl.h>
#include <fstream>

static const char *graid_dev_path = "/dev/graid_app";

using error = std::runtime_error;

static int init_giioqs(nvm_ctrl_t* ctrl, GIIOQueueState **giioqs_array);

/*
 * Device descriptor
 */
struct device
{
	int fd; /* ioctl file descriptor */
	std::mutex lock;
	cudaIpcMemHandle_t ipc_handle;
	struct graid_mem_addrs phys_addrs;
	struct graid_mem_addrs virt_aoffs;
	void *devp_bar1;

	int init();
	device() : fd(-1), devp_bar1(NULL) {}
	~device();

	int map_bar0(int dg_id, void **mm_ptr, void **mm_devp);
	int reset_ctrl(int dg_id, nvm_ctrl_t *ctrl);
	int identify_ctrl(int dg_id, struct nvm_ctrl_info *info, const nvm_ctrl_t *ctrl);
	int identify_ns(int dg_id, int ns_id, struct nvm_ns_info *ns);
	int pin_bar1(uint64_t dev_addr, unsigned page_nr, uint64_t *phy_addrs) const;
	int unpin_bar1(uint64_t dev_addr) const;

private:
	int get_bar1_res();
	int open_bar1_handle();
	void close_bar1_handle();
	void* map_address(uint64_t phys_addr, size_t len);
};

class graid_ctrl_priv : public nvm_priv {
	public:
	const int dg_id;
	GIIOQueueState *giioqs_array;

	graid_ctrl_priv(int _dg_id) :
		dg_id(_dg_id), giioqs_array(NULL) { }

	~graid_ctrl_priv() {
		if (giioqs_array)
			cudaFree(giioqs_array);
	}
};

static device *graid_dev = NULL;
static int64_t dev_usecnt;
static std::atomic_flag dev_lock;

int device::get_bar1_res()
{
	struct graid_app_get_bar1_req req;

	req.ctlr_id = 0;
	req.ipc_handle = (cuIPC_Handle_t*)&ipc_handle;
	req.phys_addrs = &phys_addrs;
	req.virt_aoffs = &virt_aoffs;

	return ioctl(fd, GRAID_APP_GET_BAR1, &req);
}

int device::reset_ctrl(int dg_id, nvm_ctrl_t *ctrl)
{
	volatile uint32_t *cc = CC(ctrl->mm_ptr);

	// FIXME: Temporary implementation, not compliant to spec

	// Set CC.EN to 0
	*cc = *cc & ~1;

	uint64_t timeout = ctrl->timeout * 1000000UL;
	uint64_t remaining = _nvm_delay_remain(timeout);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	while (CC$EN(*cc) == 0)
	{
		if (remaining == 0)
		{
			dprintf("Timeout exceeded while waiting for controller reset\n");
			return ETIME;
		}

		remaining = _nvm_delay_remain(remaining);
	}

	return 0;
}

int device::identify_ctrl(int dg_id, struct nvm_ctrl_info *info, const nvm_ctrl_t *ctrl)
{
	struct graid_ident_ctrl_req req;
	struct nvme_id_ctrl idctrl_buf = { 0 };
	int rc;

	req.dg_id = dg_id;
	req.id_ctrl_buf = &idctrl_buf;

	rc = ioctl(fd, GRAID_APP_IDENT_CTRL, &req);
	if (rc)
		return rc;

	memset(info, 0, sizeof(*info));

	info->nvme_version = (uint32_t) *VER(ctrl->mm_ptr);
	info->page_size = ctrl->page_size;
	info->db_stride = 1UL << ctrl->dstrd;
	info->timeout = ctrl->timeout;
	info->contiguous = ctrl->cqr;
	info->max_entries = ctrl->max_qs;

	memcpy(info->pci_vendor, &idctrl_buf.vid, 4);
	memcpy(info->serial_no, &idctrl_buf.sn, 20);
	memcpy(info->model_no, &idctrl_buf.mn, 40);
	memcpy(info->firmware, &idctrl_buf.fr, 8);

	info->max_data_size = (1UL << idctrl_buf.mdts) * (1UL << (12 + CAP$MPSMIN(ctrl->mm_ptr)));
	info->max_data_pages = info->max_data_size / info->page_size;
	info->sq_entry_size = 1 << _RB(idctrl_buf.sqes, 3, 0);
	info->cq_entry_size = 1 << _RB(idctrl_buf.cqes, 3, 0);
	info->max_out_cmds = idctrl_buf.maxcmd;
	info->max_n_ns = idctrl_buf.nn;

	return 0;
}

int device::identify_ns(int dg_id, int vd_id, struct nvm_ns_info *ns)
{
	struct graid_ident_ns_req req;
	struct nvme_id_ns idns_buf = { 0 };
	int rc;

	req.dg_id = dg_id;
	req.vd_id = vd_id;
	req.id_ns_buf = &idns_buf;

	rc = ioctl(fd, GRAID_APP_IDENT_NS, &req);
	if (rc)
		return rc;

	memset(ns, 0, sizeof(*ns));
	ns->ns_id = vd_id + 1;
	ns->size = idns_buf.nsze;
	ns->capacity = idns_buf.nsze;
	ns->utilization = idns_buf.nsze;

	uint8_t format_idx = idns_buf.flbas;
	struct nvme_lbaf &lba_format = idns_buf.lbaf[format_idx];
	ns->lba_data_size = 1u << lba_format.ds;
	ns->metadata_size = lba_format.ms;

	return 0;
}

int device::pin_bar1(uint64_t dev_addr, unsigned page_nr, uint64_t *phy_addrs) const
{
        int rc;
	std::unique_ptr<uint8_t[]> pin_buf(new uint8_t[sizeof(graid_gpu_bar1_req) + (sizeof(uint64_t) * page_nr)]);
        graid_gpu_bar1_req *req = reinterpret_cast<graid_gpu_bar1_req*>(pin_buf.get());
        if (!req)
                return -ENOMEM;

        req->ctlr_id = GRAID_APP_CTLR_ID;
        req->page_nr = page_nr;
        req->page_sz = GPU_PAGE_SIZE;
        req->dev_addr = dev_addr;
        rc = ioctl(fd, GRAID_APP_PIN_BAR1, req);
        if (rc == 0)
                memcpy(phy_addrs, req->paddrs, sizeof(uint64_t) * page_nr);
        return rc ? -errno : 0;
}

int device::unpin_bar1(uint64_t dev_addr) const
{
        graid_gpu_bar1_req req;
        req.ctlr_id = GRAID_APP_CTLR_ID;
        req.dev_addr = dev_addr;
        int rc = ioctl(fd, GRAID_APP_UNPIN_BAR1, &req);
        return rc ? -errno : 0;
}

int device::open_bar1_handle() {
	cudaError_t rc;

	rc = cudaIpcOpenMemHandle(&devp_bar1, ipc_handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
        if (rc != cudaSuccess) {
		devp_bar1 = 0;
		dprintf("cuIpcOpenMemHandle on buffer error: %s\n", cudaGetErrorString(rc));
        }
	return rc == cudaSuccess ? 0 : -1;
}

void device::close_bar1_handle() {
	cudaError_t rc;

	if (devp_bar1 == 0)
		return;

	rc = cudaIpcCloseMemHandle(devp_bar1);
	if (rc != cudaSuccess) {
		dprintf("Failed to close BAR1 memory handle: %s\n", cudaGetErrorString(rc));
	}
	devp_bar1 = 0;
}

int device::init() {
	fd = open(graid_dev_path, O_RDWR);
	if (fd < 0) {
		dprintf("Failed to open descriptor: %s", strerror(errno));
		goto err_out;
	}

	if (get_bar1_res() != 0) {
		dprintf("Failed to get BAR1 resource information: %s", strerror(errno));
		goto err_out;
	}

	if (open_bar1_handle() != 0) {
		goto err_out;
	}

	return 0;

err_out:
	return -EIO;
}

device::~device() {
	close_bar1_handle();
	if (fd != -1)
		close(fd);
}

void* device::map_address(uint64_t phys_addr, size_t len)
{
        if (!phys_addr)
                return NULL;

        void *hp = mmap(NULL, len, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_LOCKED | MAP_NORESERVE | MAP_POPULATE,
                        fd, phys_addr);
        if (hp == MAP_FAILED) {
                return NULL;
        }
        return hp;
}

int device::map_bar0(int dg_id, void **mm_ptr, void **mm_devpp)
{
	*mm_ptr = map_address(phys_addrs.dg_regs[dg_id], NVM_CTRL_MEM_MINSIZE);
	if (*mm_ptr == NULL)
		return -EFAULT;

	*mm_devpp = (void*)(((uint64_t)devp_bar1) + virt_aoffs.dg_regs[dg_id]);
	return 0;
}

device *get_graid_dev()
{
	device *dev = NULL;

	while (dev_lock.test_and_set())
		std::this_thread::sleep_for(std::chrono::milliseconds(1));

	++dev_usecnt;
	if (dev_usecnt <= 0) {
		dprintf("Invalid device usage count value");
		goto unlock_out;
	}
	if (dev_usecnt == 1) {
		if (graid_dev != NULL) {
			dprintf("Invalid graid_dev state");
			goto unlock_out;
		}

		int fd = open(graid_dev_path, O_RDWR);
		if (fd < 0) {
			dprintf("Failed to open descriptor: %s", strerror(errno));
			goto unlock_out;
		}

		dev = new device();
		if (dev && dev->init() != 0) {
			delete dev;
			dev = NULL;
		}
		graid_dev = dev;
	} else {
		dev = graid_dev;
	}

unlock_out:
	dev_lock.clear();
	return dev;
}

void put_graid_dev(device *dev)
{
	while (dev_lock.test_and_set())
		std::this_thread::sleep_for(std::chrono::milliseconds(1));

	if (dev != graid_dev)
		throw error("Invalid graid device pointer");
	if (dev_usecnt <= 0)
		throw error("Invalid device usage count value");
	if (--dev_usecnt == 0) {
		delete graid_dev;
		graid_dev = NULL;
	}
	dev_lock.clear();
}

/*
 * Unmap controller memory and close file descriptor.
 */
static void graid_release(struct device* dev, volatile void* mm_ptr, size_t mm_size)
{
	put_graid_dev(dev);
}

/*
 * Call kernel module ioctl and map memory for DMA.
 */
static int graid_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddrs)
{
	const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
	int rc = EINVAL;

	dprintf("%s: dev_ptr=%016lx, page_nr=%lu\n", __func__, (uint64_t)m->buffer, va->n_pages);

	switch (m->type)
	{
#ifdef _CUDA
	case MAP_TYPE_CUDA:
		if (va->page_size != 1u << 16) {
			printf("Invalid GPU page size: %lu\n", va->page_size);
			break;
		}
		rc = dev->pin_bar1((uint64_t)m->buffer, va->n_pages, ioaddrs);
		break;
#endif
	case MAP_TYPE_API:
	case MAP_TYPE_HOST:
	default:
		printf("Unsupported memory type in map for device: type=%d\n", m->type);
		return EINVAL;
	}

	if (rc < 0)
	{
		printf("Page mapping kernel request failed (ptr=%p, n_pages=%zu, ioctl_type=%u): %s\n",
			m->buffer, va->n_pages, m->type, strerror(errno));
		rc = errno;
	}

	return rc;
}

/*
 * Call kernel module ioctl and unmap memory.
 */
static void graid_unmap(const struct device* dev, const struct va_range* va)
{
	const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
	int rc = EINVAL;

	dprintf("%s: dev_ptr=%016lx\n", __func__, (uint64_t)m->buffer);

	switch (m->type)
	{
#ifdef _CUDA
	case MAP_TYPE_CUDA:
		rc = dev->unpin_bar1((uint64_t)m->buffer);
		break;
#endif
	case MAP_TYPE_API:
	case MAP_TYPE_HOST:
	default:
		printf("Unsupported memory type in map for device: type=%d\n", m->type);
		return;
	}

	if (rc < 0)
	{
		printf("Page un-map kernel request failed (ptr=%p): %s\n", m->buffer, strerror(errno));
		rc = errno;
	}
}

/*
 * Device operations
 */
static const struct device_ops graid_device_ops =
{
	.release_device = &graid_release,
	.map_range = &graid_map,
	.unmap_range = &graid_unmap,
};

int graid_ctrl_init(nvm_ctrl_t** ctrl, struct nvm_ctrl_info *info, struct nvm_ns_info *ns,
		    uint16_t *n_sqs, uint16_t *n_cqs, GIIOQueueState **p_devp_giioqs,
		    const int dg_id, const int vd_id)
{
	int err;
	struct device* dev;
	void* mm_ptr = NULL;
	void* mm_devp = NULL;
	graid_ctrl_priv *priv = NULL;

	*ctrl = NULL;

	dev = get_graid_dev();
	if (dev == NULL) {
		return EIO;
	}

	if (dev->map_bar0(dg_id, &mm_ptr, &mm_devp) != 0) {
		printf("Failed to map Graid NVMe BAR0 memory: DG%d\n", dg_id);
		goto err_put_dev_out;
	}

	priv = new graid_ctrl_priv(dg_id);
	if (priv == NULL) {
		printf("Failed to allocate graid controller private data\n");
		goto err_put_dev_out;
	}

	err = _nvm_ctrl_init(ctrl, dev, &graid_device_ops, DEVICE_TYPE_GRAID, mm_ptr, NVM_CTRL_MEM_MINSIZE, mm_devp, priv);
	if (err != 0) {
		printf("Failed to initialize nvm_ctrl_t: %s\n", strerror(errno));
		goto err_put_dev_out;
	}

	if (init_giioqs(*ctrl, &(priv->giioqs_array)) != 0) {
		printf("Failed to allocate GIIOQueueState\n");
		goto err_free_ctrl_out;
	}
	*p_devp_giioqs = priv->giioqs_array;

	if (dev->reset_ctrl(dg_id, *ctrl) != 0) {
		printf("Failed to reset controller: %s\n", strerror(errno));
		goto err_free_ctrl_out;
	}

	if (dev->identify_ctrl(dg_id, info, *ctrl) != 0) {
		printf("Failed to identify controller: %s\n", strerror(errno));
		goto err_free_ctrl_out;
	}

	if (dev->identify_ns(dg_id, vd_id, ns) != 0) {
		printf("Failed to identify name space: %s\n", strerror(errno));
		goto err_free_ctrl_out;
	}

	/* FIXME: Get it from graid driver */
	*n_sqs = NVMeMaxDGQNR;
	*n_cqs = NVMeMaxDGQNR;

	return 0;

err_free_ctrl_out:
        nvm_ctrl_free(*ctrl);
	*ctrl = NULL;
err_put_dev_out:
	put_graid_dev(dev);
	return EIO;
}

enum AddrType {
	AT_DEVP = 1,
	AT_PHYA = 2,
};

static uint64_t graid_ctrl_get_queue_addr(const nvm_ctrl_t* ctrl, const int qp_id, enum QueueType qt, enum AddrType at)
{
	const struct controller *container = _nvm_container_of(ctrl, struct controller, handle);
	const struct device *dev = container->device;
	const struct graid_ctrl_priv *priv = dynamic_cast<graid_ctrl_priv *>(ctrl->priv);
	const int dg_id = priv->dg_id;
	const int idx = (dg_id * NVMeMaxDGQNR) + qp_id - 1;
	const graid_mem_addrs *mem_addrs;
	uint64_t ret_addr = 0;

	//dprintf("%s: priv->dg_id=%u, priv->giioqs_array=%016lx\n",
	//		__func__, priv->dg_id, (uintptr_t )priv->giioqs_array);

	switch (at) {
	case AT_DEVP:
		mem_addrs = &dev->virt_aoffs;
		break;
	case AT_PHYA:
		mem_addrs = &dev->phys_addrs;
		break;
	default:
		throw error("Invalid address type");
	}

	const int socket_idx = 0;
	switch (qt) {
	case QT_SQ:
		ret_addr = mem_addrs->dg_sqs[idx];
		break;
	case QT_CQ:
		ret_addr = mem_addrs->dg_cqs[socket_idx][idx];
		break;
	case QT_GIIO_POOL:
		ret_addr = mem_addrs->giio_pools[qp_id];
		break;
	case QT_GIIO_CPLT:
		ret_addr = mem_addrs->giio_cplts[qp_id];
		break;
	case QT_GIIO_QUEUE:
		ret_addr = mem_addrs->giio_queues[qp_id];
		break;
	default:
		throw error("Invalid queue type");
	};

	if (at == AT_DEVP)
		ret_addr += (uint64_t)dev->devp_bar1;

	return ret_addr;
}

void* graid_ctrl_sq_vaddr(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return (void*)graid_ctrl_get_queue_addr(ctrl, qp_id, QT_SQ, AT_DEVP);
}

void* graid_ctrl_cq_vaddr(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return (void*)graid_ctrl_get_queue_addr(ctrl, qp_id, QT_CQ, AT_DEVP);
}

uint64_t graid_ctrl_sq_ioaddr(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return graid_ctrl_get_queue_addr(ctrl, qp_id, QT_SQ, AT_PHYA);
}

uint64_t graid_ctrl_cq_ioaddr(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return graid_ctrl_get_queue_addr(ctrl, qp_id, QT_CQ, AT_PHYA);
}

GIIOCmdPool* graid_ctrl_get_giio_pool(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return (GIIOCmdPool*)graid_ctrl_get_queue_addr(ctrl, qp_id, QT_GIIO_POOL, AT_DEVP);
}

GIIOCpltPool* graid_ctrl_get_giio_cplt(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return (GIIOCpltPool*)graid_ctrl_get_queue_addr(ctrl, qp_id, QT_GIIO_CPLT, AT_DEVP);
}

GIIOQueue* graid_ctrl_get_giio_queue(const nvm_ctrl_t* ctrl, const int qp_id)
{
	return (GIIOQueue*)graid_ctrl_get_queue_addr(ctrl, qp_id, QT_GIIO_QUEUE, AT_DEVP);
}

__global__ void init_giioqs_kernel(GIIOQueueState *giio_qs, uint32_t qid, GIIOCmdPool *pool, GIIOCpltPool *cplt, GIIOQueue *queue)
{
	uint32_t myidx = threadIdx.x;
	const uint32_t step = blockDim.x;
	constexpr uint32_t empty_cmdid = GIIO_CMDID_PREV_PHASE(0) | EMPTY_CMDID;

	if (myidx == 0) {
		giio_qs->qid = qid;
		giio_qs->queue_devp = queue;
		giio_qs->cplt_devp = cplt;
		giio_qs->cmdpool_devp = pool;
		giio_qs->cmdid_head = 0;
		giio_qs->cmdid_tail = GIIOCmdPoolSize;
		giio_qs->queue_head = GIIOQueueSize - 1;
		giio_qs->queue_tail = 0;
		giio_qs->wait_alloc_counter = 0;
		giio_qs->wait_free_counter = 0;
		giio_qs->wait_ent_phase_counter = 0;
		giio_qs->wait_ent_ack_counter = 0;
		giio_qs->wait_ack_counter = 0;
		giio_qs->wait_complete_counter = 0;
	}
	for (myidx = threadIdx.x; myidx < GIIO_CMDID_ARRSZ; myidx += step) {
		giio_qs->cmdids[myidx] = myidx < GIIOCmdPoolSize ? myidx : empty_cmdid;
	}
}

static
int init_giioqs(nvm_ctrl_t* ctrl, GIIOQueueState **p_giioqs_array)
{
	cudaError_t rc;
	//FIXME: selectDevice(), pass BaM running device id in
	rc = cudaMalloc((void**)p_giioqs_array, sizeof(GIIOQueueState) * GIIOQueueNR);
	if (rc != cudaSuccess) {
		return -1;
	}

	for (unsigned q = 0; q < GIIOQueueNR; ++q) {
		 GIIOQueueState *devp_giioqs = (*p_giioqs_array) + q;
		 GIIOCmdPool *cmdpool = graid_ctrl_get_giio_pool(ctrl, q);
		 GIIOCpltPool *cpltpool = graid_ctrl_get_giio_cplt(ctrl, q);
		 GIIOQueue *queue = graid_ctrl_get_giio_queue(ctrl, q);
		 init_giioqs_kernel<<<1, 1024>>>(devp_giioqs, q, cmdpool, cpltpool, queue);
	}
	cudaDeviceSynchronize();
	return 0;
}


static __device__ __forceinline__
uint32_t get_running_smid() {
	uint32_t smid;
	asm volatile("mov.b32 %0, %smid;" : "=r"(smid) : : );
	return smid;
}

#define __mb_block asm volatile("membar.cta;" : : : "memory")

#define __mb asm volatile("membar.gl;" : : : "memory")

#define nanosleep(nanoseconds) asm volatile("nanosleep.u32 " #nanoseconds ";" : : : )

#define nocache_load32(val, ptr)                   \
	{                                          \
		uint64_t __ptr = (uint64_t)(ptr);  \
		asm volatile("ld.cv.b32 %0, [%1];" \
			     : "=r"(val)           \
			     : "l"(__ptr)          \
			     : "memory");          \
	}

#define noL1cache_store32(ptr, val)                       \
	{                                               \
		uint64_t __ptr = (uint64_t)(ptr);       \
		uint32_t __val = val;                   \
		asm volatile("st.cs.b32 [%0], %1;"      \
			     : : "l"(__ptr), "r"(__val) \
			     : "memory");               \
	}

#define atomic_exch_relaxed(ptr, val, orig)		\
	asm volatile("atom.relaxed.global.exch.b32 %0, [%1], %2;" \
			: "=r"(orig) : "l"(ptr), "r"(val) : "memory")

#define atomic_cas_relaxed(ptr, expect, val, orig)		\
	asm volatile("atom.relaxed.global.cas.b32 %0, [%1], %2, %3;" \
			: "=r"(orig) : "l"(ptr), "r"(expect), "r"(val) : "memory")

#define atomic_next(ptr)		\
	({ \
		uint32_t orig; \
		asm volatile("atom.relaxed.global.add.u32 %0, [%1], 1;" \
			: "=r"(orig) : "l"(ptr) : "memory"); \
		orig; \
	})

#define atomic_add64(ptr, val, orig)		\
	asm volatile("atom.relaxed.global.add.u64 %0, [%1], %2;" \
		: "=l"(orig) : "l"(ptr), "l"(val) : "memory")

#define atomic_read(ptr, val)		\
	asm volatile("atom.relaxed.global.or.u32 %0, [%1], 0;" \
		: "=r"(val) : "l"(ptr) : "memory")

#define GIIOCmd_get_complete(complete, cplt_ptr) \
	nocache_load32(complete, cplt_ptr)

#if 1
#define inc_counter(counter_name, val) \
{ \
	uint64_t orig; \
	uint64_t v = val; \
	atomic_add64(&qs->counter_name, v, orig); \
	if ((orig + v) < orig) { \
		printf("Queue %u " #counter_name "64-bit wrapped around\n", qs->qid); \
	} \
}
#else
#define inc_counter(a, b)
#endif

static __device__ __forceinline__
uint32_t alloc_cmd_id(GIIOQueueState * const qs) {
	const uint32_t head = atomic_next(&qs->cmdid_head);
	const uint32_t idx = GIIO_CMDID_IDX(head);
	const uint32_t phase = GIIO_CMDID_PHASE(head);
	const uint32_t prev_phase = GIIO_CMDID_PREV_PHASE(phase);
	const uint32_t empty_cmdid = phase | EMPTY_CMDID;

	uint32_t * const p_cmdid = (qs->cmdids + idx);
	uint32_t cmdid_ent;
	__mb;
	nocache_load32(cmdid_ent, p_cmdid);
	uint32_t cmdid_phase = GIIO_CMDID_PHASE(cmdid_ent);
	while (GIIO_CMDID_IS_EMPTY(cmdid_ent) || cmdid_phase != phase) [[unlikely]] {
		/* Wait on the cmdid entry available (freed)
		 * If CMDID is empty: the entry have consumed by other allocator but not freed
		 * If CMDID is not empty and the phase mismatch: the entry have not consume by previous phase (highly unlikely)
		*/
		if (cmdid_phase != phase && cmdid_phase != prev_phase) [[unlikely]] {
			printf("ERROR: alloc_cmd_id() failed: Allocator stucked too long\n");
			__trap();
			return 0xFFFFFFFFU;
		}
		inc_counter(wait_alloc_counter, 1LU);
		nanosleep(10000);
		__mb; 
		nocache_load32(cmdid_ent, p_cmdid);
		cmdid_phase = GIIO_CMDID_PHASE(cmdid_ent);
	}
	noL1cache_store32(p_cmdid, empty_cmdid);

	return cmdid_ent;
}

static __device__ __forceinline__
void free_cmd_id(GIIOQueueState * const qs, const uint32_t cmdid, const uint32_t cmdver) {
	const uint32_t tail = atomic_next(&qs->cmdid_tail);
	const uint32_t idx = GIIO_CMDID_IDX(tail);
	const uint32_t phase = GIIO_CMDID_PHASE(tail);
	const uint32_t prev_phase = GIIO_CMDID_PREV_PHASE(phase);
	const uint32_t expect_cmdid = (prev_phase | EMPTY_CMDID);
	const uint32_t cmdid_ent = GIIO_CMDID_SET(cmdver + 1, phase, cmdid);

	uint32_t * const p_cmdid = qs->cmdids + idx;
	uint32_t old_cmdid;
	nocache_load32(old_cmdid, p_cmdid)
	while (old_cmdid != expect_cmdid) [[unlikely]] {
		/* Allocator not completed mark the entry empty
		 * Since we have doubled the size of cmdid index pool,
		 * it should be highly unlikely to happen.
		 */
		const uint32_t old_phase = GIIO_CMDID_PHASE(old_cmdid);
		if (old_phase != prev_phase) [[unlikely]] {
			printf("ERROR: free_cmd_id() failed: Allocator or Freeer stucked for too long\n");
			__trap();
			return;
		}
		nanosleep(1000);
		inc_counter(wait_free_counter, 1LU);
		nocache_load32(old_cmdid, p_cmdid)
	}
	noL1cache_store32(p_cmdid, cmdid_ent);
}

#define DECLARE_GIIO_CMD(cmdname) \
	uint32_t cmdname ## cdw0,   cmdname ## cdw1, \
		 cmdname ## slba_l, cmdname ## slba_h, \
		 cmdname ## prp1_l, cmdname ## prp1_h, \
		 cmdname ## prp2_l, cmdname ## prp2_h

#define giio_store_cmd(cmd_ptr, cmdname) \
	asm volatile("\
	st.v4.u32 [%8],      {%0, %1, %2, %3};\n\
	st.v4.u32 [%8 + 16], {%4, %5, %6, %7};" \
		: : "r"(cmdname ## cdw0),   "r"(cmdname ## cdw1), \
		    "r"(cmdname ## slba_l), "r"(cmdname ## slba_h) \
		    "r"(cmdname ## prp1_l), "r"(cmdname ## prp1_h), \
		    "r"(cmdname ## prp2_l), "r"(cmdname ## prp2_h) \
		    "l"(cmd_ptr) : )

static __device__ __forceinline__
void giio_fill_cmd(GIIOCmd * const cmdp, const uint32_t cmdver,
		   const uint64_t slba, const uint32_t nlb, uint8_t op,
		   const uint64_t prp1, const uint64_t prp2) {
	constexpr unsigned int dgid = 0;
	constexpr unsigned int vdid = 0;

	DECLARE_GIIO_CMD(GIIOC_);
	GIIOC_cdw0 = GIIOCmd_set_cdw0(nlb, vdid, dgid, op);
	GIIOC_cdw1 = GIIOCmd_set_cdw1(cmdver);
	GIIOC_slba_h = slba >> 32;
	GIIOC_slba_l = slba & 0xFFFFFFFFU;
	GIIOC_prp1_h = prp1 >> 32;
	GIIOC_prp1_l = prp1 & 0xFFFFFFFFU;
	GIIOC_prp2_h = prp2 >> 32;
	GIIOC_prp2_l = prp2 & 0xFFFFFFFFU;
	giio_store_cmd(cmdp, GIIOC_);
	__mb;
}

static __device__ __forceinline__
void giio_submit_cmd(GIIOQueueState * const qs, const uint32_t cmdid,
		     GIIOQueueEnt **p_entp, GIIOQueueEnt *p_ent) {
	const uint32_t tail = atomic_next(&qs->queue_tail);
	const uint32_t idx = tail & GIIOQMask;
	const uint32_t phase = GIIOQPhase(tail);
	const uint32_t prev_phase = GIIOQPrevPhase(phase);
	const uint32_t qent = phase | cmdid;
	GIIOQueueEnt *const entp = (*qs->queue_devp) + idx;

	uint32_t old_ent;
	__mb;
	nocache_load32(old_ent, entp);
	while (GIIOQPhase(old_ent) != prev_phase) [[unlikely]] {
		/* Wait for the previous phase to be consumed, this is highly unlikely */
		const uint32_t prev_prev_phase = GIIOQPrevPhase(prev_phase);
		if (GIIOQPhase(old_ent) != prev_prev_phase) [[unlikely]] {
			/* Should not happen, it means the sumbitter picked
			 * the entry but stucked for a long time without
			 * submitting. */
			printf("GIIO Queue submit error: Submit stuck\n");
			__trap();
			return;
		}
		nanosleep(1000);
		inc_counter(wait_ent_phase_counter, 1LU);
		__mb;
		nocache_load32(old_ent, entp);
	}
	while (GIIOQAck(old_ent) == 0) [[unlikely]] {
		/* Wait for the previous entry consumed by GRAID, this should rarely happen */
		if (GIIOQPhase(old_ent) != prev_phase) [[unlikely]] {
			// This should never happen
			printf("GIIO Queue submit error: Entry race\n");
			__trap();
			return;
		}
		nanosleep(1000);
		inc_counter(wait_ent_ack_counter, 1LU);
		__mb;
		nocache_load32(old_ent, entp);
	}

	*p_entp = entp;
	*p_ent = qent;

	// Submit the entry
	noL1cache_store32(entp, qent);
	__mb; //FIXME: Implement cross device flush memory function
}

static __device__ __forceinline__
uint32_t giio_poll_cmd_complete(GIIOQueueState * const qs, const uint32_t cmdid,
				GIIOQueueEnt * const entp, const GIIOQueueEnt ent,
				const uint32_t cmdver) {
	const GIIOCpltEnt * const cpltp = (*qs->cplt_devp) + cmdid;
	const GIIOCpltEnt acked_ent = ent | GIIOQAckFlag;
	GIIOCpltEnt cplt_ent;
	uint64_t cnt1 = 0, cnt2 = 0;

	__mb;
	GIIOCmd_get_complete(cplt_ent, cpltp);
	while (GIIOCplt_ver(cplt_ent) != cmdver || (cplt_ent & GIIOSF_DEQUEUED) == 0)
	{
		nanosleep(1000);
		++cnt1;
		__mb;
		GIIOCmd_get_complete(cplt_ent, cpltp);
	}
	if (cnt1 != 0) {
		inc_counter(wait_ack_counter, cnt1);
	}
	noL1cache_store32(entp, acked_ent);

	while ((cplt_ent & GIIOSF_DONE) == 0)
	{
		nanosleep(1000);
		++cnt2;
		__mb;
		GIIOCmd_get_complete(cplt_ent, cpltp);
	}
	if (cnt2) {
		inc_counter(wait_complete_counter, cnt2);
	}

	return GIIOCplt_flags(cplt_ent);
}

__device__ __forceinline__ static
GIIOQueueState *giioq_pick_queue(GIIOQueueState *giio_qs)
{
	uint32_t qid = get_running_smid();
	/* FIXME: Get actual GRAID active GIIO number */
	constexpr uint32_t active_giioq_nr = 114;
	while (qid >= active_giioq_nr) [[unlikely]] {
		qid -= active_giioq_nr;
	}

	return giio_qs + qid;
}

__device__ int giioq_access_data(GIIOQueueState *giio_qs,
				 const uint64_t slba, const uint32_t nlb, uint8_t op,
				 const uint64_t prp1, const uint64_t prp2)
{
	/* Pick queue based on SM-ID */
	GIIOQueueState * const qs = giioq_pick_queue(giio_qs);

	/* Allocate command entry */
	const uint32_t cmdid_ent = alloc_cmd_id(qs);
	const uint32_t cmdid = GIIO_CMDID_CID(cmdid_ent);
	const uint32_t cmdver = GIIO_CMDID_VER(cmdid_ent);
	if (cmdid_ent == 0xFFFFFFFFu) [[unlikely]] {
		/* Should never happen */
		printf("Failed to allocate command id\n");
		return -EIO;
	}

	/* Fill command */
	giio_fill_cmd((*qs->cmdpool_devp) + cmdid, cmdver, slba, nlb, op, prp1, prp2);

	/* Submit command */
	GIIOQueueEnt *entp;
	GIIOQueueEnt ent;
	giio_submit_cmd(qs, cmdid, &entp, &ent);

	/* Poll completion */
	const uint32_t status = giio_poll_cmd_complete(qs, cmdid, entp, ent, cmdver);

	/* Free command entry */
	free_cmd_id(qs, cmdid, cmdver);

	return (status & GIIOSF_ERROR) == 0 ? 0 : -EIO;
}

__global__
void giioq_print_counters(GIIOQueueState * const qss)
{
    /* FIXME: Get actual GRAID active GIIO number */
    constexpr uint32_t active_giioq_nr = GIIOQueueNR;
    for (unsigned q = 0; q < active_giioq_nr; ++q) {
        GIIOQueueState * const qs = qss + q;
        printf("Q%u: ca:%llu, cf:%llu, ep:%llu, ea:%llu, a:%llu, c:%llu\n",
                qs->qid, qs->wait_alloc_counter, qs->wait_free_counter,
                qs->wait_ent_phase_counter, qs->wait_ent_ack_counter,
                qs->wait_ack_counter, qs->wait_complete_counter);
    }
}
