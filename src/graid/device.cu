#ifndef __GRAID__
#error "Must compile with GRAID support"
#endif

#include <atomic>
#include <mutex>
#include "dprintf.h"
#include "lib_ctrl.h"
#include "nvm_ctrl.h"
#include "regs.h"
#include "ctrl.h"
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

struct graid_ctrl_priv {
	int dg_id;
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
		    uint16_t *n_sqs, uint16_t *n_cqs, const int dg_id, const int vd_id)
{
	int err;
	struct device* dev;
	void* mm_ptr = NULL;
	void* mm_devp = NULL;
	struct graid_ctrl_priv *priv = NULL;

	*ctrl = NULL;

	dev = get_graid_dev();
	if (dev == NULL) {
		return EIO;
	}

	if (dev->map_bar0(dg_id, &mm_ptr, &mm_devp) != 0) {
		printf("Failed to map Graid NVMe BAR0 memory: DG%d\n", dg_id);
		goto err_put_dev_out;
	}

	priv = (struct graid_ctrl_priv*)malloc(sizeof(*priv));
	if (priv == NULL) {
		printf("Failed to allocate graid controller private data\n");
		goto err_put_dev_out;
	}
	priv->dg_id = dg_id;

	err = _nvm_ctrl_init(ctrl, dev, &graid_device_ops, DEVICE_TYPE_GRAID, mm_ptr, NVM_CTRL_MEM_MINSIZE, mm_devp, priv);
	if (err != 0) {
		printf("Failed to initialize nvm_ctrl_t: %s\n", strerror(errno));
		goto err_put_dev_out;
	}

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
	const struct graid_ctrl_priv* priv = (struct graid_ctrl_priv*)ctrl->priv;
	const int dg_id = priv->dg_id;
	const int idx = (dg_id * NVMeMaxDGQNR) + qp_id - 1;

	if (qt == QT_SQ && at == AT_DEVP)
		return ((uint64_t)dev->devp_bar1) + dev->virt_aoffs.dg_sqs[idx];

	if (qt == QT_SQ && at == AT_PHYA)
		return dev->phys_addrs.dg_sqs[idx];

	const int socket_idx = 0;
	if (qt == QT_CQ && at == AT_DEVP)
		return ((uint64_t)dev->devp_bar1) + dev->virt_aoffs.dg_cqs[socket_idx][idx];

	if (qt == QT_CQ && at == AT_PHYA)
		return dev->phys_addrs.dg_cqs[socket_idx][idx];

	return 0llu;
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
