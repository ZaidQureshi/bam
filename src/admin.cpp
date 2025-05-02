#include <nvm_types.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_rpc.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_ctrl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "rpc.h"
#include "regs.h"
#include "lib_util.h"
#include "dprintf.h"



static void admin_cq_create(nvm_cmd_t* cmd, const nvm_queue_t* cq, uint64_t ioaddr, bool need_prp = false)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_CREATE_CQ, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (((uint32_t) cq->qs - 1) << 16) | cq->no;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | (!need_prp);
}



static void admin_cq_delete(nvm_cmd_t* cmd, const nvm_queue_t* cq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_DELETE_CQ, 0);
    cmd->dword[10] = cq->no & 0xffff;
}



static void admin_sq_create(nvm_cmd_t* cmd, const nvm_queue_t* sq, const nvm_queue_t* cq, uint64_t ioaddr, bool need_prp = false)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_CREATE_SQ, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (((uint32_t) sq->qs - 1) << 16) | sq->no;
    cmd->dword[11] = (((uint32_t) cq->no) << 16) | (0x00 << 1) | (!need_prp);
}


static void admin_sq_delete(nvm_cmd_t* cmd, const nvm_queue_t* sq)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_DELETE_SQ, 0);
    cmd->dword[10] = sq->no & 0xffff;
}



static void admin_current_num_queues(nvm_cmd_t* cmd, bool set, uint16_t n_cqs, uint16_t n_sqs)
{
    nvm_cmd_header(cmd, 0, set ? NVM_ADMIN_SET_FEATURES : NVM_ADMIN_GET_FEATURES, 0);
    nvm_cmd_data_ptr(cmd, 0, 0);

    cmd->dword[10] = (0x00 << 8) | 0x07;
    cmd->dword[11] = set ? ((n_cqs - 1) << 16) | (n_sqs - 1) : 0;
}



static void admin_identify_ctrl(nvm_cmd_t* cmd, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_IDENTIFY, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x01;
    cmd->dword[11] = 0;
}



static void admin_identify_ns(nvm_cmd_t* cmd, uint32_t ns_id, uint64_t ioaddr)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_IDENTIFY, ns_id);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (0 << 16) | 0x00;
    cmd->dword[11] = 0;
}



static void admin_get_log_page(nvm_cmd_t* cmd, uint32_t ns_id, uint64_t ioaddr, uint8_t log_id, uint64_t log_offset)
{
    nvm_cmd_header(cmd, 0, NVM_ADMIN_GET_LOG_PAGE, ns_id);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = (1024 << 16) | log_id;
    cmd->dword[11] = 0;
    cmd->dword[12] = (uint32_t)log_offset;
    cmd->dword[13] = (uint32_t)(log_offset >> 32);
}



int nvm_admin_ctrl_info(nvm_aq_ref ref, struct nvm_ctrl_info* info, void* ptr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (info == NULL || ptr == NULL || ioaddr == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(info, 0, sizeof(struct nvm_ctrl_info));
    memset(ptr, 0, 0x1000);

    nvm_cache_invalidate(ptr, 0x1000);

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    info->nvme_version = (uint32_t) *VER(ctrl->mm_ptr);
    info->page_size = ctrl->page_size;
    info->db_stride = 1UL << ctrl->dstrd;
    info->timeout = CAP$TO(ctrl->mm_ptr) * 500UL;
    info->contiguous = !!CAP$CQR(ctrl->mm_ptr);
    info->max_entries = CAP$MQES(ctrl->mm_ptr) + 1;

    admin_identify_ctrl(&command, ioaddr);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Identify controller failed: %s\n", nvm_strerror(err));
        return err;
    }

    nvm_cache_invalidate(ptr, 0x1000);

    const unsigned char* bytes = (const unsigned char*) ptr;
    memcpy(info->pci_vendor, bytes, 4);
    memcpy(info->serial_no, bytes + 4, 20);
    memcpy(info->model_no, bytes + 24, 40);
    memcpy(info->firmware, bytes + 64, 8);

    info->max_data_size = (1UL << bytes[77]) * (1UL << (12 + CAP$MPSMIN(ctrl->mm_ptr)));
    info->max_data_pages = info->max_data_size / info->page_size;
    info->sq_entry_size = 1 << _RB(bytes[512], 3, 0);
    info->cq_entry_size = 1 << _RB(bytes[513], 3, 0);
    info->max_out_cmds = *((uint16_t*) (bytes + 514));
    info->max_n_ns = *((uint32_t*) (bytes + 516));

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_ns_info(nvm_aq_ref ref, struct nvm_ns_info* info, uint32_t ns_id, void* ptr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (info == NULL || ptr == NULL || ioaddr == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(ptr, 0, 0x1000);
    memset(info, 0, sizeof(struct nvm_ns_info));

    nvm_cache_invalidate(ptr, 0x1000);

    info->ns_id = ns_id;

    admin_identify_ns(&command, ns_id, ioaddr);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Identify namespace failed: %s\n", nvm_strerror(err));
        return err;
    }
    
    nvm_cache_invalidate(ptr, 0x1000);

    const unsigned char* bytes = (const unsigned char*) ptr;
    info->size = *((uint64_t*) ptr);
    info->capacity = *((uint64_t*) ((uint64_t)ptr + 8));
    info->utilization = *((uint64_t*) ((uint64_t)ptr + 16));

    uint8_t format_idx = _RB(bytes[26], 3, 0);

    uint32_t lba_format = *((uint32_t*) (bytes + 128 + sizeof(uint32_t) * format_idx));
    info->lba_data_size = 1 << _RB(lba_format, 23, 16);
    info->metadata_size = _RB(lba_format, 15, 0);

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_get_log_page(nvm_aq_ref ref, uint32_t ns_id, void* ptr, uint64_t ioaddr, uint8_t log_id, uint64_t log_offset)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    memset(ptr, 0, 0x1000);

    nvm_cache_invalidate(ptr, 0x1000);

    admin_get_log_page(&command, ns_id, ioaddr, log_id, log_offset);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Get log page failed: %s\n", nvm_strerror(err));
        return err;
    }
    
    nvm_cache_invalidate(ptr, 0x1000);

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_cq_create(nvm_aq_ref ref, nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp)
{
    int err;
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t queue;
    size_t n_pages = 0;

    // Queue number 0 is reserved for admin queues
    if (id == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    // Get controller reference
    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);
    if (ctrl == NULL)
    {
        return NVM_ERR_PACK(NULL, ENOTTY);
    }

    // Check if a valid queue size was supplied
    if (qs == 0)
    {
        qs = _MIN(dma->page_size / sizeof(nvm_cpl_t), ctrl->max_qs);
    }

    if (qs < 2 || qs > 0x10000 || qs > ctrl->max_qs)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    n_pages = NVM_CQ_PAGES(ctrl, qs);

    // We currently only support contiguous memory
    //if (n_pages > 1 && !dma->contiguous) {
    //    dprintf("Non-contiguous queues are not supported\n");
    //    return NVM_ERR_PACK(NULL, ENOTSUP);
    //}

    // Do some sanity checking
    if (dma->vaddr == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }
    else if (n_pages == 0
            || n_pages > dma->n_ioaddrs
            || offset > dma->n_ioaddrs
            || offset + n_pages > dma->n_ioaddrs)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = nvm_queue_clear(&queue, ctrl, QT_CQ, id, qs,
            dma->local, NVM_DMA_OFFSET(dma, offset), dma->ioaddrs[offset]);
    if (err != 0)
    {
        return NVM_ERR_PACK(NULL, err);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));

    admin_cq_create(&command, &queue, dma->ioaddrs[offset], need_prp);

    err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Creating completion queue failed: %s\n", nvm_strerror(err));
        return err;
    }
    memcpy((void*) cq, (void*) &queue, sizeof(nvm_queue_t));
    //*cq = queue;
    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_cq_delete(nvm_aq_ref ref, nvm_queue_t* cq)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (cq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    admin_cq_delete(&command, cq);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Deleting completion queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    cq->db = NULL;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_sq_create(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq, uint16_t id, const nvm_dma_t* dma, size_t offset, size_t qs, bool need_prp)
{
    int err;
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t queue;
    size_t n_pages = 0;

    // Queue number 0 is reserved for admin queues
    if (id == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    // Get controller reference
    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);
    if (ctrl == NULL)
    {
        return NVM_ERR_PACK(NULL, ENOTTY);
    }

    // Check if a valid queue size was supplied
    if (qs == 0)
    {
        qs = _MIN(dma->page_size / sizeof(nvm_cmd_t), ctrl->max_qs);
    }

    if (qs < 2 || qs > 0x10000 || qs > ctrl->max_qs)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    n_pages = NVM_SQ_PAGES(ctrl, qs);


    // We currently only support contiguous memory
    //if (n_pages > 1 && !dma->contiguous) {
    //    dprintf("Non-contiguous queues are not supported\n");
    //    return NVM_ERR_PACK(NULL, ENOTSUP);
    //}


    // Do some sanity checking
    if (dma->vaddr == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }
    else if (n_pages == 0
            || n_pages > dma->n_ioaddrs
            || offset > dma->n_ioaddrs
            || offset + n_pages > dma->n_ioaddrs)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = nvm_queue_clear(&queue, ctrl, QT_SQ, id, qs,
            dma->local, NVM_DMA_OFFSET(dma, offset), dma->ioaddrs[offset]);
    if (err != 0)
    {
        return NVM_ERR_PACK(NULL, err);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    admin_sq_create(&command, &queue, cq, dma->ioaddrs[offset], need_prp);

    err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Creating submission queue failed: %s\n", nvm_strerror(err));
        return err;
    }
    memcpy((void*) sq, (void*) &queue, sizeof(nvm_queue_t));
    //*sq = queue;
    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_sq_delete(nvm_aq_ref ref, nvm_queue_t* sq, const nvm_queue_t* cq)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (cq == NULL || cq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    if (sq->db == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));
    admin_sq_delete(&command, sq);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Deleting submission queue failed: %s\n", nvm_strerror(err));
        return err;
    }

    sq->db = NULL;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_get_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));

    admin_current_num_queues(&command, false, 0, 0);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (!nvm_ok(err))
    {
        dprintf("Failed to get current number of queues: %s\n", nvm_strerror(err));
        return err;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;

    return NVM_ERR_PACK(NULL, 0);
}



int nvm_admin_set_num_queues(nvm_aq_ref ref, uint16_t n_cqs, uint16_t n_sqs)
{
    return nvm_admin_request_num_queues(ref, &n_cqs, &n_sqs);
}



int nvm_admin_request_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (*n_cqs == 0 || *n_sqs == 0)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    memset(&command, 0, sizeof(command));
    memset(&completion, 0, sizeof(completion));

    admin_current_num_queues(&command, true, *n_cqs, *n_sqs);

    int err = nvm_raw_rpc(ref, &command, &completion);
    if (err != 0)
    {
        dprintf("Failed to set current number of queues: %s\n", nvm_strerror(err));
        return err;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;

    return NVM_ERR_PACK(NULL, 0);
}

