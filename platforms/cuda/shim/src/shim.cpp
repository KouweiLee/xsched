#include <cstdint>
#include <list>
#include <atomic>

#include "xsched/utils/xassert.h"
#include "xsched/xqueue.h"
#include "xsched/utils/map.h"
#include "xsched/protocol/def.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/xqueue/xqueue.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/shim/shim.h"
#include "xsched/cuda/hal/common/levels.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::preempt;

namespace xsched::cuda
{

static utils::ObjectMap<CUevent, std::shared_ptr<CudaEventRecordCommand>> g_events;
std::mutex g_shim_sync_mtx;

std::shared_ptr<XQueue> GetXQueueForStream(CUstream stream)
{
    auto hwq_h = GetHwQueueHandle(stream);
    auto xq = HwQueueManager::GetXQueue(hwq_h);
    if (xq == nullptr) {
        XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, stream); });
        xq = HwQueueManager::GetXQueue(hwq_h);
    }
    return xq;
}

void ShimSyncStream(CUstream stream, std::shared_ptr<XQueue> xq)
{
    std::lock_guard<std::mutex> lock(g_shim_sync_mtx);
    if (stream == nullptr) {
        WaitBlockingXQueues();
    } else if (xq != nullptr) {
        auto hwq = xq->GetHwQueue();
        auto cuda_q = std::dynamic_pointer_cast<CudaQueueLv1>(hwq);
        if (cuda_q && !(cuda_q->GetStreamFlags() & CU_STREAM_NON_BLOCKING)) {
            auto default_xq = HwQueueManager::GetXQueue(GetHwQueueHandle(nullptr));
            if (default_xq) default_xq->WaitAll();
        }
    } else {
        XASSERT(false, "fail to sync stream %p", stream);
    }
}

// This function waits for all blocking CUDA streams managed by XSched to complete their queued commands.
// It iterates over all XQueues, identifies those associated with blocking CUDA streams (i.e., streams that are not non-blocking),
// submits a wait-all command to each, and then waits for all these commands to finish.
// Non-blocking streams are skipped as they do not require synchronization here.
void WaitBlockingXQueues()
{
    std::list<std::shared_ptr<XQueueWaitAllCommand>> wait_cmds;
    XResult res = XQueueManager::ForEach([&](std::shared_ptr<XQueue> xq)->XResult {
        auto hwq = xq->GetHwQueue();
        // Skip default stream to avoid self-serialization on host side during launch.
        if (hwq->GetHandle() == GetHwQueueHandle(nullptr)) return kXSchedSuccess;

        auto cuda_q = std::dynamic_pointer_cast<CudaQueueLv1>(hwq);
        if (cuda_q == nullptr) return kXSchedErrorUnknown;
        // Skip non-blocking streams, as they do not require waiting
        if (cuda_q->GetStreamFlags() & CU_STREAM_NON_BLOCKING) return kXSchedSuccess;
        auto wait_cmd = xq->SubmitWaitAll();
        if (wait_cmd == nullptr) return kXSchedErrorUnknown;
        wait_cmds.push_back(wait_cmd);
        return kXSchedSuccess;
    });
    XASSERT(res == kXSchedSuccess, "Fail to submit wait all commands");
    for (auto &cmd : wait_cmds) cmd->Wait();
}

CUresult XLaunchKernel(CUfunction f,
                       unsigned int gdx, unsigned int gdy, unsigned int gdz,
                       unsigned int bdx, unsigned int bdy, unsigned int bdz,
                       unsigned int shmem, CUstream stream, void **params, void **extra)
{
    XDEBG("XLaunchKernel(func: %p, stream: %p, grid: [%u, %u, %u], block: [%u, %u, %u], "
          "shm: %u, params: %p, extra: %p)", f, stream, gdx, gdy, gdz, bdx, bdy, bdz,
          shmem, params, extra);

    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);

    auto kernel = std::make_shared<CudaKernelLaunchCommand>(
        f, gdx, gdy, gdz, bdx, bdy, bdz, shmem, params, extra, true);

    xq->Submit(kernel);
    return CUDA_SUCCESS;
}

CUresult XLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **params, void **extra)
{
    XDEBG("XLaunchKernelEx(cfg: %p, func: %p, params: %p, extra: %p)", config, f, params, extra);
    if (config == nullptr) return Driver::LaunchKernelEx(config, f, params, extra);

    CUstream stream = config->hStream;
    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);
    
    auto kn = std::make_shared<CudaKernelLaunchExCommand>(config, f, params, extra, true);

    xq->Submit(kn);
    return CUDA_SUCCESS;
}

CUresult XLaunchHostFunc(CUstream stream, CUhostFn fn, void *data)
{
    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);

    auto hw_cmd = std::make_shared<CudaHostFuncCommand>(fn, data);
    xq->Submit(hw_cmd);
    return CUDA_SUCCESS;
}

CUresult XEventQuery(CUevent event)
{
    XDEBG("XEventQuery(event: %p)", event);
    if (event == nullptr) return Driver::EventQuery(event);
    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventQuery(event);

    auto state = xevent->GetState();
    if (state >= kCommandStateCompleted) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult XEventRecord(CUevent event, CUstream stream)
{
    XDEBG("XEventRecord(event: %p, stream: %p)", event, stream);
    if (event == nullptr) return Driver::EventRecord(event, stream);

    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);

    auto xevent = std::make_shared<CudaEventRecordCommand>(event);
    xq->Submit(xevent);

    g_events.Add(event, xevent);
    return CUDA_SUCCESS;
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream, unsigned int flags)
{
    XDEBG("XEventRecordWithFlags(event: %p, stream: %p, flags: %u)", event, stream, flags);
    if (event == nullptr) return Driver::EventRecordWithFlags(event, stream, flags);

    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);

    auto xevent = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);
    xq->Submit(xevent);

    g_events.Add(event, xevent);
    return CUDA_SUCCESS;
}

CUresult XEventSynchronize(CUevent event)
{
    XDEBG("XEventSynchronize(event: %p)", event);
    if (event == nullptr) return Driver::EventSynchronize(event);

    auto xevent = g_events.Get(event, nullptr);
    if (xevent == nullptr) return Driver::EventSynchronize(event);

    xevent->Wait();
    return CUDA_SUCCESS;
}

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    XDEBG("XStreamWaitEvent(stream: %p, event: %p, flags: %u)", stream, event, flags);
    if (event == nullptr)return Driver::StreamWaitEvent(stream, event, flags);

    auto xevent = g_events.Get(event, nullptr);
    // the event is not recorded yet
    if (xevent == nullptr) return Driver::StreamWaitEvent(stream, event, flags);

    auto xq = GetXQueueForStream(stream);
    ShimSyncStream(stream, xq);
    XASSERT(xq != nullptr, "fail to get XQueue for stream %p", stream);

    auto cmd = std::make_shared<CudaEventWaitCommand>(xevent, flags);
    xq->Submit(cmd);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy(CUevent event)
{
    XDEBG("XEventDestroy(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
        // According to CUDA driver API documentation, if the event is waiting in XQueues,
        // we should not destroy it immediately. Instead, we shall set a flag to destroy
        // the CUevent in the destructor of the xevent.
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy(event);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy_v2(CUevent event)
{
    XDEBG("XEventDestroy_v2(event: %p)", event);
    if (event == nullptr) return Driver::EventDestroy_v2(event);

    auto xevent = g_events.DoThenDel(event, nullptr, [](auto xevent) {
        // Same as XEventDestroy.
        xevent->DestroyEvent();
    });
    if (xevent == nullptr) return Driver::EventDestroy_v2(event);
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize(CUstream stream)
{
    XDEBG("XStreamSynchronize(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) return Driver::StreamSynchronize(stream);
    xq->WaitAll();
    return CUDA_SUCCESS;
}

CUresult XStreamQuery(CUstream stream)
{
    XDEBG("XStreamQuery(stream: %p)", stream);
    auto xq = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xq == nullptr) Driver::StreamQuery(stream);

    switch (xq->Query())
    {
    case kQueueStateIdle:
        return CUDA_SUCCESS;
    case kQueueStateReady:
        return CUDA_ERROR_NOT_READY;
    default:
        return Driver::StreamQuery(stream);
    }
}
CUresult XCtxSynchronize()
{
    XDEBG("XCtxSynchronize()");
    XQueueManager::ForEachWaitAll();
    return Driver::CtxSynchronize();
}

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    CUresult res = Driver::StreamCreate(stream, flags);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreate(stream: %p, flags: 0x%x)", *stream, flags);
    return res;
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags, int priority)
{
    CUresult res = Driver::StreamCreateWithPriority(stream, flags, priority);
    if (res != CUDA_SUCCESS) return res;
    XQueueManager::AutoCreate([&](HwQueueHandle *hwq) { return CudaQueueCreate(hwq, *stream); });
    XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)",
          *stream, flags, priority);
    return res;
}

CUresult XStreamDestroy(CUstream stream)
{
    XDEBG("XStreamDestroy(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy(stream);
}

CUresult XStreamDestroy_v2(CUstream stream)
{
    XDEBG("XStreamDestroy_v2(stream: %p)", stream);
    XQueueManager::AutoDestroy(GetHwQueueHandle(stream));
    return Driver::StreamDestroy_v2(stream);
}

} // namespace xsched::cuda
