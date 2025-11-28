#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/level2/cuda_queue.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

CudaQueueLv2::CudaQueueLv2(CUstream stream): CudaQueueLv1(stream)
{
    instrument_manager_ = std::make_unique<InstrumentManager>(context_, cudevice_);
}

void CudaQueueLv2::Launch(std::shared_ptr<HwCommand> hw_cmd)
{
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) return instrument_manager_->Launch(kernel, kStream, level_);
    
    auto cuda_cmd = std::dynamic_pointer_cast<CudaCommand>(hw_cmd);
    XASSERT(cuda_cmd != nullptr, "hw_cmd is not a CudaCommand");
    CUDA_ASSERT(cuda_cmd->LaunchWrapper(kStream));
}

void CudaQueueLv2::Deactivate()
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Deactivate() not supported on level-%d", level_);
    instrument_manager_->Deactivate();
}

void CudaQueueLv2::Reactivate(const preempt::CommandLog &log)
{
    XASSERT(level_ >= kPreemptLevelDeactivate, "Reactivate() not supported on level-%d", level_);
    this->Synchronize();

    uint64_t resume_cmd_idx = instrument_manager_->Reactivate();
    if (resume_cmd_idx == 0) return;

    for (auto cmd : log) {
        if (cmd->GetIdx() < (int64_t)resume_cmd_idx) continue;
        this->Launch(cmd);
    }
}

void CudaQueueLv2::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level <= kPreemptLevelDeactivate, "unsupported level: %d", level);
    level_ = level;
}

void CudaQueueLv2::OnHwCommandSubmit(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    if (level_ < kPreemptLevelDeactivate) return;
    auto kernel = std::dynamic_pointer_cast<CudaKernelCommand>(hw_cmd);
    if (kernel != nullptr) instrument_manager_->Instrument(kernel);
}
// 这里之所以不直接launch，是因为v2和v3的launch，其实是作者通过逆向工程发现的GPU的特性
// 本身CUDA官方文档是没有给出这些特性的，所以不能直接使用CUDA的API来launch
CUresult CudaQueueLv2::DirectLaunch(std::shared_ptr<CudaKernelCommand> kernel,
                                    CUcontext ctx, CUstream stream)
{
    auto instrument_ctx = InstrumentContext::Instance(ctx);
    return instrument_ctx->Launch(kernel, stream, kKernelLaunchOriginal);
}
