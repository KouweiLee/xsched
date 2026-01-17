#pragma once

#include <map>
#include <chrono>

#include "xsched/types.h"
#include "xsched/protocol/def.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class UtilizationPartitionPolicy : public Policy
{
public:
    UtilizationPartitionPolicy() : Policy(kPolicyUtilizationPartition) {}
    virtual ~UtilizationPartitionPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    std::chrono::microseconds GetBudget(Utilization util);
    void SwitchToAny(const Status &status);
    void SwitchTo(XQueueHandle handle, Utilization util, const Status &status);

    XQueueHandle cur_running_ = 0;
    std::chrono::system_clock::time_point cur_end_;
    /// @brief The utilization(the proportion of total time slice) of each XQueue.
    std::map<XQueueHandle, Utilization> utils_;
    /// @brief The total time slice of the system. (us)
    std::chrono::microseconds timeslice_ = 
        std::chrono::microseconds(TIMESLICE_DEFAULT);
};

} // namespace xsched::sched
