#include "sdfg/passes/schedule/allocation_inference.h"

namespace sdfg {
namespace passes {

AllocationInference::AllocationInference()
    : Pass() {

      };

std::string AllocationInference::name() { return "AllocationInference"; };

bool AllocationInference::run_pass(Schedule& schedule) {
    bool applied = false;

    auto& analysis_manager = schedule.analysis_manager();
    auto& sdfg = schedule.sdfg();

    auto& users = analysis_manager.get<analysis::Users>();
    for (auto& container : sdfg.containers()) {
        if (!sdfg.is_transient(container)) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Pointer*>(&type)) {
            schedule.allocation_type(container, AllocationType::ALLOCATE);
            applied = true;
            continue;
        }

        bool is_moved_before_access = true;
        auto sources = users.sources(container);
        for (auto& source : sources) {
            if (source->use() != analysis::Use::MOVE) {
                is_moved_before_access = false;
                break;
            }
        }

        if (!is_moved_before_access) {
            schedule.allocation_type(container, AllocationType::ALLOCATE);
            applied = true;
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
