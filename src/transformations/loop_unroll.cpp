#include "sdfg/transformations/loop_unroll.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"

namespace sdfg {
namespace transformations {

LoopUnroll::LoopUnroll(structured_control_flow::Sequence& parent,
                       structured_control_flow::For& loop)
    : parent_(parent), loop_(loop) {

      };

std::string LoopUnroll::name() { return "LoopUnroll"; };

bool LoopUnroll::can_be_applied(Schedule& schedule) {
    // Criterion: Check if the loop iteration count is known and an Integer
    auto iteration_count = get_iteration_count(this->loop_);
    if (iteration_count == SymEngine::null) {
        return false;
    }

    // Criterion: Check if the loop has a known init and update expression
    auto& update = loop_.update();
    if (!symbolic::eq(update, symbolic::add(loop_.indvar(), symbolic::integer(1)))) {
        return false;
    }

    // Criterion: Check if the loop indvar is not used as an access_node in the loop body
    auto& analysis_manager = schedule.analysis_manager();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& body = loop_.root();
    analysis::UsersView body_users(users, body);
    for (auto user : body_users.uses(loop_.indvar()->get_name())) {
        if (auto AccessNode = dynamic_cast<data_flow::AccessNode*>(user->element())) {
            return false;
        }
    }

    return true;
};

void LoopUnroll::apply(Schedule& schedule) {
    auto& builder = schedule.builder();
    auto& sdfg = builder.subject();
    auto& analysis_manager = schedule.analysis_manager();
    auto iteration_count = get_iteration_count(this->loop_);

    auto& init = loop_.init();
    auto update = symbolic::integer(1);

    for (int i = 0; i < iteration_count->as_int(); i++) {
        auto& branch = builder.add_if_else(parent_);
        auto pseudo_iterator = symbolic::add(init, symbolic::mul(symbolic::integer(i), update));
        auto branch_cond =
            symbolic::subs(this->loop_.condition(), this->loop_.indvar(), pseudo_iterator);
        auto& branch_case = builder.add_case(branch, branch_cond);

        deepcopy::StructuredSDFGDeepCopy copier(builder, branch_case, loop_.root());
        copier.copy();
        branch_case.replace(this->loop_.indvar(), pseudo_iterator);
    }

    builder.remove_child(parent_, loop_);

    analysis_manager.invalidate_all();
};

}  // namespace transformations
}  // namespace sdfg