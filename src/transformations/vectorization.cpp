#include "sdfg/transformations/vectorization.h"

#include "sdfg/types/utils.h"

namespace sdfg {
namespace transformations {

bool Vectorization::tasklet_supported(data_flow::TaskletCode c) {
    switch (c) {
        case data_flow::TaskletCode::assign:
            return true;
        case data_flow::TaskletCode::neg:
            return true;
        case data_flow::TaskletCode::add:
            return true;
        case data_flow::TaskletCode::sub:
            return true;
        case data_flow::TaskletCode::mul:
            return true;
        case data_flow::TaskletCode::div:
            return true;
        case data_flow::TaskletCode::fma:
            return true;
        case data_flow::TaskletCode::mod:
            return true;
        case data_flow::TaskletCode::abs:
            return true;
        case data_flow::TaskletCode::fabs:
            return true;
        case data_flow::TaskletCode::max:
            return true;
        case data_flow::TaskletCode::min:
            return true;
        case data_flow::TaskletCode::sqrt:
            return true;
        case data_flow::TaskletCode::sqrtf:
            return true;
        case data_flow::TaskletCode::sin:
            return true;
        case data_flow::TaskletCode::cos:
            return true;
        case data_flow::TaskletCode::tan:
            return true;
        case data_flow::TaskletCode::pow:
            return true;
        case data_flow::TaskletCode::exp:
            return true;
        case data_flow::TaskletCode::expf:
            return true;
        case data_flow::TaskletCode::exp2:
            return true;
        case data_flow::TaskletCode::log:
            return true;
        case data_flow::TaskletCode::log2:
            return true;
        case data_flow::TaskletCode::log10:
            return true;
        case data_flow::TaskletCode::floor:
            return true;
        case data_flow::TaskletCode::ceil:
            return true;
        case data_flow::TaskletCode::trunc:
            return true;
        case data_flow::TaskletCode::round:
            return true;
        case data_flow::TaskletCode::olt:
            return true;
        case data_flow::TaskletCode::ole:
            return true;
        case data_flow::TaskletCode::oeq:
            return true;
        case data_flow::TaskletCode::one:
            return true;
        case data_flow::TaskletCode::ogt:
            return true;
        case data_flow::TaskletCode::oge:
            return true;
        case data_flow::TaskletCode::bitwise_and:
            return true;
        case data_flow::TaskletCode::bitwise_or:
            return true;
        case data_flow::TaskletCode::bitwise_xor:
            return true;
        case data_flow::TaskletCode::bitwise_not:
            return true;
    }

    return false;
};

Vectorization::Vectorization(structured_control_flow::Sequence& parent,
                             structured_control_flow::For& loop)
    : loop_(loop) {

      };

std::string Vectorization::name() { return "Vectorization"; };

bool Vectorization::can_be_applied(Schedule& schedule) {
    auto& sdfg = schedule.builder().subject();
    auto& analysis_manager = schedule.analysis_manager();
    auto& body = this->loop_.root();

    // Section: CFG

    // Criterion: Body must only consists of blocks
    std::unordered_set<const data_flow::Tasklet*> tasklets;
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&body};
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        if (auto block = dynamic_cast<const structured_control_flow::Block*>(node)) {
            for (auto& child : block->dataflow().nodes()) {
                if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&child)) {
                    tasklets.insert(tasklet);
                }
            }
        } else if (auto sequence = dynamic_cast<const structured_control_flow::Sequence*>(node)) {
            for (size_t i = 0; i < sequence->size(); i++) {
                if (sequence->at(i).second.assignments().size() > 0) {
                    return false;
                }
                queue.push_back(&sequence->at(i).first);
            }
        } else {
            return false;
        }
    }

    // Section: Operations

    // Criterion: All tasklets must be mappable to vector instructions
    for (auto& tasklet : tasklets) {
        if (!tasklet_supported(tasklet->code())) {
            return false;
        }

        // Criterion: All connectors must have same type
        types::PrimitiveType operation_type = types::PrimitiveType::Void;
        for (auto& inp : tasklet->inputs()) {
            const types::Scalar& type = inp.second;
            if (operation_type == types::PrimitiveType::Void) {
                operation_type = type.primitive_type();
            } else if (operation_type != type.primitive_type()) {
                return false;
            }
        }
    }

    // Section: Memory accesses

    // Criterion: Loop is contiguous
    auto indvar = this->loop_.indvar();
    if (!analysis::DataParallelismAnalysis::is_contiguous(this->loop_)) {
        return false;
    }

    // Criterion: No pointer operations
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users(all_users, body);
    if (!users.views().empty() || !users.moves().empty()) {
        return false;
    }

    // Determine moving symbols, i.e., pseudo loop iterators
    std::unordered_set<std::string> moving_symbols;
    auto write_subset = users.write_subsets();
    for (auto& entry : write_subset) {
        auto& data = entry.first;
        auto& type = sdfg.type(data);
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }
        if (!types::is_integer(type.primitive_type())) {
            continue;
        }
        moving_symbols.insert(data);
    }

    size_t bit_width = 0;

    // Criterion: All memory accesses must be vectorizable
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(this->loop_);
    if (dependencies.size() == 0) {
        return false;
    }

    auto read_subset = users.read_subsets();
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        auto& type = sdfg.type(container);
        auto& dep_type = dep.second;

        // Criterion: Non-parallel accesses must be valid scatter operations
        if (dep_type == analysis::Parallelism::DEPENDENT) {
            return false;
        }

        // Criterion: All accesses are vectorizable and have matching bit width
        for (auto& subset : read_subset[container]) {
            auto access_type = classify_access(subset, indvar, moving_symbols);
            if (access_type == AccessType::UNKNOWN) {
                return false;
            }

            // Symbolic expressions are casted to the right bit width
            if (type.is_symbol() && types::is_integer(type.primitive_type())) {
                continue;
            }

            auto& inferred_type = types::infer_type(sdfg, type, subset);
            auto primitive_type = inferred_type.primitive_type();
            if (primitive_type != types::PrimitiveType::Bool) {
                auto inferred_bit_width = types::bit_width(primitive_type);
                if (bit_width == 0) {
                    bit_width = inferred_bit_width;
                } else if (bit_width != inferred_bit_width) {
                    return false;
                }
            }
        }
        for (auto& subset : write_subset[container]) {
            auto access_type = classify_access(subset, indvar, moving_symbols);
            if (access_type == AccessType::UNKNOWN) {
                return false;
            }

            auto& inferred_type = types::infer_type(sdfg, type, subset);
            auto primitive_type = inferred_type.primitive_type();
            if (primitive_type != types::PrimitiveType::Bool) {
                auto inferred_bit_width = types::bit_width(primitive_type);
                if (bit_width == 0) {
                    bit_width = inferred_bit_width;
                } else if (bit_width != inferred_bit_width) {
                    return false;
                }
            }
        }

        // Criterion: No inter-loop dependencies

        // Readonly, private and parallel inter-loop dependencies are trivially safe
        if (dep_type == analysis::Parallelism::READONLY ||
            dep_type == analysis::Parallelism::PRIVATE ||
            dep_type == analysis::Parallelism::PARALLEL) {
            continue;
        }

        // Criterion: Reductions must be supported operations
        if (dep_type == analysis::Parallelism::REDUCTION) {
            // Find reduction template
            auto writes = users.writes(container);
            if (writes.size() != 1) {
                return false;
            }
            auto element = writes[0]->element();
            if (!dynamic_cast<const data_flow::AccessNode*>(element)) {
                return false;
            }
            auto access_node = static_cast<const data_flow::AccessNode*>(element);
            auto& graph = access_node->get_parent();
            auto& edge = *graph.in_edges(*access_node).begin();
            auto& tasklet = static_cast<const data_flow::Tasklet&>(edge.src());
            if (tasklet.is_conditional()) {
                return false;
            }

            auto reads = users.reads(container);
            if (reads.size() == 0) {
                continue;
            }
            if (reads.size() > 1) {
                return false;
            }

            if (tasklet.code() != data_flow::TaskletCode::add &&
                tasklet.code() != data_flow::TaskletCode::min &&
                tasklet.code() != data_flow::TaskletCode::max &&
                tasklet.code() != data_flow::TaskletCode::fma &&
                tasklet.code() != data_flow::TaskletCode::assign) {
                return false;
            }
            if (tasklet.code() == data_flow::TaskletCode::fma) {
                // Must be the additive part
                for (auto& iedge : graph.in_edges(tasklet)) {
                    if (iedge.dst_conn() == tasklet.input(2).first) {
                        auto& read_node = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
                        if (read_node.data() != container) {
                            return false;
                        }
                    }
                }
            } else {
                bool found_input = false;
                for (auto& iedge : graph.in_edges(tasklet)) {
                    auto& read_node = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
                    if (read_node.data() == container) {
                        found_input = true;
                        break;
                    }
                }
                if (!found_input) {
                    return false;
                }
            }
        }
    }
    if (bit_width != 8 && bit_width != 16 && bit_width != 32 && bit_width != 64) {
        return false;
    }

    return true;
};

void Vectorization::apply(Schedule& schedule) {
    schedule.loop_schedule(&this->loop_, LoopSchedule::VECTORIZATION);
};

AccessType Vectorization::classify_access(const data_flow::Subset& subset,
                                          const symbolic::Symbol& indvar,
                                          const std::unordered_set<std::string>& moving_symbols) {
    // Leading dimensions must be constant
    for (size_t i = 0; i < subset.size() - 1; i++) {
        auto& dim = subset[i];
        if (symbolic::uses(dim, indvar->get_name())) {
            return AccessType::UNKNOWN;
        }
        for (auto& scalar : moving_symbols) {
            if (symbolic::uses(dim, scalar)) {
                return AccessType::UNKNOWN;
            }
        }
    }

    // Three supported cases: Contiguous, Indirection, Constant
    auto& last_dim = subset.back();

    // Case: INDIRECTION
    symbolic::Symbol gather_variable = SymEngine::null;
    for (auto& scalar : moving_symbols) {
        if (symbolic::uses(last_dim, scalar)) {
            assert(gather_variable == SymEngine::null);
            gather_variable = symbolic::symbol(scalar);
        }
    }
    if (gather_variable != SymEngine::null) {
        return AccessType::INDIRECTION;
    }

    // Case: Contiguous
    if (symbolic::uses(last_dim, indvar->get_name())) {
        auto match = symbolic::affine(last_dim, indvar);
        if (match.first == SymEngine::null) {
            return AccessType::UNKNOWN;
        }
        if (!symbolic::eq(match.first, symbolic::integer(1))) {
            return AccessType::UNKNOWN;
        }
        return AccessType::CONTIGUOUS;
    }

    // Case: Constant
    return AccessType::CONSTANT;
};

}  // namespace transformations
}  // namespace sdfg
