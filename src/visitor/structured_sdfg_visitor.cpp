#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace visitor {

StructuredSDFGVisitor::StructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder,
                                             analysis::AnalysisManager& analysis_manager)
    : builder_(builder), analysis_manager_(analysis_manager) {}

bool StructuredSDFGVisitor::visit() {
    if (this->accept(builder_.subject().root(), builder_.subject().root())) {
        return true;
    }
    return this->visit(builder_.subject().root());
}

bool StructuredSDFGVisitor::visit(structured_control_flow::Sequence& parent) {
    std::list<structured_control_flow::ControlFlowNode*> queue;
    for (size_t i = 0; i < parent.size(); i++) {
        queue.push_back(&parent.at(i).first);
    }
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(current)) {
            if (this->accept(parent, *block_stmt)) {
                return true;
            }
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            if (this->accept(parent, *sequence_stmt)) {
                return true;
            }

            if (this->visit(*sequence_stmt)) {
                return true;
            }
        } else if (auto return_stmt = dynamic_cast<structured_control_flow::Return*>(current)) {
            if (this->accept(parent, *return_stmt)) {
                return true;
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            if (this->accept(parent, *if_else_stmt)) {
                return true;
            }

            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                if (this->visit(if_else_stmt->at(i).first)) {
                    return true;
                }
            }
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            if (this->accept(parent, *for_stmt)) {
                return true;
            }

            if (this->visit(for_stmt->root())) {
                return true;
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            if (this->accept(parent, *while_stmt)) {
                return true;
            }

            if (this->visit(while_stmt->root())) {
                return true;
            }
        } else if (auto continue_stmt = dynamic_cast<structured_control_flow::Continue*>(current)) {
            if (this->accept(parent, *continue_stmt)) {
                return true;
            }
        } else if (auto break_stmt = dynamic_cast<structured_control_flow::Break*>(current)) {
            if (this->accept(parent, *break_stmt)) {
                return true;
            }
        } else if (auto kern_stmt = dynamic_cast<structured_control_flow::Kernel*>(current)) {
            if (this->accept(parent, *kern_stmt)) {
                return true;
            }

            if (this->visit(kern_stmt->root())) {
                return true;
            }
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(current)) {
            if (this->accept(parent, *map_stmt)) {
                return true;
            }

            if (this->visit(map_stmt->root())) {
                return true;
            }
        }
    }

    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Block& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Sequence& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Return& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::IfElse& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::While& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Continue& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Break& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::For& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Kernel& node) {
    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& parent,
                                   structured_control_flow::Map& node) {
    return false;
};

}  // namespace visitor
}  // namespace sdfg
