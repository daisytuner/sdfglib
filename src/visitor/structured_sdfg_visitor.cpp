#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace visitor {

StructuredSDFGVisitor::
    StructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : builder_(builder), analysis_manager_(analysis_manager) {}

bool StructuredSDFGVisitor::visit() { return this->visit(builder_.subject().root()); }

bool StructuredSDFGVisitor::visit(structured_control_flow::Sequence& parent) {
    if (this->accept(parent)) {
        return true;
    }

    for (size_t i = 0; i < parent.size(); i++) {
        auto& current = parent.at(i).first;

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(&current)) {
            if (this->accept(*block_stmt)) {
                return true;
            }
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(&current)) {
            if (this->visit(*sequence_stmt)) {
                return true;
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&current)) {
            if (this->accept(*if_else_stmt)) {
                return true;
            }

            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                if (this->visit(if_else_stmt->at(i).first)) {
                    return true;
                }
            }
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(&current)) {
            if (this->accept(*for_stmt)) {
                return true;
            }

            if (this->visit(for_stmt->root())) {
                return true;
            }
        } else if (auto for_each_stmt = dynamic_cast<structured_control_flow::ForEach*>(&current)) {
            if (this->accept(*for_each_stmt)) {
                return true;
            }

            if (this->visit(for_each_stmt->root())) {
                return true;
            }
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&current)) {
            if (this->accept(*map_stmt)) {
                return true;
            }

            if (this->visit(map_stmt->root())) {
                return true;
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&current)) {
            if (this->accept(*while_stmt)) {
                return true;
            }

            if (this->visit(while_stmt->root())) {
                return true;
            }
        } else if (auto continue_stmt = dynamic_cast<structured_control_flow::Continue*>(&current)) {
            if (this->accept(*continue_stmt)) {
                return true;
            }
        } else if (auto break_stmt = dynamic_cast<structured_control_flow::Break*>(&current)) {
            if (this->accept(*break_stmt)) {
                return true;
            }
        } else if (auto return_stmt = dynamic_cast<structured_control_flow::Return*>(&current)) {
            if (this->accept(*return_stmt)) {
                return true;
            }
        }
    }

    return false;
};

bool StructuredSDFGVisitor::accept(structured_control_flow::Block& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::Sequence& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::Return& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::IfElse& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::While& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::Continue& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::Break& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::For& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::ForEach& node) { return false; };

bool StructuredSDFGVisitor::accept(structured_control_flow::Map& node) { return false; };

} // namespace visitor
} // namespace sdfg
