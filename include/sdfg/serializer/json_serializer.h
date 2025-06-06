#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"
#include "symengine/logic.h"
#include "symengine/printers/codegen.h"

namespace sdfg {
namespace serializer {

class JSONSerializer {
   public:
    JSONSerializer() {}

    nlohmann::json serialize(std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

    std::unique_ptr<sdfg::StructuredSDFG> deserialize(nlohmann::json& j);

    void structure_definition_to_json(nlohmann::json& j,
                                      const sdfg::types::StructureDefinition& definition);
    void type_to_json(nlohmann::json& j, const sdfg::types::IType& type);
    void dataflow_to_json(nlohmann::json& j, const sdfg::data_flow::DataFlowGraph& dataflow);

    void sequence_to_json(nlohmann::json& j,
                          const sdfg::structured_control_flow::Sequence& sequence);
    void block_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Block& block);
    void for_to_json(nlohmann::json& j, const sdfg::structured_control_flow::For& for_node);
    void if_else_to_json(nlohmann::json& j,
                         const sdfg::structured_control_flow::IfElse& if_else_node);
    void while_node_to_json(nlohmann::json& j,
                            const sdfg::structured_control_flow::While& while_node);
    void break_node_to_json(nlohmann::json& j,
                            const sdfg::structured_control_flow::Break& break_node);
    void continue_node_to_json(nlohmann::json& j,
                               const sdfg::structured_control_flow::Continue& continue_node);
    void return_node_to_json(nlohmann::json& j,
                             const sdfg::structured_control_flow::Return& return_node);
    void map_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Map& map_node);

    void debug_info_to_json(nlohmann::json& j, const sdfg::DebugInfo& debug_info);

    void json_to_structure_definition(const nlohmann::json& j,
                                      sdfg::builder::StructuredSDFGBuilder& builder);
    void json_to_dataflow(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                          sdfg::structured_control_flow::Block& parent);

    void json_to_sequence(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                          sdfg::structured_control_flow::Sequence& sequence);
    void json_to_block_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                            sdfg::structured_control_flow::Sequence& parent,
                            symbolic::Assignments& assignments);
    void json_to_for_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                          sdfg::structured_control_flow::Sequence& parent,
                          symbolic::Assignments& assignments);
    void json_to_if_else_node(const nlohmann::json& j,
                              sdfg::builder::StructuredSDFGBuilder& builder,
                              sdfg::structured_control_flow::Sequence& parent,
                              symbolic::Assignments& assignments);
    void json_to_while_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                            sdfg::structured_control_flow::Sequence& parent,
                            symbolic::Assignments& assignments);
    void json_to_break_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                            sdfg::structured_control_flow::Sequence& parent,
                            symbolic::Assignments& assignments);
    void json_to_continue_node(const nlohmann::json& j,
                               sdfg::builder::StructuredSDFGBuilder& builder,
                               sdfg::structured_control_flow::Sequence& parent,
                               symbolic::Assignments& assignments);
    void json_to_return_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                             sdfg::structured_control_flow::Sequence& parent,
                             symbolic::Assignments& assignments);
    void json_to_map_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                          sdfg::structured_control_flow::Sequence& parent,
                          symbolic::Assignments& assignments);
    std::unique_ptr<sdfg::types::IType> json_to_type(const nlohmann::json& j);
    std::vector<std::pair<std::string, types::Scalar>> json_to_arguments(const nlohmann::json& j);
    DebugInfo json_to_debug_info(const nlohmann::json& j);

    std::string expression(const symbolic::Expression& expr);
};

class JSONSymbolicPrinter
    : public SymEngine::BaseVisitor<JSONSymbolicPrinter, SymEngine::CodePrinter> {
   public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    // Logical expressions
    void bvisit(const SymEngine::Equality& x);
    void bvisit(const SymEngine::Unequality& x);

    void bvisit(const SymEngine::LessThan& x);
    void bvisit(const SymEngine::StrictLessThan& x);

    // Min and Max
    void bvisit(const SymEngine::Min& x);
    void bvisit(const SymEngine::Max& x);
};

}  // namespace serializer
}  // namespace sdfg
