#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace serializer {

class JSONSerializer {
   private:
    std::string filename_;
    nlohmann::json json_data_;

    std::unique_ptr<sdfg::StructuredSDFG>& sdfg_;

    std::unordered_map<long, sdfg::structured_control_flow::While&> while_map_;

   public:
    JSONSerializer(const std::string& filename, std::unique_ptr<sdfg::StructuredSDFG>& sdfg)
        : filename_(filename), sdfg_(sdfg) {}

    nlohmann::json serialize();

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
    void kernel_to_json(nlohmann::json& j,
                        const sdfg::structured_control_flow::Kernel& kernel_node);
    void return_node_to_json(nlohmann::json& j,
                             const sdfg::structured_control_flow::Return& return_node);

    void json_to_structure_definition(const nlohmann::json& j,
                                      sdfg::builder::StructuredSDFGBuilder& builder);
    void json_to_containers(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder);
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
                              sdfg::structured_control_flow::Sequence& parent);
    void json_to_while_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                            sdfg::structured_control_flow::Sequence& parent,
                            symbolic::Assignments& assignments);
    void json_to_break_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                            sdfg::structured_control_flow::Sequence& parent);
    void json_to_continue_node(const nlohmann::json& j,
                               sdfg::builder::StructuredSDFGBuilder& builder,
                               sdfg::structured_control_flow::Sequence& parent);
    void json_to_kernel_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                             sdfg::structured_control_flow::Sequence& parent);
    void json_to_return_node(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
                             sdfg::structured_control_flow::Sequence& parent,
                             symbolic::Assignments& assignments);
    std::unique_ptr<sdfg::types::IType> json_to_type(const nlohmann::json& j);
    std::vector<std::pair<std::string, types::Scalar>> json_to_arguments(const nlohmann::json& j);
};

}  // namespace serializer
}  // namespace sdfg
