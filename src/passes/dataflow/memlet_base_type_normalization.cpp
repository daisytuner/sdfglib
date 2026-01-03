#include "sdfg/passes/dataflow/memlet_base_type_normalization.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

/**
 * @brief Visitor to normalize memlet base types across the SDFG
 */
class MemletNormalizationVisitor : public visitor::ActualStructuredSDFGVisitor {
private:
    bool applied_;

    /**
     * @brief Normalizes all memlets in a dataflow graph
     * @param dfg The dataflow graph to process
     * @return true if any memlets were normalized
     */
    bool normalize_memlets_in_dataflow(data_flow::DataFlowGraph& dfg) {
        bool local_applied = false;

        // Collect all memlets that need normalization
        std::vector<data_flow::Memlet*> memlets_to_normalize;

        for (auto& memlet : dfg.edges()) {
            auto& base_type = memlet.base_type();

            // Check if base_type is a pointer to nested arrays
            if (base_type.type_id() != types::TypeID::Pointer) {
                continue;
            }

            auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);
            if (!pointer_type.has_pointee_type()) {
                continue;
            }

            // Check if pointee type contains arrays
            auto& pointee_type = pointer_type.pointee_type();
            const types::IType* current_type = &pointee_type;
            std::vector<symbolic::Expression> array_dimensions;

            // Collect all array dimensions
            while (current_type->type_id() == types::TypeID::Array) {
                auto array_type = dynamic_cast<const types::Array*>(current_type);
                array_dimensions.push_back(array_type->num_elements());
                current_type = &array_type->element_type();
            }

            // If we found nested arrays, mark this memlet for normalization
            if (!array_dimensions.empty()) {
                memlets_to_normalize.push_back(const_cast<data_flow::Memlet*>(&memlet));
            }
        }

        // Apply normalization to collected memlets
        for (auto* memlet : memlets_to_normalize) {
            auto& base_type = memlet->base_type();
            auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);
            auto& pointee_type = pointer_type.pointee_type();

            // Get innermost element type and collect array dimensions
            const types::IType* current_type = &pointee_type;
            std::vector<symbolic::Expression> array_dimensions;

            while (current_type->type_id() == types::TypeID::Array) {
                auto array_type = dynamic_cast<const types::Array*>(current_type);
                array_dimensions.push_back(array_type->num_elements());
                current_type = &array_type->element_type();
            }

            // Create new pointer type with innermost element type
            types::Pointer new_pointer_type(
                pointer_type.storage_type(), pointer_type.alignment(), pointer_type.initializer(), *current_type
            );

            // Linearize subset
            auto old_subset = memlet->subset();
            data_flow::Subset new_subset;

            // If subset is empty or doesn't match array dimensions, keep it as is
            if (!old_subset.empty() && old_subset.size() <= array_dimensions.size()) {
                // Calculate linearized index: idx = i0 * (d1 * d2 * ... * dn-1) + i1 * (d2 * ... * dn-1) + ... + in-1
                symbolic::Expression linearized_index = symbolic::zero();

                for (size_t i = 0; i < old_subset.size(); ++i) {
                    symbolic::Expression stride = symbolic::one();

                    // Calculate stride: product of all dimensions after current dimension
                    for (size_t j = i + 1; j < array_dimensions.size(); ++j) {
                        stride = symbolic::mul(stride, array_dimensions[j]);
                    }

                    linearized_index = symbolic::add(linearized_index, symbolic::mul(old_subset[i], stride));
                }

                new_subset.push_back(linearized_index);

                // Keep any remaining subset dimensions that were beyond the array dimensions
                for (size_t i = array_dimensions.size(); i < old_subset.size(); ++i) {
                    new_subset.push_back(old_subset[i]);
                }
            } else {
                // Keep original subset if it doesn't fit the pattern
                new_subset = old_subset;
            }

            // Update memlet with new base type and subset
            memlet->set_base_type(new_pointer_type);
            memlet->set_subset(new_subset);

            local_applied = true;
        }

        return local_applied;
    }

public:
    MemletNormalizationVisitor() : ActualStructuredSDFGVisitor(), applied_(false) {}

    bool visit(structured_control_flow::Block& node) override {
        auto& dfg = node.dataflow();
        if (normalize_memlets_in_dataflow(dfg)) {
            applied_ = true;
        }
        return true;
    }

    bool was_applied() const { return applied_; }
};

} // namespace

MemletBaseTypeNormalization::MemletBaseTypeNormalization() : Pass() {}

std::string MemletBaseTypeNormalization::name() { return "MemletBaseTypeNormalization"; }

bool MemletBaseTypeNormalization::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    MemletNormalizationVisitor visitor;
    visitor.dispatch(sdfg.root());

    return visitor.was_applied();
};

} // namespace passes
} // namespace sdfg
