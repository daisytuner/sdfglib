#include "sdfg/codegen/dispatchers/schedules/highway_dispatcher.h"

#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/transformations/vectorization.h"

namespace sdfg {
namespace codegen {

void HighwayDispatcher::dispatch_declarations(const structured_control_flow::ControlFlowNode& node,
                                              PrettyPrinter& stream) {
    auto& sdfg = schedule_.sdfg();
    for (auto& container : schedule_.node_allocations(&node)) {
        if (container == node_.indvar()->get_name()) {
            continue;
        }
        auto& type = sdfg.type(container);
        stream << language_extension_.declaration(container, type) << ";" << std::endl;
    }
};

void HighwayDispatcher::dispatch_declarations_vector(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) {
    auto& sdfg = schedule_.sdfg();
    for (auto& container : schedule_.allocations(&node)) {
        if (container == node_.indvar()->get_name()) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (type.primitive_type() == types::PrimitiveType::Bool) {
            stream << "auto " << container << " = "
                   << "hn::MaskFalse("
                   << "daisy_vec_fp"
                   << ");" << std::endl;
        } else if (types::is_floating_point(type.primitive_type())) {
            stream << "auto " << container << " = "
                   << "hn::Undefined("
                   << "daisy_vec_fp"
                   << ");" << std::endl;
        } else {
            stream << "auto " << container << " = "
                   << "hn::Undefined("
                   << "daisy_vec_i"
                   << ");" << std::endl;
        }
        this->declared_vectors_.insert(container);
    }
};

void HighwayDispatcher::dispatch(structured_control_flow::ControlFlowNode& node,
                                 PrettyPrinter& stream) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        this->dispatch(*block, stream);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        this->dispatch(*sequence, stream);
    } else {
        throw std::runtime_error("Unsupported control flow node");
    }
}

void HighwayDispatcher::dispatch(structured_control_flow::Sequence& node, PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    for (size_t i = 0; i < node.size(); i++) {
        assert(node.at(i).second.assignments().empty());
        auto& child = node.at(i).first;
        this->dispatch(child, stream);
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void HighwayDispatcher::dispatch(structured_control_flow::Block& node, PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& graph = node.dataflow();
    for (auto& dnode : graph.topological_sort()) {
        if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(dnode)) {
            this->dispatch_tasklet(graph, *tasklet, stream);
        }
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
};

void HighwayDispatcher::dispatch_tasklet(const data_flow::DataFlowGraph& graph,
                                         const data_flow::Tasklet& tasklet, PrettyPrinter& stream) {
    auto& sdfg = schedule_.sdfg();

    // Start of tasklet
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    PrettyPrinter tasklet_stream;
    tasklet_stream.setIndent(stream.indent());
    PrettyPrinter constants_stream;
    constants_stream.setIndent(stream.indent());

    // Analyze output
    auto& output_edge = *graph.out_edges(tasklet).begin();
    auto& output_subset = output_edge.subset();
    auto& output_lastdim = output_subset.at(output_subset.size() - 1);
    auto output_access_type =
        transformations::Vectorization::classify_access(output_subset, indvar_, moving_symbols_);
    auto& output_access_node = static_cast<const data_flow::AccessNode&>(output_edge.dst());
    auto& output_data = output_access_node.data();
    auto& output_type = sdfg.type(output_data);
    auto& output_inferred_type = types::infer_type(sdfg, output_type, output_subset);
    std::string output_vec_type;
    if (types::is_floating_point(output_inferred_type.primitive_type())) {
        output_vec_type = "daisy_vec_fp";
    } else if (types::is_unsigned(output_inferred_type.primitive_type())) {
        output_vec_type = "daisy_vec_ui";
    } else {
        output_vec_type = "daisy_vec_i";
    }

    types::PrimitiveType operation_type = types::PrimitiveType::Void;
    if (tasklet.inputs().empty()) {
        operation_type = tasklet.output(0).second.primitive_type();
    } else {
        operation_type = tasklet.input(0).second.primitive_type();
    }
    std::string operation_type_str;
    if (types::is_floating_point(operation_type)) {
        operation_type_str = "daisy_vec_fp";
    } else if (types::is_unsigned(operation_type)) {
        operation_type_str = "daisy_vec_ui";
    } else {
        operation_type_str = "daisy_vec_i";
    }

    bool output_fp_cast = false;
    output_fp_cast |= types::is_floating_point(operation_type) &&
                      types::is_integer(output_inferred_type.primitive_type());
    output_fp_cast |= types::is_integer(operation_type) &&
                      types::is_floating_point(output_inferred_type.primitive_type());
    bool output_signed_cast = false;
    if (types::is_integer(operation_type) &&
        types::is_integer(output_inferred_type.primitive_type())) {
        output_signed_cast |= types::is_signed(operation_type) &&
                              types::is_unsigned(output_inferred_type.primitive_type());
        output_signed_cast |= types::is_unsigned(operation_type) &&
                              types::is_signed(output_inferred_type.primitive_type());
    }
    if (output_inferred_type.primitive_type() == types::PrimitiveType::Bool) {
        output_fp_cast = false;
        output_signed_cast = false;
    }

    // Potential reduction
    bool is_reduction = false;
    std::string reduction_conn;

    // Generate code for inputs
    for (auto& iedge : graph.in_edges(tasklet)) {
        // Analyze input
        auto& access_node = static_cast<const data_flow::AccessNode&>(iedge.src());
        auto& subset = iedge.subset();
        auto& last_dim = subset.at(subset.size() - 1);
        auto access_type =
            transformations::Vectorization::classify_access(subset, indvar_, moving_symbols_);
        auto& data = access_node.data();
        auto& type = sdfg.type(data);
        auto& inferred_type = types::infer_type(sdfg, type, subset);
        std::string vec_type;
        if (types::is_floating_point(inferred_type.primitive_type())) {
            vec_type = "daisy_vec_fp";
        } else if (types::is_unsigned(inferred_type.primitive_type())) {
            vec_type = "daisy_vec_ui";
        } else {
            vec_type = "daisy_vec_i";
        }

        // Determine reduction
        if (data == output_data) {
            if (output_access_type == transformations::AccessType::CONSTANT) {
                assert(tasklet.code() == data_flow::TaskletCode::add ||
                       tasklet.code() == data_flow::TaskletCode::min ||
                       tasklet.code() == data_flow::TaskletCode::max ||
                       tasklet.code() == data_flow::TaskletCode::fma);
                assert(is_reduction == false);
                if (tasklet.code() == data_flow::TaskletCode::fma) {
                    assert(tasklet.input(2).first == iedge.dst_conn());
                }
                is_reduction = true;
                reduction_conn = iedge.dst_conn();

                // Handled separately
                continue;
            }
        }

        // Casting
        bool fp_cast = false;
        fp_cast |= types::is_floating_point(operation_type) &&
                   types::is_integer(inferred_type.primitive_type());
        fp_cast |= types::is_integer(operation_type) &&
                   types::is_floating_point(inferred_type.primitive_type());
        bool signed_cast = false;
        if (types::is_integer(operation_type) &&
            types::is_integer(inferred_type.primitive_type())) {
            signed_cast |= types::is_signed(operation_type) &&
                           types::is_unsigned(inferred_type.primitive_type());
            signed_cast |= types::is_unsigned(operation_type) &&
                           types::is_signed(inferred_type.primitive_type());
        }

        // Start code

        // Case 1: Local w.r.t. the loop body
        if (declared_vectors_.find(data) != declared_vectors_.end()) {
            tasklet_stream << "const auto " << iedge.dst_conn() << " = ";
            if (fp_cast) {
                tasklet_stream << "hn::ConvertTo(" << operation_type_str << ", ";
            } else if (signed_cast) {
                tasklet_stream << "hn::BitCast(" << operation_type_str << ", ";
            }
            tasklet_stream << data;
            if (fp_cast || signed_cast) {
                tasklet_stream << ")";
            }
            tasklet_stream << ";" << std::endl;
            continue;
        }

        // Case 2: Indvar
        if (data == indvar_->get_name()) {
            tasklet_stream << "const auto " << iedge.dst_conn() << " = ";
            if (fp_cast) {
                tasklet_stream << "hn::ConvertTo(" << operation_type_str << ", ";
            } else if (signed_cast) {
                tasklet_stream << "hn::BitCast(" << operation_type_str << ", ";
            }
            tasklet_stream << "hn::Iota(" << vec_type << ", " << data << ")";
            if (fp_cast || signed_cast) {
                tasklet_stream << ")";
            }
            tasklet_stream << ";" << std::endl;
            continue;
        }

        // We are left with external values that were not seen before
        // Hence, we need to load them into vectors

        switch (access_type) {
            case transformations::AccessType::CONSTANT: {
                tasklet_stream << "const auto " << iedge.dst_conn() << " = ";
                if (fp_cast) {
                    tasklet_stream << "hn::ConvertTo(" << operation_type_str << ", ";
                } else if (signed_cast) {
                    tasklet_stream << "hn::BitCast(" << operation_type_str << ", ";
                }
                tasklet_stream << "hn::Set(" << vec_type << ", " << data
                               << language_extension_.subset(schedule_.sdfg(), type, subset) << ")";
                if (fp_cast || signed_cast) {
                    tasklet_stream << ")";
                }
                tasklet_stream << ";" << std::endl;
                break;
            }
            case transformations::AccessType::CONTIGUOUS: {
                data_flow::Subset prefix(subset.begin(), subset.end() - 1);
                tasklet_stream << "const auto " << iedge.dst_conn() << " = ";
                if (fp_cast) {
                    tasklet_stream << "hn::ConvertTo(" << operation_type_str << ", ";
                } else if (signed_cast) {
                    tasklet_stream << "hn::BitCast(" << operation_type_str << ", ";
                }
                tasklet_stream << "hn::LoadU(" << vec_type << ", " << data
                               << language_extension_.subset(schedule_.sdfg(), type, prefix)
                               << " + (" << language_extension_.expression(last_dim) << "))";
                if (fp_cast || signed_cast) {
                    tasklet_stream << ")";
                }
                tasklet_stream << ";" << std::endl;
                break;
            }
            case transformations::AccessType::INDIRECTION: {
                std::string index = indirection_to_cpp(last_dim, tasklet_stream);

                data_flow::Subset prefix(subset.begin(), subset.end() - 1);
                tasklet_stream << "const auto " << iedge.dst_conn() << " = ";
                if (fp_cast) {
                    tasklet_stream << "hn::ConvertTo(" << operation_type_str << ", ";
                } else if (signed_cast) {
                    tasklet_stream << "hn::BitCast(" << operation_type_str << ", ";
                }
                tasklet_stream << "hn::GatherIndex(" << vec_type << ", " << data
                               << language_extension_.subset(schedule_.sdfg(), type, prefix);
                tasklet_stream << ", " << index << ")";
                if (fp_cast || signed_cast) {
                    tasklet_stream << ")";
                }
                tasklet_stream << ";" << std::endl;
                break;
            }
            default:
                throw std::runtime_error("Unsupported access type");
        }
    }

    tasklet_stream << "" << std::endl;

    // Generate code for tasklet
    std::string vectorized_code = "";
    if (!is_reduction) {
        // Special handling of fma:
        if (tasklet.code() == data_flow::TaskletCode::fma) {
            vectorized_code = "const auto " + output_edge.src_conn() + "_mul" + " = " + "hn::Mul(";
            for (size_t i = 0; i < 2; i++) {
                if (i > 0) {
                    vectorized_code += ", ";
                }
                auto arg = tasklet.input(i).first;

                if (!tasklet.needs_connector(i)) {
                    // Define a constant vector on-the-fly
                    constants_stream << "const auto "
                                     << "daisy_constant_" << constant_counter << " = "
                                     << "hn::Set(" << operation_type_str + ", " << arg << ")"
                                     << ";" << std::endl;
                    arg = "daisy_constant_" + std::to_string(constant_counter);
                    constant_counter++;
                }

                vectorized_code += arg;
            }
            vectorized_code += ");\n";
            vectorized_code += "const auto " + output_edge.src_conn() + " = " + "hn::Add(";
            vectorized_code += output_edge.src_conn() + "_mul" + ", ";

            auto arg = tasklet.input(2).first;
            if (!tasklet.needs_connector(2)) {
                // Define a constant vector on-the-fly
                constants_stream << "const auto "
                                 << "daisy_constant_" << constant_counter << " = "
                                 << "hn::Set(" << operation_type_str + ", " << arg << ")"
                                 << ";" << std::endl;
                arg = "daisy_constant_" + std::to_string(constant_counter);
                constant_counter++;
            }

            vectorized_code += arg;
            vectorized_code += ")";
        } else {
            vectorized_code += "const auto " + output_edge.src_conn() + " = ";
            vectorized_code += tasklet_to_simd_instruction(tasklet.code(), output_vec_type);
            for (size_t i = 0; i < tasklet.inputs().size(); i++) {
                if (i > 0) {
                    vectorized_code += ", ";
                }
                auto arg = tasklet.input(i).first;

                if (!tasklet.needs_connector(i)) {
                    // Define a constant vector on-the-fly
                    constants_stream << "const auto "
                                     << "daisy_constant_" << constant_counter << " = "
                                     << "hn::Set(" << operation_type_str + ", " << arg << ")"
                                     << ";" << std::endl;
                    arg = "daisy_constant_" + std::to_string(constant_counter);
                    constant_counter++;
                }

                vectorized_code += arg;
            }
            vectorized_code += ")";
        }
    } else {
        vectorized_code += "const auto " + output_edge.src_conn() + " = ";
        if (tasklet.code() == data_flow::TaskletCode::add ||
            tasklet.code() == data_flow::TaskletCode::min ||
            tasklet.code() == data_flow::TaskletCode::max) {
            // Tasklet degenerates to an assignment to out connector
            // The out connector holds a vector which is reduced to the output later

            // Determine non-reduction argument
            size_t index = 0;
            std::string arg = tasklet.input(0).first;
            if (arg == reduction_conn) {
                index = 1;
                arg = tasklet.input(1).first;
            }

            if (!tasklet.needs_connector(index)) {
                // Define a constant vector on-the-fly
                constants_stream << "const auto "
                                 << "daisy_constant_" << constant_counter << " = "
                                 << "hn::Set(" << operation_type_str + ", " << arg << ")"
                                 << ";" << std::endl;
                arg = "daisy_constant_" + std::to_string(constant_counter);
                constant_counter++;
            }

            vectorized_code += arg;
        } else if (tasklet.code() == data_flow::TaskletCode::fma) {
            // Tasklet degenerates to a multiplication to the out connector
            // The out connector holds a vector which is reduced to the output later

            vectorized_code +=
                tasklet_to_simd_instruction(data_flow::TaskletCode::mul, output_vec_type);
            for (size_t i = 0; i < 2; i++) {
                if (i > 0) {
                    vectorized_code += ", ";
                }
                auto arg = tasklet.input(i).first;

                if (!tasklet.needs_connector(i)) {
                    // Define a constant vector on-the-fly
                    constants_stream << "const auto "
                                     << "daisy_constant_" << constant_counter << " = "
                                     << "hn::Set(" << operation_type_str + ", " << arg << ")"
                                     << ";" << std::endl;
                    arg = "daisy_constant_" + std::to_string(constant_counter);
                    constant_counter++;
                }

                vectorized_code += arg;
            }
            vectorized_code += ")";
        }
    }

    tasklet_stream << vectorized_code << ";" << std::endl;

    tasklet_stream << "" << std::endl;

    // Case: Local w.r.t. the loop body
    // The local variable was declared already (at allocating node)
    // and can be used directly.
    if (declared_vectors_.find(output_data) != declared_vectors_.end()) {
        if (!tasklet.is_conditional()) {
            tasklet_stream << output_data << " = ";
            if (output_fp_cast) {
                tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", "
                               << output_edge.src_conn() << ")";
            } else if (output_signed_cast) {
                tasklet_stream << "hn::BitCast(" << output_vec_type << ", "
                               << output_edge.src_conn() << ")";
            } else {
                tasklet_stream << output_edge.src_conn();
            }
            tasklet_stream << ";" << std::endl;
        } else {
            auto& condition = tasklet.condition();
            auto mask = condition_to_simd_instruction(condition);
            mask = "hn::RebindMask(" + output_vec_type + ", " + mask + ")";

            // Else value
            std::string else_value = output_edge.src_conn() + "_else";
            tasklet_stream << "const auto " << else_value << " = " << output_data << ";"
                           << std::endl;

            // Result
            std::string value = output_edge.src_conn() + "_result";
            tasklet_stream << "const auto " << value << " = "
                           << "hn::IfThenElse(" + mask + ", " + output_edge.src_conn() + ", " +
                                  else_value + ")"
                           << ";" << std::endl;

            // Assign
            tasklet_stream << output_data << " = ";
            if (output_fp_cast) {
                tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", " << value << ")";
            } else if (output_signed_cast) {
                tasklet_stream << "hn::BitCast(" << output_vec_type << ", " << value << ")";
            } else {
                tasklet_stream << value;
            }
            tasklet_stream << ";" << std::endl;
        }

        stream << constants_stream.str();
        stream << tasklet_stream.str();
        stream << "}" << std::endl;
        return;
    }

    // We are left with external values that were not seen before
    // Hence, we need to store them into vectors

    switch (output_access_type) {
        case transformations::AccessType::CONSTANT: {
            if (!is_reduction) {
                tasklet_stream << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type,
                                                             output_subset)
                               << " = ";
                if (output_fp_cast) {
                    tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", ";
                    tasklet_stream << "hn::ExtractLane(" << output_edge.src_conn()
                                   << ", hn::Lanes(daisy_vec_i) - 1)";
                    tasklet_stream << ")";
                } else if (output_signed_cast) {
                    tasklet_stream << "hn::BitCast(" << output_vec_type << ", ";
                    tasklet_stream << "hn::ExtractLane(" << output_edge.src_conn()
                                   << ", hn::Lanes(daisy_vec_i) - 1)";
                    tasklet_stream << ")";
                } else {
                    tasklet_stream << "hn::ExtractLane(" << output_edge.src_conn()
                                   << ", hn::Lanes(daisy_vec_i) - 1)";
                }
                tasklet_stream << ";" << std::endl;
            } else {
                std::string hwy_op;
                std::string op;
                if (tasklet.code() == data_flow::TaskletCode::add ||
                    tasklet.code() == data_flow::TaskletCode::fma) {
                    hwy_op = "hn::ReduceSum";
                    op = "+";
                } else if (tasklet.code() == data_flow::TaskletCode::min) {
                    hwy_op = "hn::ReduceMin";
                    op += "__daisy_min";
                } else if (tasklet.code() == data_flow::TaskletCode::max) {
                    hwy_op = "hn::ReduceMax";
                    op += "__daisy_max";
                } else {
                    throw std::runtime_error("Unsupported reduction");
                }
                std::string reduced_out = output_edge.src_conn() + "_reduced";

                tasklet_stream << "const auto " << reduced_out << " = " << hwy_op << "("
                               << output_vec_type << ", " << output_edge.src_conn() << ")"
                               << ";" << std::endl;

                bool is_infix = data_flow::is_infix(tasklet.code());
                if (tasklet.code() == data_flow::TaskletCode::fma) {
                    op = "+";
                    is_infix = true;
                }

                tasklet_stream << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type,
                                                             output_subset)
                               << " = ";
                if (is_infix) {
                    tasklet_stream
                        << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, output_subset)
                        << " " << op << " " << reduced_out << ";" << std::endl;
                } else {
                    tasklet_stream
                        << op << "(" << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, output_subset)
                        << ", " << reduced_out << ");" << std::endl;
                }
            }
            break;
        }
        case transformations::AccessType::CONTIGUOUS: {
            data_flow::Subset prefix(output_subset.begin(), output_subset.end() - 1);
            if (!tasklet.is_conditional()) {
                if (output_fp_cast) {
                    tasklet_stream << "hn::StoreU(";
                    tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", "
                                   << output_edge.src_conn() << ")";
                    tasklet_stream
                        << ", " << output_vec_type << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                } else if (output_signed_cast) {
                    tasklet_stream << "hn::StoreU(";
                    tasklet_stream << "hn::BitCast(" << output_vec_type << ", "
                                   << output_edge.src_conn() << ")";
                    tasklet_stream
                        << ", " << output_vec_type << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                } else {
                    tasklet_stream
                        << "hn::StoreU(" << output_edge.src_conn() << ", " << output_vec_type
                        << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                }
            } else {
                auto& condition = tasklet.condition();
                auto mask = condition_to_simd_instruction(condition);
                mask = "hn::RebindMask(" + output_vec_type + ", " + mask + ")";

                // Else value
                std::string else_value = output_edge.src_conn() + "_else";
                tasklet_stream << "const auto " << else_value << " = "
                               << "hn::LoadU(" << output_vec_type << ", " << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                               << " + (" << language_extension_.expression(output_lastdim) << "));"
                               << std::endl;

                // Result
                std::string value = output_edge.src_conn() + "_result";
                tasklet_stream << "const auto " << value << " = "
                               << "hn::IfThenElse(" + mask + ", " + output_edge.src_conn() + ", " +
                                      else_value + ")"
                               << ";" << std::endl;

                if (output_fp_cast) {
                    tasklet_stream << "hn::StoreU(";
                    tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", " << value << ")";
                    tasklet_stream
                        << ", " << output_vec_type << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                } else if (output_signed_cast) {
                    tasklet_stream << "hn::StoreU(";
                    tasklet_stream << "hn::BitCast(" << output_vec_type << ", " << value << ")";
                    tasklet_stream
                        << ", " << output_vec_type << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                } else {
                    tasklet_stream
                        << "hn::StoreU(" << value << ", " << output_vec_type << ", " << output_data
                        << language_extension_.subset(schedule_.sdfg(), output_type, prefix)
                        << " + (" << language_extension_.expression(output_lastdim) << "));"
                        << std::endl;
                }
            }
            break;
        }
        case transformations::AccessType::INDIRECTION: {
            std::string index = indirection_to_cpp(output_lastdim, tasklet_stream);

            data_flow::Subset prefix(output_subset.begin(), output_subset.end() - 1);
            tasklet_stream << "hn::ScatterIndex(";
            if (output_fp_cast) {
                tasklet_stream << "hn::ConvertTo(" << output_vec_type << ", "
                               << output_edge.src_conn() << ")";
                tasklet_stream << ", " << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type, prefix);
                tasklet_stream << ", " << index;
                tasklet_stream << ");" << std::endl;
            } else if (output_signed_cast) {
                tasklet_stream << "hn::BitCast(" << output_vec_type << ", "
                               << output_edge.src_conn() << ")";
                tasklet_stream << ", " << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type, prefix);
                tasklet_stream << ", " << index;
                tasklet_stream << ");" << std::endl;
            } else {
                tasklet_stream << ", " << output_data
                               << language_extension_.subset(schedule_.sdfg(), output_type, prefix);
                tasklet_stream << ", " << index;
                tasklet_stream << ");" << std::endl;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported access type");
    }

    stream << constants_stream.str();
    stream << tasklet_stream.str();

    // End of tasklet
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

std::string HighwayDispatcher::indirection_to_cpp(const symbolic::Expression& access,
                                                  PrettyPrinter& stream) {
    std::string new_index_name = "daisy_indirection_" + std::to_string(constant_counter++);
    if (SymEngine::is_a<SymEngine::Integer>(*access)) {
        stream << "const auto " << new_index_name << " = "
               << "hn::Set(daisy_vec_i, " << language_extension_.expression(access) << ");"
               << std::endl;
    } else if (symbolic::eq(access, indvar_)) {
        stream << "const auto " << new_index_name << " = "
               << "hn::Iota(daisy_vec_i, " << indvar_->get_name() << ");" << std::endl;
    } else if (SymEngine::is_a<SymEngine::Symbol>(*access)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(access);
        if (this->declared_vectors_.find(symbol->get_name()) != this->declared_vectors_.end()) {
            stream << "const auto " << new_index_name << " = " << symbol->get_name() << ";"
                   << std::endl;
        } else {
            stream << "const auto " << new_index_name << " = "
                   << "hn::Set(daisy_vec_i, " << symbol->get_name() << ");" << std::endl;
        }
    } else if (SymEngine::is_a<SymEngine::Add>(*access)) {
        auto args = access->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);
        auto op1 = indirection_to_cpp(lhs, stream);
        auto op2 = indirection_to_cpp(rhs, stream);
        stream << "const auto " << new_index_name << " = "
               << "hn::Add(" << op1 << ", " << op2 << ");" << std::endl;
    } else if (SymEngine::is_a<SymEngine::Mul>(*access)) {
        auto args = access->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);
        auto op1 = indirection_to_cpp(lhs, stream);
        auto op2 = indirection_to_cpp(rhs, stream);
        stream << "const auto " << new_index_name << " = "
               << "hn::Mul(" << op1 << ", " << op2 << ");" << std::endl;
    } else {
        throw std::runtime_error("Unsupported indirection");
    }

    return new_index_name;
};

std::string HighwayDispatcher::function_name(const structured_control_flow::ControlFlowNode& node) {
    return node.name();
};

std::vector<std::string> HighwayDispatcher::function_arguments(
    Schedule& schedule, structured_control_flow::ControlFlowNode& node) {

    // Local variables
    auto allocations = schedule.allocations(&node);

    // All read and write sets
    auto& analysis_manager = schedule.analysis_manager();
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView node_users(users, node);

    std::unordered_set<std::string> arguments_unique;
    for (auto& use : node_users.uses()) {
        if (allocations.find(use->container()) == allocations.end()) {
            arguments_unique.insert(use->container());
        }
    }
    std::vector<std::string> arguments_vector(arguments_unique.begin(), arguments_unique.end());
    std::sort(arguments_vector.begin(), arguments_vector.end());

    return arguments_vector;
};

std::string HighwayDispatcher::condition_to_simd_instruction(
    const symbolic::Expression& condition) {
    if (SymEngine::is_a<SymEngine::Symbol>(*condition)) {
        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(condition);
        if (declared_vectors_.find(symbol->get_name()) != declared_vectors_.end()) {
            return symbol->get_name();
        } else {
            return "hn::Set(daisy_vec_i, " + symbol->get_name() + ")";
        }
    } else if (SymEngine::is_a<SymEngine::Integer>(*condition)) {
        return "hn::Set(daisy_vec_i, " + language_extension_.expression(condition) + ")";
    } else if (symbolic::is_true(condition)) {
        return "hn::MaskTrue(daisy_vec_i)";
    } else if (symbolic::is_false(condition)) {
        return "hn::MaskFalse(daisy_vec_i)";
    }

    if (SymEngine::is_a<SymEngine::Equality>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        // Lhs is constant
        if (symbolic::is_true(lhs)) {
            return condition_to_simd_instruction(rhs);
        }
        if (symbolic::is_false(lhs)) {
            return "hn::Not(" + condition_to_simd_instruction(rhs) + ")";
        }

        // Rhs is constant
        if (symbolic::is_true(rhs)) {
            return condition_to_simd_instruction(lhs);
        }
        if (symbolic::is_false(rhs)) {
            return "hn::Not(" + condition_to_simd_instruction(lhs) + ")";
        }

        return "hn::Eq(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else if (SymEngine::is_a<SymEngine::Unequality>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        // Lhs is constant
        if (symbolic::is_true(lhs)) {
            return "hn::Not(" + condition_to_simd_instruction(rhs) + ")";
        }
        if (symbolic::is_false(lhs)) {
            return condition_to_simd_instruction(rhs);
        }

        // Rhs is constant
        if (symbolic::is_true(rhs)) {
            return "hn::Not(" + condition_to_simd_instruction(lhs) + ")";
        }
        if (symbolic::is_false(rhs)) {
            return condition_to_simd_instruction(lhs);
        }

        return "hn::Ne(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else if (SymEngine::is_a<SymEngine::And>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        if (symbolic::is_true(rhs)) {
            return condition_to_simd_instruction(lhs);
        } else if (symbolic::is_true(lhs)) {
            return condition_to_simd_instruction(rhs);
        } else if (symbolic::is_false(rhs)) {
            return "hn::MaskFalse(daisy_vec_i)";
        } else if (symbolic::is_false(lhs)) {
            return "hn::MaskFalse(daisy_vec_i)";
        }

        return "hn::And(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else if (SymEngine::is_a<SymEngine::Or>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        if (symbolic::is_true(rhs) || symbolic::is_true(lhs)) {
            return "hn::MaskTrue(daisy_vec_i)";
        } else if (symbolic::is_false(rhs)) {
            return condition_to_simd_instruction(lhs);
        } else if (symbolic::is_false(lhs)) {
            return condition_to_simd_instruction(rhs);
        }

        return "hn::Or(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        return "hn::Lt(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else if (SymEngine::is_a<SymEngine::LessThan>(*condition)) {
        auto args = condition->get_args();
        auto lhs = args.at(0);
        auto rhs = args.at(1);

        return "hn::Le(" + condition_to_simd_instruction(lhs) + ", " +
               condition_to_simd_instruction(rhs) + ")";
    } else {
        throw std::runtime_error("Unsupported condition");
    }
}

std::string HighwayDispatcher::tasklet_to_simd_instruction(data_flow::TaskletCode c,
                                                           std::string vec_type) {
    switch (c) {
        case data_flow::TaskletCode::assign:
            return "(";
        case data_flow::TaskletCode::neg:
            return "hn::Neg(";
        case data_flow::TaskletCode::add:
            return "hn::Add(";
        case data_flow::TaskletCode::sub:
            return "hn::Sub(";
        case data_flow::TaskletCode::mul:
            return "hn::Mul(";
        case data_flow::TaskletCode::div:
            return "hn::Div(";
        case data_flow::TaskletCode::mod:
            return "hn::Mod(";
        case data_flow::TaskletCode::abs:
            return "hn::Abs(";
        case data_flow::TaskletCode::max:
            return "hn::Max(";
        case data_flow::TaskletCode::min:
            return "hn::Min(";
        case data_flow::TaskletCode::fabs:
            return "hn::Abs(";
        case data_flow::TaskletCode::sqrt:
            return "hn::Sqrt(";
        case data_flow::TaskletCode::sqrtf:
            return "hn::Sqrt(";
        case data_flow::TaskletCode::sin:
            return "hn::Sin(" + vec_type + ", ";
        case data_flow::TaskletCode::cos:
            return "hn::Cos(" + vec_type + ", ";
        case data_flow::TaskletCode::tan:
            return "hn::Tan(" + vec_type + ", ";
        case data_flow::TaskletCode::pow:
            return "hn::Pow(" + vec_type + ", ";
        case data_flow::TaskletCode::exp:
            return "hn::Exp(" + vec_type + ", ";
        case data_flow::TaskletCode::expf:
            return "hn::Exp(" + vec_type + ", ";
        case data_flow::TaskletCode::exp2:
            return "hn::Exp2(" + vec_type + ", ";
        case data_flow::TaskletCode::log:
            return "hn::Log(" + vec_type + ", ";
        case data_flow::TaskletCode::log2:
            return "hn::Log2(" + vec_type + ", ";
        case data_flow::TaskletCode::log10:
            return "hn::Log10(" + vec_type + ", ";
        case data_flow::TaskletCode::fma:
            return "hn::MulAdd(";
        case data_flow::TaskletCode::floor:
            return "hn::Floor(" + vec_type + ", ";
        case data_flow::TaskletCode::ceil:
            return "hn::Ceil(" + vec_type + ", ";
        case data_flow::TaskletCode::trunc:
            return "hn::Trunc(" + vec_type + ", ";
        case data_flow::TaskletCode::round:
            return "hn::Round(" + vec_type + ", ";
        case data_flow::TaskletCode::olt:
            return "hn::Lt(";
        case data_flow::TaskletCode::ole:
            return "hn::Le(";
        case data_flow::TaskletCode::oeq:
            return "hn::Eq(";
        case data_flow::TaskletCode::one:
            return "hn::Ne(";
        case data_flow::TaskletCode::ogt:
            return "hn::Gt(";
        case data_flow::TaskletCode::oge:
            return "hn::Ge(";
        case data_flow::TaskletCode::bitwise_and:
            return "hn::And(";
        case data_flow::TaskletCode::bitwise_or:
            return "hn::Or(";
        case data_flow::TaskletCode::bitwise_xor:
            return "hn::Xor(";
        case data_flow::TaskletCode::bitwise_not:
            return "hn::Not(";
    }

    throw std::runtime_error("Unknown tasklet class");
};

HighwayDispatcher::HighwayDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                     structured_control_flow::For& node, Instrumentation& instrumentation)
    : NodeDispatcher(language_extension, schedule, node, instrumentation),
      node_(node),
      indvar_(node.indvar()) {
    auto& body = node_.root();

    // Determine moving symbols
    auto& sdfg = schedule.builder().subject();
    auto& all_users = schedule.analysis_manager().get<analysis::Users>();
    analysis::UsersView users(all_users, body);
    for (auto entry : users.write_subsets()) {
        auto& data = entry.first;
        auto& type = sdfg.type(data);
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }
        if (!types::is_integer(type.primitive_type())) {
            continue;
        }
        moving_symbols_.insert(data);
    }

    // Determine bit width
    this->bit_width_ = 0;
    for (auto& entry : users.write_subsets()) {
        auto& data = entry.first;
        for (auto& subset : entry.second) {
            auto access_type =
                transformations::Vectorization::classify_access(subset, indvar_, moving_symbols_);
            if (access_type == transformations::AccessType::INDIRECTION ||
                access_type == transformations::AccessType::CONTIGUOUS) {
                auto& type = sdfg.type(data);
                auto& inferred_type = types::infer_type(sdfg, type, subset);
                if (inferred_type.primitive_type() != types::PrimitiveType::Bool) {
                    this->bit_width_ = types::bit_width(inferred_type.primitive_type());
                    break;
                }
            }
        }
        if (this->bit_width_ > 0) {
            break;
        }
    }
    if (this->bit_width_ == 0) {
        for (auto& entry : users.read_subsets()) {
            auto& data = entry.first;
            for (auto& subset : entry.second) {
                auto access_type = transformations::Vectorization::classify_access(subset, indvar_,
                                                                                   moving_symbols_);
                if (access_type == transformations::AccessType::INDIRECTION ||
                    access_type == transformations::AccessType::CONTIGUOUS) {
                    auto& type = sdfg.type(data);
                    auto& inferred_type = types::infer_type(sdfg, type, subset);
                    if (inferred_type.primitive_type() != types::PrimitiveType::Bool) {
                        this->bit_width_ = types::bit_width(inferred_type.primitive_type());
                        break;
                    }
                }
            }
            if (this->bit_width_ > 0) {
                break;
            }
        }
    }
};

void HighwayDispatcher::dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                 PrettyPrinter& library_stream) {
    auto& sdfg = schedule_.sdfg();

    // Collect arguments
    auto function_name = this->function_name(node_);
    auto function_arguments = this->function_arguments(schedule_, node_);

    // Begin namespace
    library_stream << "namespace HWY_NAMESPACE {" << std::endl;
    library_stream << "namespace hn = hwy::HWY_NAMESPACE;" << std::endl << std::endl;

    // Pass by reference
    std::vector<std::string> arglist_references;
    for (auto& arg : function_arguments) {
        auto& base_type = sdfg.type(arg);
        if (dynamic_cast<const types::Scalar*>(&base_type) ||
            dynamic_cast<const types::Structure*>(&base_type)) {
            auto ref_type = std::make_unique<Reference>(sdfg.type(arg));
            arglist_references.push_back(language_extension_.declaration(arg, *ref_type));
        } else if (auto array_type = dynamic_cast<const types::Array*>(&base_type)) {
            auto ref_type = std::make_unique<types::Pointer>(array_type->element_type());
            arglist_references.push_back(language_extension_.declaration(arg, *ref_type));
        } else {
            arglist_references.push_back(language_extension_.declaration(arg, base_type));
        }
    }

    // Define vectorized function
    library_stream << "HWY_ATTR void ";
    library_stream << function_name;
    library_stream << "(";
    library_stream << sdfg::helpers::join(arglist_references, ", ");
    library_stream << ")" << std::endl;

    // Dispatch vectorized function body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    // Declare vector width
    if (this->bit_width_ == 64 || this->bit_width_ == 0) {
        library_stream << "const hn::ScalableTag<double> daisy_vec_fp;" << std::endl;
        library_stream << "const hn::ScalableTag<int64_t> daisy_vec_i;" << std::endl;
        library_stream << "const hn::ScalableTag<uint64_t> daisy_vec_ui;" << std::endl;
    } else if (this->bit_width_ == 32) {
        library_stream << "const hn::ScalableTag<float> daisy_vec_fp;" << std::endl;
        library_stream << "const hn::ScalableTag<int32_t> daisy_vec_i;" << std::endl;
        library_stream << "const hn::ScalableTag<uint32_t> daisy_vec_ui;" << std::endl;
    } else if (this->bit_width_ == 16) {
        library_stream << "const hn::ScalableTag<int16_t> daisy_vec_i;" << std::endl;
        library_stream << "const hn::ScalableTag<uint16_t> daisy_vec_ui;" << std::endl;
    } else if (this->bit_width_ == 8) {
        library_stream << "const hn::ScalableTag<int8_t> daisy_vec_i;" << std::endl;
        library_stream << "const hn::ScalableTag<uint8_t> daisy_vec_ui;" << std::endl;
    }

    PrettyPrinter dummy_stream(0, true);
    this->dispatch_node(library_stream, dummy_stream, dummy_stream);

    // End function body
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;

    // End namespace
    library_stream << std::endl;
    library_stream << "}" << std::endl;

    // Declare lib function

    // Pass by pointer (C-interoperability)
    std::vector<std::string> arglist_pointers;
    std::vector<std::string> calllist_dereferenced;
    std::vector<std::string> calllist_referenced;
    for (auto& arg : function_arguments) {
        auto& base_type = sdfg.type(arg);
        if (dynamic_cast<const types::Scalar*>(&base_type) ||
            dynamic_cast<const types::Structure*>(&base_type)) {
            auto ref_type = std::make_unique<types::Pointer>(sdfg.type(arg));
            arglist_pointers.push_back(language_extension_.declaration(arg, *ref_type));

            calllist_dereferenced.push_back("*" + arg);
            calllist_referenced.push_back("&" + arg);
        } else if (auto array_type = dynamic_cast<const types::Array*>(&base_type)) {
            auto ref_type = std::make_unique<types::Pointer>(array_type->element_type());
            arglist_pointers.push_back(language_extension_.declaration(arg, *ref_type));

            calllist_dereferenced.push_back(arg);
            calllist_referenced.push_back(arg);
        } else {
            arglist_pointers.push_back(language_extension_.declaration(arg, base_type));

            calllist_dereferenced.push_back(arg);
            calllist_referenced.push_back(arg);
        }
    }

    globals_stream << "extern ";
    if (dynamic_cast<CPPLanguageExtension*>(&this->language_extension_)) {
        globals_stream << "\"C\" ";
    }
    globals_stream << "void " << function_name << "_lib";
    globals_stream << "(";
    globals_stream << sdfg::helpers::join(arglist_pointers, ", ");
    globals_stream << ");" << std::endl;

    // Define lib function
    library_stream << "#if HWY_ONCE" << std::endl << std::endl;

    library_stream << "HWY_EXPORT(" << function_name << ");" << std::endl;

    library_stream << "extern \"C\" void " << function_name << "_lib";
    library_stream << "(";
    library_stream << sdfg::helpers::join(arglist_pointers, ", ");
    library_stream << ")" << std::endl;
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);
    library_stream << "HWY_STATIC_DISPATCH";
    library_stream << "(" << function_name << ")";
    library_stream << "(";
    library_stream << sdfg::helpers::join(calllist_dereferenced, ", ");
    library_stream << ");" << std::endl;
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl << std::endl;
    library_stream << "#endif" << std::endl;

    // Call lib function
    main_stream << function_name << "_lib";
    main_stream << "(";
    main_stream << sdfg::helpers::join(calllist_referenced, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;
};

void HighwayDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                      PrettyPrinter& library_stream) {
    auto& sdfg = schedule_.sdfg();

    // Indvar and bound
    auto indvar = node_.indvar();
    auto new_condition =
        symbolic::subs(node_.condition(), indvar,
                       symbolic::symbol("(" + indvar->get_name() + " + hn::Lanes(daisy_vec_i))"));

    if (schedule_.allocation_lifetime(indvar->get_name()) == &this->node_) {
        main_stream << language_extension_.declaration(indvar->get_name(),
                                                       sdfg.type(indvar->get_name()))
                    << ";" << std::endl;
    }
    main_stream << indvar->get_name() << " = " << language_extension_.expression(node_.init())
                << ";" << std::endl;

    // 1. Vector loop
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Locals
    this->dispatch_declarations_vector(node_, main_stream);

    // Loop header
    main_stream << "for";
    main_stream << "(";
    main_stream << ";";
    main_stream << language_extension_.expression(new_condition);
    main_stream << ";";
    main_stream << indvar->get_name();
    main_stream << " = ";
    main_stream << indvar->get_name() + " + " + "hn::Lanes(daisy_vec_i)";
    main_stream << ")" << std::endl;

    // Loop body
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    this->dispatch(node_.root(), main_stream);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // End of vector loop
    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // 2. Remainder loop
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Declarations as scalars
    this->dispatch_declarations(node_, main_stream);

    // Loop header
    main_stream << "for";
    main_stream << "(";
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << indvar->get_name();
    main_stream << " = ";
    main_stream << indvar->get_name() + " + 1";
    main_stream << ")" << std::endl;

    // Loop body
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    PrettyPrinter dummy_stream(0, true);
    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), instrumentation_);
    dispatcher.dispatch(main_stream, dummy_stream, dummy_stream);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // End of remainder loop
    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
