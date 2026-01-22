#include "mlir/Target/SDFG/TranslateToSDFG.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/TypeSwitch.h>
#include <memory>

#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/c_code_generator.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace mlir {
namespace sdfg {

class SDFGTranslator {
    using Builder = ::sdfg::builder::StructuredSDFGBuilder;
    using Sequence = ::sdfg::structured_control_flow::Sequence;
    using Block = ::sdfg::structured_control_flow::Block;
    std::list<Builder> builders_;

    using ValueMap = llvm::ScopedHashTableScope<Value, std::string>;
    llvm::ScopedHashTable<Value, std::string> value_map_;
    size_t value_counter_ = 0;

public:
    std::string get_or_create_container(Builder& builder, Value val, bool argument = false) {
        if (!this->value_map_.count(val)) {
            this->value_map_.insert(val, "_" + std::to_string(value_counter_++));
        }
        std::string container = *this->value_map_.begin(val);
        auto type = convertType(val.getType());
        if (builder.subject().exists(container)) {
            assert(builder.subject().type(container) == *type);
            assert(!argument || builder.subject().is_argument(container));
        } else {
            builder.add_container(container, *type, argument);
        }
        return container;
    }

    std::unique_ptr<::sdfg::types::IType> convertType(const Type& mlir_type) {
        if (mlir_type.isInteger(1)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Bool);
        } else if (mlir_type.isSignedInteger(8) || mlir_type.isSignlessInteger(8)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int8);
        } else if (mlir_type.isSignedInteger(16) || mlir_type.isSignlessInteger(16)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int16);
        } else if (mlir_type.isSignedInteger(32) || mlir_type.isSignlessInteger(32)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int32);
        } else if (mlir_type.isSignedInteger(64) || mlir_type.isSignlessInteger(64)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int64);
        } else if (mlir_type.isSignedInteger(128) || mlir_type.isSignlessInteger(128)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Int128);
        } else if (mlir_type.isUnsignedInteger(8)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt8);
        } else if (mlir_type.isUnsignedInteger(16)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt16);
        } else if (mlir_type.isUnsignedInteger(32)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt32);
        } else if (mlir_type.isUnsignedInteger(64)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt64);
        } else if (mlir_type.isUnsignedInteger(128)) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::UInt128);
        } else if (mlir_type.isF16()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Half);
        } else if (mlir_type.isBF16()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::BFloat);
        } else if (mlir_type.isF32()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Float);
        } else if (mlir_type.isF64()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::Double);
        } else if (mlir_type.isF80()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::X86_FP80);
        } else if (mlir_type.isF128()) {
            return std::make_unique<::sdfg::types::Scalar>(::sdfg::types::PrimitiveType::FP128);
        }
        return nullptr;
    }

    LogicalResult translate(Operation* op) {
        return llvm::TypeSwitch<Operation*, LogicalResult>(op)
            .Case<ModuleOp>([&](ModuleOp module_op) { return this->translateModuleOp(&module_op); })
            .Default([&](Operation* op) { return op->emitError("A module op is required here"); });
    }

    LogicalResult translateModuleOp(ModuleOp* module_op) {
        for (auto& op : module_op->getRegion().getOps()) {
            LogicalResult status = llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                                       .Case<SDFGOp>([&](SDFGOp sdfg_op) { return this->translateSDFGOp(&sdfg_op); })
                                       .Default([&](Operation* op) {
                                           return op->emitError("A sdfg op is required here");
                                       });
            if (failed(status)) {
                return failure();
            }
        }
        return success();
    }

    LogicalResult translateSDFGOp(SDFGOp* sdfg_op) {
        this->builders_.push_back(Builder(sdfg_op->getSymName().data(), ::sdfg::FunctionType_CPU));
        auto& builder = this->builders_.back();
        ValueMap value_map_scope(this->value_map_);

        // Return type
        auto function_type = sdfg_op->getFunctionType();
        assert(function_type.getNumResults() <= 1);
        if (function_type.getNumResults() == 1) {
            auto return_type = this->convertType(function_type.getResult(0));
            if (!return_type) {
                return sdfg_op->emitError("Could not convert type ") << function_type.getResult(0) << " to SDFG type";
            }
            builder.set_return_type(*return_type);
        }

        // Arguments
        for (auto arg : sdfg_op->getRegion().getArguments()) {
            this->get_or_create_container(builder, arg, true);
        }

        // Region
        for (auto& op : sdfg_op->getRegion().getOps()) {
            if (failed(this->translateOperation(builder, builder.subject().root(), &op))) {
                return failure();
            }
        }

        return success();
    }

    LogicalResult translateOperation(Builder& builder, Sequence& parent, Operation* op) {
        return llvm::TypeSwitch<Operation*, LogicalResult>(op)
            .Case<BlockOp>([&](BlockOp block_op) { return this->translateBlockOp(builder, parent, &block_op); })
            .Case<ReturnOp>([&](ReturnOp return_op) { return this->translateReturnOp(builder, parent, &return_op); })
            .Default([&](Operation* op) {
                return op->emitError("Unknown control flow operation. Could not translate.");
            });
    }

    LogicalResult translateBlockOp(Builder& builder, Sequence& parent, BlockOp* block_op) {
        auto& block = builder.add_block(parent);
        std::unordered_map<mlir::Attribute::ImplType*, ::sdfg::data_flow::Tasklet*> tasklets;
        std::unordered_map<mlir::detail::ValueImpl*, ::sdfg::data_flow::AccessNode*> access_nodes;

        // Capture results
        std::vector<Value> results;
        for (auto value : block_op->getResults()) {
            results.push_back(value);
        }

        // Create all nodes
        for (auto& op : block_op->getRegion().getOps()) {
            LogicalResult result =
                llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                    .Case<TaskletOp>([&](TaskletOp tasklet_op) {
                        auto code = static_cast<::sdfg::data_flow::TaskletCode>(tasklet_op.getCode());
                        std::vector<std::string> inputs;
                        switch (arity(tasklet_op.getCode())) {
                            case 1:
                                inputs.push_back("_in");
                                break;
                            case 2:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                break;
                            case 3:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                inputs.push_back("_in3");
                                break;
                            default:
                                return failure();
                        }
                        tasklets
                            .insert({tasklet_op.getLoc()->getImpl(), &builder.add_tasklet(block, code, "_out", inputs)}
                            );
                        return success();
                    })
                    .Case<MemletOp>([&](MemletOp memlet_op) {
                        if (dyn_cast_or_null<TaskletOp>(memlet_op.getInput().getDefiningOp())) {
                            access_nodes.insert(
                                {memlet_op.getOutput().getImpl(),
                                 &builder
                                      .add_access(block, this->get_or_create_container(builder, memlet_op.getOutput()))}
                            );
                        } else {
                            access_nodes.insert(
                                {memlet_op.getInput().getImpl(),
                                 &builder.add_access(block, this->get_or_create_container(builder, memlet_op.getInput()))
                                }
                            );
                        }
                        return success();
                    })
                    .Case<YieldOp>([&](YieldOp yield_op) {
                        unsigned int num_operands = op.getNumOperands();
                        assert(num_operands == results.size());
                        for (unsigned int i = 0; i < num_operands; i++) {
                            this->value_map_
                                .insert(results.at(i), this->get_or_create_container(builder, yield_op.getOperand(i)));
                        }
                        return success();
                    })
                    .Default([&](Operation* op) {
                        return op->emitError("Unknown data flow operation. Could not translate.");
                    });
            if (failed(result)) {
                return failure();
            }
        }

        // Create all edges
        for (auto& op : block_op->getRegion().getOps()) {
            LogicalResult result =
                llvm::TypeSwitch<Operation*, LogicalResult>(&op)
                    .Case<TaskletOp>([&](TaskletOp tasklet_op) {
                        std::vector<std::string> inputs;
                        switch (arity(tasklet_op.getCode())) {
                            case 1:
                                inputs.push_back("_in");
                                break;
                            case 2:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                break;
                            case 3:
                                inputs.push_back("_in1");
                                inputs.push_back("_in2");
                                inputs.push_back("_in3");
                                break;
                            default:
                                return failure();
                        }
                        for (unsigned int i = 0, e = tasklet_op.getNumOperands(); i < e; i++) {
                            if (auto memlet_op = dyn_cast_or_null<MemletOp>(tasklet_op.getOperand(i).getDefiningOp())) {
                                builder.add_computational_memlet(
                                    block,
                                    *access_nodes.at(memlet_op.getInput().getImpl()),
                                    *tasklets.at(tasklet_op.getLoc()->getImpl()),
                                    inputs.at(i),
                                    {}
                                );
                            } else {
                                return failure();
                            }
                        }
                        return success();
                    })
                    .Case<MemletOp>([&](MemletOp memlet_op) {
                        if (auto tasklet_op = dyn_cast_or_null<TaskletOp>(memlet_op.getInput().getDefiningOp())) {
                            builder.add_computational_memlet(
                                block,
                                *tasklets.at(tasklet_op.getLoc()->getImpl()),
                                "_out",
                                *access_nodes.at(memlet_op.getOutput().getImpl()),
                                {}
                            );
                        }
                        return success();
                    })
                    .Case<YieldOp>([&](YieldOp yield_op) { return success(); })
                    .Default([&](Operation* op) {
                        return op->emitError("Unknown data flow operation. Could not translate.");
                    });
            if (failed(result)) {
                return failure();
            }
        }

        return success();
    }

    LogicalResult translateReturnOp(Builder& builder, Sequence& parent, ReturnOp* return_op) {
        builder.add_return(parent, this->get_or_create_container(builder, return_op->getOperand()));
        return success();
    }

    LogicalResult emitCode(raw_ostream& os) {
        for (auto& builder : this->builders_) {
            ::sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
            auto instrumentation_plan = ::sdfg::codegen::InstrumentationPlan::none(builder.subject());
            auto arg_capture_plan = ::sdfg::codegen::ArgCapturePlan::none(builder.subject());
            ::sdfg::codegen::CCodeGenerator
                generator(builder.subject(), analysis_manager, *instrumentation_plan, *arg_capture_plan);
            if (!generator.generate()) {
                return failure();
            }
            os << generator.includes().str() << "\n"
               << generator.globals().str() << "\n"
               << generator.function_definition() << " {\n"
               << generator.main().str() << "}\n";
        }
        return success();
    }
};

LogicalResult translateToSDFG(Operation* op, raw_ostream& os) {
    ::sdfg::codegen::register_default_dispatchers();
    ::sdfg::serializer::register_default_serializers();

    SDFGTranslator translator;
    if (failed(translator.translate(op))) {
        return emitError(op->getLoc(), "Could not translate to SDFG");
    }

    if (failed(translator.emitCode(os))) {
        return emitError(op->getLoc(), "Could not generate code");
    }

    return success();
}

void registerToSDFGTranslation() {
    TranslateFromMLIRRegistration registration(
        "mlir-to-sdfg",
        "translate MLIR in SDFG dialect to SDFG",
        [](Operation* op, raw_ostream& os) { return translateToSDFG(op, os); },
        [](mlir::DialectRegistry& registry) { registry.insert<mlir::sdfg::SDFGDialect>(); }
    );
}

} // namespace sdfg
} // namespace mlir
