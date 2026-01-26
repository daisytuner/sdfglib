#include <boost/program_options.hpp>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
// #include <sdfg/cuda/plugin.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_sdfg.h>
#include "sdfg/transformations/rpc_transfer_tuning_transform.h"

using json = nlohmann::json;
namespace po = boost::program_options;
using namespace sdfg;

int main(int argc, char* argv[]) {
    std::filesystem::path sdfg_path;
    std::filesystem::path output_prefix;
    std::string category;
    std::string target;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("output,o", po::value<std::filesystem::path>(&output_prefix)->default_value("./"), "prefix of generated files")
        ("category,c", po::value<std::string>(&category)->default_value("server"), "category to tune for")
        ("target", po::value<std::string>(&target)->default_value("SEQUENTIAL"), "target device");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Register default dispatchers
    codegen::register_default_dispatchers();
    // sdfg::serializer::register_default_serializers();
    //  sdfg::cuda::register_cuda_plugin();

    // Create sdfg

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

    auto& root = builder->subject().root();
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::integer(64));
    types::Pointer desc_2(desc_1);

    builder->add_container("A", desc_2, true);
    builder->add_container("B", desc_2, true);
    builder->add_container("C", desc_2, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder->add_container("K", sym_desc, true);
    builder->add_container("N", sym_desc, true);
    builder->add_container("M", sym_desc, true);
    builder->add_container("i", sym_desc);
    builder->add_container("j", sym_desc);
    builder->add_container("k", sym_desc);

    // Define loop 1
    auto bound = symbolic::integer(64);
    auto indvar = symbolic::symbol("i");

    auto& loop = builder->add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::integer(64);
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder->add_for(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), bound_2),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& body_2 = loop_2.root();

    // Define loop 3
    auto bound_3 = symbolic::integer(64);
    auto indvar_3 = symbolic::symbol("k");

    auto& loop_3 = builder->add_map(
        body_2,
        indvar_3,
        symbolic::Lt(symbolic::symbol("k"), bound_3),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& body_3 = loop_3.root();

    // Add computation
    auto& block = builder->add_block(body_3);
    auto& a_in = builder->add_access(block, "A");
    auto& b_in = builder->add_access(block, "B");
    auto& c_in = builder->add_access(block, "C");
    auto& c_out = builder->add_access(block, "C");

    {
        auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
        builder->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("k")});
    }

    auto sdfg_initial = builder->subject().clone();

    // Transfer tuning replayer

    sdfg::analysis::AnalysisManager analysis_manager(builder->subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();

    size_t loopnest_index = 0;
    for (auto loopnest : outer_loops) {
        sdfg::transformations::RPCTransferTuningTransform
            transfer_tuning(target, category, &builder->subject(), loop_analysis.loop_info(loopnest));

        if (!transfer_tuning.can_be_applied(*builder, analysis_manager)) {
            continue;
        }
        transfer_tuning.apply(*builder, analysis_manager);

        analysis_manager.invalidate_all();

        loopnest_index++;
    }

    // Generate code for initial sdfg

    auto localPrefix = output_prefix.filename().string();
    auto parent = output_prefix.parent_path();
    auto init_source_path = parent / (localPrefix + sdfg_initial->name() + "_init.cpp");
    auto init_header_path = parent / (localPrefix + sdfg_initial->name() + "_init.h");

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg_initial);
    auto arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(*sdfg_initial);
    analysis::AnalysisManager analysis_manager_initial(*sdfg_initial);
    codegen::CPPCodeGenerator
        code_generator(*sdfg_initial, analysis_manager_initial, *instrumentation_plan, *arg_capture_plan);
    bool success = code_generator.generate();
    if (!success) {
        std::cerr << "Code generation for initial sdfg failed" << std::endl;
        return 1;
    }

    // generate code for tuned sdfg

    auto sdfg_final = builder->move();

    auto source_path = parent / (localPrefix + sdfg_final->name() + "_opt.cpp");
    auto header_path = parent / (localPrefix + sdfg_final->name() + "_opt.h");

    auto instrumentation_plan_opt = codegen::InstrumentationPlan::none(*sdfg_final);
    auto arg_capture_plan_opt = sdfg::codegen::ArgCapturePlan::none(*sdfg_final);
    analysis::AnalysisManager analysis_manager_opt(*sdfg_final);
    codegen::CPPCodeGenerator
        code_generator_opt(*sdfg_final, analysis_manager_opt, *instrumentation_plan_opt, *arg_capture_plan_opt);

    success = code_generator_opt.generate();
    if (!success) {
        std::cerr << "Code generation failed" << std::endl;
        return 1;
    }

    return 0;
};
