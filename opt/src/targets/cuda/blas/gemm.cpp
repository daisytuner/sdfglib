#include "sdfg/targets/cuda/blas/gemm.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/cuda/blas/utils.h"

namespace sdfg::cuda::blas {

GEMMNodeDispatcher_CUBLASWithTransfers::GEMMNodeDispatcher_CUBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_CUBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    globals_stream << "#include <cuda.h>" << std::endl;
    globals_stream << "#include <cublas_v2.h>" << std::endl;

    std::string type, type2;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            type2 = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            type2 = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS GEMM node");
    }

    std::string size_A = this->language_extension_.expression(symbolic::mul(gemm_node.m(), gemm_node.k())) +
                         " * sizeof(" + type + ")";

    std::string size_B = this->language_extension_.expression(symbolic::mul(gemm_node.k(), gemm_node.n())) +
                         " * sizeof(" + type + ")";

    std::string size_C = this->language_extension_.expression(symbolic::mul(gemm_node.m(), gemm_node.n())) +
                         " * sizeof(" + type + ")";

    stream << type << " *dA, *dB, *dC;" << std::endl;

    stream << "cudaMalloc((void**) &dA, " << size_A << ");" << std::endl;
    stream << "cudaMalloc((void**) &dB, " << size_B << ");" << std::endl;
    stream << "cudaMalloc((void**) &dC, " << size_C << ");" << std::endl;

    stream << "cudaMemcpy(dA, __A, " << size_A << ", cudaMemcpyHostToDevice);" << std::endl;
    stream << "cudaMemcpy(dB, __B, " << size_B << ", cudaMemcpyHostToDevice);" << std::endl;

    create_blas_handle(stream, this->language_extension_);
    stream << "cublasStatus_t err;" << std::endl;

    bool invert_transpose = false;
    auto first_dim = gemm_node.m();
    auto second_dim = gemm_node.n();
    if (gemm_node.layout() == sdfg::math::blas::BLAS_Layout::RowMajor) {
        invert_transpose = true;
        // first_dim = gemm_node.n();
        // second_dim = gemm_node.m();
    }

    auto trans_a = gemm_node.trans_a();
    auto trans_b = gemm_node.trans_b();
    if (invert_transpose) {
        trans_a = (trans_a == sdfg::math::blas::BLAS_Transpose::No) ? sdfg::math::blas::BLAS_Transpose::Trans
                                                                    : sdfg::math::blas::BLAS_Transpose::No;
        trans_b = (trans_b == sdfg::math::blas::BLAS_Transpose::No) ? sdfg::math::blas::BLAS_Transpose::Trans
                                                                    : sdfg::math::blas::BLAS_Transpose::No;
    }

    std::string trans_a_str = (trans_a == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";
    std::string trans_b_str = (trans_b == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";

    stream << "err = cublas" << type2 << "gemm(handle, " << trans_a_str << ", " << trans_b_str << ", "
           << this->language_extension_.expression(first_dim) << ", "
           << this->language_extension_.expression(second_dim) << ", "
           << this->language_extension_.expression(gemm_node.k()) << ", "
           << "&__alpha, "
           << "dA, " << this->language_extension_.expression(gemm_node.lda()) << ", "
           << "dB, " << this->language_extension_.expression(gemm_node.ldb()) << ", "
           << "&__beta, "
           << "dC, " << this->language_extension_.expression(gemm_node.k()) << ");" << std::endl;
    cublas_error_checking(stream, this->language_extension_, "err");

    stream << "cudaMemcpy(__C, dC, " << size_C << ", cudaMemcpyDeviceToHost);" << std::endl;

    stream << "cudaFree(dA);" << std::endl;
    stream << "cudaFree(dB);" << std::endl;
    stream << "cudaFree(dC);" << std::endl;

    destroy_blas_handle(stream, this->language_extension_);
}

GEMMNodeDispatcher_CUBLASWithoutTransfers::GEMMNodeDispatcher_CUBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_CUBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    globals_stream << "#include <cuda.h>" << std::endl;
    globals_stream << "#include <cublas_v2.h>" << std::endl;

    std::string type;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS GEMM node");
    }

    create_blas_handle(stream, this->language_extension_);
    stream << "cublasStatus_t err;" << std::endl;

    bool invert_transpose = false;
    auto first_dim = gemm_node.m();
    auto second_dim = gemm_node.n();
    if (gemm_node.layout() == sdfg::math::blas::BLAS_Layout::RowMajor) {
        invert_transpose = true;
        first_dim = gemm_node.n();
        second_dim = gemm_node.m();
    }

    auto trans_a = gemm_node.trans_a();
    auto trans_b = gemm_node.trans_b();
    if (invert_transpose) {
        trans_a = (trans_a == sdfg::math::blas::BLAS_Transpose::No) ? sdfg::math::blas::BLAS_Transpose::Trans
                                                                    : sdfg::math::blas::BLAS_Transpose::No;
        trans_b = (trans_b == sdfg::math::blas::BLAS_Transpose::No) ? sdfg::math::blas::BLAS_Transpose::Trans
                                                                    : sdfg::math::blas::BLAS_Transpose::No;
    }

    std::string trans_a_str = (trans_a == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";
    std::string trans_b_str = (trans_b == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";

    stream << "err = cublas" << type << "gemm(handle, " << trans_a_str << ", " << trans_b_str << ", "
           << this->language_extension_.expression(first_dim) << ", "
           << this->language_extension_.expression(second_dim) << ", "
           << this->language_extension_.expression(gemm_node.k()) << ", "
           << "&__alpha, "
           << "__A, " << this->language_extension_.expression(gemm_node.lda()) << ", "
           << "__B, " << this->language_extension_.expression(gemm_node.ldb()) << ", "
           << "&__beta, "
           << "__C, " << this->language_extension_.expression(gemm_node.ldc()) << ");" << std::endl;
    cublas_error_checking(stream, this->language_extension_, "err");

    destroy_blas_handle(stream, this->language_extension_);
}

} // namespace sdfg::cuda::blas
