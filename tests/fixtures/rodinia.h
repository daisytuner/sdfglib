#pragma once

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

inline std::unique_ptr<StructuredSDFG> srad() {
    /**
        for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) {
                Jc = J[i * cols + j];

                // directional derivates
                dN[i * cols + j] = J[iN[i] * cols + j] - Jc;
                dS[i * cols + j] = J[iS[i] * cols + j] - Jc;
                dW[i * cols + j] = J[i * cols + jW[j]] - Jc;
                dE[i * cols + j] = J[i * cols + jE[j]] - Jc;

                G2 = (dN[i * cols + j]*dN[i * cols + j] + dS[i * cols + j]*dS[i * cols + j]
                    + dW[i * cols + j]*dW[i * cols + j] + dE[i * cols + j]*dE[i * cols + j]) / Jc;

                L = (dN[i * cols + j] + dS[i * cols + j] + dW[i * cols + j] + dE[i * cols + j]) /
       Jc;

                // diffusion coefficent (equ 33)
                c[i * cols + j] = L - q0sqr;

                // saturate diffusion coefficent
                if (c[i * cols + j] < 0) {c[i * cols + j] = 0;}
                else if (c[i * cols + j] > 1) {c[i * cols + j] = 1;}
            }
        }
    */

    builder::StructuredSDFGBuilder builder("srad");

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("rows", sym_desc, true);
    builder.add_container("cols", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("iN_i", sym_desc);
    builder.add_container("iS_i", sym_desc);
    builder.add_container("jW_j", sym_desc);
    builder.add_container("jE_j", sym_desc);

    types::Scalar element_desc(types::PrimitiveType::Float);
    builder.add_container("G2", element_desc);
    builder.add_container("L", element_desc);
    builder.add_container("Jc", element_desc);
    builder.add_container("q0sqr", element_desc, true);

    types::Pointer ptr_desc(element_desc);
    builder.add_container("J", ptr_desc, true);
    builder.add_container("dN", ptr_desc, true);
    builder.add_container("dS", ptr_desc, true);
    builder.add_container("dW", ptr_desc, true);
    builder.add_container("dE", ptr_desc, true);
    builder.add_container("c", ptr_desc, true);

    types::Pointer ptr_sym(sym_desc);
    builder.add_container("iN", ptr_sym, true);
    builder.add_container("iS", ptr_sym, true);
    builder.add_container("jW", ptr_sym, true);
    builder.add_container("jE", ptr_sym, true);

    auto& root = builder.subject().root();

    auto& loop_i = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("rows")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& body_loop_i = loop_i.root();

    auto& loop_j = builder.add_for(body_loop_i, symbolic::symbol("j"),
                                   symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("cols")),
                                   symbolic::integer(0),
                                   symbolic::add(symbolic::symbol("j"), symbolic::integer(1)));
    auto& body_loop_j = loop_j.root();

    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& iN_i_node = builder.add_access(block1, "iN_i");
        auto& iN_node = builder.add_access(block1, "iN");
        auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block1, iN_node, "void", tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_memlet(block1, tasklet1, "_out", iN_i_node, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& iS_i_node = builder.add_access(block2, "iS_i");
        auto& iS_node = builder.add_access(block2, "iS");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block2, iS_node, "void", tasklet2, "_in", {symbolic::symbol("i")});
        builder.add_memlet(block2, tasklet2, "_out", iS_i_node, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& jW_j_node = builder.add_access(block3, "jW_j");
        auto& jW_node = builder.add_access(block3, "jW");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block3, jW_node, "void", tasklet3, "_in", {symbolic::symbol("j")});
        builder.add_memlet(block3, tasklet3, "_out", jW_j_node, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& jE_j_node = builder.add_access(block4, "jE_j");
        auto& jE_node = builder.add_access(block4, "jE");
        auto& tasklet4 = builder.add_tasklet(block4, data_flow::TaskletCode::assign,
                                             {"_out", sym_desc}, {{"_in", sym_desc}});
        builder.add_memlet(block4, jE_node, "void", tasklet4, "_in", {symbolic::symbol("j")});
        builder.add_memlet(block4, tasklet4, "_out", jE_j_node, "void", {});
    }

    {
        auto& block = builder.add_block(body_loop_j);
        auto& Jc = builder.add_access(block, "Jc");
        auto& J = builder.add_access(block, "J");
        auto& tasklet11 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                              {"_out", element_desc}, {{"_in", element_desc}});
        builder.add_memlet(
            block, J, "void", tasklet11, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block, tasklet11, "_out", Jc, "void", {});
    }

    // directional derivates
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& dN = builder.add_access(block1, "dN");
        auto& J_node1 = builder.add_access(block1, "J");
        auto& Jc_node1 = builder.add_access(block1, "Jc");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, J_node1, "void", tasklet1, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("iN_i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, Jc_node1, "void", tasklet1, "_in2", {});
        builder.add_memlet(
            block1, tasklet1, "_out", dN, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block2 = builder.add_block(body_loop_j);
        auto& dS = builder.add_access(block2, "dS");
        auto& J_node2 = builder.add_access(block2, "J");
        auto& Jc_node2 = builder.add_access(block2, "Jc");
        auto& tasklet2 =
            builder.add_tasklet(block2, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block2, J_node2, "void", tasklet2, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("iS_i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, Jc_node2, "void", tasklet2, "_in2", {});
        builder.add_memlet(
            block2, tasklet2, "_out", dS, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block3 = builder.add_block(body_loop_j);
        auto& dW = builder.add_access(block3, "dW");
        auto& J_node3 = builder.add_access(block3, "J");
        auto& Jc_node3 = builder.add_access(block3, "Jc");
        auto& tasklet3 =
            builder.add_tasklet(block3, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block3, J_node3, "void", tasklet3, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("jW_j"))});
        builder.add_memlet(block3, Jc_node3, "void", tasklet3, "_in2", {});
        builder.add_memlet(
            block3, tasklet3, "_out", dW, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block4 = builder.add_block(body_loop_j);
        auto& dE = builder.add_access(block4, "dE");
        auto& J_node4 = builder.add_access(block4, "J");
        auto& Jc_node4 = builder.add_access(block4, "Jc");
        auto& tasklet4 =
            builder.add_tasklet(block4, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block4, J_node4, "void", tasklet4, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("jE_j"))});
        builder.add_memlet(block4, Jc_node4, "void", tasklet4, "_in2", {});
        builder.add_memlet(
            block4, tasklet4, "_out", dE, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    // G2
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& G2_node1 = builder.add_access(block1, "G2");
        auto& dN_node1 = builder.add_access(block1, "dN");
        auto& tasklet2 =
            builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet2, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet2, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet2, "_out", G2_node1, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& G2_node2 = builder.add_access(block2, "G2");
        auto& G2_node2_out = builder.add_access(block2, "G2");
        auto& dS_node2 = builder.add_access(block2, "dS");
        auto& tasklet3 = builder.add_tasklet(
            block2, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block2, dS_node2, "void", tasklet3, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block2, dS_node2, "void", tasklet3, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, G2_node2, "void", tasklet3, "_in3", {symbolic::integer(0)});
        builder.add_memlet(block2, tasklet3, "_out", G2_node2_out, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& G2_node3 = builder.add_access(block3, "G2");
        auto& G2_node3_out = builder.add_access(block3, "G2");
        auto& dW_node3 = builder.add_access(block3, "dW");
        auto& tasklet4 = builder.add_tasklet(
            block3, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block3, dW_node3, "void", tasklet4, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block3, dW_node3, "void", tasklet4, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block3, G2_node3, "void", tasklet4, "_in3", {symbolic::integer(0)});
        builder.add_memlet(block3, tasklet4, "_out", G2_node3_out, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& G2_node4 = builder.add_access(block4, "G2");
        auto& G2_node4_out = builder.add_access(block4, "G2");
        auto& dE_node4 = builder.add_access(block4, "dE");
        auto& tasklet5 = builder.add_tasklet(
            block4, data_flow::TaskletCode::fma, {"_out", element_desc},
            {{"_in1", element_desc}, {"_in2", element_desc}, {"_in3", element_desc}});
        builder.add_memlet(
            block4, dE_node4, "void", tasklet5, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block4, dE_node4, "void", tasklet5, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block4, G2_node4, "void", tasklet5, "_in3", {symbolic::integer(0)});
        builder.add_memlet(block4, tasklet5, "_out", G2_node4_out, "void", {});

        auto& block5 = builder.add_block(body_loop_j);
        auto& G2_node5 = builder.add_access(block5, "G2");
        auto& G2_node5_out = builder.add_access(block5, "G2");
        auto& Jc_node5 = builder.add_access(block5, "Jc");
        auto& tasklet6 =
            builder.add_tasklet(block5, data_flow::TaskletCode::div, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block5, G2_node5, "void", tasklet6, "_in1", {symbolic::integer(0)});
        builder.add_memlet(block5, Jc_node5, "void", tasklet6, "_in2", {});
        builder.add_memlet(block5, tasklet6, "_out", G2_node5_out, "void", {});
    }

    // L
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& L_node1 = builder.add_access(block1, "L");
        auto& dN_node1 = builder.add_access(block1, "dN");
        auto& dS_node1 = builder.add_access(block1, "dS");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(
            block1, dN_node1, "void", tasklet1, "_in1",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(
            block1, dS_node1, "void", tasklet1, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet1, "_out", L_node1, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& L_node2 = builder.add_access(block2, "L");
        auto& L_node2_out = builder.add_access(block2, "L");
        auto& dW_node2 = builder.add_access(block2, "dW");
        auto& tasklet2 =
            builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block2, L_node2, "void", tasklet2, "_in1", {symbolic::integer(0)});
        builder.add_memlet(
            block2, dW_node2, "void", tasklet2, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block2, tasklet2, "_out", L_node2_out, "void", {});

        auto& block3 = builder.add_block(body_loop_j);
        auto& L_node3 = builder.add_access(block3, "L");
        auto& L_node3_out = builder.add_access(block3, "L");
        auto& dE_node3 = builder.add_access(block3, "dE");
        auto& tasklet3 =
            builder.add_tasklet(block3, data_flow::TaskletCode::add, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block3, L_node3, "void", tasklet3, "_in1", {symbolic::integer(0)});
        builder.add_memlet(
            block3, dE_node3, "void", tasklet3, "_in2",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block3, tasklet3, "_out", L_node3_out, "void", {});

        auto& block4 = builder.add_block(body_loop_j);
        auto& L_node4 = builder.add_access(block4, "L");
        auto& L_node4_out = builder.add_access(block4, "L");
        auto& Jc_node4 = builder.add_access(block4, "Jc");
        auto& tasklet4 =
            builder.add_tasklet(block4, data_flow::TaskletCode::div, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block4, L_node4, "void", tasklet4, "_in1", {symbolic::integer(0)});
        builder.add_memlet(block4, Jc_node4, "void", tasklet4, "_in2", {});
        builder.add_memlet(block4, tasklet4, "_out", L_node4_out, "void", {});
    }

    // diffusion coefficent (equ 33)
    {
        auto& block1 = builder.add_block(body_loop_j);
        auto& c_node1 = builder.add_access(block1, "c");
        auto& L_node1 = builder.add_access(block1, "L");
        auto& q0sqr_node1 = builder.add_access(block1, "q0sqr");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::sub, {"_out", element_desc},
                                {{"_in1", element_desc}, {"_in2", element_desc}});
        builder.add_memlet(block1, L_node1, "void", tasklet1, "_in1", {symbolic::integer(0)});
        builder.add_memlet(block1, q0sqr_node1, "void", tasklet1, "_in2", {});
        builder.add_memlet(
            block1, tasklet1, "_out", c_node1, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    // saturate diffusion coefficent
    // if (c[i * cols + j] < 0) {c[i * cols + j] = 0;}
    // else if (c[i * cols + j] > 1) {c[i * cols + j] = 1;}
    {
        types::Scalar bool_desc(types::PrimitiveType::Bool);
        builder.add_container("tmp_0", bool_desc);
        builder.add_container("tmp_1", bool_desc);

        auto& block0 = builder.add_block(body_loop_j);
        auto& c_node0 = builder.add_access(block0, "c");
        auto& tmp_0_node = builder.add_access(block0, "tmp_0");
        auto& tasklet0 =
            builder.add_tasklet(block0, data_flow::TaskletCode::olt, {"_out", element_desc},
                                {{"_in", element_desc}, {"0", element_desc}});
        builder.add_memlet(
            block0, c_node0, "void", tasklet0, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block0, tasklet0, "_out", tmp_0_node, "void", {});

        auto& block1 = builder.add_block(body_loop_j);
        auto& c_node1 = builder.add_access(block1, "c");
        auto& tmp_1_node = builder.add_access(block1, "tmp_1");
        auto& tasklet1 =
            builder.add_tasklet(block1, data_flow::TaskletCode::ogt, {"_out", element_desc},
                                {{"_in", element_desc}, {"1", element_desc}});
        builder.add_memlet(
            block1, c_node1, "void", tasklet1, "_in",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
        builder.add_memlet(block1, tasklet1, "_out", tmp_1_node, "void", {});

        auto& block2 = builder.add_block(body_loop_j);
        auto& c_node2 = builder.add_access(block2, "c");
        auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                             {"_out", element_desc}, {{"0", element_desc}});
        tasklet2.condition() = symbolic::Eq(symbolic::symbol("tmp_0"), symbolic::__true__());
        builder.add_memlet(
            block2, tasklet2, "_out", c_node2, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});

        auto& block3 = builder.add_block(body_loop_j);
        auto& c_node3 = builder.add_access(block3, "c");
        auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                             {"_out", element_desc}, {{"1", element_desc}});
        tasklet3.condition() = symbolic::Eq(symbolic::symbol("tmp_1"), symbolic::__true__());
        builder.add_memlet(
            block3, tasklet3, "_out", c_node3, "void",
            {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::symbol("cols")),
                           symbolic::symbol("j"))});
    }

    return builder.move();
};
