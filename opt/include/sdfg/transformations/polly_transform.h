//===- polly/ScheduleOptimizer.h - The Schedule Optimizer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is based on the original source files which were modified for sdfglib.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sdfg/analysis/scop_analysis.h>
#include <sdfg/transformations/transformation.h>

namespace sdfg {
namespace transformations {

class PollyTransform : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

    std::unique_ptr<analysis::Scop> scop_;

    std::unique_ptr<analysis::Dependences> dependences_;

    bool tile_;

    bool applied_ = false;

public:
    PollyTransform(structured_control_flow::StructuredLoop& loop, bool tile = true);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static PollyTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
