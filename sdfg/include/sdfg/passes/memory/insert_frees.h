#pragma once

#include "sdfg/analysis/users.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class InsertFrees : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    std::unordered_set<std::string> freed_containers_;

    void insert_free_after(const std::string& container, analysis::User* last_use);

public:
    InsertFrees(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "InsertFrees"; };

    bool visit() override;
};

typedef VisitorPass<InsertFrees> InsertFreesPass;

} // namespace passes
} // namespace sdfg
