#include "sdfg/analysis/type_analysis.h"

#include "sdfg/analysis/users.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/exceptions.h"

namespace sdfg {
namespace analysis {

void TypeAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    for (auto container : this->sdfg_.containers()) {
        if (sdfg_.type(container).type_id() == sdfg::types::TypeID::Pointer) {
            auto pointer_type = static_cast<const sdfg::types::Pointer*>(&sdfg_.type(container));
            if (!pointer_type->has_pointee_type()) {
                continue;
            }
        }
        type_map_.insert({container, &sdfg_.type(container)});
    }

    auto& users = analysis_manager.get<Users>();
    codegen::CLanguageExtension c_lang;

    // iterate over writes
    for (auto user : users.writes()) {
        auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
        if (access_node == nullptr) {
            continue;
        }

        for (auto& memlet : user->parent()->in_edges(*access_node)) {
            auto base_type = &memlet.base_type();
            if (base_type->type_id() == types::TypeID::Pointer) {
                auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                if (!pointer_type->has_pointee_type()) {
                    continue;
                }
            }

            if (memlet.type() == data_flow::MemletType::Dereference_Src) {
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    base_type = &pointer_type->pointee_type();

                } else if (base_type->type_id() == types::TypeID::Array) {
                    auto array_type = dynamic_cast<const types::Array*>(base_type);
                    base_type = &array_type->element_type();
                }

                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!inner_pointer_type->has_pointee_type()) {
                        continue;
                    }
                }
            }

            if (this->type_map_.find(user->container()) == this->type_map_.end()) {
                this->type_map_.insert({user->container(), base_type});
                continue;
            }

            if (*this->type_map_.at(user->container()) != *base_type) {
                throw InvalidSDFGException(
                    "Type mismatch for container '" + user->container() + "': expected " +
                    c_lang.declaration("", *this->type_map_.at(user->container())) + ", got " +
                    c_lang.declaration("", *base_type)
                );
            }
        }
    }

    // iterate over reads
    for (auto user : users.reads()) {
        // Pointers may be used in symbolic conditions
        auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
        if (access_node == nullptr) {
            continue;
        }

        for (auto& memlet : user->parent()->out_edges(*access_node)) {
            auto base_type = &memlet.base_type();
            if (base_type->type_id() == types::TypeID::Pointer) {
                auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                if (!pointer_type->has_pointee_type()) {
                    continue;
                }
            }

            if (memlet.type() == data_flow::MemletType::Dereference_Dst) {
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    base_type = &pointer_type->pointee_type();

                } else if (base_type->type_id() == types::TypeID::Array) {
                    auto array_type = dynamic_cast<const types::Array*>(base_type);
                    base_type = &array_type->element_type();
                }

                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!inner_pointer_type->has_pointee_type()) {
                        continue;
                    }
                }
            }
            if (this->type_map_.find(user->container()) == this->type_map_.end()) {
                this->type_map_.insert({user->container(), base_type});
                continue;
            }

            if (*this->type_map_.at(user->container()) != *base_type) {
                throw InvalidSDFGException(
                    "Type mismatch for container '" + user->container() + "': expected " +
                    c_lang.declaration("", *this->type_map_.at(user->container())) + ", got " +
                    c_lang.declaration("", *base_type)
                );
            }
        }
    }

    // iterate over views
    for (auto user : users.views()) {
        auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
        if (access_node == nullptr) {
            continue;
        }
        for (auto& memlet : user->parent()->out_edges(*access_node)) {
            auto base_type = &memlet.base_type();
            if (base_type->type_id() == types::TypeID::Pointer) {
                auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                if (!pointer_type->has_pointee_type()) {
                    continue;
                }
            }

            if (memlet.type() == data_flow::MemletType::Dereference_Dst) {
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    base_type = &pointer_type->pointee_type();

                } else if (base_type->type_id() == types::TypeID::Array) {
                    auto array_type = dynamic_cast<const types::Array*>(base_type);
                    base_type = &array_type->element_type();
                }

                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!inner_pointer_type->has_pointee_type()) {
                        continue;
                    }
                }
            }
            if (this->type_map_.find(user->container()) == this->type_map_.end()) {
                this->type_map_.insert({user->container(), base_type});
                continue;
            }
        }
    }

    // iterate over moves
    for (auto user : users.moves()) {
        auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
        if (access_node == nullptr) {
            continue;
        }
        for (auto& memlet : user->parent()->in_edges(*access_node)) {
            auto base_type = &memlet.base_type();
            if (base_type->type_id() == types::TypeID::Pointer) {
                auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                if (!pointer_type->has_pointee_type()) {
                    continue;
                }
            }

            if (memlet.type() == data_flow::MemletType::Dereference_Src) {
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    base_type = &pointer_type->pointee_type();

                } else if (base_type->type_id() == types::TypeID::Array) {
                    auto array_type = dynamic_cast<const types::Array*>(base_type);
                    base_type = &array_type->element_type();
                }

                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!inner_pointer_type->has_pointee_type()) {
                        continue;
                    }
                }
            }
            if (this->type_map_.find(user->container()) == this->type_map_.end()) {
                this->type_map_.insert({user->container(), base_type});
                continue;
            }
        }
    }
}

TypeAnalysis::TypeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

const sdfg::types::IType* TypeAnalysis::get_outer_type(const std::string& container) const {
    auto it = type_map_.find(container);
    if (it != type_map_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace analysis
} // namespace sdfg
