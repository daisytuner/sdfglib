#include "sdfg/analysis/type_analysis.h"

#include "sdfg/analysis/users.h"
#include "sdfg/exceptions.h"

namespace sdfg {
namespace analysis {

void TypeAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    std::vector<std::string> containers;
    for (auto container : this->sdfg_.containers()) {
        auto& contType = sdfg_.type(container);
        if (contType.type_id() == sdfg::types::TypeID::Pointer) {
            auto pointer_type = static_cast<const sdfg::types::Pointer*>(&sdfg_.type(container));
            if (!pointer_type->has_pointee_type()) {
                containers.push_back(container);
            }
        } else if (contType.type_id() == sdfg::types::TypeID::Structure) { // we model std::shared_ptr as pointer-like
                                                                           // structure
            // if we find uses of it AS a pointer those can also be used
            auto struct_type = static_cast<const sdfg::types::Structure&>(contType);
            if (struct_type.is_pointer_like()) {
                containers.push_back(container);
            }
        }
    }

    auto& users = analysis_manager.get<Users>();
    for (auto container : containers) {
        // iterate over writes
        for (auto user : users.writes(container)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                continue;
            }

            for (auto& memlet : access_node->get_parent().in_edges(*access_node)) {
                auto base_type = &memlet.base_type();
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!pointer_type->has_pointee_type()) {
                        continue;
                    }
                } else if (base_type->type_id() == types::TypeID::Structure) { // pointer-likes may also be used raw.
                                                                               // Cannot glean anything from those
                                                                               // accesses directly
                    continue;
                }

                if (memlet.type() == data_flow::MemletType::Dereference_Src) {
                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        base_type = &pointer_type->pointee_type();

                    } /* else if (base_type->type_id() == types::TypeID::Array) {
                        auto array_type = dynamic_cast<const types::Array*>(base_type);
                        base_type = &array_type->element_type();
                    } */

                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        if (!inner_pointer_type->has_pointee_type()) {
                            continue;
                        }
                    }
                }

                if (this->type_map_.find(container) == this->type_map_.end()) {
                    this->type_map_.insert({container, base_type});
                    continue;
                }
            }
        }

        // iterate over reads
        for (auto user : users.reads(container)) {
            // Pointers may be used in symbolic conditions
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                continue;
            }

            for (auto& memlet : access_node->get_parent().out_edges(*access_node)) {
                auto base_type = &memlet.base_type();
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!pointer_type->has_pointee_type()) {
                        continue;
                    }
                } else if (base_type->type_id() == types::TypeID::Structure) { // pointer-likes may also be used raw.
                                                                               // Cannot glean anything from those
                                                                               // accesses directly
                    continue;
                }

                if (memlet.type() == data_flow::MemletType::Dereference_Dst) {
                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        base_type = &pointer_type->pointee_type();

                    } /* else if (base_type->type_id() == types::TypeID::Array) {
                        auto array_type = dynamic_cast<const types::Array*>(base_type);
                        base_type = &array_type->element_type();
                    } */

                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        if (!inner_pointer_type->has_pointee_type()) {
                            continue;
                        }
                    }
                }
                if (this->type_map_.find(container) == this->type_map_.end()) {
                    this->type_map_.insert({container, base_type});
                    continue;
                }
            }
        }

        // iterate over views
        for (auto user : users.views(container)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                continue;
            }
            for (auto& memlet : access_node->get_parent().out_edges(*access_node)) {
                auto base_type = &memlet.base_type();
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!pointer_type->has_pointee_type()) {
                        continue;
                    }
                } else if (base_type->type_id() == types::TypeID::Structure) { // pointer-likes may also be used raw.
                                                                               // Cannot glean anything from those
                                                                               // accesses directly
                    continue;
                }

                if (memlet.type() == data_flow::MemletType::Dereference_Dst) {
                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        base_type = &pointer_type->pointee_type();

                    } /* else if (base_type->type_id() == types::TypeID::Array) {
                        auto array_type = dynamic_cast<const types::Array*>(base_type);
                        base_type = &array_type->element_type();
                    } */

                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        if (!inner_pointer_type->has_pointee_type()) {
                            continue;
                        }
                    }
                }
                if (this->type_map_.find(container) == this->type_map_.end()) {
                    this->type_map_.insert({container, base_type});
                    continue;
                }
            }
        }

        // iterate over moves
        for (auto user : users.moves(container)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                continue;
            }
            for (auto& memlet : access_node->get_parent().in_edges(*access_node)) {
                auto base_type = &memlet.base_type();
                if (base_type->type_id() == types::TypeID::Pointer) {
                    auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                    if (!pointer_type->has_pointee_type()) {
                        continue;
                    }
                } else if (base_type->type_id() == types::TypeID::Structure) {
                    continue;
                }

                if (memlet.type() == data_flow::MemletType::Dereference_Src) {
                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        base_type = &pointer_type->pointee_type();

                    } /* else if (base_type->type_id() == types::TypeID::Array) {
                        auto array_type = dynamic_cast<const types::Array*>(base_type);
                        base_type = &array_type->element_type();
                    } */

                    if (base_type->type_id() == types::TypeID::Pointer) {
                        auto inner_pointer_type = dynamic_cast<const types::Pointer*>(base_type);
                        if (!inner_pointer_type->has_pointee_type()) {
                            continue;
                        }
                    }
                }
                if (this->type_map_.find(container) == this->type_map_.end()) {
                    this->type_map_.insert({container, base_type});
                    continue;
                }
            }
        }
    }
}

TypeAnalysis::TypeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

const sdfg::types::IType* TypeAnalysis::get_outer_type(const std::string& container) const {
    auto& contType = sdfg_.type(container);
    if (contType.type_id() == sdfg::types::TypeID::Pointer) {
        auto pointer_type = static_cast<const sdfg::types::Pointer*>(&contType);
        if (pointer_type->has_pointee_type()) {
            return pointer_type;
        }
    } else if (contType.type_id() == sdfg::types::TypeID::Structure &&
               static_cast<const types::Structure&>(contType).is_pointer_like()) {
    } else {
        return &contType;
    }

    auto it = type_map_.find(container);
    if (it != type_map_.end()) {
        return it->second;
    }
    return nullptr;
}

} // namespace analysis
} // namespace sdfg
