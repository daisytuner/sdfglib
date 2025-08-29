#include "sdfg/element.h"

namespace sdfg {

Element::Element(size_t element_id, const DebugInfoRegion& debug_info)
    : element_id_(element_id), debug_info_(debug_info) {};

size_t Element::element_id() const { return this->element_id_; };

const DebugInfoRegion& Element::debug_info() const { return this->debug_info_; };

void Element::set_debug_info(const DebugInfoRegion& debug_info) { this->debug_info_ = debug_info; }

} // namespace sdfg
