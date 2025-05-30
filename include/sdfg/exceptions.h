#pragma once

#include <exception>
#include <string>

namespace sdfg {

class InvalidSDFGException : public std::exception {
   private:
    std::string message_;

   public:
    InvalidSDFGException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

}  // namespace sdfg
