//
//  base64 encoding and decoding with C++.
//  See impl. for further notes
//

#ifndef BASE64_H_C0CE2A47_D10E_42C9_A27C_C883944E704A
#define BASE64_H_C0CE2A47_D10E_42C9_A27C_C883944E704A

#include <cstdint>
#include <string>
#include <vector>

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);

std::vector<uint8_t> base64_decode(std::string const& encoded_string);

#endif /* BASE64_H_C0CE2A47_D10E_42C9_A27C_C883944E704A */
