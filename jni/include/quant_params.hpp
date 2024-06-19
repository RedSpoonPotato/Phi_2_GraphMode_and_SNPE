#ifndef QUANT_PARAMS_H_
#define QUANT_PARAMS_H_

#include <vector>
#include <map>
#include "Util.hpp"
#include "main_macros.h"

/* directions: 
    use a python script to generate a text file that defines the params
*/

// dummy
std::vector<std::map<std::string, quantParams>> quantizationParams() {
    std::vector<std::map<std::string, quantParams>> params(DECODERS);

    std::map<std::string, quantParams> default_values = {
        {"hidden_states",               {0,0}},
        {"fc1_out",                     {0,0}},
        {"gelu_fc1_out",                {0,0}},
        {"query_states",                {0,0}},
        {"key_states",                  {0,0}},
        {"attn_weights",                {0,0}},
        {"p4_1_out",                    {0,0}},
        {"feed_forward_hidden_states",  {0,0}}
    };

    for (size_t i = 0; i < DECODERS; i++) {
        params[i] = default_values;
    }

    return params;
}

#endif