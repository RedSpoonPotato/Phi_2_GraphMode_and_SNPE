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
        {"decoder_layernorm",           {0,0}},
        {"gelu_out",                    {0,0}},
        {"p3_attn_weights",             {0,0}},
        {"fc1_out",                     {0,0}},
        {"query_states",                {0,0}},
        {"key_states",                  {0,0}},
        {"p4_1_out",                    {0,0}},
        {"p4_ff",                       {0,0}}
    };

    for (size_t i = 0; i < DECODERS; i++) {
        params[i] = default_values;
    }

    return params;
}


std::vector<std::map<std::string, quantParams>> parseFile(const std::string& filename) {
    std::vector<std::map<std::string, quantParams>> result;
    std::ifstream infile(filename);

    if (!infile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return result;
    }

    std::string line;
    std::map<std::string, quantParams> currentMap;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string key;
        char delimiter;
        int stepEquivalentTo0;
        float quantizedStepSize;

        if (iss >> key >> delimiter >> stepEquivalentTo0 >> delimiter >> quantizedStepSize) {
            currentMap[key] = quantParams{static_cast<unsigned char>(stepEquivalentTo0), quantizedStepSize};
        } else {
            // Assuming a blank line or an invalid line indicates the end of a map entry
            if (!currentMap.empty()) {
                result.push_back(currentMap);
                currentMap.clear();
            }
        }
    }

    // Don't forget to add the last map if it's non-empty
    if (!currentMap.empty()) {
        result.push_back(currentMap);
    }

    return result;
}

// example for parseFile()
// int main() {
//     std::string filename = "data.txt"; // Replace with your file name
//     auto data = parseFile(filename);

//     // Example of how to access and print the parsed data
//     for (const auto& mapEntry : data) {
//         for (const auto& pair : mapEntry) {
//             std::cout << "Key: " << pair.first 
//                       << ", stepEquivalentTo0: " << static_cast<int>(pair.second.stepEquivalentTo0) 
//                       << ", quantizedStepSize: " << pair.second.quantizedStepSize << std::endl;
//         }
//         std::cout << "-----" << std::endl;
//     }

//     return 0;
// }

#endif