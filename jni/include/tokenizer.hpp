#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <vector>
#include <string>

// replace later
void tokenize_generate(const std::string& input_txt, std::vector<uint32_t>& input_ids) {
    input_ids = std::vector<uint32_t> {
        2061, 318, 534, 4004, 3124, 30, 13, 11517, 318, 2266, 13
    };
}

// replace later
void tokenize_decode(const std::vector<uint32_t>& input_ids, std::string& input_txt) {
    input_txt = std::string();
    for (auto& id: input_ids) {
        input_txt += std::to_string(id) + " ";
    }
}


#endif