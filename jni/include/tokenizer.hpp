#pragma once

#include <unordered_map>
#include <iostream>
#include <string>
#include <codecvt>
#include <locale>
#include <vector>
#include <regex>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

#include <map>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/regex.hpp>

#define REGEX_TOOL boost

std::string UnicodeToUTF8(unsigned int codepoint)
{
    std::string out;

    if (codepoint <= 0x7f)
        out.append(1, static_cast<char>(codepoint));
    else if (codepoint <= 0x7ff)
    {
        out.append(1, static_cast<char>(0xc0 | ((codepoint >> 6) & 0x1f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else if (codepoint <= 0xffff)
    {
        out.append(1, static_cast<char>(0xe0 | ((codepoint >> 12) & 0x0f)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    else
    {
        out.append(1, static_cast<char>(0xf0 | ((codepoint >> 18) & 0x07)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        out.append(1, static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return out;
}

uint32_t utf_to_int(std::string utf8String) {
    // Convert UTF-8 to wide string (UTF-32)
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf32conv;
    std::u32string utf32String = utf32conv.from_bytes(utf8String);

    // Convert UTF-32 to integers
    uint32_t intValue;
    for (char32_t codePoint : utf32String) {
        intValue = static_cast<int>(codePoint);
        break;
    }
    return intValue;
}

std::string codePointToUTF8(uint32_t codepoint) {
    std::string utf8;

    if (codepoint <= 0x7F) {
        // 1-byte sequence
        utf8.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        // 2-byte sequence
        utf8.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        utf8.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        // 3-byte sequence
        utf8.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        utf8.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        utf8.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0x10FFFF) {
        // 4-byte sequence
        utf8.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        utf8.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        utf8.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        utf8.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        throw std::invalid_argument("Invalid Unicode codepoint");
    }

    return utf8;
}

std::unordered_map<uint32_t, std::string> bytes_to_unicode() {
    std::unordered_map<uint32_t, std::string> byte_encoder;
    std::string string1;
    std::string string2;
    for (uint32_t b = utf_to_int("!"); b <= utf_to_int("~"); ++b) {
        byte_encoder[b] = codePointToUTF8(b);
    }
    for (uint32_t b = utf_to_int("¡"); b <= utf_to_int("¬"); ++b) {
        byte_encoder[b] = codePointToUTF8(b);
    }
    for (uint32_t b = utf_to_int("®"); b <= utf_to_int("ÿ"); ++b) {
        byte_encoder[b] = codePointToUTF8(b);
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (byte_encoder.find(b) == byte_encoder.end()) {
            byte_encoder[b] = codePointToUTF8(n+256);
            n += 1;
        }
    }
    return byte_encoder;
}


// Function to split a string by a delimiter
std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void getPairsFromFile(
    const std::string& merges_file,
    std::vector<std::tuple<std::string, std::string>>& bpe_merges) {
    // Read the merges file
    std::ifstream merges_handle(merges_file);
    std::string line;
    bpe_merges = std::vector<std::tuple<std::string, std::string>>();

    if (merges_handle.is_open()) {
        // Skip the first line
        std::getline(merges_handle, line);
        while (std::getline(merges_handle, line)) {
            if (!line.empty()) {
                std::vector<std::string> merge = split(line, ' ');
                if (merge.size() == 2) {
                    bpe_merges.emplace_back(merge[0], merge[1]);
                }
            }
        }
        merges_handle.close();
    } else {
        std::cerr << "Unable to open file";
    }
}

std::vector<std::string> split_utf8(const std::string& utf8_string) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < utf8_string.size()) {
        size_t char_len = 1;
        unsigned char lead = utf8_string[i];
        if (lead < 0x80) {
            char_len = 1;
        } else if ((lead >> 5) == 0x6) {
            char_len = 2;
        } else if ((lead >> 4) == 0xe) {
            char_len = 3;
        } else if ((lead >> 3) == 0x1e) {
            char_len = 4;
        } else {
            throw std::invalid_argument("Invalid UTF-8 encoding");
        }

        if (i + char_len > utf8_string.size()) {
            throw std::invalid_argument("Incomplete UTF-8 character at the end of the string");
        }

        result.push_back(utf8_string.substr(i, char_len));
        i += char_len;
    }
    return result;
}

uint32_t utf8_to_utf32(const std::string& utf8_char) {
    if (utf8_char.empty()) {
        throw std::invalid_argument("Input string is empty.");
    }

    // Convert UTF-8 to UTF-32
    uint32_t utf32_char = 0;
    size_t length = utf8_char.size();

    if (length == 1) {
        // 1-byte UTF-8 (ASCII)
        utf32_char = static_cast<unsigned char>(utf8_char[0]);
    } else if (length == 2) {
        // 2-byte UTF-8
        utf32_char = ((static_cast<unsigned char>(utf8_char[0]) & 0x1F) << 6) |
                     (static_cast<unsigned char>(utf8_char[1]) & 0x3F);
    } else if (length == 3) {
        // 3-byte UTF-8
        utf32_char = ((static_cast<unsigned char>(utf8_char[0]) & 0x0F) << 12) |
                     ((static_cast<unsigned char>(utf8_char[1]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(utf8_char[2]) & 0x3F);
    } else if (length == 4) {
        // 4-byte UTF-8
        utf32_char = ((static_cast<unsigned char>(utf8_char[0]) & 0x07) << 18) |
                     ((static_cast<unsigned char>(utf8_char[1]) & 0x3F) << 12) |
                     ((static_cast<unsigned char>(utf8_char[2]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(utf8_char[3]) & 0x3F);
    } else {
        throw std::invalid_argument("Input string is not a valid UTF-8 encoded character.");
    }

    return utf32_char;
}

std::string utf32_to_utf8(uint32_t utf32_char) {
    std::string utf8_str;

    if (utf32_char <= 0x7F) {
        // 1-byte UTF-8 (ASCII)
        utf8_str.push_back(static_cast<char>(utf32_char));
    } else if (utf32_char <= 0x7FF) {
        // 2-byte UTF-8
        utf8_str.push_back(static_cast<char>((utf32_char >> 6) | 0xC0));
        utf8_str.push_back(static_cast<char>((utf32_char & 0x3F) | 0x80));
    } else if (utf32_char <= 0xFFFF) {
        // 3-byte UTF-8
        utf8_str.push_back(static_cast<char>((utf32_char >> 12) | 0xE0));
        utf8_str.push_back(static_cast<char>(((utf32_char >> 6) & 0x3F) | 0x80));
        utf8_str.push_back(static_cast<char>((utf32_char & 0x3F) | 0x80));
    } else if (utf32_char <= 0x10FFFF) {
        // 4-byte UTF-8
        utf8_str.push_back(static_cast<char>((utf32_char >> 18) | 0xF0));
        utf8_str.push_back(static_cast<char>(((utf32_char >> 12) & 0x3F) | 0x80));
        utf8_str.push_back(static_cast<char>(((utf32_char >> 6) & 0x3F) | 0x80));
        utf8_str.push_back(static_cast<char>((utf32_char & 0x3F) | 0x80));
    } else {
        throw std::invalid_argument("Input character is not a valid UTF-32 character.");
    }

    return utf8_str;
}

class Tokenizer {
    public:
        Tokenizer(std::string vocab_file, std::string merges_file);
        std::vector<uint32_t> generate(const std::string& text);
        std::string decode(const std::vector<uint32_t>& input_ids);
    private:

        // boost::property_tree::ptree encoder;
        std::unordered_map<std::string, uint32_t> encoder;
        std::unordered_map<uint32_t, std::string> decoder;
        std::unordered_map<uint32_t, std::string> byte_encoder;
        std::unordered_map<std::string, uint32_t> byte_decoder;
        REGEX_TOOL::regex pat;
        std::unordered_map<std::string, std::string> cache;
        std::map<std::tuple<std::string, std::string>, int> bpe_ranks; // using map for now b/c of wierd error with unordered
        std::string unk_token = "<|endoftext|>";
        std::string errors = "ERROR";

        std::vector<std::string> _tokenize(const std::string& text);
        void _convert_tokens_to_ids(
            const std::vector<std::string>& str_vec,
            std::vector<uint32_t>& ids
        );
        std::string bpe(const std::string& token);
};

void get_pairs(
    const std::vector<std::string>& word,
    std::set<std::pair<std::string, std::string>>& pairs
) {
    pairs = std::set<std::pair<std::string, std::string>>();
    std::string prev_char = word[0];
    std::string current_char;
    for (size_t i = 1; i < word.size(); i++) {
        current_char = word[i];
        pairs.insert({prev_char, current_char});
        prev_char = current_char;
    }
}

// NEED TO TEST THISSSS
std::string Tokenizer::bpe(const std::string& token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    std::vector<std::string> word = split_utf8(token);
    std::set<std::pair<std::string, std::string>> pairs;
    get_pairs(word, pairs);
    
    if (pairs.empty()) {
        return token;
    }

    while (true) {
        auto bigram_it = std::min_element(pairs.begin(), pairs.end(), 
            [this](const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) {
                return bpe_ranks.count({a.first, a.second}) && (!bpe_ranks.count({b.first, b.second}) || bpe_ranks[{a.first, a.second}] < bpe_ranks[{b.first, b.second}]);
            }
        );

        std::pair<std::string, std::string> bigram = *bigram_it;

        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }

        std::string first = bigram.first;
        std::string second = bigram.second;
        std::vector<std::string> new_word;

        for (size_t i = 0; i < word.size();) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it != word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, it);
                i = std::distance(word.begin(), it);
            } else {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;

        if (word.size() == 1) {
            break;
        } else {
            get_pairs(word, pairs);
        }
    }

    std::string result = "";
    for (const auto& w : word) {
        result += w + " ";
    }
    result.pop_back(); // remove the trailing space

    cache[token] = result;
    return result;
}

// TEST TO SEE IF THIS WORKS
std::vector<std::string> Tokenizer::_tokenize(const std::string& text) {
    // // Initialize the regex pattern
    // std::regex pat(R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");

    std::vector<std::string> bpe_tokens;

    // Iterate over the matches of the regex pattern in the text
    auto words_begin = REGEX_TOOL::sregex_iterator(text.begin(), text.end(), pat);
    auto words_end = REGEX_TOOL::sregex_iterator();

    for (REGEX_TOOL::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string token = (*i).str();
        std::cout << "token in re.findall: " << token << std::endl;

        // Encode the token into UTF-8 and map using byte_encoder
        std::string encoded_token;
        for (unsigned char b : token) {
            encoded_token += byte_encoder[b];
        }
        std::cout << "token after join: " << encoded_token << std::endl;

        // Apply the BPE process (assuming bpe is a member function)
        std::string bpe_result = bpe(encoded_token);

        // Split the BPE result by spaces and add to the tokens list
        size_t pos = 0;
        while ((pos = bpe_result.find(' ')) != std::string::npos) {
            bpe_tokens.push_back(bpe_result.substr(0, pos));
            bpe_result.erase(0, pos + 1);
        }
        bpe_tokens.push_back(bpe_result);
    }
    // remove later
    std::cout << "['";
    for (size_t i = 0; i < bpe_tokens.size(); i++) {
        auto str = bpe_tokens[i];
        std::cout << str;
        if (i == bpe_tokens.size()-1) { 
            std::cout << "']\n";
            break; 
        }
        std::cout << "', '";
    }

    return bpe_tokens;
}

void Tokenizer::_convert_tokens_to_ids(
    const std::vector<std::string>& str_vec,
    std::vector<uint32_t>& ids
) {
    for (const std::string& id : str_vec) {
        if (encoder.find(id) == encoder.end()) {
            ids.push_back(encoder[unk_token]);
        }
        else {
            ids.push_back(encoder[id]);
        }
    }
}

std::unordered_map<std::string, uint32_t> to_map(const boost::property_tree::ptree& encoder_tree) {
    std::unordered_map<std::string, uint32_t> result;
    for (const auto& item : encoder_tree) {
        try {
            result[item.first] = item.second.get_value<uint32_t>();
        } catch (const boost::property_tree::ptree_bad_data& e) {
            std::cerr << "Error converting value to int for key " << item.first << ": " << e.what() << std::endl;
        }
    }
    return result;
}

Tokenizer::Tokenizer(std::string vocab_file, std::string merges_file) {

    // Intialize encoder
    std::ifstream file(vocab_file);
    boost::property_tree::ptree encoder_tree;
    boost::property_tree::read_json(file, encoder_tree);
    encoder = to_map(encoder_tree);

    // Intialize decoder
    for (auto& pair : encoder) {
        decoder[pair.second] = pair.first;
    }

    // remove later
    std::cout << decoder[42344] << "\n";

    // Create {codepoint, uint32_t mappings}
    byte_encoder = bytes_to_unicode();
    for (auto& pair : byte_encoder) {
        byte_decoder[pair.second] = pair.first;
    }

    std::cout << "Compiling Regex\n";
    // python version: re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    // std version 1 (does not work well)
    // pat = REGEX_TOOL::regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)");
    // boost verision, seems to work in the basic example, but unsure if it will work everywhere
    pat = REGEX_TOOL::regex(
        // R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)",
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)", // seems to work
        boost::regex::perl | boost::regex::icase
    );
    std::cout << "Done\n";

    // Intialize bpe_ranks dictionary from merges file
    std::vector<std::tuple<std::string, std::string>> bpe_merges;
    getPairsFromFile(merges_file, bpe_merges);
    for (size_t i = 0; i < bpe_merges.size(); ++i) {
        bpe_ranks[bpe_merges[i]] = i;
    }
}

std::vector<uint32_t> Tokenizer::generate(const std::string& text) {
    std::vector<std::string> tokenized_text = _tokenize(text);
    std::vector<uint32_t> token_ids;
    _convert_tokens_to_ids(tokenized_text, token_ids);
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& input_ids) {
    std::vector<std::string> tokens;
    
    // Convert input IDs to tokens
    for (int id : input_ids) {
        auto it = decoder.find(id);
        if (it != decoder.end()) {
            tokens.push_back(it->second);
        } else {
            // Handle the case where the ID is not found in the decoder
            tokens.push_back(""); // or any default value
        }
    }

    // Concatenate tokens to form the text
    std::string text;
    for (const std::string& token : tokens) {
        text += token;
    }

    // this may fail
    std::vector<std::string> text_utf8_chars = split_utf8(text);
    std::string final_text;
    for (const std::string& utf8_char : text_utf8_chars) {
        if (byte_decoder.find(utf8_char) != byte_decoder.end()) {
            final_text += static_cast<char>(byte_decoder[utf8_char]);
        }
        else {
            final_text += errors;
        }
    }

    for (auto& pair: byte_decoder) {
        assert(pair.second < 256);
    }

    return final_text;
}


// MAIN **********************************

// int main() {

//     Tokenizer tokenizer("vocab.json", "merges.txt");
//     auto tokens = tokenizer.generate("What is your favorite color?. Mine is red.");
//     std::cout << "[";
//     for (auto token : tokens) {std::cout << token << ", ";}
//     std::cout << "]\n";
    

//     std::cout << tokenizer.decode(tokens) << "\n";
// }