// built for testing load times of DLCs

#include "android_main.h"
#include "main_stuff.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {

    // assert(argc == 6);
    // std::string input_txt = "this test does not matter";
    std::string input_txt = "What is your favorite color?. Mine is red.";

    // std::vector<zdl::DlSystem::Runtime_t> runtime_modes;
    std::map<std::string, RuntimeParams> runtimes;
    // auto runtime_type = zdl::DlSystem::Runtime_t::DSP;
    auto runtime_type = zdl::DlSystem::Runtime_t::CPU;

    // grab max_iterations
    char* end;
    uint32_t max_iterations = static_cast<uint32_t>(strtoul(argv[1], &end, 10));
    std::cout << "max iterations: " << max_iterations << "\n";
    assert(*end == '\0');

    // grab paths
    std::string path;
    path = std::string(argv[2]);

    runtimeArgParse(argv, argc, runtimes);

    // setting dlc paths
    std::set<std::pair<std::string, std::string>> ModelNameAndPaths;

    std::string dlcDir;
    std::string dlcSuffix;
    std::string model_name;
    
    for (const auto pair : runtimes) {
        model_name = pair.first;
        if (pair.second.runtime_type == zdl::DlSystem::Runtime_t::DSP) {
            dlcDir = path + "/q_dlc/";
            dlcSuffix = "q_model_";
        }
        else {
            dlcDir = path + "/dlc/";
            dlcSuffix = "model_";
        }
        ModelNameAndPaths.insert({model_name, dlcDir + dlcSuffix + model_name + ".dlc"});
    }


    // setting sin, cos, embedding paths
    std::map<std::string, std::string> otherPaths;
    // std::string dataPath = "./data/";
    otherPaths["sin"] = "sin_cached.bin"; // 32-bit
    otherPaths["cos"] = "cos_cached.bin"; // 32-bit
    otherPaths["embedding"] = "embedding.bin"; // 16-bit
    otherPaths["token_vocab"] = "vocab.json";
    otherPaths["token_merges"] = "merges.txt";
    // otherPaths[""] = "./fp16_test/model_split/data/merges.txt";
    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        otherPaths["layernorm_weight_" + i_str] = "layernorm_weight_" + i_str + ".bin";
        otherPaths["layernorm_bias_" + i_str] = "layernorm_bias_" + i_str + ".bin";
    }
    otherPaths["final_layernorm_weight"]    = "final_layernorm_weight.bin";
    otherPaths["final_layernorm_bias"]      = "final_layernorm_bias.bin";
    for (auto& pair : otherPaths) {
        pair.second = path + "/data/" + pair.second;
    }

    // other params
    int debugReturnCode = 20;
    uint32_t end_token_id = 1; // CHANGE THIS TO WHATEVER IT SHOULD BE
    bool use_end_token_id = false; // CHANGE THIS TO WHATEVER IT SHOULD BE

    // final result
    std::string output_str;

    output_str = modelLaunch(
        input_txt,
        runtimes,
        ModelNameAndPaths, // abs paths
        otherPaths, // abs path of sin, cos, embeddingFIle
        max_iterations, 
        Free_Status::run_and_free, // modded
        debugReturnCode,
        end_token_id,
        use_end_token_id);

    std::cout << "final result: " << output_str << "\n";

    // example of running with the previous result as input, and freeing everything ()
    // output_str = modelLaunch(
    //     output_str,
    //     runtime_modes,
    //     datasize,
    //     isTFBuffer,
    //     ModelNameAndPaths, // abs paths
    //     otherPaths, // abs path of sin, cos, embeddingFIle
    //     max_iterations, 
    //     Free_Status::run_and_free,
    //     debugReturnCode,
    //     end_token_id,
    //     use_end_token_id);

    // std::cout << "final result: " << output_str << "\n";

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    return 0;
}
