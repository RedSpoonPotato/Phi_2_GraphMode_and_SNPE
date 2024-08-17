#ifndef MAIN_STUFF
#define MAIN_STUFF

#include "android_main.h"
#include <cassert>
#include <iostream>
#include <cstring>


bool readInput(int argc, char* argv[], std::string &input) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" && i + 1 < argc) {
            input = argv[i + 1];
            return true;
        }
    }
    std::cerr << "Error: -i flag not found or missing argument.\n";
    return false;
}

std::map<std::string, RuntimeParams> runtimeArgParse(
    char** argv, 
    int argc
    ) 
{
    std::map<std::string, RuntimeParams> model_runtimes;
    int i = 0;
    std::string model_name;
    std::cout << "Reading model names: ";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) != "-m") {
            continue;
        }
        model_name = argv[i + 1];
        std::cout << model_name << ", ";
        model_runtimes[model_name] = RuntimeParams();
        if (std::strncmp(argv[i + 2], "dsp08", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::DSP;
            model_runtimes[model_name].datasize = 1;
            model_runtimes[model_name].isTFBuffer = true;
            model_runtimes[model_name].dataDir = "u8_weights/";
        }
        else if (std::strncmp(argv[i + 2], "gpu32", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::GPU;
            model_runtimes[model_name].datasize = 4;
            model_runtimes[model_name].isTFBuffer = false;
            model_runtimes[model_name].dataDir = "fp32_weights/";
        }
        else if (std::strncmp(argv[i + 2], "gpu16", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
            model_runtimes[model_name].datasize = 2;
            model_runtimes[model_name].isTFBuffer = false;
            model_runtimes[model_name].dataDir = "fp16_weights/";
        }
        else if (std::strncmp(argv[i + 2], "cpu32", 5) == 0) { 
            model_runtimes[model_name].runtime_type = zdl::DlSystem::Runtime_t::CPU;
            model_runtimes[model_name].datasize = 4;
            model_runtimes[model_name].isTFBuffer = false;
            model_runtimes[model_name].dataDir = "fp32_weights/";
        }
        else {
            std::cerr << "problem with parsing argv[index]: " << argv[i + 2] << "\n";
        }
    }
    std::cout << "\n";
    return model_runtimes;
}

std::set<std::pair<std::string, std::string>> setDlcPaths (
    const std::map<std::string, RuntimeParams>& runtimes,
    const std::string& path
)
{
    std::set<std::pair<std::string, std::string>> ModelNameAndPaths;

    std::string dlcDir;
    std::string model_name;
    
    for (const auto pair : runtimes) {
        model_name = pair.first;
        if (pair.second.runtime_type == zdl::DlSystem::Runtime_t::DSP) {
            dlcDir = path + "/dlc/u8/";
        }
        else if (pair.second.runtime_type == zdl::DlSystem::Runtime_t::GPU_FLOAT16) {
            dlcDir = path + "/dlc/fp16/";
        }
        else {
            dlcDir = path + "/dlc/fp32/";
        }
        ModelNameAndPaths.insert({model_name, dlcDir + model_name + ".dlc"});
    }
    return ModelNameAndPaths;
}

std::map<std::string, std::string> setOtherPaths(
    const std::map<std::string, RuntimeParams>& runtimes,
    const std::string& path
) {
    std::map<std::string, std::string> otherPaths;

    otherPaths["sin"] = "sin_cached.bin"; // 32-bit
    otherPaths["cos"] = "cos_cached.bin"; // 32-bit
    otherPaths["embedding"] = "embedding.bin"; // 16-bit
    otherPaths["token_vocab"] = "vocab.json";
    otherPaths["token_merges"] = "merges.txt";
    otherPaths["decoder_params"] = "decoder_params.txt";

    // set weight and layernorm paths
    std::map<std::string, std::string> fp16_layernorm_Paths;
    std::map<std::string, std::string> weightPaths;
    std::string layerNormDir = "fp16_layernorm/";
    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        fp16_layernorm_Paths["layernorm_weight_" + i_str] = layerNormDir + "layernorm_weight_" + i_str + ".bin";
        weightPaths["q_weight_" + i_str]    = runtimes.at("P1_Q_reshaped_with_bias").dataDir +  i_str + "_q_proj_weight.bin";
        weightPaths["k_weight_" + i_str]    = runtimes.at("P1_K_reshaped_with_bias").dataDir + i_str + "_k_proj_weight.bin";
        weightPaths["v_weight_" + i_str]    = runtimes.at("P1_V_reshaped_with_bias").dataDir + i_str + "_v_proj_weight.bin";
        weightPaths["fc1_weight_" + i_str]  = runtimes.at("FC1_reshaped_with_bias").dataDir + i_str + "_mlp_fc1_weight.bin";
        weightPaths["fc2_weight_" + i_str]  = runtimes.at("FC2_reshaped_with_bias").dataDir + i_str + "_mlp_fc2_weight.bin";
        weightPaths["p4_weight_" + i_str]   = runtimes.at("P4_1_reshaped_with_bias").dataDir + i_str + "_dense_weight.bin";

        fp16_layernorm_Paths["layernorm_bias_" + i_str] = layerNormDir + "layernorm_bias_" + i_str + ".bin";
        weightPaths["q_bias_" + i_str]      = runtimes.at("P1_Q_reshaped_with_bias").dataDir +  i_str + "_q_proj_bias.bin";
        weightPaths["k_bias_" + i_str]      = runtimes.at("P1_K_reshaped_with_bias").dataDir + i_str + "_k_proj_bias.bin";
        weightPaths["v_bias_" + i_str]      = runtimes.at("P1_V_reshaped_with_bias").dataDir + i_str + "_v_proj_bias.bin";
        weightPaths["fc1_bias_" + i_str]    = runtimes.at("FC1_reshaped_with_bias").dataDir + i_str + "_mlp_fc1_bias.bin";
        weightPaths["fc2_bias_" + i_str]    = runtimes.at("FC2_reshaped_with_bias").dataDir + i_str + "_mlp_fc2_bias.bin";
        weightPaths["p4_bias_" + i_str]     = runtimes.at("P4_1_reshaped_with_bias").dataDir + i_str + "_dense_bias.bin";
    }
    fp16_layernorm_Paths["final_layernorm_weight"]    = layerNormDir + "final_layernorm_weight.bin";
    fp16_layernorm_Paths["final_layernorm_bias"]      = layerNormDir + "final_layernorm_bias.bin";

    // merge mappings
    otherPaths.insert(weightPaths.begin(), weightPaths.end());
    otherPaths.insert(fp16_layernorm_Paths.begin(), fp16_layernorm_Paths.end());

    // set parent path
    for (auto& pair : otherPaths) {
        pair.second = path + "/data/" + pair.second;
    }
    
    return otherPaths;
}

#endif