// built for testing load times of DLCs

#include "android_main.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {

    // assert(argc == 6);
    // std::string input_txt = "this test does not matter";
    std::string input_txt = "What is your favorite color?. Mine is red.";

    // std::vector<zdl::DlSystem::Runtime_t> runtime_modes;
    std::map<std::string, zdl::DlSystem::Runtime_t> runtime_modes;
    // auto runtime_type = zdl::DlSystem::Runtime_t::DSP;
    auto runtime_type = zdl::DlSystem::Runtime_t::CPU;

    size_t datasize;
    bool isTFBuffer;
    if (std::strncmp(argv[1], "dsp08", 5) == 0) { 
        runtime_type = zdl::DlSystem::Runtime_t::DSP;
        datasize = 1;
        isTFBuffer = true;
    }
    else if (std::strncmp(argv[1], "gpu32", 5) == 0) { 
        runtime_type = zdl::DlSystem::Runtime_t::GPU;
        datasize = 4;
        isTFBuffer = false;
    }
    else if (std::strncmp(argv[1], "gpu16", 5) == 0) { 
        runtime_type = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
        datasize = 2;
        isTFBuffer = false;
    }
    else if (std::strncmp(argv[1], "cpu32", 5) == 0) { 
        runtime_type = zdl::DlSystem::Runtime_t::CPU;
        datasize = 4;
        isTFBuffer = false;
    }
    else {
        std::cerr << "problem with parsing, argv[1]: " << argv[1]; return 1; 
    }

    // remove this if u add cpu and/or gpu support
    // assert(runtime_type == zdl::DlSystem::Runtime_t::DSP);

    // setting names
    std::set<std::string> model_names;
    for (int i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        model_names.insert("P1_1_reshaped_layer_" + i_str);
        model_names.insert("P1_2_reshaped_layer_" + i_str);
        model_names.insert("P4_1_reshaped_layer_" + i_str);
    }
    model_names.insert("gelu"); // might fail to reshape, if so, just implement manually in executable
    model_names.insert("P2_1_first_buffered");
    model_names.insert("P2_not_first_reshaped");
    model_names.insert("P3_first_buffered");
    // model_names.insert("P3_not_first_reshaped"); // this had problems so using buffered version instead to avoid reshaping
    model_names.insert("P3_not_first_buffered");
    model_names.insert("P4_2_reshaped");

    // setting runtime
    for (const std::string& model_name : model_names) {
        runtime_modes[model_name] = runtime_type;
    }

    // setting dlc paths
    std::set<std::pair<std::string, std::string>> ModelNameAndPaths;
    // std::string dlcDir = "./fp16_test/model_split/q_dlc/q_model_"; // restore this
    std::string dlcDir = "./fp16_test/model_split/dlc/model_"; // remove this
    for (const std::string& model_name : model_names) {
        ModelNameAndPaths.insert({model_name, dlcDir + model_name + ".dlc"});
    }

    // remove this later
    // for (const std::string& model_name : model_names) {
    //     if (model_name == "P3_not_first_reshaped") {
    //         ModelNameAndPaths.insert({model_name, dlcDir + "P3_not_first_reshaped_test.dlc"});
    //     }
    //     else if (model_name == "P1_reshaped_layer_0") {
    //         ModelNameAndPaths.insert({model_name, dlcDir + "P1_reshaped_test.dlc"});
    //     }
    //     else {
    //         ModelNameAndPaths.insert({model_name, dlcDir + model_name + ".dlc"});
    //     }
    // }


    // setting sin, cos, embedding paths
    std::map<std::string, std::string> otherPaths;
    otherPaths["sin"] = "./fp16_test/model_split/data/sin_cached.bin"; // 32-bit
    otherPaths["cos"] = "./fp16_test/model_split/data/cos_cached.bin"; // 32-bit
    otherPaths["embedding"] = "./fp16_test/model_split/data/embedding.bin"; // 16-bit
    otherPaths["token_vocab"] = "./fp16_test/model_split/data/vocab.json";
    otherPaths["token_merges"] = "./fp16_test/model_split/data/merges.txt";
    // otherPaths[""] = "./fp16_test/model_split/data/merges.txt";
    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        otherPaths["layernorm_weight_" + i_str] = "./fp16_test/model_split/data/layernorm_weight_" + i_str + ".bin";
        otherPaths["layernorm_bias_" + i_str] = "./fp16_test/model_split/data/layernorm_bias_" + i_str + ".bin";
    }

    // other params
    uint32_t max_iterations = 2;  // CHANGE THIS TO WHATEVER IT SHOULD BE
    int debugReturnCode = 20;
    uint32_t end_token_id = 1; // CHANGE THIS TO WHATEVER IT SHOULD BE
    bool use_end_token_id = false; // CHANGE THIS TO WHATEVER IT SHOULD BE

    // final result
    std::string output_str;

    output_str = modelLaunch(
        input_txt,
        runtime_modes,
        datasize,
        isTFBuffer,
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

    std::cout << "final result: " << output_str << "\n";

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    return 0;
}