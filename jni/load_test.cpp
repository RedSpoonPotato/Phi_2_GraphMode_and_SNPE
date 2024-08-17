// built for testing load times of DLCs

#include "android_main.h"
#include "main_stuff.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {

    // load input text
    // std::string input_txt = "What is your favorite color?. Mine is red.";
    std::string input_txt;
    bool result = readInput(argc, argv, input_txt);
    assert(result);
    std::cout << "input text: \"" << input_txt << "\"\n";

    auto runtime_type = zdl::DlSystem::Runtime_t::CPU;

    // grab max_iterations
    char* end;
    uint32_t max_iterations = static_cast<uint32_t>(strtoul(argv[1], &end, 10));
    std::cout << "max iterations: " << max_iterations << "\n";
    assert(*end == '\0');

    // grab decoder_store number
    uint8_t decoder_cache_size = static_cast<uint8_t>(strtoul(argv[2], &end, 10));
    std::cout << "decoder_cache_size: " << int(decoder_cache_size) << "\n";
    assert(*end == '\0');
    assert(decoder_cache_size <= DECODERS);

    /* grab paths */
    auto base_dir = std::string(argv[3]);
    // parse cmd line
    std::map<std::string, RuntimeParams> runtimes = runtimeArgParse(argv, argc);
    // setting dlc paths
    std::set<std::pair<std::string, std::string>> ModelNameAndPaths = setDlcPaths(runtimes, base_dir);
    // set weight, layernorm, and miscellaneous paths
    std::map<std::string, std::string> otherPaths = setOtherPaths(runtimes, base_dir);

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
        decoder_cache_size,
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
