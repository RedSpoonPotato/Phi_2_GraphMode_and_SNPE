#include "android_main.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {

    assert(argc == 6);
    std::string input_txt = "this test does not matter";
    // std::vector<zdl::DlSystem::Runtime_t> runtime_modes = {
    //     zdl::DlSystem::Runtime_t::DSP //  8-bit fixed point
    // };
    std::vector<zdl::DlSystem::Runtime_t> runtime_modes;
    size_t data_size;
    if (std::strncmp(argv[1], "dsp08", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::DSP);
        data_size = 1;
    }
    else if (std::strncmp(argv[1], "gpu32", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::GPU); 
        data_size = 4;
    }
    else if (std::strncmp(argv[1], "gpu16", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::GPU_FLOAT16); 
        data_size = 2;
    }
    else if (std::strncmp(argv[1], "cpu32", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::CPU); 
        data_size = 4;
    }
    else {
        std::cerr << "problem with parsing, argv[1]: " << argv[1]; return 1; 
    }
    std::string dlcName = argv[2]; // "matmul_model.dlc"
    std::string srcDIR = ".";
    std::vector<std::string> inputList = {
        std::string(argv[3]),
        std::string(argv[4])
    };
    size_t N = std::strtoul(argv[5], nullptr, 10);
    std::vector<size_t> first_model_input_sizes = {
        N * data_size,
        N * N * data_size
    };
    std::vector<size_t> first_model_output_sizes = {
        N * data_size,
    };
    std::vector<std::string> outputNames = {"matmul_out:0"};
    uint32_t NUM_ITERS = 1;
    std::string udo_path = "";
    std::string embeddingFile = "";
    bool use_udo = false;
    bool first_run = true;
    std::string outputDir = ".";
    int debugReturnCode = 20;

    std::string output_str;

    output_str = modelLaunch(
        input_txt,
        runtime_modes,
        srcDIR, 
        inputList, 
        first_model_input_sizes,
        first_model_output_sizes,
        data_size,
        embeddingFile,
        dlcName, 
        outputNames,
        NUM_ITERS,
        udo_path,
        use_udo,
        first_run,
        outputDir,
        Free_Status::run,
        debugReturnCode);

    std::cout << "final result: " << output_str << "\n";

    // output_str = modelLaunch(
    //     input_txt,
    //     srcDIR, 
    //     inputList, 
    //     first_model_input_sizes,
    //     embeddingFile,
    //     dlcName, 
    //     outputNames,
    //     NUM_ITERS,
    //     udo_path,
    //     use_udo,
    //     false,
    //     Free_Status::run_and_free);

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    return 0;
}