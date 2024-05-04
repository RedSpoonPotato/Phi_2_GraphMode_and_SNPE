// built for testing load times of DLCs

#include "android_main.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {

    // assert(argc == 6);
    std::string input_txt = "this test does not matter";

    std::vector<zdl::DlSystem::Runtime_t> runtime_modes;
    size_t data_size;
    bool isTFBuffer;
    if (std::strncmp(argv[1], "dsp08", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::DSP);
        data_size = 1;
        isTFBuffer = true;
    }
    else if (std::strncmp(argv[1], "dsp16", 5) == 0) { 
        runtime_modes.push_back(static_cast<zdl::DlSystem::Runtime_t>(6));
        data_size = 2;
        isTFBuffer = true;
    }
    else if (std::strncmp(argv[1], "gpu32", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::GPU); 
        data_size = 4;
        isTFBuffer = false;
    }
    else if (std::strncmp(argv[1], "gpu16", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::GPU_FLOAT16);
        data_size = 2;
        isTFBuffer = false;
    }
    else if (std::strncmp(argv[1], "cpu32", 5) == 0) { 
        runtime_modes.push_back(zdl::DlSystem::Runtime_t::CPU); 
        data_size = 4;
        isTFBuffer = false;
    }
    else {
        std::cerr << "problem with parsing, argv[1]: " << argv[1]; return 1; 
    }
    std::string dlcPath = argv[2]; // "matmul_model.dlc"
    std::string srcDIR = "./model_generation";
    std::vector<std::string> inputList = {
        "vector.dat",
        // "matrix.dat"
    };
    size_t N = std::strtoul(argv[3], nullptr, 10);
    // size_t N = size_t(1e4);
    // file_size
    std::vector<size_t> first_model_input_sizes = {
        1 * N * data_size,
        // N * N * data_size
    };
    // can be large as you want
    std::vector<size_t> first_model_buffer_sizes = {
        // 3 * N * data_size,
        4 * N * 4
        // N * N * data_size
    };
    // can be large as you want
    std::vector<size_t> first_model_output_buffer_sizes = {
        // 4 * N * data_size, //  the good one
        4 * N * 4
    };
    std::vector<std::string> outputNames = {"matmul_out:0"};
    uint32_t NUM_ITERS = 2;
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
        first_model_buffer_sizes,
        first_model_output_buffer_sizes,
        data_size,
        isTFBuffer,
        embeddingFile,
        dlcPath, 
        outputNames,
        NUM_ITERS,
        udo_path,
        use_udo,
        first_run,
        outputDir,
        Free_Status::run,
        debugReturnCode);

    std::cout << "final result: " << output_str << "\n";

    output_str = modelLaunch(
        input_txt,
        runtime_modes,
        srcDIR, 
        inputList, 
        first_model_input_sizes,
        first_model_buffer_sizes,
        first_model_output_buffer_sizes,
        data_size,
        isTFBuffer,
        embeddingFile,
        dlcPath, 
        outputNames,
        NUM_ITERS,
        udo_path,
        use_udo,
        false,
        outputDir,
        Free_Status::run_and_free,
        debugReturnCode);

    std::cout << "final result: " << output_str << "\n";

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    return 0;
}