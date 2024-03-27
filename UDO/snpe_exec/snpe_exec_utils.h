#ifndef SNPE_EXEC_UTILS_H_
#define SNPE_EXEC_UTILS_H_

// #include <cstring>
// #include <iostream>
// #include <getopt.h>
// #include <fstream>
// #include <cstdlib>
#include <vector>
// #include <string>
#include <iterator>
#include <unordered_map>
#include <cassert>
#include <cstdlib>
#include <iostream>


#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DiagLog/IDiagLog.hpp"

#include "SNPE/SNPEBuilder.hpp"

#include "snpe_exec_utils.h"
#include "main_macros.h"


// 1 per model
struct ModelRunetime {
    // model info stuff
    std::string model;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
    std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers;
    zdl::DlSystem::UserBufferMap inputMap;
    zdl::DlSystem::UserBufferMap outputMap;
        
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> input_user_buff_vec;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> output_user_buff_vec;

    // static zdl::DlSystem::Runtime_t runtime;
    zdl::DlSystem::Runtime_t runtime;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
};

void parse_argv_models(std::vector<ModelRunetime>& models, int argc, char* argv[]) {
    std::string temp_str;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            // std::cout << argv[i] << "\n";
            // Identify the option type
            switch (argv[i][1]) {
                case 'm':
                    // New model
                    models.emplace_back();
                    models.back().model = argv[++i];
                    continue;
                case 'o':
                    // Output node(s)
                    temp_str = argv[++i];
                    models.back().outputs.push_back(temp_str + ":0");
                    continue;
                case 'i':
                    // Input(s)
                    temp_str = argv[++i];
                    temp_str = temp_str.substr(0, temp_str.find(".dat"));
                    models.back().inputs.push_back(temp_str); // ex: "input1.dat" --> "input1"
                    continue;
                default:
                continue;
            }
        }
    }
}

void parse_argv_other(
    int argc, char* argv[],
    uint32_t& num_iterations) 
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                case 'N':
                    num_iterations = std::atoi(argv[++i]);
                    continue;
                default:
                    continue;
            }
        }
    }
}

void print_model_runtimes(const std::vector<ModelRunetime>& models) {
    std::cout << "Model Information Parsed:\n";
    for (const auto& model : models) {
        std::cout << model.model << ",";
        for (const auto& input : model.inputs)    {std::cout << " " << input;}
        std::cout << ",";
        for (const auto& output : model.outputs)  {std::cout << " " << output;}
        std::cout << "\n";
   }
   std::cout << "---------------------------\n";
}

void intialize_model_runtime(std::vector<ModelRunetime>& runtimes) {
    int num_models = runtimes.size();
    for (int i = 0; i < num_models; i++) {
        runtimes[i].applicationInputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
        runtimes[i].applicationOutputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
        runtimes[i].inputMap = zdl::DlSystem::UserBufferMap();
        runtimes[i].outputMap = zdl::DlSystem::UserBufferMap();
        runtimes[i].runtime = checkRuntime();
        runtimes[i].container = loadContainerFromFile(runtimes[i].model);
        runtimes[i].snpe = setBuilderOptions(
            runtimes[i].container, runtimes[i].runtime, 
            true, (runtimes[i].inputs[0]+":0").c_str());
    }
}

// currently only built for a single model, see test.cpp for other code for mulitple models
void allocate_model_input_buffers(
    std::vector<ModelRunetime>& runtimes,
    std::vector<int> model_input_sizes,
    bool debug)
{
    int num_models = runtimes.size();
    for (int i = 0; i < num_models; i++) {
        for (int j = 0; j < runtimes[i].inputs.size(); j++) {
            runtimes[i].applicationInputBuffers[runtimes[i].inputs[j] + ":0"] 
               = std::vector<u_int8_t>(model_input_sizes[j]);
            if (debug) {std::cout << "calling createUserBuffer()\n";}
            createUserBuffer(runtimes[i].inputMap, runtimes[i].applicationInputBuffers,
                runtimes[i].input_user_buff_vec, runtimes[i].snpe, (runtimes[i].inputs[j] + ":0").c_str());
            if (debug) {std::cout << "finished\n";}
        }
    }
}

// currently only built for a single model, see test.cpp for other code for mulitple models
void allocate_model_output_buffers(
    std::vector<ModelRunetime>& runtimes,
    std::vector<int> model_output_sizes,
    bool debug)
{
    std::cout << "runtimes[0].outputs.size():" << runtimes[0].outputs.size() << "\n";
    int num_models = runtimes.size();
    for (int i = 0; i < num_models; i++) {
        for (int j = 0; j < runtimes[i].outputs.size(); j++) {
            runtimes[i].applicationOutputBuffers[runtimes[i].outputs[j]] 
               = std::vector<u_int8_t>(model_output_sizes[j]);
            if (debug) {std::cout << "calling createUserBuffer()\n";}
            createUserBuffer(runtimes[i].outputMap, runtimes[i].applicationOutputBuffers,
                runtimes[i].output_user_buff_vec, runtimes[i].snpe, (runtimes[i].outputs[j]).c_str());
            if (debug) {std::cout << "finished\n";}
        }
    }
}

void intialize_input_buffers(
    std::vector<ModelRunetime>& runtimes,
    bool debug) 
{
    int num_models = runtimes.size();
    assert(num_models == 1);
    std::string fileLine;
    for (int i = 0; i < num_models; i++) {
        fileLine = std::string();
        if (debug) { 
            std::cout << "reading files for 1st model input\n"; 
        }
        for (int j = 0; j < runtimes[i].inputs.size(); j++) {fileLine += runtimes[i].inputs[j] + ".dat ";}
        fileLine.pop_back(); // remove last space
        if (debug) {
            std::cout << "fileLine: " << fileLine << "\n";
            std::cout << "calling loadInputUserBuffer()\n";
        }
        loadInputUserBuffer(runtimes[i].applicationInputBuffers, runtimes[i].snpe, fileLine);
    }
}


// generate mask and position_ids; also iteration_num should first be 0
template <typename T>
void prepareInputs(T* mask, int* position_ids, uint32_t seq_len, uint32_t iteration_num)
{
    // must adjust indexing if this is not true
    assert(sizeof(T) == 2);
    if (iteration_num == 0) {
        // set position_ids shape
        position_ids[MAX_SEQ_LEN + 0] = 1;
        position_ids[MAX_SEQ_LEN + 1] = 1;
        position_ids[MAX_SEQ_LEN + 2] = 1;
        position_ids[MAX_SEQ_LEN + 3] = seq_len;
        // set position_ids
        for (int i = 0; i < seq_len; ++i) { position_ids[i] = i; }
        std::cout << "set mask1\n";
        // set mask shape
        uint16_t* ptr16 = (uint16_t*)mask;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = (uint16_t)seq_len;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = (uint16_t)seq_len;
        // set mask
        std::cout << "writing mask\n";
        T lowest = std::numeric_limits<T>::lowest();
        for (uint32_t row = 0; row < seq_len; row++) {
            for (uint32_t col = 0; col < seq_len; col++) {
                std::cout << "(row, col): (" << row << ", " << col << ")\n";
                if (row >= col) { mask[row*seq_len + col] = 0; }
                else            { mask[row*seq_len + col] = lowest; } 
            }
        }
    }
    else {
        // set position_ids shape
        position_ids[MAX_SEQ_LEN + 0] = 1;
        position_ids[MAX_SEQ_LEN + 1] = 1;
        position_ids[MAX_SEQ_LEN + 2] = 1;
        position_ids[MAX_SEQ_LEN + 3] = 1;
        // set position_ids
        position_ids[0] = seq_len-1;
        std::cout << "set mask2\n";
        // set mask shape
        uint16_t* ptr16 = (uint16_t*)mask;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = 1;
        ptr16[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = (uint16_t)seq_len;
        // set mask
        for (uint32_t i = 0; i < seq_len; i++) { mask[i] = 0; }
    }
}

template <typename T>
void printV(const std::string& str, const std::vector<T>& vec) {
    std::cout << str;
    std::cout << ": [";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i != vec.size()-1) { std::cout << ", ";}
    }
    std::cout << "]\n";
}

#endif