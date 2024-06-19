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
#include <iomanip>
#include <chrono>
#include <set>



// #include "SNPE/SNPE.hpp"
// #include "SNPE/SNPEFactory.hpp"
// #include "SNPE/SNPEBuilder.hpp"

// #include "DlSystem/DlError.hpp"
// #include "DlSystem/RuntimeList.hpp"

// #include "DlSystem/UserBufferMap.hpp"
// #include "DlSystem/IUserBuffer.hpp"
// #include "DlContainer/IDlContainer.hpp"
// #include "DiagLog/IDiagLog.hpp"


// #include "snpe_exec_utils.h"
#include "main_macros.h"
#include "snpe_tutorial_utils.h"
#include <map>

// 1 per model
struct ModelRuntime {
    // model info stuff
    std::string dlc_path;
    std::vector<std::string> input_names; // should match the dlc input names (verified by VerifyIO())
    // std::map<std::string, std::string> inputNameToFileMap;  


    std::vector<std::string> output_names;

    std::unordered_map<std::string, std::vector<uint8_t>*> applicationInputBuffers;
    std::unordered_map<std::string, std::vector<uint8_t>*> applicationOutputBuffers;
    zdl::DlSystem::UserBufferMap inputMap;
    zdl::DlSystem::UserBufferMap outputMap;
        
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> input_user_buff_vec;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> output_user_buff_vec;

    zdl::DlSystem::Runtime_t runtime;
    zdl::DlSystem::RuntimeList runtimeList;
    zdl::DlSystem::PlatformConfig platformConfig;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    std::unique_ptr<zdl::SNPE::SNPE> snpe;

    // new thing
    std::vector<uint8_t> dlc_buff;



    // ModelRuntime(const ModelRuntime& other) {
    //     model = other.model;
    //     inputs = other.inputs;
    //     inputNameToFileMap = other.inputNameToFileMap;
    //     outputs = other.outputs;
    //     applicationInputBuffers = other.applicationInputBuffers;
    //     applicationOutputBuffers = other.applicationOutputBuffers;
    //     inputMap = other.inputMap;
    //     outputMap = other.outputMap;
    //     runtime = other.runtime;
    //     runtimeList = other.runtimeList;
    //     // platformConfig = other.platformConfig;
    //     dlc_buff = other.dlc_buff;
        
    //     // Deep copy for unique_ptr members
    //     for (const auto& buffer : other.input_user_buff_vec) {
    //         input_user_buff_vec.push_back(std::make_unique<zdl::DlSystem::IUserBuffer>(*buffer));
    //     }

    //     for (const auto& buffer : other.output_user_buff_vec) {
    //         output_user_buff_vec.push_back(std::make_unique<zdl::DlSystem::IUserBuffer>(*buffer));
    //     }

    //     // Deep copy or transfer of ownership for unique_ptr members
    //     if (other.container) {
    //         // container = other.container->clone();
    //         container = std::make_unique<zdl::DlContainer::IDlContainer>(*other.container);
    //     }

    //     if (other.snpe) {
    //         snpe = std::make_unique<zdl::SNPE::SNPE>(*other.snpe);
    //     }
    // }

};


// rebuilds snpe to shape
// void reshapeSnpe(ModelRuntime& runtime, std::map<std::string, std::vector<size_t>> dims_dict) 
// {
//     // runtime.snpe = setBuilderOptions(runtime.container, runtime.runtimeList, true, dims_dict);
//     setBuilderOptions(runtime.container, runtime.runtimeList, true, dims_dict);
//     return;
// }

// setting model_name(key), and the absolute dlc path
std::map<std::string, ModelRuntime>* modelDictCreator(
    const std::set<std::pair<std::string, std::string>>& ModelNameAndPaths)   
{
    auto map_ptr = new std::map<std::string, ModelRuntime>();
    for (const auto& pair : ModelNameAndPaths) {
        const std::string& model_name = pair.first;
        const std::string& dlcPath = pair.second;
        (*map_ptr).emplace(model_name, ModelRuntime());
        (*map_ptr).at(model_name).dlc_path = dlcPath;
    }
    return map_ptr;
}

// could comment in later

template<typename T>
bool haveSameElements(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    // Check if the sizes are the same
    if (vec1.size() != vec2.size()) { return false; }
    // Create copies of the vectors to sort
    std::vector<T> sortedVec1 = vec1;
    std::vector<T> sortedVec2 = vec2;
    // Sort both vectors
    std::sort(sortedVec1.begin(), sortedVec1.end());
    std::sort(sortedVec2.begin(), sortedVec2.end());
    // Compare the sorted vectors
    return sortedVec1 == sortedVec2;
}

// ensures names of each ModelRuntime IO matches with their dlc
// bool verifyModelsIO(const std::map<std::string, ModelRuntime>& models) {
//     for (const auto& pair : models) {
//         const std::string& model_name = pair.first;
//         // verify inputs
//         std::vector<std::string> input_vec = StringListToVector(models.at(model_name).snpe->getInputTensorNames());
//         assert(input_vec.size() == models.at(model_name).inputNameToFileMap.size());
//         for (int i = 0; i < input_vec.size(); i++) {
//             const std::string& input_name = input_vec[i];
//             auto iter = models.at(model_name).inputNameToFileMap.find(input_name);
//             if (iter == models.at(model_name).inputNameToFileMap.end()) {
//                 return false;
//             }
//         }
//         // verify outputs
//         std::vector<std::string> output_vec = StringListToVector(models.at(model_name).snpe->getOutputTensorNames());
//         if (!haveSameElements(output_vec, models.at(model_name).outputs)) {
//             return false;
//         }
//     }
//     return true;
// }



// void loadInputMap(
//     std::map<std::string, ModelRuntime>& models,
//     const std::map<std::string, std::map<std::string, std::string>>& fileNameToBufferMaps) 
// {
//     assert(models.size() == fileNameToBufferMaps.size());
//     for (const auto& pair : models) {
//         const std::string& str = pair.first;
//         models[str].inputNameToFileMap = fileNameToBufferMaps.at(str);
//     }
// }

// void parse_argv_models(std::vector<ModelRuntime>& models, int argc, char* argv[]) {
//     std::string temp_str;
//     for (int i = 1; i < argc; ++i) {
//         if (argv[i][0] == '-') {
//             // std::cout << argv[i] << "\n";
//             // Identify the option type
//             switch (argv[i][1]) {
//                 case 'm':
//                     // New model
//                     models.emplace_back();
//                     models.back().model = argv[++i];
//                     continue;
//                 case 'o':
//                     // Output node(s)
//                     temp_str = argv[++i];
//                     models.back().outputs.push_back(temp_str + ":0");
//                     continue;
//                 case 'i':
//                     // Input(s)
//                     temp_str = argv[++i];
//                     temp_str = temp_str.substr(0, temp_str.find(".dat"));
//                     models.back().inputs.push_back(temp_str); // ex: "input1.dat" --> "input1"
//                     continue;
//                 default:
//                 continue;
//             }
//         }
//     }
// }

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

// void print_model_runtimes(const std::vector<ModelRuntime>& models) {
//     std::cout << "Model Information Parsed:\n";
//     for (const auto& model : models) {
//         std::cout << model.model << ",";
//         for (const auto& input : model.inputs)    {std::cout << " " << input;}
//         std::cout << ",";
//         for (const auto& output : model.outputs)  {std::cout << " " << output;}
//         std::cout << "\n";
//    }
//    std::cout << "---------------------------\n";
// }

void loadFile(std::vector<uint8_t>& vec, const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }
    // Determine the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    // Resize the vector to fit the file content
    vec.resize(fileSize);
    // Load the file content into the vector
    if (!file.read(reinterpret_cast<char*>(vec.data()), fileSize)) {
        throw std::runtime_error("Error reading file: " + filePath);
    }
    file.close();
}

// void quantizeSinCos(std::vector<uint8_t>& sin, std::vector<uint8_t> cos) {

// }


void loadAndQuantize(
    std::vector<uint8_t>& buff, 
    const std::string& filePath
) {
    // grab fp32 data
    std::vector<uint8_t> temp_buff;
    loadFile(temp_buff, filePath);
    // quantize to uint8
    // (this might be a horrible result, print the results)
    unsigned char stepEquivalentTo0 = 0;
    float quantizedStepSize = 0;
    FloatToTfN(buff.data(),
                stepEquivalentTo0,
                quantizedStepSize,
                false,
                (float*)temp_buff.data(),
                temp_buff.size() / 4,
                8);
    // can remove this later
    std::cout << "testing the results of the qunantization:\n";
    for (int i = 0; i < 5; i++) { std::cout << buff[i] << " "; } 
    std::cout << "\n";
}




void linkBuffers(
    std::map<std::string, ModelRuntime> *models,
    std::vector<uint8_t>& buff_1,
    // std::vector<uint8_t>& buff_2,
    std::vector<uint8_t>& buff_3,
    std::vector<uint8_t>& buff_4,
    std::vector<uint8_t>& buff_5,
    std::vector<uint8_t>& buff_6,
    std::vector<uint8_t>& buff_7,
    std::vector<uint8_t>& buff_8
) {

    // this may not be able to be reshaped
    (*models)["gelu"].applicationInputBuffers["input:0"] = &buff_8;
    (*models)["gelu"].applicationOutputBuffers["gelu_out:0"] = &buff_8;
    
    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"] = &buff_8; // new
        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["query_states:0"] = &buff_3;
        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["key_states:0"] = &buff_4;
        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["value_states:0"] = &buff_5;
        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["fc1_out:0"] = &buff_6;

        (*models)["P1_2_reshaped_layer_" + i_str].applicationInputBuffers["gelu_fc1_out:0"] = &buff_8;
        (*models)["P1_2_reshaped_layer_" + i_str].applicationOutputBuffers["feed_forward_hidden_states:0"] = &buff_6;
    }

    (*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"] = &buff_3;
    (*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"] = &buff_4;
    (*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"] = &buff_7;
    (*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"] = &buff_8;

    (*models)["P2_not_first_reshaped"].applicationInputBuffers["query_states_0:0"] = &buff_3;
    (*models)["P2_not_first_reshaped"].applicationInputBuffers["key_states_0:0"] = &buff_4;
    (*models)["P2_not_first_reshaped"].applicationInputBuffers["attention_mask:0"] = &buff_7;
    (*models)["P2_not_first_reshaped"].applicationOutputBuffers["attn_weights:0"] = &buff_8;

    (*models)["P3_first_buffered"].applicationInputBuffers["value_states_0:0"] = &buff_5;
    (*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"] = &buff_8;
    (*models)["P3_first_buffered"].applicationOutputBuffers["attn_output:0"] = &buff_3;

    (*models)["P3_not_first_buffered"].applicationInputBuffers["value_states:0"] = &buff_5;
    (*models)["P3_not_first_buffered"].applicationInputBuffers["attn_weights:0"] = &buff_8;
    (*models)["P3_not_first_buffered"].applicationOutputBuffers["attn_output:0"] = &buff_3;

    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        (*models)["P4_1_reshaped_layer_" + i_str].applicationInputBuffers["p3_out:0"] = &buff_3;
        (*models)["P4_1_reshaped_layer_" + i_str].applicationOutputBuffers["p4_1_out:0"] = &buff_4;
    }

    (*models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"] = &buff_8;
    (*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"] = &buff_1;
    (*models)["P4_2_reshaped"].applicationInputBuffers["p4_1_out:0"] = &buff_4;
    (*models)["P4_2_reshaped"].applicationOutputBuffers["decoder_output:0"] = &buff_3;
}

// std::string intialize_model_runtime(
//     std::vector<ModelRuntime>& runtimes, 
//     const std::vector<DlSystem::Runtime_t>& runtime_modes) 
// {
//     std::cout << "runtimes.size(): " << runtimes.size() << "\n"; // remove later
//     int num_models = runtimes.size();
//     assert(runtime_modes.size() == num_models);
//     for (int i = 0; i < num_models; i++) {
//         runtimes[i].applicationInputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
//         runtimes[i].applicationOutputBuffers = std::unordered_map<std::string, std::vector<uint8_t>>();
//         runtimes[i].inputMap = zdl::DlSystem::UserBufferMap();
//         runtimes[i].outputMap = zdl::DlSystem::UserBufferMap();
//         runtimes[i].runtime = checkRuntime(runtime_modes[i]);
//         runtimes[i].runtimeList.add(runtimes[i].runtime);
//         std::cout << "platform config options: " << runtimes[i].platformConfig.getPlatformOptions() << "\n";
//         // runtimes[i].platformConfig.

//         /* old way, chagen later back to new way */
//         // runtimes[i].container = loadContainerFromFile(runtimes[i].model);


//         std::cout << "loading file: " << runtimes[i].model << "\n";
//         #ifdef DEBUG
//             auto start = std::chrono::high_resolution_clock::now();
//         #endif 

//         /* new way of loading dlc */
//         loadFile(runtimes[i].dlc_buff, runtimes[i].model);
//         runtimes[i].container = loadContainerFromVector(runtimes[i].dlc_buff);

//         #ifdef DEBUG
//             auto end = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//             std::cout << "time to load model " << runtimes[i].model << ": " << duration << "ms\n";
//         #endif

//         #ifdef DEBUG
//             auto start_2 = std::chrono::high_resolution_clock::now();
//         #endif

//         // runtimes[i].snpe = setBuilderOptions(runtimes[i].container, runtimes[i].runtime, true);
        
//         /* testing reshaping */
//         std::unordered_map<std::string, std::vector<size_t>> new_map;
//         // new_map["vector:0"] = {2, 1, 1, int(1e3)};
//         new_map["vector:0"] = {4, 100 / 2};
//         runtimes[i].snpe = setBuilderOptions(runtimes[i].container, runtimes[i].runtime, true, new_map);

//         /* testing IO changes */
//         // std::unordered_map<std::string, std::tuple<std::vector<size_t>, zdl::DlSystem::IOBufferDataType_t>> dims_dict;
//         // dims_dict["vector:0"] = {{1, 1, 1, 1000}, zdl::DlSystem::IOBufferDataType_t::UINT_8};
//         // std::unordered_map<std::string, zdl::DlSystem::IOBufferDataType_t> output_datatypes;
//         // output_datatypes["matmul_out:0"] = zdl::DlSystem::IOBufferDataType_t::UINT_8;
//         // runtimes[i].snpe = setBuilderOptions(runtimes[i].container, runtimes[i].runtime, true, dims_dict, output_datatypes);


        
//         /* testing platformConfig*/
//         std::cout << "runtime: " <<  DlSystem::RuntimeList::runtimeToString(runtimes[i].runtime) << "\n";
//         std::cout << "using platformConfig case\n";
//         // runtimes[i].snpe = setBuilderOptions(runtimes[i].container, runtimes[i].runtimeList, true, runtimes[i].platformConfig);

//         // using modified one from example (remove later)
//         bool useUserSuppliedBuffers = true;
//         bool useCaching = false;
//         bool cpuFixedPointMode = false;
//         // runtimes[i].snpe = setBuilderOptions_ex(runtimes[i].container,
//         //                                         runtimes[i].runtime,
//         //                                         runtimes[i].runtimeList,
//         //                                         useUserSuppliedBuffers,
//         //                                         runtimes[i].platformConfig,
//         //                                         useCaching, cpuFixedPointMode);
        
        


//     //    runtimes[i].inputMap = zdl::DlSystem::UserBufferMap();
//     //     runtimes[i].outputMap = zdl::DlSystem::UserBufferMap();

//         #ifdef DEBUG
//             auto end_2 = std::chrono::high_resolution_clock::now();
//             auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count();
//             std::cout << "time to build model " << runtimes[i].model << ": " << duration_2 << "ms\n";
//         #endif
//     }
//     return std::to_string(reinterpret_cast<std::uintptr_t>(runtimes[0].container.get())) + "\n" + 
//             std::to_string(reinterpret_cast<std::uintptr_t>(runtimes[0].snpe.get()));
// }

// assume runtime_modes
std::string intialize_model_runtime(
    std::map<std::string, ModelRuntime>& runtimes, 
    const std::map<std::string, zdl::DlSystem::Runtime_t>& runtime_modes)
{
    std::cout << "runtimes.size(): " << runtimes.size() << "\n"; // remove later
    int num_models = runtimes.size();
    assert(runtime_modes.size() == num_models);

    for (auto const& pair : runtimes) {
        const std::string& model = pair.first;
        #ifdef DEBUG
            std::cout << "iterating through model: " << model << "\n";
        #endif
        runtimes[model].applicationInputBuffers = std::unordered_map<std::string, std::vector<uint8_t>*>();
        runtimes[model].applicationOutputBuffers = std::unordered_map<std::string, std::vector<uint8_t>*>();
        runtimes[model].inputMap = zdl::DlSystem::UserBufferMap();
        runtimes[model].outputMap = zdl::DlSystem::UserBufferMap();
        // runtime_modes[str];
        runtimes[model].runtime = checkRuntime(runtime_modes.at(model));
        std::cout << "checkedRunteim\n";
        runtimes[model].runtimeList.add(runtimes[model].runtime);
        std::cout << "platform config options: " << runtimes[model].platformConfig.getPlatformOptions() << "\n";
        // runtimes[str].platformConfig.

        /* old way, chagen later back to new way */
        // runtimes[str].container = loadContainerFromFile(runtimes[str].model);

        


        std::cout << "loading file: " << runtimes[model].dlc_path << "\n";
        #ifdef DEBUG
            auto start = std::chrono::high_resolution_clock::now();
        #endif 

        /* new way of loading dlc */
        loadFile(runtimes[model].dlc_buff, runtimes[model].dlc_path);
        runtimes[model].container = loadContainerFromVector(runtimes[model].dlc_buff);

        #ifdef DEBUG
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "time to load model " << runtimes[model].dlc_path << ": " << duration << "ms\n";
        #endif

        #ifdef DEBUG
            auto start_2 = std::chrono::high_resolution_clock::now();
        #endif

        // runtimes[str].snpe = setBuilderOptions(runtimes[str].container, runtimes[str].runtime, true);
        
        /* testing reshaping */
        // std::unordered_map<std::string, std::vector<size_t>> new_map;
        // new_map["vector:0"] = {2, 1, 1, int(1e3)};
        // new_map["vector:0"] = {4, 100 / 2};
        // runtimes[str].snpe = setBuilderOptions(runtimes[str].container, runtimes[str].runtime, true, new_map);

        /* testing IO changes */
        // std::unordered_map<std::string, std::tuple<std::vector<size_t>, zdl::DlSystem::IOBufferDataType_t>> dims_dict;
        // dims_dict["vector:0"] = {{1, 1, 1, 1000}, zdl::DlSystem::IOBufferDataType_t::UINT_8};
        // std::unordered_map<std::string, zdl::DlSystem::IOBufferDataType_t> output_datatypes;
        // output_datatypes["matmul_out:0"] = zdl::DlSystem::IOBufferDataType_t::UINT_8;
        // runtimes[str].snpe = setBuilderOptions(runtimes[str].container, runtimes[str].runtime, true, dims_dict, output_datatypes);


        
        /* testing platformConfig*/
        // std::cout << "runtime: " <<  DlSystem::RuntimeList::runtimeToString(runtimes[model].runtime) << "\n";
        // std::cout << "using platformConfig case\n";
        // runtimes[str].snpe = setBuilderOptions(runtimes[str].container, runtimes[str].runtimeList, true, runtimes[str].platformConfig);

        // using modified one from example (remove later)
        bool useUserSuppliedBuffers = true;
        bool useCaching = false;
        bool cpuFixedPointMode = false;
        runtimes[model].snpe = setBuilderOptions_ex(
            runtimes[model].container,
            runtimes[model].runtime,
            runtimes[model].runtimeList,
            useUserSuppliedBuffers,
            runtimes[model].platformConfig,
            useCaching, cpuFixedPointMode);
        
        


    //    runtimes[str].inputMap = zdl::DlSystem::UserBufferMap();
    //     runtimes[str].outputMap = zdl::DlSystem::UserBufferMap();

        runtimes[model].input_names = StringListToVector(runtimes[model].snpe->getInputTensorNames());
        std::cout << "finished with StringListToVector\n";
        runtimes[model].output_names = StringListToVector(runtimes[model].snpe->getOutputTensorNames());
        std::cout << "finished with StringListToVector\n";

        #ifdef DEBUG
            auto end_2 = std::chrono::high_resolution_clock::now();
            auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count();
            std::cout << "time to build model " << model << ": " << duration_2 << "ms\n\n";
        #endif
    }
    return "done";
    // return std::to_string(reinterpret_cast<std::uintptr_t>(runtimes[0].container.get())) + "\n" + 
    //         std::to_string(reinterpret_cast<std::uintptr_t>(runtimes[0].snpe.get()));
}

// currently only built for a single model, see test.cpp for other code for mulitple models
// void allocate_model_input_buffers(
//     std::vector<ModelRuntime>& runtimes,
//     std::vector<size_t> model_buffer_sizes,
//     bool debug,
//     size_t datasize, bool isTFBuffer)
// {
//     std::cout << "runtimes.size(): " << runtimes.size() << "\n"; // remove later
//     int num_models = runtimes.size();
//     for (int i = 0; i < num_models; i++) {
//         for (int j = 0; j < runtimes[i].inputs.size(); j++) {

//             /* restore this later */ 
//             std::string name = (*(runtimes[i].snpe->getInputTensorNames())).at(j);
//             runtimes[i].applicationInputBuffers[name] = std::vector<uint8_t>(model_buffer_sizes[j], 1);

//             if (debug) {std::cout << "calling createUserBuffer()\n";} 
//             createUserBuffer(runtimes[i].inputMap, runtimes[i].applicationInputBuffers,
//                 runtimes[i].input_user_buff_vec, runtimes[i].snpe, name.c_str(), datasize, isTFBuffer);
//             // return "Called createUserBuffer";
//             if (debug) {std::cout << "finished\n";}
//         }
//     }
// }



/* 
    NOTE: the reason b/c the two functions below are commented out are b/c of the 
            shared buffers changed to ModelRuntime
*/
// works for multiple models
// void allocate_model_input_buffers(
//     std::map<std::string, ModelRuntime>& runtimes,
//     const std::map<std::string, std::map<std::string, size_t>>& model_buffer_sizes,
//     size_t datasize, bool isTFBuffer,
//     uint8_t default_value)
// {
//     for (auto& pair : runtimes) {
//         std::string model_name = pair.first;
//         for (const auto& name_pair : model_buffer_sizes.at(model_name)) {
//             const std::string& name = name_pair.first;
//             runtimes[model_name].applicationInputBuffers[name] 
//                 = std::vector<uint8_t>(model_buffer_sizes.at(model_name).at(name), default_value);
//             #ifdef DEBUG
//                 std::cout << "calling createUserBuffer()\n";
//             #endif
//             createUserBuffer(runtimes[model_name].inputMap, runtimes[model_name].applicationInputBuffers,
//                 runtimes[model_name].input_user_buff_vec, runtimes[model_name].snpe, name.c_str(), datasize, isTFBuffer);
//             #ifdef DEBUG
//                 std::cout << "finished\n";
//             #endif
//         }
//     }
// }

// void allocate_model_output_buffers(
//     std::map<std::string, ModelRuntime>& runtimes,
//     const std::map<std::string, std::map<std::string, size_t>>& model_buffer_sizes,
//     size_t datasize, bool isTFBuffer,
//     uint8_t default_value)
// {
//     for (auto& pair : runtimes) {
//         std::string model_name = pair.first;
//         for (const auto& name_pair : model_buffer_sizes.at(model_name)) {
//             const std::string& name = name_pair.first;
//             runtimes[model_name].applicationOutputBuffers[name] 
//                 = std::vector<uint8_t>(model_buffer_sizes.at(model_name).at(name), default_value);
//             #ifdef DEBUG
//                 std::cout << "calling createUserBuffer()\n";
//             #endif
//             createUserBuffer(runtimes[model_name].outputMap, runtimes[model_name].applicationOutputBuffers,
//                 runtimes[model_name].output_user_buff_vec, runtimes[model_name].snpe, name.c_str(), datasize, isTFBuffer);
//             #ifdef DEBUG
//                 std::cout << "finished\n";
//             #endif
//         }
//     }
// }



// currently only built for a single model, see test.cpp for other code for mulitple models
// void allocate_model_output_buffers(
//     std::vector<ModelRuntime>& runtimes,
//     std::vector<size_t> model_output_sizes,
//     bool debug,
//     size_t datasize, bool isTFBuffer)
// {
//     std::cout << "runtimes.size(): " << runtimes.size() << "\n"; // remove later
//     // these assertions will be wrong later
//     int num_models = runtimes.size();
//     assert(num_models != 0);
//     assert(runtimes[0].outputs.size() == 1);
//     for (int i = 0; i < num_models; i++) {
//         for (int j = 0; j < runtimes[i].outputs.size(); j++) {
//             /* restore this later */
//             std::string name = (*(runtimes[i].snpe->getOutputTensorNames())).at(j);
//             std::cout << "output_name: " << name << "\n";
//             runtimes[i].applicationOutputBuffers[name] = std::vector<uint8_t>(model_output_sizes[j], 2);
//             if (debug) {std::cout << "calling createUserBuffer()\n";}


//             createUserBuffer(runtimes[i].outputMap, runtimes[i].applicationOutputBuffers,
//                 runtimes[i].output_user_buff_vec, runtimes[i].snpe, name.c_str(), datasize, isTFBuffer);
//             if (debug) {std::cout << "finished\n";}
//         }
//     }
// }


void create_user_buffers(
    std::map<std::string, ModelRuntime>& runtimes,
    size_t datasize, 
    bool isTFBuffer
) {
    for (auto& pair : runtimes) {
        const std::string& model_name = pair.first;
        #ifdef DEBUG
            std::cout << "\nMODELNAME: " << model_name << "\n";
        #endif
        // for (const auto& name_pair : model_buffer_sizes.at(model_name)) {
        for (const auto& input_name : runtimes[model_name].input_names) {
            #ifdef DEBUG
                std::cout << "\n";
            #endif
            createUserBuffer(runtimes[model_name].inputMap, runtimes[model_name].applicationInputBuffers,
                runtimes[model_name].input_user_buff_vec, runtimes[model_name].snpe, input_name.c_str(), datasize, isTFBuffer);
        }
        for (const std::string& output_name : runtimes[model_name].output_names) {
            #ifdef DEBUG
                std::cout << "\n";
            #endif
            createUserBuffer(runtimes[model_name].outputMap, runtimes[model_name].applicationOutputBuffers,
                runtimes[model_name].output_user_buff_vec, runtimes[model_name].snpe, output_name.c_str(), datasize, isTFBuffer);
        }
    }
}

void reshapeModels(
    std::map<std::string, ModelRuntime>& models,
    std::string model_name,
    const std::unordered_map<std::string, std::vector<size_t>>& new_map,
    size_t datasize
) {

    #ifdef DEBUG
        std::cout << "\t\treshapeModels() MODELNAME: " << model_name << "\n";
        std::cout << "\t\tmodels[model_name].snpe: "  
            << models[model_name].snpe.get() << "\n";
    #endif
    bool useUserSuppliedBuffers = true;
    bool useCaching = false;
    bool cpuFixedPointMode = false;
    models[model_name].snpe = setBuilderOptions_reshape(
        models[model_name].container,
        models[model_name].runtime,
        models[model_name].runtimeList,
        useUserSuppliedBuffers,
        useCaching, 
        cpuFixedPointMode,
        new_map);

    int i = 0;
    for (const auto& input_name : models[model_name].input_names) {
        #ifdef DEBUG
            std::cout << "input_name: " << input_name << "\n";
        #endif
        modifyUserBuffer(models[model_name].inputMap, models[model_name].applicationInputBuffers,
            models[model_name].input_user_buff_vec, models[model_name].snpe, input_name.c_str(), datasize, i);
        i++;
    }
    i = 0;
    for (const std::string& output_name : models[model_name].output_names) {
        #ifdef DEBUG
            std::cout << "output_name: " << output_name << "\n";
        #endif
        modifyUserBuffer(models[model_name].outputMap, models[model_name].applicationOutputBuffers,
            models[model_name].output_user_buff_vec, models[model_name].snpe, output_name.c_str(), datasize, i);
    }
}

void reshapeStuff(
    std::map<std::string, ModelRuntime> *models,
    const uint32_t iteration_num,
    const size_t datasize,
    const size_t tot_seq_len
) {
    if (iteration_num == 0) {
        reshapeModels(*models, "gelu",
            {
                {"input:0", {1, INTERMEDIATE_SIZE}}
            }, datasize);

        reshapeModels(*models, "P4_2_reshaped",
            {
                {"p4_1_out:0", {1, HIDDEN_SIZE}},
                {"feed_forward_hidden_states:0", {1, HIDDEN_SIZE}},
                {"residual:0", {1, HIDDEN_SIZE}},
            }, datasize);

        for (size_t i = 0; i < DECODERS; i++) {
            std::string i_str = std::to_string(i);

            reshapeModels(*models, "P1_1_reshaped_layer_" + i_str,
                {
                    {"hidden_states:0", {1, HIDDEN_SIZE}}
                }, datasize);

            reshapeModels(*models, "P1_2_reshaped_layer_" + i_str,
                {
                    {"gelu_fc1_out:0", {1, INTERMEDIATE_SIZE}}
                }, datasize);

            reshapeModels(*models, "P4_1_reshaped_layer_" + i_str,
                {
                    {"p3_out:0", {1, HIDDEN_SIZE}}
                }, datasize);
        }
    }

    reshapeModels(*models, "P2_not_first_reshaped",
        {
            {"query_states_0:0", {1, 32, 80}},
            {"key_states_0:0", {tot_seq_len, 32, 80}},
            {"attention_mask:0", {tot_seq_len}},
        }, datasize);
}

void freeModels(std::vector<ModelRuntime>* models) {
    for (int i = 0; i < (*models).size(); i++) {
        zdl::SNPE::SNPEFactory::terminateLogging();
        (*models)[i].snpe.reset();
    }
    delete models;
}



void freeModels(std::map<std::string, ModelRuntime>* models) {
    for (auto const& pair : *models)
    {
        zdl::SNPE::SNPEFactory::terminateLogging();
        const std::string& str = pair.first;
        (*models)[str].snpe.reset();
    }
    delete models;
}

template <typename T>
int loadFileIntoVec(std::string filePath, size_t numBytes, std::vector<T>& dataVec) {
    // check if buffer is large enough
    size_t buffer_size = sizeof(T) * dataVec.size();
    std::cout << "sizeof T: " << sizeof(T) << "\n";
    std::cout << "dataVec.size()" << dataVec.size() << "\n";
    if (numBytes > buffer_size) {
        std::cerr  << "buffer_size: " << buffer_size 
            << " while numBytes: " << numBytes << "\n";
        return 1;
    }
    // open file
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 2;
    }
    // Get the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    // Check if the requested number of bytes is greater than the file size
    if (numBytes > fileSize) {
        std::cerr << "Requested number of bytes exceeds file size." << std::endl;
        return 3;
    }
    // Read data into buffer
    file.read((char*)dataVec.data(), numBytes);
    // Check for errors during read
    if (!file) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return 4;
    }
    // Close the file
    file.close();
    return 0;
}

template <typename T>
int loadFileIntoVec(std::string filePath, std::vector<T>& dataVec) {
    // check if buffer is large enough
    size_t buffer_size = sizeof(T) * dataVec.size();
    std::cout << "sizeof T: " << sizeof(T) << "\n";
    std::cout << "dataVec.size()" << dataVec.size() << "\n";
    // open file
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 2;
    }
    // Get the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout << "fileSize: " << fileSize << "\n";
    // Check if the requested number of bytes is greater than the file size
    if (fileSize > buffer_size) {
        std::cerr << "File size exceeds buffer size." << std::endl;
        return 3;
    }
    // Read data into buffer
    file.read((char*)dataVec.data(), fileSize);
    // Check for errors during read
    if (!file) {
        std::cerr << "Error reading file: " << filePath << std::endl;
        return 4;
    }
    // Close the file
    file.close();
    return 0;
}

// made for a single model
// void intialize_input_buffers_custom(
//     std::vector<ModelRuntime>& runtimes,
//     const std::string& srcDir,
//     const std::vector<size_t>& inputFileSizes,
//     bool debug) 
// {
//     int num_models = runtimes.size();
//     assert(num_models == 1);
//     std::string filePath;
//     for (int i = 0; i < num_models; i++) {
//         for (int j = 0; j < runtimes[i].inputs.size(); j++) {
//             filePath = srcDir + "/" + runtimes[i].inputs[j];
//             if (debug) {
//                 std::cout << "filepath to read: " << filePath << "\n";
//                 // std::cout << "key: " << runtimes[i].inputs[j] << "\n";
//                 // std::cout << "vector_size: " 
//                 //     << runtimes[i].applicationInputBuffers[runtimes[i].inputs[j]].size() << "\n";
//             }
//             int ret_code = loadFileIntoVec(
//                 filePath, 
//                 inputFileSizes[j], 
//                 runtimes[i].applicationInputBuffers[(*(runtimes[i].snpe->getInputTensorNames())).at(j)]); 
//                 // runtimes[i].applicationInputBuffers[runtimes[i].inputs[j] + ":0"]); 
//             if (debug) {std::cout << "loadFilesIntoVec ret_code: " << ret_code << "\n";} 
//             assert(ret_code == 0);
//         }
//     }
// }

// works for multiple models
// COMEBACK
// void intialize_input_buffers_custom(
//     std::map<std::string, ModelRuntime>& runtimes,
//     const std::string& srcDir)
// {
//     std::string filePath;
//     for (auto& pair : runtimes) {
//         const std::string& model_name = pair.first;
//         for (const auto& name_pair : runtimes[model_name].inputNameToFileMap) {
//             const std::string& input_name = name_pair.first;
//             const std::string& file_name = name_pair.second;
//             filePath = srcDir + "/" + file_name;
//             #ifdef DEBUG
//                 std::cout << "for model(" << model_name << "), for input(" << input_name 
//                     << "), loading from file path(" << filePath  << ")\n";
//             #endif
//             int ret_code = loadFileIntoVec(
//                 filePath, 
//                 *(runtimes[model_name].applicationInputBuffers[input_name]));
//         }
//     }
// }

// void saveBuffer(const ModelRuntime& model_runtime, const std::string& OutputDir) {
//     #ifdef DEBUG
//         std::cout << "calling outputMap.getuserBufferNames\n";
//     #endif
//     const zdl::DlSystem::StringList& outputBufferNames = model_runtime.outputMap.getUserBufferNames();
//     #ifdef DEBUG
//         std::cout << "finished\n";
//     #endif 
    
//     // Iterate through output buffers and print each output to a raw file
//     int i = 0;
//     std::for_each(outputBufferNames.begin(), outputBufferNames.end(), [&](const char* name)
//     {
//         #ifdef DEBUG
//             std::cout << "saveBuffer start iteration:" << i << "\n";
//         #endif
//         std::ostringstream path;
//         path << OutputDir << "/" << model_runtime.outputs[i] << ".raw";
//         SaveUserBuffer(path.str(), model_runtime.applicationOutputBuffers.at(name));
//         #ifdef DEBUG
//             std::cout << "saveBuffer end iteration:" << i << "\n";
//         #endif
//         i++;
//     });
//     if (i <= 0) {
//         std::cerr << "Error: No outputs were written to\n";
//     }
//     // assert(i > 0); 
// }


void execute(std::map<std::string, ModelRuntime>& models, const std::string& model_name) {
    #ifdef DEBUG
        std::cout << "\nEXECUTION STAGE <" << model_name << ">\n";
        for (const std::string& input_name : models[model_name].input_names) {
            printN(input_name, models[model_name].applicationInputBuffers[input_name]->data(), N_PRINT);
        }
        models[model_name].snpe->execute(models[model_name].inputMap, models[model_name].outputMap);
        for (const std::string& output_name : models[model_name].output_names) {
            printN(output_name, models[model_name].applicationOutputBuffers[output_name]->data(), N_PRINT);
    }
    #else
        models[model_name].snpe->execute(models[model_name].inputMap, models[model_name].outputMap);
    #endif
}



template <typename T>
void resetKV(T* in) {
    uint8_t* in_8 = (uint8_t*)in;
    T* in_past_keys;
    T* in_past_values;
    uint32_t* in_key_dims;
    uint32_t* in_value_dims;
    for (uint64_t i = 0; i < DECODERS; i++) {
        in_past_keys =   (T*)&in_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*i)];
        in_past_values = (T*)&in_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (2 + 2*i)];
        in_key_dims = (uint32_t*)(&in_past_keys[MAX_SEQ_LEN * HIDDEN_SIZE]);
        in_value_dims = (uint32_t*)(&in_past_values[MAX_SEQ_LEN * HIDDEN_SIZE]);
        for (uint64_t j = 0; j < 4; j++) {
            in_key_dims[j]   = 0;
            in_value_dims[j] = 0;
        }
    }
}

template <typename T>
size_t multiReduce(const std::vector<T>& dims) {
    size_t total_elements = 1;
    for (auto i : dims) { total_elements *= i; }
    return total_elements;
}

void init_dims_uint32_t(std::vector<uint32_t>& vec, uint8_t *ptr, int num) {
  vec = std::vector<uint32_t>();
  uint32_t* ptr_32 = (uint32_t*)ptr;
  std::cout << "grabbing dimensions: ";
  assert(num <= 4);
  for (int i = 4 - num; i < 4; i++) { 
    std::cout << ptr_32[i] << " ";
    vec.push_back(ptr_32[i]); 
    }
    std::cout << "\n";
}

template <typename T>
void copyKV(T* in, T* out) {
    uint8_t* in_8 = (uint8_t*)in;
    uint8_t* out_8 = (uint8_t*)out;
    T *in_past_keys, *in_past_values, *out_past_keys, *out_past_values;
    uint32_t *in_key_dims, *in_value_dims, *out_key_dims, *out_value_dims;
    std::vector<uint32_t> key_dims_vec;
    std::vector<uint32_t> val_dims_vec;
    size_t k_size, v_size;

    for (uint64_t i = 0; i < DECODERS; i++) {
        /* calculating locations */
        in_past_keys =   (T*)&in_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*i)];
        in_past_values = (T*)&in_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (2 + 2*i)];
        out_past_keys =   (T*)&out_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*i)];
        out_past_values = (T*)&out_8[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (2 + 2*i)];

        in_key_dims = (uint32_t*)(&in_past_keys[MAX_SEQ_LEN * HIDDEN_SIZE]);
        in_value_dims = (uint32_t*)(&in_past_values[MAX_SEQ_LEN * HIDDEN_SIZE]);
        out_key_dims = (uint32_t*)(&out_past_keys[MAX_SEQ_LEN * HIDDEN_SIZE]);
        out_value_dims = (uint32_t*)(&out_past_values[MAX_SEQ_LEN * HIDDEN_SIZE]);

        /* copying dims */
        for (uint64_t j = 0; j < 4; j++) {
            out_key_dims[j] = in_key_dims[j];
            out_value_dims[j] = in_value_dims[j];
        }

        /* ensure KV cache is not empty */
        assert(in_key_dims[3] != 0);
        assert(in_value_dims[3] != 0);

        /* grabbing dims */
        init_dims_uint32_t(key_dims_vec, (uint8_t*)in_key_dims , 4);
        init_dims_uint32_t(val_dims_vec, (uint8_t*)in_value_dims , 4);

        /* writing data */
        k_size = multiReduce(key_dims_vec);
        v_size = multiReduce(val_dims_vec);
        assert(k_size != 0 && v_size != 0);
        for (size_t j = 0; j < k_size; j++) {out_past_keys[j] = in_past_keys[j];}
        for (size_t j = 0; j < v_size; j++) {out_past_values[j] = in_past_values[j];}
    }
}


// generate mask and position_ids; also iteration_num should first be 0
void prepareInputs_old(float* mask, int* position_ids, uint32_t seq_len, uint32_t iteration_num)
{
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
        uint32_t* ptr32 = (uint32_t*)mask;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = seq_len;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = seq_len;
        // set mask
        std::cout << "writing mask\n";
        float lowest = std::numeric_limits<float>::lowest();
        for (uint32_t row = 0; row < seq_len; row++) {
            for (uint32_t col = 0; col < seq_len; col++) {
                // std::cout << "(row, col): (" << row << ", " << col << ")\n";
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
        uint32_t* ptr32 = (uint32_t*)mask;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 0] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 1] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 2] = 1;
        ptr32[MAX_SEQ_LEN * MAX_SEQ_LEN + 3] = seq_len;
        // set mask
        for (uint32_t i = 0; i < seq_len; i++) { mask[i] = 0; }
    }
}

// iteration_num should first be 0
template <typename T>
void prepareMask(T* mask, size_t seq_len, size_t iteration_num)
{
    if (iteration_num == 0) {
        // set mask
        T lowest = std::numeric_limits<T>::lowest();
        for (size_t row = 0; row < seq_len; row++) {
            for (size_t col = 0; col < seq_len; col++) {
                // std::cout << "(row, col): (" << row << ", " << col << ")\n";
                if (row >= col) { mask[row*seq_len + col] = 0; }
                else            { mask[row*seq_len + col] = lowest; } 
            }
        }
    }
    else {
        // set mask
        for (size_t i = 0; i < seq_len; i++) { mask[i] = 0; }
    }
}

// could template
template <typename T>
void reshapeToBufferedBeforeP2first(
    size_t seq_len, 
    size_t tot_seq_len,
    T* temp_buff, 
    std::map<std::string, ModelRuntime> *models
) {
    reshaped_to_buffered(
        {1, 32, seq_len, 80},
        {1, 32, MAX_SEQ_LEN, 80},
        static_cast<T>(0),
        (T*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
        temp_buff,
        (T*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data()
    );
    reshaped_to_buffered(
        {1, 32, tot_seq_len, 80},
        {1, 32, MAX_SEQ_LEN, 80},
        static_cast<T>(0),
        (T*)(*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"]->data(),
        temp_buff,
        (T*)(*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"]->data()
    );
    reshaped_to_buffered(
        {1, 32, tot_seq_len, 80},
        {1, 32, MAX_SEQ_LEN, 80},
        static_cast<T>(0),
        (T*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states_0:0"]->data(),
        temp_buff,
        (T*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states_0:0"]->data()
    );
    reshaped_to_buffered(
        {1, 32, seq_len, seq_len},
        {1, 32, MAX_SEQ_LEN, MAX_SEQ_LEN},
        std::numeric_limits<T>::lowest(),
        (T*)(*models)["P2_not_first_reshaped"].applicationInputBuffers["attention_mask:0"]->data(),
        temp_buff,
        (T*)(*models)["P2_not_first_reshaped"].applicationInputBuffers["attention_mask:0"]->data()
    );
}

template <typename T>
void reshapeToBufferedBeforeP3notFirst(
    size_t tot_seq_len,
    T* temp_buff,
    std::map<std::string, ModelRuntime> *models
) {
    reshaped_to_buffered(
        {1, 32, tot_seq_len, 80},
        {1, 32, MAX_SEQ_LEN, 80},
        static_cast<T>(0),
        (T*)(*models)["P3_not_first_buffered"].applicationInputBuffers["value_states:0"]->data(),
        temp_buff,
        (T*)(*models)["P3_not_first_buffered"].applicationInputBuffers["value_states:0"]->data()
    );
    reshaped_to_buffered(
        {1, 32, 1, tot_seq_len},
        {1, 32, 1, MAX_SEQ_LEN},
        static_cast<T>(0),
        (T*)(*models)["P3_not_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
        temp_buff,
        (T*)(*models)["P3_not_first_buffered"].applicationInputBuffers["attn_weights:0"]->data()
    );
}

template <typename T>
void bufferedToReshapeBeforeP4(
    size_t seq_len,
    const std::string& i_str,
    T* temp_buff,
    std::map<std::string, ModelRuntime> *models
) {
    buffered_to_reshaped(
        {MAX_SEQ_LEN, HIDDEN_SIZE},
        {seq_len, HIDDEN_SIZE},
        (T*)(*models)["P4_1_reshaped_layer_" + i_str].applicationInputBuffers["p3_out:0"]->data(),
        temp_buff,
        (T*)(*models)["P4_1_reshaped_layer_" + i_str].applicationInputBuffers["p3_out:0"]->data()
    );
}

typedef unsigned short ushort;
typedef unsigned int uint;
uint as_uint(const float x) {
    return *(uint*)&x;
}
float as_float(const uint x) {
    return *(float*)&x;
}
float half_to_float(const ushort x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x&0x7C00)>>10; // exponent
    const uint m = (x&0x03FF)<<13; // mantissa
    const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}
ushort float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint e = (b&0x7F800000)>>23; // exponent
    const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

void fp16_to_fp32(ushort* in, float* out, const std::vector<uint32_t>& dims) {
    assert((void*)in != (void*)out);
    size_t num_elem = dims.end()[-1];
    for (int i = 0; i < dims.size()-1; i++) { num_elem *= dims[i]; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = half_to_float(in[i]);
    }
}
void fp32_to_fp16(float* in, ushort* out, const std::vector<uint32_t>& dims) {
    assert((void*)in != (void*)out);
    size_t num_elem = dims.end()[-1];
    for (int i = 0; i < dims.size()-1; i++) { num_elem *= dims[i]; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = float_to_half(in[i]);
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

template <typename T>
void printN(const std::string& str, const T* vec, const size_t N) {
    std::cout << str;
    std::cout << ": [";
    for (size_t i = 0; i < N; i++) {
        std::cout << vec[i];
        std::cout << ", ";
    }
    std::cout << "]\n";
}

// does not really work with every datatype
template <typename T>
void printT(
    const std::string& str, const std::vector<uint32_t>& dims, const T* tensor, 
    bool precise, size_t max_elem
    ) {
    // print dims
    printV(str, dims);
    // calculate size of each dimension
    std::vector<uint32_t> dim_sizes = {dims.end()[-1]};
    for (int i = dims.size()-2; i >= 0; i--) {
        uint32_t offset = dims[i] * dim_sizes.end()[-1];
        dim_sizes.push_back(offset);
    }
    // calculate total number of elements
    size_t tot_elements = 1;
    for (auto i : dims) { tot_elements *= i; }
    // print everything
    for (auto i : dims) { std::cout << "["; }
    for (size_t i = 0; i < tot_elements; i++) {
        if (i > max_elem) {break;}
        for (auto j : dim_sizes) {
            if (i % j == 0 && i != 0) {
                std::cout << "]\n[";
            }
        }
        if (precise) { std::cout << std::setprecision(30) 
            << half_to_float((ushort)tensor[i]) << ", "; }
        else { std::cout << half_to_float((ushort)tensor[i]) << ", "; }
    }
    for (auto i : dims) { std::cout << "]"; }
    std::cout << "\n";
}



#endif