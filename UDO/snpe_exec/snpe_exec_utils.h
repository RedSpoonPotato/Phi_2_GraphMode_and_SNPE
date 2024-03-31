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
    std::vector<size_t> model_input_sizes,
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
    size_t model_output_size,
    bool debug)
{
    std::cout << "runtimes[0].outputs.size():" << runtimes[0].outputs.size() << "\n";
    int num_models = runtimes.size();
    for (int i = 0; i < num_models; i++) {
        for (int j = 0; j < runtimes[i].outputs.size(); j++) {
            runtimes[i].applicationOutputBuffers[runtimes[i].outputs[j]] 
               = std::vector<u_int8_t>(model_output_size);
            if (debug) {std::cout << "calling createUserBuffer()\n";}
            createUserBuffer(runtimes[i].outputMap, runtimes[i].applicationOutputBuffers,
                runtimes[i].output_user_buff_vec, runtimes[i].snpe, (runtimes[i].outputs[j]).c_str());
            if (debug) {std::cout << "finished\n";}
        }
    }
}

template <typename T>
int loadFilesIntoVec(std::string filePath, size_t numBytes, std::vector<T>& dataVec) {
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

// made for a single model
void intialize_input_buffers_custom(
    std::vector<ModelRunetime>& runtimes,
    const std::string& srcDir,
    const std::vector<size_t>& inputSizes,
    bool debug) 
{
    int num_models = runtimes.size();
    assert(num_models == 1);
    std::string filePath;
    for (int i = 0; i < num_models; i++) {
        for (int j = 0; j < runtimes[i].inputs.size(); j++) {
            filePath = srcDir + "/" + runtimes[i].inputs[j] + ".dat";
            if (debug) {
                std::cout << "filepath to read: " << filePath << "\n";
                // std::cout << "key: " << runtimes[i].inputs[j] << "\n";
                // std::cout << "vector_size: " 
                //     << runtimes[i].applicationInputBuffers[runtimes[i].inputs[j]].size() << "\n";
            }
            int ret_code = loadFilesIntoVec(
                filePath, 
                inputSizes[j], 
                runtimes[i].applicationInputBuffers[runtimes[i].inputs[j] + ":0"]);
            if (debug) {std::cout << "loadFilesIntoVec ret_code: " << ret_code << "\n";}
            assert(ret_code == 0);
        }
    }
}

// dont use this
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
void prepareInputs(float* mask, int* position_ids, uint32_t seq_len, uint32_t iteration_num)
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


void printT(
    const std::string& str, const std::vector<uint32_t>& dims, const datatype* tensor, 
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
