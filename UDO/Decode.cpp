//==============================================================================
// Auto Generated Code for DecodePackage
//==============================================================================
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace decode {

typedef uint16_t datatype; // idk if this should be outside the namespace

// MACROS
#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560
#define QUERY_STATES_BUFF_SIZE  (BATCH_SIZE*MAX_SEQ_LEN*HIDDEN_SIZE)

#define INTERMEDIATE_SIZE 10240
#define INTERMEDIATE_STATES_BUFF_SIZE (BATCH_SIZE*MAX_SEQ_LEN*INTERMEDIATE_SIZE)

// partial_rotary_factor: 0.4
// head_dim = hidden_size // num_attention_heads i.e. (2560 / 32) = 80
// sin_cos(param:dim) = head_dim * partial_rotary_factor = 80 * .4 = 32
#define SIN_COS_DIM 32
#define SIN_COS_MAX_SEQ_LEN 2048 // a temporary solution
#define SIN_COS_BUFF_SIZE  (SIN_COS_DIM*SIN_COS_MAX_SEQ_LEN)

#define ATTN_WEIGHTS_SIZE (MAX_SEQ_LEN*MAX_SEQ_LEN*32)

#define LARGE_BUFFER_SIZE ATTN_WEIGHTS_SIZE
#define DATASIZE 2

#define DECODER_WEIGHT_SIZE 78671360
#define LM_WEIGHTS_SIZE 131128320

#define VOCAB_SIZE 51200

#define DECODERS 4

// taken from : https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion/60047308#60047308
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



void add_32f(const float* ten1, const float* ten2, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] + ten2[i];
    }
}

void sub_32f(const float* ten1, const float* ten2, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] - ten2[i];
    }
}

void mul_32f(const float* ten1, const float* ten2, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] * ten2[i];
    }
}

void div_32f(const float* ten1, const float* ten2, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] / ten2[i];
    }
}

// vector print function
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

bool range(int a, int b, int c) {
    return (a >= b && a <= c);
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

void printT_Alt(
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
    for (size_t i = 0; i < max_elem; i++) {
       std::cout << half_to_float((ushort)tensor[i]) << ", ";
    }
    std::cout << " ... ";
    for (size_t i = tot_elements-max_elem; i < tot_elements; i++) {
       std::cout << half_to_float((ushort)tensor[i]) << ", ";
    }
    for (auto i : dims) { std::cout << "]"; }
    std::cout << "\n";
}


void printT_32f(
    const std::string& str, const std::vector<uint32_t>& dims, const float* tensor, 
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
            << tensor[i] << ", "; }
        else { std::cout << tensor[i] << ", "; }
    }
    for (auto i : dims) { std::cout << "]"; }
    std::cout << "\n";
}

void printT_32f_Alt(
    const std::string& str, const std::vector<uint32_t>& dims, const float* tensor, 
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
    for (size_t i = 0; i < max_elem; i++) {
       std::cout << tensor[i] << ", ";
    }
    std::cout << " ... ";
    for (size_t i = tot_elements-max_elem; i < tot_elements; i++) {
       std::cout << tensor[i] << ", ";
    }
    for (auto i : dims) { std::cout << "]"; }
    std::cout << "\n";
}

void DownUpCast(float* tensor, std::vector<uint32_t> dims) {
    size_t tot_elem = 1;
    for (auto i : dims) { tot_elem *= i; }
    datatype* temp_buff = (datatype*)malloc(tot_elem * sizeof(datatype));
    fp32_to_fp16(tensor, temp_buff, dims);
    fp16_to_fp32(temp_buff, tensor, dims);
    free(temp_buff);
}

void add_32f_general(
    const float* ten1, const float* ten2, float* out, 
    const std::vector<uint32_t>& ten1_dims, const std::vector<uint32_t>& ten2_dims,
    std::vector<uint32_t>& out_dims
) {
    assert(ten1_dims.size() == ten2_dims.size());
    // compute out_dims
    out_dims = std::vector<uint32_t>();
    for (size_t i = 0; i < ten1_dims.size(); ++i) {
        assert(ten1_dims[i] == 1 || ten2_dims[i] == 1 || ten1_dims[i] == ten2_dims[i]);
        out_dims.push_back(std::max(ten1_dims[i], ten2_dims[i]));
    }
    // Perform tensor addition
    size_t total_elements = 1;
    for (size_t i = 0; i < out_dims.size(); ++i) {
        total_elements *= out_dims[i];
    }
    for (size_t i = 0; i < total_elements; ++i) {
        size_t index = i;
        size_t ten1_index = 0;
        size_t ten2_index = 0;
        for (int dim = out_dims.size() - 1; dim >= 0; --dim) {
            size_t dim_size = out_dims[dim];
            size_t ten1_dim_size = ten1_dims[dim];
            size_t ten2_dim_size = ten2_dims[dim];
            size_t ten1_coord = index % dim_size;
            size_t ten2_coord = index % dim_size;
            ten1_index += ten1_coord * (ten1_dim_size == 1 ? 0 : 1);
            ten2_index += ten2_coord * (ten2_dim_size == 1 ? 0 : 1);
            index /= dim_size;
        }
        out[i] = ten1[ten1_index] + ten2[ten2_index];
    }
}

// can add tensors of different shapes assuming they are compatible
void add_general_16f_cpu(
    const datatype* ten1_q, const datatype* ten2_q, datatype* out_q,
    float* ten1, float* ten2, float* out,
    const std::vector<uint32_t>& ten1_dims, const std::vector<uint32_t>& ten2_dims,
    std::vector<uint32_t>& out_dims
) {
    fp16_to_fp32((ushort*)ten1_q, (float*)ten1, ten1_dims);
    fp16_to_fp32((ushort*)ten2_q, (float*)ten2, ten2_dims);
    add_32f_general(ten1, ten2, out, ten1_dims, ten2_dims, out_dims);
    fp32_to_fp16((float*)out, (ushort*)out_q, out_dims);
}

void mul_32f_general(
    const float* ten1, const float* ten2, float* out, 
    const std::vector<uint32_t>& ten1_dims, const std::vector<uint32_t>& ten2_dims,
    std::vector<uint32_t>& out_dims
) {
    assert(ten1_dims.size() == ten2_dims.size());
    // compute out_dims
    out_dims = std::vector<uint32_t>();
    for (size_t i = 0; i < ten1_dims.size(); ++i) {
        assert(ten1_dims[i] == 1 || ten2_dims[i] == 1 || ten1_dims[i] == ten2_dims[i]);
        out_dims.push_back(std::max(ten1_dims[i], ten2_dims[i]));
    }
    // Perform tensor addition
    size_t total_elements = 1;
    for (size_t i = 0; i < out_dims.size(); ++i) {
        total_elements *= out_dims[i];
    }
    for (size_t i = 0; i < total_elements; ++i) {
        size_t index = i;
        size_t ten1_index = 0;
        size_t ten2_index = 0;
        for (int dim = out_dims.size() - 1; dim >= 0; --dim) {
            size_t dim_size = out_dims[dim];
            size_t ten1_dim_size = ten1_dims[dim];
            size_t ten2_dim_size = ten2_dims[dim];
            size_t ten1_coord = index % dim_size;
            size_t ten2_coord = index % dim_size;
            ten1_index += ten1_coord * (ten1_dim_size == 1 ? 0 : 1);
            ten2_index += ten2_coord * (ten2_dim_size == 1 ? 0 : 1);
            index /= dim_size;
        }
        out[i] = ten1[ten1_index] * ten2[ten2_index];
    }
}

void mul_general_16f_cpu(
    const datatype* ten1_q, const datatype* ten2_q, datatype* out_q,
    float* ten1, float* ten2, float* out,
    const std::vector<uint32_t>& ten1_dims, const std::vector<uint32_t>& ten2_dims,
    std::vector<uint32_t>& out_dims
) {
    fp16_to_fp32((ushort*)ten1_q, (float*)ten1, ten1_dims);
    fp16_to_fp32((ushort*)ten2_q, (float*)ten2, ten2_dims);
    mul_32f_general(ten1, ten2, out, ten1_dims, ten2_dims, out_dims);
    fp32_to_fp16((float*)out, (ushort*)out_q, out_dims);
}

// vertical expansion
void flatten_along_row(std::vector<uint32_t>& dims) {
  assert(dims.size() >= 2);
  long long int num_rows = dims.end()[-2]; // could cause issues for gpu
  long long int num_cols = dims.end()[-1];
  for (int i = 0; i < dims.size()-2; i++) { num_rows *= dims[i];}
  dims = std::vector<u_int32_t>(2);
  dims[0] = num_rows;
  dims[1] = num_cols;
//   for (auto i : dims) {std::cout << i << " ";}
}

// horizontal expansion
void flatten_along_col(std::vector<uint32_t>& dims) {
  assert(dims.size() >= 2);
  long long int num_rows = dims.end()[-2];
  long long int num_cols = dims.end()[-1]; // could cause issues for gpu
  for (int i = 0; i < dims.size()-2; i++) { num_cols *= dims[i];}
  dims = std::vector<u_int32_t>(2);
  dims[0] = num_rows;
  dims[1] = num_cols;
//   for (auto i : dims) {std::cout << i << " ";}
}

void flatten_to_3d(std::vector<uint32_t>& dims) {
    std::vector<uint32_t> temp_dims = {1,1,1};
    for (int i = 0; i < dims.size()-2; i++) { 
        temp_dims[0] *= dims[i];
    }
    temp_dims.end()[-1] = dims.end()[-1];
    temp_dims.end()[-2] = dims.end()[-2];
    dims = temp_dims;
}

// does not set output dims, use constrained version for that
void matmul_Nd_32f(const float* ten1, const float* ten2, float* out, 
                   std::vector<uint32_t> dims1, std::vector<uint32_t> dims2) {
    assert(dims1.end()[-1] == dims2.end()[-2]); // rule of matrix multiplication
    // use ints, as uint32_t's seem to segfault if they go negative
    printV("dims1", dims1);
    printV("dims2", dims2);
    // flatten
    flatten_to_3d(dims1);
    flatten_to_3d(dims2);
    std::cout << "dims after flattening:\n";
    printV("dims1", dims1);
    printV("dims2", dims2);
    uint32_t rows1 = dims1.end()[-2];
    uint32_t cols1 = dims1.end()[-1];
    uint32_t rows2 = dims2.end()[-2];
    uint32_t cols2 = dims2.end()[-1];
    assert(cols1 == rows2); // rule of matrix multiplication
    assert(dims1[0] == dims1[0]);
    assert(dims1.size() == 3);
    int offset = dims1.end()[-2] * dims2.end()[-1];
    for (uint32_t z = 0; z < dims1[0]; z++) {
        // 2d matmul algorithm
        for (uint32_t i = 0; i < rows1; ++i) {
            for (uint32_t j = 0; j < cols2; ++j) {
                float sum = 0.0;
                for (uint32_t k = 0; k < rows2; ++k) {
                    sum += ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset];
                }
                out[i*cols2 + j + z*offset] = sum;
            }
        }
    }
}

// makes certain assumptions to set output dimensions
void matmul_Nd_32f_constrained(const float* ten1, const float* ten2, float* out, 
                   std::vector<uint32_t> dims1, std::vector<uint32_t> dims2,
                   std::vector<uint32_t>& out_dims) {

    assert(dims1.end()[-1] == dims2.end()[-2]); // rule of matrix multiplication
    // set out_dims (assuming additional contraints below)
    assert(dims1.size() == dims2.size());
    int rank = dims1.size();
    out_dims = std::vector<uint32_t>();
    for (int i = 0; i < rank - 1; i++) { out_dims.push_back(dims1[i]); }
    out_dims.push_back(dims2.end()[-1]);
    // calling matmul
    matmul_Nd_32f(ten1, ten2, out, dims1, dims2);
}

void matmul_Nd_constrained_16f_cpu(
    const datatype* ten1_q, const datatype* ten2_q, datatype* out_q,
    float* ten1, float* ten2, float* out, 
    std::vector<uint32_t> dims1, std::vector<uint32_t> dims2,
    std::vector<uint32_t>& out_dims
) {
    fp16_to_fp32((ushort*)ten1_q, (float*)ten1, dims1);
    fp16_to_fp32((ushort*)ten2_q, (float*)ten2, dims2);
    matmul_Nd_32f_constrained(ten1, ten2, out, dims1, dims2, out_dims);
    fp32_to_fp16((float*)out, (ushort*)out_q, out_dims);
}



// set bias to nullptr if None
void linear_Nd_32f(const float* ten, const float* weight, const float* bias, float* out,
                   const std::vector<uint32_t>& ten_dims, const std::vector<uint32_t>& weight_dims,
                   std::vector<uint32_t>& out_dims) {
    assert(ten_dims.size() >= 2);
    assert(weight_dims.size() >= 2);
    // weight: (in, out)
    assert(ten_dims.end()[-1] == weight_dims.end()[-2]);
    // matmul
    matmul_Nd_32f(ten, weight, out, ten_dims, weight_dims);
    if (bias == nullptr) { return; }
    // bias
    int num_vectors = 1;
    int out_size = weight_dims.end()[-1];
    for (int i = 0; i < ten_dims.size() - 1; i++) { num_vectors *= ten_dims[i]; }
    for (int i = 0; i < num_vectors; i++) {
      for (int j = 0; j < out_size; j++) {
        out[i*out_size + j] += bias[j];
      }
    }
    // output tensor size
    out_dims = std::vector<uint32_t>();
    for (int i = 0; i < ten_dims.size()-1; i++) { out_dims.push_back(ten_dims[i]); }
    out_dims.push_back(out_size);
}


void linear_Nd_16f_cpu(
    const datatype* ten_q, const datatype* weight_q, const datatype* bias_q, datatype* out_q,
    float* ten, float* weight, float* out,
    const std::vector<uint32_t>& ten_dims, const std::vector<uint32_t>& weight_dims,
    std::vector<uint32_t>& out_dims
) {
    fp16_to_fp32((ushort*)ten_q, (float*)ten, ten_dims);
    fp16_to_fp32((ushort*)weight_q, (float*)weight, weight_dims);
    // linear code
    assert(ten_dims.size() >= 2);
    assert(weight_dims.size() >= 2);
    // weight: (in, out)
    assert(ten_dims.end()[-1] == weight_dims.end()[-2]);
    // matmul
    matmul_Nd_32f(ten, weight, out, ten_dims, weight_dims);
    if (bias_q == nullptr) { return; }
    // reuse weight buffer to hold 32-bit bias
    float* bias = weight;
    std::vector<uint32_t> bias_dims = {weight_dims.end()[-1]};
    fp16_to_fp32((ushort*)bias_q, (float*)bias, bias_dims);
    // bias
    int num_vectors = 1;
    int out_size = weight_dims.end()[-1];
    for (int i = 0; i < ten_dims.size() - 1; i++) { num_vectors *= ten_dims[i]; }
    for (int i = 0; i < num_vectors; i++) {
      for (int j = 0; j < out_size; j++) {
        out[i*out_size + j] += bias[j];
      }
    }
    // output tensor size
    out_dims = std::vector<uint32_t>();
    for (int i = 0; i < ten_dims.size()-1; i++) { out_dims.push_back(ten_dims[i]); }
    out_dims.push_back(out_size);
    // convert to fp16    
    fp32_to_fp16((float*)out, (ushort*)out_q, out_dims);
}

// 1d
void layernorm_1d_32f(
    const float* vec, const float* weight, const float* bias, float* out,
    const int vec_len, const int weight_len, const float eps
) {
    // mean
    float mean = 0;
    for (int i = 0; i < vec_len; i++) { mean += vec[i]; }
    mean = mean / vec_len;
    // variance
    float variance = 0;
    for (int i = 0; i < vec_len; i++) { 
        variance += pow(vec[i] - mean, 2); 
    }
    variance = variance / vec_len;
    float variance_plus_eps_sprt = sqrt(variance + eps);
    // output
    for (int i = 0; i < vec_len; i++) { 
        out[i] = (vec[i] - mean) / variance_plus_eps_sprt; 
        out[i] = (out[i] * weight[i]) + bias[i];
    }
}

void layernorm_Nd_32f(
    const float* tensor, const float* weight, const float* bias, float* out,
    const std::vector<uint32_t>& tensor_dims, const int weight_len,
    const float eps
) {
    int num_vectors = 1;
    int vec_len = tensor_dims.end()[-1];
    for (int i = 0; i < tensor_dims.size() - 1; i++) { num_vectors *= tensor_dims[i]; }
    for (int i = 0; i < num_vectors; i++) {
        layernorm_1d_32f(
            &tensor[i*weight_len], weight, bias, &out[i*weight_len],
            vec_len, weight_len, eps);
    }
}

void layernorm_Nd_16f_cpu(
    const datatype* tensor_q, const datatype* weight_q, const datatype* bias_q, datatype* out_q,
    float* tensor, float* weight, float* out,
    const std::vector<uint32_t>& tensor_dims, const uint32_t weight_len,
    const float eps
) {
    // b/c of small size of weights for layernorm, we will combine weight and bias into a single buffer
    fp16_to_fp32((ushort*)tensor_q, (float*)tensor, tensor_dims);
    fp16_to_fp32((ushort*)weight_q, (float*)weight, std::vector<uint32_t>{weight_len});
    float* bias = &(weight[weight_len]);
    fp16_to_fp32((ushort*)bias_q, (float*)bias, std::vector<uint32_t>{weight_len});
    layernorm_Nd_32f(tensor, weight, bias, out, tensor_dims, weight_len, eps);
    fp32_to_fp16((float*)out, (ushort*)out_q, tensor_dims);
}



void transpose_16f(
    const datatype* tensor, datatype* out, const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& tensor_dims, std::vector<uint32_t>& out_dims
) {
    // not safe to do inplace
    assert(tensor != out);
    int rank = tensor_dims.size();
    assert(rank == 4);
    // set out_dims
    out_dims.resize(4);
    for (uint32_t i = 0; i < 4; i++) {
        out_dims[i] = tensor_dims[perm[i]];
    }
    // code
    int num_elements;
    std::vector<int> old_indices(4);
    std::vector<int> new_indices(4);
    std::vector<int> dim_offsets = {1,1,1,1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * tensor_dims[i+1];
    }
    num_elements = dim_offsets[0];
    // out_dim stuff
    std::vector<int> out_dim_offsets = {1,1,1,1};
    for (int i = rank - 2; i >= 0; i--) {
        out_dim_offsets[i] = out_dim_offsets[i+1] * out_dims[i+1];
    }
    
    for (int i = 0; i < num_elements; i++) {
        // get old indices
        uint32_t index = i;
        for (int j = 0; j < 4; j++) {
            old_indices[j] = index / dim_offsets[j];
            index = index % dim_offsets[j];
        }
        // get new indices
        for (int j = 0; j < 4; j++) {
            new_indices[j] = old_indices[perm[j]];
        }
        // write data
        out[
            new_indices[0] +
            new_indices[1] * out_dim_offsets[1] +
            new_indices[2] * out_dim_offsets[2] +
            new_indices[3] * out_dim_offsets[3]
        ] = tensor[i];
    }
}

void transpose_32f(
    const float* tensor, float* out, const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& tensor_dims, std::vector<uint32_t>& out_dims
) {
    // not safe to do inplace
    assert(tensor != out);
    int rank = tensor_dims.size();
    assert(rank == 4);
    // set out_dims
    out_dims.resize(4);
    for (uint32_t i = 0; i < 4; i++) {
        out_dims[i] = tensor_dims[perm[i]];
    }
    // code
    int num_elements;
    std::vector<int> old_indices(4);
    std::vector<int> new_indices(4);
    std::vector<int> dim_offsets = {1,1,1,1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * tensor_dims[i+1];
    }
    num_elements = dim_offsets[0];
    // out_dim stuff
    std::vector<int> out_dim_offsets = {1,1,1,1};
    for (int i = rank - 2; i >= 0; i--) {
        out_dim_offsets[i] = out_dim_offsets[i+1] * out_dims[i+1];
    }
    
    for (int i = 0; i < num_elements; i++) {
        // get old indices
        uint32_t index = i;
        for (int j = 0; j < 4; j++) {
            old_indices[j] = index / dim_offsets[j];
            index = index % dim_offsets[j];
        }
        // get new indices
        for (int j = 0; j < 4; j++) {
            new_indices[j] = old_indices[perm[j]];
        }
        // write data
        out[
            new_indices[0] +
            new_indices[1] * out_dim_offsets[1] +
            new_indices[2] * out_dim_offsets[2] +
            new_indices[3] * out_dim_offsets[3]
        ] = tensor[i];
    }
}

// not input-to-output safe
void truncate_16f(
    const datatype* input, const std::vector<uint32_t>& input_dims,
    datatype* output, std::vector<uint32_t>& output_dims,
    const std::vector<int>& dims_to_split,
    const std::vector<int>& values,
    const std::vector<int>& colon_lefts
) {
    // assertions
    const uint32_t rank = input_dims.size();
    assert(rank <= 4); // assume 4d or less
    assert(values.size() <= rank);
    assert(dims_to_split.size() == values.size() == colon_lefts.size());
    assert(input != output);
    for (int i = 0; i < values.size(); i++) {
        assert(values[i] < input_dims[dims_to_split[i]]);
    }
    // indice computation
    int indice_start[] = {0,0,0,0};
    int indice_end[]   = {0,0,0,0}; // including
    for (int i = 0; i < rank; i++) { indice_end[i] = input_dims[i] - 1; }
    for (int i = 0; i < values.size(); i++) {
        if (colon_lefts[i]) { indice_end[dims_to_split[i]] = values[i] - 1; }
        else                { indice_start[dims_to_split[i]] = values[i]; }
    }

    for (int i=0;i<4;i++) {std::cout << indice_start[i] << " ";}
    for (int i=0;i<4;i++) {std::cout << indice_end[i] << " ";}

    // writing output
    unsigned long long elements_written = 0;
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * input_dims[i+1];
    }
    std::cout << "\ttruncation(): about to enter quad for-loop\n";
    for (int a = indice_start[0]; a <= indice_end[0]; a++) {
        for (int b = indice_start[1]; b <= indice_end[1]; b++) {
            for (int c = indice_start[2]; c <= indice_end[2]; c++) {
                for (int d = indice_start[3]; d <= indice_end[3]; d++) {
                    output[elements_written] = input[
                        d + 
                        c*dim_offsets[2] +
                        b*dim_offsets[1] +
                        a*dim_offsets[0]];
                    elements_written++;
                }
            }
        }
    }
    // writing output_dims
    output_dims = std::vector<uint32_t>();
    for (int i = 0; i < input_dims.size(); i++) {
        output_dims.push_back(indice_end[i]+1 - indice_start[i]);
    }
}

// not input-to-output safe
void truncate_32f(
    const float* input, const std::vector<uint32_t>& input_dims,
    float* output, std::vector<uint32_t>& output_dims,
    const std::vector<int>& dims_to_split,
    const std::vector<int>& values,
    const std::vector<int>& colon_lefts
) {
    // assertions
    const uint32_t rank = input_dims.size();
    assert(rank <= 4); // assume 4d or less
    assert(values.size() <= rank);
    assert(dims_to_split.size() == values.size() == colon_lefts.size());
    assert(input != output);
    for (int i = 0; i < values.size(); i++) {
        assert(values[i] < input_dims[dims_to_split[i]]);
    }
    // indice computation
    int indice_start[] = {0,0,0,0};
    int indice_end[]   = {0,0,0,0}; // including
    for (int i = 0; i < rank; i++) { indice_end[i] = input_dims[i] - 1; }
    for (int i = 0; i < values.size(); i++) {
        if (colon_lefts[i]) { indice_end[dims_to_split[i]] = values[i] - 1; }
        else                { indice_start[dims_to_split[i]] = values[i]; }
    }
    // writing output
    unsigned long long elements_written = 0;
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * input_dims[i+1];
    }
    std::cout << "\ttruncation(): about to enter quad for-loop\n";
    for (int a = indice_start[0]; a <= indice_end[0]; a++) {
        for (int b = indice_start[1]; b <= indice_end[1]; b++) {
            for (int c = indice_start[2]; c <= indice_end[2]; c++) {
                for (int d = indice_start[3]; d <= indice_end[3]; d++) {
                    output[elements_written] = input[
                        d + 
                        c*dim_offsets[2] +
                        b*dim_offsets[1] +
                        a*dim_offsets[0]];
                    elements_written++;
                }
            }
        }
    }
    // writing output_dims
    output_dims = std::vector<uint32_t>();
    for (int i = 0; i < input_dims.size(); i++) {
        output_dims.push_back(indice_end[i]+1 - indice_start[i]);
    }
}

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
void rotary_emb_16f(
    const datatype* x, const std::vector<uint32_t>& x_dims,
    const int seq_len,
    const datatype* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const datatype* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    datatype* sin, std::vector<uint32_t>& sin_dims,
    datatype* cos, std::vector<uint32_t>& cos_dims
) {
    assert(seq_len <= SIN_COS_MAX_SEQ_LEN); // TEMP solution
    assert(sin_cached_dims.size() == cos_cached_dims.size());
    for (int i = 0; i < sin_cached_dims.size(); i++) {
        // idk if this needs to be true, but if it does not, change code below
        assert(sin_cached_dims[i] == cos_cached_dims[i]);
    }
    std::vector<int> dims_to_split = {0};
    std::vector<int> values = {seq_len};
    std::vector<int> colon_lefts = {1};
    std::cout << "\tCalling truncate()\n";
    printV("x", x_dims);
    std::cout << "\tseq_len in truncate: " << seq_len << "\n";
    truncate_16f(
        cos_cached, cos_cached_dims, cos, cos_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
    truncate_16f(
        sin_cached, sin_cached_dims, sin, sin_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
}

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
void rotary_emb_32f(
    const float* x, const std::vector<uint32_t>& x_dims,
    const int seq_len,
    const float* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const float* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    float* sin, std::vector<uint32_t>& sin_dims,
    float* cos, std::vector<uint32_t>& cos_dims
) {
    assert(seq_len <= SIN_COS_MAX_SEQ_LEN); // TEMP solution
    assert(sin_cached_dims.size() == cos_cached_dims.size());
    for (int i = 0; i < sin_cached_dims.size(); i++) {
        // idk if this needs to be true, but if it does not, change code below
        assert(sin_cached_dims[i] == cos_cached_dims[i]);
    }
    std::vector<int> dims_to_split = {0};
    std::vector<int> values = {seq_len};
    std::vector<int> colon_lefts = {1};
    std::cout << "\tCalling truncate()\n";
    printV("x", x_dims);
    truncate_32f(
        cos_cached, cos_cached_dims, cos, cos_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
    truncate_32f(
        sin_cached, sin_cached_dims, sin, sin_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
}

// cannot do in place
void gather_16f(
    const datatype* x, const std::vector<uint32_t>& x_dims,
    const std::vector<int>& indices,
    datatype* out, std::vector<uint32_t>& out_dims
) {
    // offset computation
    int rank = x_dims.size();
    assert(rank <= 4);
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * x_dims[i+1];
    }
    unsigned long long offset = dim_offsets[0];
    // writing to out
    for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < offset; j++) {
            out[i*offset + j] = x[indices[i]*offset + j];
        }
    }
    // writing out_dims
    out_dims = std::vector<uint32_t>();
    out_dims.push_back(indices.size());
    for (int i = 1; i < rank; i++) {
        out_dims.push_back(x_dims[i]);
    }
}

void gather_32f(
    const float* x, const std::vector<uint32_t>& x_dims,
    const std::vector<int>& indices,
    float* out, std::vector<uint32_t>& out_dims
) {
    // offset computation
    int rank = x_dims.size();
    assert(rank <= 4);
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * x_dims[i+1];
    }
    unsigned long long offset = dim_offsets[0];
    std::cout << "writing data\n";
    // writing to out
    for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < offset; j++) {
            out[i*offset + j] = x[indices[i]*offset + j];
        }
    }
    // writing out_dims
    out_dims = std::vector<uint32_t>();
    out_dims.push_back(indices.size());
    for (int i = 1; i < rank; i++) {
        out_dims.push_back(x_dims[i]);
    }
}

void concat_16f(
    const datatype* x1, const std::vector<uint32_t>& x1_dims,
    const datatype* x2, const std::vector<uint32_t>& x2_dims,
    const int axis,
    datatype* out, std::vector<uint32_t>& out_dims
) {
    // assertions
    assert((out != x1) && (out != x2));
    assert(x1_dims.size() == x2_dims.size());
    int rank = x1_dims.size();
    assert(axis < rank);
    for (int i = 0; i < rank; i++) {
        if (i != axis) { assert(x1_dims[i] == x2_dims[i]); }
    }
    assert(rank <= 4);
    // writing out_dims
    out_dims = x1_dims;
    out_dims[axis] += x2_dims[axis];
    // offset computations
    std::vector<uint32_t> x1_4d_dims = {1, 1, 1, 1};
    std::vector<uint32_t> x2_4d_dims = {1, 1, 1, 1};
    unsigned long long x1_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long x2_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long out_dim_offsets[4] = {1, 1, 1, 1};
    for (int i = 0; i < 4; i++) {
        x1_4d_dims[i] = x1_dims.end()[i - 4];
        x2_4d_dims[i] = x2_dims.end()[i - 4];
    }
    for (int i = rank - 2; i >= 0; i--) {
        x1_dim_offsets[i]  = x1_dim_offsets[i+1]  * x1_dims[i+1];
        x2_dim_offsets[i]  = x2_dim_offsets[i+1]  * x2_dims[i+1];
        out_dim_offsets[i] = out_dim_offsets[i+1] * out_dims[i+1];
    }
    for (int a = 0; a < x1_4d_dims[0]; a++) {
        for (int b = 0; b < x1_4d_dims[1]; b++) {
            for (int c = 0; c < x1_4d_dims[2]; c++) {
                for (int d = 0; d < x1_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[2] + 
                        b*out_dim_offsets[1] +
                        a*out_dim_offsets[0]] 
                    = x1[
                        d + 
                        c*x1_dim_offsets[2] +
                        b*x1_dim_offsets[1] +
                        a*x1_dim_offsets[0]];
                }
            }
        }
    }
    std::cout << "\twriting x2 to out\n";
    // writing x2 to out
    unsigned long long x2_write_offset = x1_dims[axis] * out_dim_offsets[axis];
    for (int a = 0; a < x2_4d_dims[0]; a++) {
        for (int b = 0; b < x2_4d_dims[1]; b++) {
            for (int c = 0; c < x2_4d_dims[2]; c++) {
                for (int d = 0; d < x2_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[2] + 
                        b*out_dim_offsets[1] +
                        a*out_dim_offsets[0] + 
                        x2_write_offset]
                    = x2[
                        d + 
                        c*x2_dim_offsets[2] + 
                        b*x2_dim_offsets[1] +
                        a*x2_dim_offsets[0]];
                }
            }
        }
    }
}

void concat_32f(
    const float* x1, const std::vector<uint32_t>& x1_dims,
    const float* x2, const std::vector<uint32_t>& x2_dims,
    const int axis,
    float* out, std::vector<uint32_t>& out_dims
) {
    // assertions
    assert((out != x1) && (out != x2));
    assert(x1_dims.size() == x2_dims.size());
    int rank = x1_dims.size();
    assert(axis < rank);
    for (int i = 0; i < rank; i++) {
        if (i != axis) { assert(x1_dims[i] == x2_dims[i]); }
    }
    assert(rank <= 4);
    // writing out_dims
    out_dims = x1_dims;
    out_dims[axis] += x2_dims[axis];
    // offset computations
    std::vector<uint32_t> x1_4d_dims = {1, 1, 1, 1};
    std::vector<uint32_t> x2_4d_dims = {1, 1, 1, 1};
    unsigned long long x1_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long x2_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long out_dim_offsets[4] = {1, 1, 1, 1};
    for (int i = 0; i < 4; i++) {
        x1_4d_dims[i] = x1_dims.end()[i - 4];
        x2_4d_dims[i] = x2_dims.end()[i - 4];
    }
    for (int i = rank - 2; i >= 0; i--) {
        x1_dim_offsets[i]  = x1_dim_offsets[i+1]  * x1_dims[i+1];
        x2_dim_offsets[i]  = x2_dim_offsets[i+1]  * x2_dims[i+1];
        out_dim_offsets[i] = out_dim_offsets[i+1] * out_dims[i+1];
    }
    for (int a = 0; a < x1_4d_dims[0]; a++) {
        for (int b = 0; b < x1_4d_dims[1]; b++) {
            for (int c = 0; c < x1_4d_dims[2]; c++) {
                for (int d = 0; d < x1_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[2] + 
                        b*out_dim_offsets[1] +
                        a*out_dim_offsets[0]] 
                    = x1[
                        d + 
                        c*x1_dim_offsets[2] +
                        b*x1_dim_offsets[1] +
                        a*x1_dim_offsets[0]];
                }
            }
        }
    }
    std::cout << "\twriting x2 to out\n";
    // writing x2 to out
    unsigned long long x2_write_offset = x1_dims[axis] * out_dim_offsets[axis];
    for (int a = 0; a < x2_4d_dims[0]; a++) {
        for (int b = 0; b < x2_4d_dims[1]; b++) {
            for (int c = 0; c < x2_4d_dims[2]; c++) {
                for (int d = 0; d < x2_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[2] + 
                        b*out_dim_offsets[1] +
                        a*out_dim_offsets[0] + 
                        x2_write_offset]
                    = x2[
                        d + 
                        c*x2_dim_offsets[2] + 
                        b*x2_dim_offsets[1] +
                        a*x2_dim_offsets[0]];
                }
            }
        }
    }
}

// uses an internal buffer for now (be careful if you remove)
// could optimize by writing directly to out rather than using truncate()
void rotate_half_16f(
    const datatype* x, const std::vector<uint32_t>& x_dims,
    datatype* out, std::vector<uint32_t>& out_dims
) {
    // internal buffers
    // datatype x1[QUERY_STATES_BUFF_SIZE];
    datatype* x1 = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype x2[QUERY_STATES_BUFF_SIZE];
    datatype* x2 = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    std::vector<uint32_t> x1_dims, x2_dims;

    int x1_len = x_dims.end()[-1] / 2;
    std::vector<int> dims_to_split = {(int)x_dims.size()-1};
    std::vector<int> values = {x1_len};
    std::vector<int> colon_left = {1};
    std::vector<int> colon_right = {0};
    truncate_16f(
        x, x_dims, x1, x1_dims, dims_to_split,
        values, colon_left
    );
    truncate_16f(
        x, x_dims, x2, x2_dims, dims_to_split,
        values, colon_right
    );
    std::cout << "\t\tfinished both truncate()s in rotate_half()\n";
    // negate x2
    unsigned long long x2_size = 1;
    for (auto i : x2_dims) { x2_size *= i; }
    for (int i = 0; i < x2_size; i++) {x2[i] = -1 * x2[i];}
    // concat
    std::cout << "\t\tCalling concat()\n";
    concat_16f(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
    std::cout << "\t\tfreeing memory\n";
    free(x1);
    free(x2);
}

void rotate_half_32f(
    const float* x, const std::vector<uint32_t>& x_dims,
    float* out, std::vector<uint32_t>& out_dims
) {
    // internal buffers
    // float x1[QUERY_STATES_BUFF_SIZE];
    float* x1 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float x2[QUERY_STATES_BUFF_SIZE];
    float* x2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> x1_dims, x2_dims;

    int x1_len = x_dims.end()[-1] / 2;
    std::vector<int> dims_to_split = {(int)x_dims.size()-1};
    std::vector<int> values = {x1_len};
    std::vector<int> colon_left = {1};
    std::vector<int> colon_right = {0};
    truncate_32f(
        x, x_dims, x1, x1_dims, dims_to_split,
        values, colon_left
    );
    truncate_32f(
        x, x_dims, x2, x2_dims, dims_to_split,
        values, colon_right
    );
    std::cout << "\t\tfinished both truncate()s in rotate_half()\n";
    // negate x2
    unsigned long long x2_size = 1;
    for (auto i : x2_dims) { x2_size *= i; }
    for (int i = 0; i < x2_size; i++) {x2[i] = -1 * x2[i];}
    // concat
    std::cout << "\t\tCalling concat()\n";
    concat_32f(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
    std::cout << "\t\tfreeing memory\n";
    free(x1);
    free(x2);
}


/* NOTE: b/c q and q_embed to same buffer in PhiAttetion, we will create a additional buffer
            so that the computation is correct, and we can mirror the python code. */
// are using an internal buffer
void apply_rotary_pos_emb_32f(
    const float* q, const std::vector<uint32_t>& q_dims,
    const float* k, const std::vector<uint32_t>& k_dims,
    const float* cos, const std::vector<uint32_t>& cos_dims,
    const float* sin, const std::vector<uint32_t>& sin_dims,
    const std::vector<int>& position_ids, const int unsqueeze_dim,
    float* q_embed, std::vector<uint32_t>& q_embed_dims,
    float* k_embed, std::vector<uint32_t>& k_embed_dims
) {
    // buffers prbably dont need to be this big
    // float cos_buff[SIN_COS_BUFF_SIZE];
    float* cos_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    // float sin_buff[SIN_COS_BUFF_SIZE];
    float* sin_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> cos_buff_dims, sin_buff_dims;
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    std::cout << "cos[0]: " << cos[0] << "\n";
    printV("cos_dims", cos_dims);
    for (auto i : position_ids) {std::cout << i << ", ";}
    std::cout << "\n";
    gather_32f(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    gather_32f(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
    std::cout << "unsqueeze_dim: " << unsqueeze_dim << "\n";
    cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1);
    sin_buff_dims.insert(sin_buff_dims.begin() + unsqueeze_dim, 1);
    printV("cos_buff_dims", cos_buff_dims);
    printV("q_dims", q_dims);
    printV("k_dims", k_dims);
    // computing embedding
    // assume the last 2 dims are the same size for q and cos
    assert(cos_buff_dims.end()[-1] == q_dims.end()[-1]);
    assert(cos_buff_dims.end()[-1] == k_dims.end()[-1]);
    // assert(q_dims.end()[-2] == cos_buff_dims.end()[-2] == k_dims.end()[-2]);
    assert(q_dims.end()[-2] == k_dims.end()[-2]);
    int q_rank = q_dims.size();
    assert(q_rank <= 4);
    unsigned long long q_dim_offsets[4] = {1, 1, 1, 1};
    for (int i = q_rank - 2; i >= 0; i--) {
        q_dim_offsets[i] = q_dim_offsets[i+1] * q_dims[i+1]; //ugly fix
    }
    assert(q_dims[0] == k_dims[0]); // we can adjust if this needs to be false
    assert(q_dims[1] == k_dims[1]); // would need to create another collection of for loops
    std::cout << "\tallocating interal buffers of k and q\n";
    // float q_temp_buff[QUERY_STATES_BUFF_SIZE];
    float* q_temp_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float k_temp_buff[QUERY_STATES_BUFF_SIZE];
    float* k_temp_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::cout << "\tCalling Multiply Loop\n";
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_32f(
                &(q[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff, 
                q_temp_buff, cos_buff_dims);
            mul_32f(
                &(k[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff,
                k_temp_buff, cos_buff_dims);
        }
    }
    std::cout << "\tCalling Rotate_Half\n";
    rotate_half_32f(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
    std::cout << "\tCalling Rotate_Half\n";
    printV("k_dims", k_dims);
    rotate_half_32f(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims
    std::cout << "\tCalling Mutliply Loop\n";
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_32f(
                &(q_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
                q_embed, sin_buff_dims);
            mul_32f(
                &(k_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
                k_embed, sin_buff_dims);
        }
    }
    std::cout << "\tCalling add_32f\n";
    add_32f(q_embed, q_temp_buff, q_embed, q_embed_dims);
    std::cout << "\tCalling add_32f\n";
    add_32f(k_embed, k_temp_buff, k_embed, k_embed_dims);
    //free
    free(cos_buff);
    free(sin_buff);
}

void allocate_32f(const datatype* in, float*& out, const std::vector<uint32_t>& dims) {
    size_t ten_size = 1;
    for (auto i : dims) { ten_size *= i; }
    std::cout << ten_size << "\n";
    out = (float*)malloc(ten_size * sizeof(float));
    fp16_to_fp32((ushort*)in, out, dims);
}

// uses a lot of internal buffers
void apply_rotary_pos_emb_16f_cpu(
    /* 16 bit */
    const datatype* q_q, const std::vector<uint32_t>& q_dims,
    const datatype* k_q, const std::vector<uint32_t>& k_dims,
    const datatype* cos_q, const std::vector<uint32_t>& cos_dims,
    const datatype* sin_q, const std::vector<uint32_t>& sin_dims,
    const std::vector<int>& position_ids, const int unsqueeze_dim,
    datatype* q_embed_q, std::vector<uint32_t>& q_embed_dims,
    datatype* k_embed_q, std::vector<uint32_t>& k_embed_dims
) {
    float *q, *k, *cos, *sin, *q_embed, *k_embed;
    allocate_32f(q_q, q, q_dims);
    allocate_32f(k_q, k, k_dims);
    allocate_32f(cos_q, cos, cos_dims);
    allocate_32f(sin_q, sin, sin_dims);
    allocate_32f(q_embed_q, q_embed, q_embed_dims);
    allocate_32f(k_embed_q, k_embed, k_embed_dims);
    std::cout << "calling apply_rot_pos_emb_32f\n";
    apply_rotary_pos_emb_32f(
    q, q_dims, k, k_dims, cos, cos_dims, sin, sin_dims, 
    position_ids, unsqueeze_dim, q_embed, q_embed_dims, k_embed, k_embed_dims);
    // freeing
    free(q);
    free(k);
    free(cos);
    free(sin);
    free(q_embed);
    free(k_embed);
}


void copyTensor_16f(const datatype* ten1, datatype* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i];
    }
}

void copyTensor_32f(const float* ten1, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    for (uint32_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i];
    }
}

// code taken mostly from Softmax.cpp in SNPE example code
// assumed to be done on most inner dimensions
void softmax(const float* tensor, float* out, const std::vector<uint32_t>& dims) {
    const uint32_t rank = dims.size();
    const size_t depth = dims[rank - 1];
    uint32_t tensorLength = 1;
    for(uint32_t i = 0; i < rank; i++) { tensorLength *= dims[i]; }
    const size_t numPixels = tensorLength/depth;
    for( size_t pix = 0; pix < numPixels; ++pix ) {
        const float* in = (float*)tensor+pix*depth;
        float* out_temp = (float*)out+pix*depth;
        // find the max element for max subtraction
        float maxElt = std::numeric_limits<float>::lowest();
        for( size_t i = 0; i < depth; ++i ) { maxElt = std::max( maxElt, in[i] ); }
        // compute exponentiations
        float expSum = 0.0;
        for( size_t i = 0; i < depth; ++i ) {
            const float ei = expf( in[i] - maxElt );
            out_temp[i] = ei;
            expSum += ei;
        }
        // normalize
        for( size_t i = 0; i < depth; ++i ) { out_temp[i] = out_temp[i] / expSum; }
    }
}

void mySoftmax(const float* tensor, float* out, const std::vector<uint32_t>& dims) {
    int rank = dims.size();
    float maxElt = std::numeric_limits<float>::lowest();
    float expSum;
    std::cout << "max Elt: " << maxElt << "\n";
    // compute number of iterations
    unsigned long long iterations = 1; // total_elements / dims[-1]
    for (int i = 0; i < rank - 1; i++) {iterations *= dims[i];}
    int inner_dim_size = dims.end()[-1];
    // compute
    for (auto i = 0; i < iterations; i++) {
        maxElt = std::numeric_limits<float>::lowest();
        // find max element
        for (int j = 0; j < inner_dim_size; j++) {
            // maxElt = std::max(maxElt, tensor[i*inner_dim_size + j]);
            maxElt = (maxElt > tensor[i*inner_dim_size + j]) ? 
                    maxElt : tensor[i*inner_dim_size + j];
        }
        // exponentiate
        expSum = 0.0;
        for (int j = 0; j < inner_dim_size; j++) {
            const float ei = expf(tensor[i*inner_dim_size + j] - maxElt);
            out[i*inner_dim_size + j] = ei;
            expSum += ei;
        }
        // normalize
        for (int j = 0; j < inner_dim_size; j++) {
            out[i*inner_dim_size + j] /= expSum;
        }
    }
}



void PhiAttention_16f_cpu(
    /* inputs */
    const datatype* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    const float* attention_mask, const std::vector<uint32_t>& attention_mask_dims,
    const std::vector<int>& position_ids,
    const datatype* old_past_keys, const std::vector<uint32_t>& old_past_keys_dims,
    const datatype* old_past_values, const std::vector<uint32_t>& old_past_values_dims,
    /* weights */
    const datatype* q_proj_weights, const std::vector<uint32_t>& q_proj_weights_dims,
    const datatype* q_proj_bias,
    const datatype* k_proj_weights, const std::vector<uint32_t>& k_proj_weights_dims,
    const datatype* k_proj_bias,
    const datatype* v_proj_weights, const std::vector<uint32_t>& v_proj_weights_dims,
    const datatype* v_proj_bias,
    // const datatype* q_layernorm_weights, const int q_layernorm_weights_len,
    // const datatype* q_layernorm_bias,
    // const datatype* k_layernorm_weights, const int k_layernorm_weights_len,
    // const datatype* k_layernorm_bias,
    // const datatype eps,
    /* LARGE buffers */
    float* buff_1, float* buff_2, float* buff_3,

    const int num_heads, const int head_dim, const int num_kv_heads,
    const datatype* dense_weights, const std::vector<uint32_t>& dense_weights_dims,
    const datatype* dense_bias,
    /* init params */
    const int layer_idx,
    const datatype* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const datatype* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    int rotary_emb_dim,
    /* outputs */
    datatype* attn_output, std::vector<uint32_t>& attn_output_dims,
    datatype* past_keys, std::vector<uint32_t>& past_keys_dims,
    datatype* past_values, std::vector<uint32_t>& past_values_dims
) {
    int bsz = hidden_states_dims[0];
    int q_len = hidden_states_dims[1];
    // datatype query_states_buff[QUERY_STATES_BUFF_SIZE];
    datatype* query_states_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype key_states_buff[QUERY_STATES_BUFF_SIZE];
    datatype* key_states_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype value_states_buff[QUERY_STATES_BUFF_SIZE];
    datatype* value_states_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));

    std::vector<uint32_t> query_states_dims;
    std::vector<uint32_t> key_states_dims;
    std::vector<uint32_t> value_states_dims;

    std::cout << "Calling Projection Linear Layers\n";

    linear_Nd_16f_cpu(
        hidden_states, q_proj_weights, q_proj_bias, query_states_buff,
        buff_1, buff_2, buff_3,
        hidden_states_dims, q_proj_weights_dims, query_states_dims);

    linear_Nd_16f_cpu(
        hidden_states, k_proj_weights, k_proj_bias, key_states_buff,
        buff_1, buff_2, buff_3,
        hidden_states_dims, k_proj_weights_dims, key_states_dims);

    linear_Nd_16f_cpu(
        hidden_states, v_proj_weights, v_proj_bias, value_states_buff,
        buff_1, buff_2, buff_3,
        hidden_states_dims, v_proj_weights_dims, value_states_dims);

    printT_Alt("query_states after linear", query_states_dims, query_states_buff, 
    false, 5);
    
    printV("value_states after before reshape:", value_states_dims);
    // reshape
    query_states_dims = std::vector<uint32_t>{
        (uint32_t)bsz, (uint32_t)q_len, (uint32_t)num_heads, (uint32_t)head_dim};
    key_states_dims = std::vector<uint32_t>{
        (uint32_t)bsz, (uint32_t)q_len, (uint32_t)num_kv_heads, (uint32_t)head_dim};
    value_states_dims = key_states_dims;
    printV("value_states", value_states_dims);

    // transpose
    // cant use same buffers (unless transpose() uses a temporary buffer)
    // datatype query_states_buff_2[QUERY_STATES_BUFF_SIZE];
    datatype* query_states_buff_2 = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype key_states_buff_2[QUERY_STATES_BUFF_SIZE];
    datatype* key_states_buff_2 = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype value_states_buff_2[QUERY_STATES_BUFF_SIZE];
    datatype* value_states_buff_2 = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    std::vector<uint32_t> temp_dims;
    std::vector<uint32_t> perm = {0, 2, 1, 3};
    std::cout << "Calling tranpose() 3 times\n";
    transpose_16f(
        query_states_buff, query_states_buff_2, 
        perm, 
        query_states_dims, temp_dims);
    query_states_dims = temp_dims;
    transpose_16f(
        key_states_buff, key_states_buff_2, 
        perm, 
        key_states_dims, temp_dims);
    key_states_dims = temp_dims;
    transpose_16f(
        value_states_buff, value_states_buff_2, 
        perm, 
        value_states_dims, temp_dims);
    value_states_dims = temp_dims;
    printV("value_states", value_states_dims);
    // Cache
    // old_past_keys_dims: (seq_len, something), (seq_len, something)
    int kv_seq_len = key_states_dims.end()[-2];
    if (old_past_keys_dims.size() > 0) { // not first run
        kv_seq_len += old_past_keys_dims.end()[-2]; // seq_len MAKE SURE THIS IS RIGHT
    }
    std::cout << "kv_seq_len:" << kv_seq_len << "\n";

    printT_Alt("query_states after transpose", query_states_dims, query_states_buff_2, 
    false, 5);

    // partial rotary embedding
    // datatype sin_buff[SIN_COS_BUFF_SIZE];
    datatype* sin_buff = (datatype*)malloc(SIN_COS_BUFF_SIZE * sizeof(datatype));
    // datatype cos_buff[SIN_COS_BUFF_SIZE];
    datatype* cos_buff = (datatype*)malloc(SIN_COS_BUFF_SIZE * sizeof(datatype));
    std::vector<uint32_t> sin_buff_dims;
    std::vector<uint32_t> cos_buff_dims;
    printV("value_states", value_states_dims);
    std::cout << "calling rot_emb() \n";
    rotary_emb_16f(
        value_states_buff_2, value_states_dims, kv_seq_len,
        sin_cached, sin_cached_dims, cos_cached, cos_cached_dims,
        sin_buff, sin_buff_dims, cos_buff, cos_buff_dims);
    std::cout << "finished calling rot_emb()\n";
    printV("sin_buff_dims", sin_buff_dims);
    printV("cos_buff_dims", cos_buff_dims);
    printT_Alt("sin_buff after rot_emb_16f:", sin_buff_dims, sin_buff,
    false, 5);
    //rotations
    // datatype query_rot_buff[QUERY_STATES_BUFF_SIZE];
    datatype* query_rot_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype query_pass_buff[QUERY_STATES_BUFF_SIZE];
    datatype* query_pass_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype key_rot_buff[QUERY_STATES_BUFF_SIZE];
    datatype* key_rot_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    // datatype key_pass_buff[QUERY_STATES_BUFF_SIZE];
    datatype* key_pass_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
    std::vector<uint32_t> query_rot_buff_dims;
    std::vector<uint32_t> query_pass_buff_dims;
    std::vector<uint32_t> key_rot_buff_dims;
    std::vector<uint32_t> key_pass_buff_dims;
    std::cout << "Calling truncate 4 times for rot and pass\n";
    printV("query_states_buff_2",query_states_dims);
    printV("key_states_buff_2",key_states_dims);
    truncate_16f(
        query_states_buff_2, query_states_dims,
        query_rot_buff, query_rot_buff_dims, 
        std::vector<int> {int(query_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
        );
    truncate_16f(
        query_states_buff_2, query_states_dims,
        query_pass_buff, query_pass_buff_dims, 
        std::vector<int> {int(query_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0});
    truncate_16f(
        key_states_buff_2, key_states_dims,
        key_rot_buff, key_rot_buff_dims, 
        std::vector<int> {int(key_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1});
    truncate_16f(
        key_states_buff_2, key_states_dims,
        key_pass_buff, key_pass_buff_dims, 
        std::vector<int> {int(key_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0});
    printV("query_rot_buff_dims", query_rot_buff_dims);
    printV("query_pass_buff_dims", query_pass_buff_dims);
    printT_Alt("query_rot after truncation", query_rot_buff_dims, query_rot_buff,
    false, 5);
    printT_Alt("key_pass after truncation", key_pass_buff_dims, key_pass_buff,
    false, 5);
    // applying rot_pos-emb
    // [batch_size, seq_length, num_heads, head_dim // partial_rotary_factor]
    std::cout << "Calling apply_rotary_pos_emb()\n";
    apply_rotary_pos_emb_16f_cpu(
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims,
        cos_buff, cos_buff_dims, sin_buff, sin_buff_dims, 
        position_ids, 1,
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims);
    printV("key_rot_buff_dims", key_rot_buff_dims);
    printT_Alt("key_rot_buff after apply_rot_pos_emb", key_rot_buff_dims, key_rot_buff,
    false, 5);
    // concatting rot and pass
    std::cout << "Concatenating rot and pass\n";
    concat_16f(
        query_rot_buff, query_rot_buff_dims,
        query_pass_buff, query_pass_buff_dims, 
        query_rot_buff_dims.size()-1, 
        query_states_buff, query_states_dims);
    concat_16f(
    key_rot_buff, key_rot_buff_dims,
    key_pass_buff, key_pass_buff_dims, 
    key_rot_buff_dims.size()-1, 
    key_states_buff, key_states_dims);

    printT_Alt("key-states after concatenation", key_states_dims, key_states_buff,
    false, 5);

    // updating cache
    // past_key_value_old: (seq_len, something), (seq_len, something)

    // COMMENT BACK IN
    std::cout << "--Going to Cache--\n";
    printV("old_past_keys", old_past_keys_dims);
    printV("key_states_dims", key_states_dims);
    if (old_past_keys_dims.size() == 0) { 
        std::cout << "first run\n";
        // first run
        copyTensor_16f(key_states_buff, past_keys, key_states_dims);
        past_keys_dims = key_states_dims;
        copyTensor_16f(value_states_buff_2, past_values, value_states_dims);
        past_values_dims = value_states_dims;
    }
    else { 
        // not first run
        std::cout << "not first run\n";
        std::cout << "calling first concat\n";
        concat_16f(
            old_past_keys, old_past_keys_dims, key_states_buff, key_states_dims,
            key_states_dims.size()-2, 
            past_keys, past_keys_dims);
        std::cout << "calling second concat\n";
        concat_16f(
            old_past_values, old_past_values_dims, value_states_buff_2, value_states_dims,
            value_states_dims.size()-2, 
            past_values, past_values_dims);
        std::cout << "calling first copyTensor()\n";
        copyTensor_16f(past_keys, key_states_buff, past_keys_dims);
        key_states_dims = past_keys_dims;
        copyTensor_16f(past_values, value_states_buff_2, past_values_dims);
        value_states_dims = past_values_dims;
    }

    printV("key_states_dims after grabbing from cache", key_states_dims);

    // dont have to implement repeat_kv b/c num_atten_heads / num_kv_heads = 32/32 = 1

    /*
    could split the model right here if you want-------------------
    but wouldn't that lead to ineffcieny due to execute() in SNPE having 
    to load the weights twice?
    */

    /*
    1, seq, 32, 80 --> 1, 32, seq, 80 (querys)
    1, seq, 32, 80 --> 1, 32, seq, 80 --> 1, 32, 80, seq (keys)
    result: 1, 32, seq, seq
    ------LATER after masking-----------
    attn_output = attn_weights(1, 32, seq, seq) x value_states(1, 32, seq, 80)
    attn_output = (1, 32, seq, 80)
    */

    // matmul
    std::cout << "calling tranpose() before matmul()\n";
    fp16_to_fp32(query_states_buff, buff_1, query_states_dims);
    transpose_16f(
        key_states_buff, key_states_buff_2, 
        std::vector<uint32_t> {0, 1, 3, 2},
        key_states_dims, temp_dims
        );
    key_states_dims = temp_dims;
    fp16_to_fp32(key_states_buff_2, buff_2, key_states_dims);
    std::vector<uint32_t> attn_weights_dims;
    printV("query_states", query_states_dims);
    printV("key_states_tranposed", key_states_dims);
    std::cout << "Calling 1st matmul\n";
    matmul_Nd_32f_constrained(
        buff_1, buff_2, buff_3,
        query_states_dims, key_states_dims, attn_weights_dims);
    printV("attn_weights", attn_weights_dims);

    printT_32f_Alt("attn_weights after matmul but before sqrt", attn_weights_dims, buff_3,
    false, 5);

    // WHAT ABOUT THE DIVIDING BY SQRT
    const float sqrt_head_dim = 1 / sqrt(80.0);
    size_t attn_weight_elements = 1;
    for (auto i : attn_weights_dims) { attn_weight_elements *= i; }
    for (size_t i = 0; i < attn_weight_elements; i++) {
        buff_3[i] *= sqrt_head_dim;
    }

    printT_32f_Alt("attn_weights after first matmul and sqrt", attn_weights_dims, buff_3,
    false, 5);

    // masking
    std::cout << "Calling Masking\n";
    printV("attention_mask_dims", attention_mask_dims);
    add_32f_general(
        buff_3, attention_mask, buff_1,
        attn_weights_dims, attention_mask_dims, attn_output_dims);
    printV("post masking attn_weights(attn_output)", attn_output_dims);
    printT_32f("attn_weights after masking", attn_output_dims, buff_1,
    false, 5);

    
    // softmax
    std::cout << "Calling softmax\n";
    mySoftmax(buff_1, buff_2, attn_output_dims);
    attn_weights_dims = attn_output_dims;
    printV("post softmaxing attn_weights", attn_weights_dims);
    printT_32f_Alt("attn_weights after softmaxing", attn_weights_dims, buff_2,
    false, 5);

    // downcast to fp16 (not doing for this for this code b/c cpu must upcast anyway)
    // fp32_to_fp16(buff_2, attn_output, attn_output_dims); // dont use for cpu-only implementation

    //temp solution to be closer to TF-fp16 computation
    // downcast, then upcast to 
    DownUpCast(buff_2, attn_weights_dims);



    // matmul (to attn_output) (supposed to be 16 bit)
    fp16_to_fp32(value_states_buff_2, buff_1, value_states_dims);
    std::cout << "Calling 2nd matmul\n";
    matmul_Nd_32f_constrained(
        buff_2, buff_1, buff_3,
        attn_weights_dims, value_states_dims, attn_output_dims
    );


    // now downcast attn_output back to fp16
    fp32_to_fp16(buff_3, attn_output, attn_output_dims);

    printT_Alt("attn_output after matmul", attn_output_dims, attn_output,
    false, 5);

    // bad solution:
    datatype* attn_weights = (datatype*)malloc(ATTN_WEIGHTS_SIZE * sizeof(datatype));

    // tranpose (to attn_output)
    std::cout << "Calling transpose\n";
    transpose_16f(
        attn_output, attn_weights, 
        std::vector<uint32_t> {0, 2, 1, 3},
        attn_output_dims, attn_weights_dims);

    // reshape (attn_output)
    std::cout << "Reshaping\n";
    attn_weights_dims = std::vector<uint32_t> {
        (uint32_t)bsz, 
        (uint32_t)q_len, 
        HIDDEN_SIZE};
    
    printV("attn_weights_dims", attn_weights_dims);
    printV("dense_weights_dims", dense_weights_dims);
    printV("attn_output_dims", attn_output_dims);
    printT_Alt("attn_output after transposing and reshaping", attn_weights_dims, attn_weights,
    false, 5);

    std::cout << "Calling Final Dense Layer\n";

    // dense layer
    linear_Nd_16f_cpu(
        attn_weights, dense_weights, dense_bias, attn_output,
        buff_1, buff_2, buff_3,
        attn_weights_dims, dense_weights_dims, attn_output_dims);
    
    printT_Alt("attn_output after dense layer", attn_output_dims, attn_output,
    false, 5);


    std::cout << "Freeing PhiAttentionBuffers\n";
    // freeing
    free(query_states_buff);
    free(key_states_buff);
    free(value_states_buff);
    free(query_states_buff_2);
    free(key_states_buff_2);
    free(value_states_buff_2);
    free(sin_buff);
    free(cos_buff);
    free(query_rot_buff);
    free(query_pass_buff);
    free(key_rot_buff);
    free(key_pass_buff);
    free(attn_weights);
}

void NewGELU_32f(
    const float* input, float* output, const std::vector<uint32_t>& dims
) {
    float const_1 = sqrt(2.0f / M_PI);
    // calculate total number of elements
    uint64_t tot_elem = 1;
    for (auto i : dims) {tot_elem *= i;}
    // compute
    for (uint64_t i = 0; i < tot_elem; i++) {
        output[i] = 
            0.5 * input[i] * 
            tanh(const_1 * (input[i] + 0.044715 + pow(input[i], 3.0f)));
    }
}

void NewGELU_16f(
    datatype* input_q, datatype* output_q,
    float* input, float* output, const std::vector<uint32_t>& dims
) {
    fp16_to_fp32((ushort*)input_q, input, dims);
    NewGELU_32f(input, output, dims);
    fp32_to_fp16(output, (ushort*)output_q, dims);
}

// uses input as a buffer
void PhiMLP_16f_cpu(
    datatype* input, const std::vector<uint32_t>& input_dims,
    datatype* output, std::vector<uint32_t>& output_dims,
    /* weights */
    const datatype* fc1_weights, const std::vector<uint32_t>& fc1_weights_dims,
    const datatype* fc1_bias,
    const datatype* fc2_weights, const std::vector<uint32_t>& fc2_weights_dims,
    const datatype* fc2_bias,

    float* buff_1, float* buff_2, float* buff_3
) {
    // buffer to store larger intermediate size
    std::vector<uint32_t> intermediate_buff_dims;
    linear_Nd_16f_cpu(
        input, fc1_weights, fc1_bias, output, 
        buff_1, buff_2, buff_3, 
        input_dims, fc1_weights_dims, intermediate_buff_dims);
    NewGELU_16f(output, input, buff_1, buff_2, intermediate_buff_dims);
    linear_Nd_16f_cpu(
        input, fc2_weights, fc2_bias, output, 
        buff_1, buff_2, buff_3, 
        intermediate_buff_dims, fc2_weights_dims, output_dims);
}

// NOTE: Could be optimized
// using an internal buffer for holding the residual
void PhiDecoderLayer_16f_cpu(
    /* inputs */
    const datatype* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    const float* attention_mask, const std::vector<uint32_t>& attention_mask_dims,
    const std::vector<int>& position_ids,
    const datatype* old_past_keys, const std::vector<uint32_t>& old_past_keys_dims,
    const datatype* old_past_values, const std::vector<uint32_t>& old_past_values_dims,
    /* Decoder Weights */
    const datatype* input_layernorm_weights, const int input_layernorm_weights_len,
    const datatype* input_layernorm_bias, const float decoder_eps,
    const datatype* fc1_weights, const std::vector<uint32_t>& fc1_weights_dims,
    const datatype* fc1_bias,
    const datatype* fc2_weights, const std::vector<uint32_t>& fc2_weights_dims,
    const datatype* fc2_bias,
    /* LARGE buffers */
    float* buff_1, float* buff_2, float* buff_3,
    datatype* buff_4, datatype* buff_5, datatype* buff_6,

    /* PhiAttention weights */
    const datatype* q_proj_weights, const std::vector<uint32_t>& q_proj_weights_dims,
    const datatype* q_proj_bias,
    const datatype* k_proj_weights, const std::vector<uint32_t>& k_proj_weights_dims,
    const datatype* k_proj_bias,
    const datatype* v_proj_weights, const std::vector<uint32_t>& v_proj_weights_dims,
    const datatype* v_proj_bias,
    // const datatype* q_layernorm_weights, const int q_layernorm_weights_len,
    // const datatype* q_layernorm_bias,
    // const datatype* k_layernorm_weights, const int k_layernorm_weights_len,
    // const datatype* k_layernorm_bias,
    // const float eps,
    const int num_heads, const int head_dim, const int num_kv_heads,
    const datatype* dense_weights, const std::vector<uint32_t>& dense_weights_dims,
    const datatype* dense_bias,
    /* init params */
    const int layer_idx,
    const datatype* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const datatype* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    int rotary_emb_dim,
    /* outputs */
    datatype* attn_output, std::vector<uint32_t>& attn_output_dims,
    datatype* past_keys, std::vector<uint32_t>& past_keys_dims,
    datatype* past_values, std::vector<uint32_t>& past_values_dims
) {
    printT("residual", hidden_states_dims, hidden_states, 
    false, 5);

    // layernorm
    layernorm_Nd_16f_cpu(
        hidden_states, input_layernorm_weights, input_layernorm_bias, buff_4,
        buff_1, buff_2, buff_3, hidden_states_dims,input_layernorm_weights_len, decoder_eps);
    std::vector<uint32_t> hidden_states_dims_2 = hidden_states_dims;
    // Phi Attention
    // look into implementing inplace addition so you dont have to allocate this much memory
    std::vector<uint32_t> attn_output_buff_2_dims;
    PhiAttention_16f_cpu(
        /* inputs */
        buff_4,  hidden_states_dims_2, // could possibly be optmized
        attention_mask, attention_mask_dims,
        position_ids,
        old_past_keys,  old_past_keys_dims,
        old_past_values,  old_past_values_dims,
        /* weights */
        q_proj_weights, q_proj_weights_dims,
        q_proj_bias,
        k_proj_weights, k_proj_weights_dims,
        k_proj_bias,
        v_proj_weights, v_proj_weights_dims,
        v_proj_bias,
        /* LARGE buffers */
        buff_1, buff_2, buff_3,

        num_heads, head_dim, num_kv_heads,
        dense_weights, dense_weights_dims,
        dense_bias,
        /* init params */
        layer_idx,
        sin_cached,  sin_cached_dims,
        cos_cached,  cos_cached_dims,
        rotary_emb_dim,
        /* outputs */
        buff_5, attn_output_buff_2_dims, // this could be dangerous
        past_keys, past_keys_dims,
        past_values, past_values_dims
    );
    // MLP
    std::vector<uint32_t> feed_forward_hidden_states_dims;
    // writing over buff_4
    PhiMLP_16f_cpu
    (
        buff_4, hidden_states_dims_2, buff_6, feed_forward_hidden_states_dims,
        fc1_weights, fc1_weights_dims, fc1_bias, 
        fc2_weights, fc2_weights_dims, fc2_bias,
        buff_1, buff_2, buff_3
        );
    // buff_5 + buff_6 + hidden_states
    // hidden_states is the residual in this case
    std::vector<uint32_t> temp_dims;
    add_general_16f_cpu(
        buff_5, buff_6, buff_4, 
        buff_1, buff_2, buff_3,
        attn_output_buff_2_dims, feed_forward_hidden_states_dims,
        temp_dims);
    add_general_16f_cpu(
        buff_4, hidden_states, attn_output, 
        buff_1, buff_2, buff_3,
        temp_dims, hidden_states_dims,
        attn_output_dims);
}

void LM_Head_16f(
    const datatype* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    /* weights */
    const datatype* final_layernorm_weights, const int final_layernorm_weights_len,
    const datatype* final_layernorm_bias, const float eps,
    const datatype* lm_head_weights, const std::vector<uint32_t>& lm_head_weights_dims,
    const datatype* lm_head_bias,
    /* buffers */
    float* buff_1, float* buff_2, float* buff_3,
    datatype* buff_4, datatype* buff_5,
    /* output */
    uint32_t& token_selected
) {
    // buffers 2,3,5 have to be big to hold vocab tensors

    // layer norm
    layernorm_Nd_16f_cpu(
        hidden_states, final_layernorm_weights, final_layernorm_bias, buff_4,
        buff_1, buff_2, buff_3,
        hidden_states_dims, final_layernorm_weights_len, eps);
    // lm head linear layer
    std::vector<uint32_t> temp_dims;
    linear_Nd_16f_cpu(
        buff_4, lm_head_weights, lm_head_bias, buff_5,
        buff_1, buff_2, buff_3,
        hidden_states_dims, lm_head_weights_dims, temp_dims);
    std::cout << "finished with the final linear!\n";
    // argmax
    // temp_dims: 1, 1, seq_len, vocab_size or 1,seq-len,vocab_size
    fp16_to_fp32((ushort*)buff_5, buff_2, temp_dims);
    uint32_t word_index = temp_dims.end()[-2] - 1;
    uint32_t vocab_size = temp_dims.end()[-1];
    uint32_t offset = word_index * vocab_size;
    float maxElt = std::numeric_limits<float>::lowest();
    uint32_t argmax = 0;
    std::cout << "going into loop\n";
    for (int i = 0; i < vocab_size; i++) {
        if (maxElt < buff_2[i + offset]) {
            maxElt = buff_2[i + offset];
            argmax = i;
        }
    }
    token_selected = argmax;
    std::cout << "token selected: " << token_selected << "\n";
}


// initalizes a vector to 4 values
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



Qnn_ErrorHandle_t execute(CustomOp* operation) {

  auto m_Input  = operation->getInput(0);
  auto m_Outputs = operation->getOutput(0);

  const uint32_t rank = m_Outputs->rank;
  const size_t depth = m_Outputs->currentDimensions[rank - 1];

  std::cout << "depth: " << depth << "\n";

  // this appears to work

  ////////////////////////////////////////////////////////////

  /*
  assumed inputs (NOTE: THIS ORDER MAY HAVE OT CHANGE DRAMATICALLY) (all 16-bit except mask and pos_ids)

  SIZES IN MEMORY (in bytes)

  attn_and_kv_in[(4*4+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*DECODERS)]
  attention_mask[4*4 + 4*MAX_SEQ_LEN^2] // 32bit
  position_ids_1[4*4 + MAX_SEQ_LEN] // 32bit
  decoder_weights[DECODERS * DECODER_WEIGHT_SIZE * DATASIZE)]
  decoder_weights_dims[DECODERS * DECODER_WEIGHT_DIMS_SIZE * DATASIZE]
  decoder_init_pararms[DECODERS * DECODER_INIT_PARAMS_SIZE * DATASIZE]
  */

  // udo input pointers
  uint8_t* const Attn_And_Kv_In = (uint8_t*)operation->getInput(0)->data;
  uint8_t* const AttentionMask = (uint8_t*)operation->getInput(1)->data;
  uint8_t* const Position_Ids_1 = (uint8_t*)operation->getInput(2)->data;
  uint8_t* const Decoder_Weights = (uint8_t*)operation->getInput(3)->data;
  uint8_t* const LM_Head_Weights = (uint8_t*)operation->getInput(4)->data;
  // added output
  datatype* sin_cached = (datatype*)operation->getInput(5)->data;
  datatype* cos_cached = (datatype*)operation->getInput(6)->data;
  

  // udo output pointer
  uint8_t* const UdoOutput = (uint8_t*)operation->getOutput(0)->data;


  /* Decoder-Only Stuff */
  // data
  datatype *input_layernorm_weights, *input_layernorm_bias, 
            *fc1_weights, *fc1_bias,
            *fc2_weights, *fc2_bias;
  // dims
  std::vector<uint32_t> fc1_weights_dims, fc2_weights_dims;
  int input_layernorm_weights_len;
  float decoder_eps;

  /* Phi Attention Stuff */
  // data
  datatype *hidden_states, *old_past_keys, *old_past_values,
           *q_proj_weights, *q_proj_bias,
           *k_proj_weights, *k_proj_bias,
           *v_proj_weights, *v_proj_bias,
           *dense_weights, *dense_bias,
          *attn_output, *past_keys, *past_values;
  // mask
  float* attention_mask;
  std::vector<int> position_ids;
  // dims
  std::vector<uint32_t> hidden_states_dims, attention_mask_dims, 
  old_past_keys_dims, old_past_values_dims,
  q_proj_weights_dims, k_proj_weights_dims, v_proj_weights_dims,
  dense_weights_dims,
  attn_output_dims, past_keys_dims, past_values_dims;
  // std::vector<uint32_t> proj_weights_dims = {HIDDEN_SIZE, HIDDEN_SIZE}; // remenant odl ops.cpp code

  std::vector<uint32_t> sin_cached_dims, cos_cached_dims;
  // params (do these need to be grabbed?)
  const int num_heads = 32;
  const int head_dim = HIDDEN_SIZE / num_heads;
  const int num_kv_heads = 32;
  int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)

  /* Buffers */
  float* buff_1 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  float* buff_2 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  float* buff_3 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  datatype* buff_4 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));
  datatype* buff_5 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));
  datatype* buff_6 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));


  /* constant intializations */
  // data
  hidden_states = (datatype*)Attn_And_Kv_In;
  attention_mask = (float*)AttentionMask;
  int* ptr_32 = (int*)(&Position_Ids_1[MAX_SEQ_LEN * 4]);
  int pos_ids_length = ptr_32[3];
  std::cout << "pos_ids_length: " << pos_ids_length << "\n";
  ptr_32 = (int*)(Position_Ids_1);
  for (int i = 0; i < pos_ids_length; i++) {
    position_ids.push_back(ptr_32[i]);
  }
  // dims
  std::cout << "calling init_dims_4_uint32_t\n";
  init_dims_uint32_t(hidden_states_dims, (uint8_t*)hidden_states + (MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE), 3);
  init_dims_uint32_t(attention_mask_dims, (uint8_t*)(attention_mask) + (MAX_SEQ_LEN * MAX_SEQ_LEN * 4), 4);
  input_layernorm_weights_len = HIDDEN_SIZE;
  decoder_eps = 1e-5;
  q_proj_weights_dims = std::vector<uint32_t> {HIDDEN_SIZE, HIDDEN_SIZE};
  k_proj_weights_dims = std::vector<uint32_t> {HIDDEN_SIZE, HIDDEN_SIZE};
  v_proj_weights_dims = std::vector<uint32_t> {HIDDEN_SIZE, HIDDEN_SIZE};
  dense_weights_dims = std::vector<uint32_t> {HIDDEN_SIZE, HIDDEN_SIZE};
  fc1_weights_dims = std::vector<uint32_t> {HIDDEN_SIZE, INTERMEDIATE_SIZE};
  fc2_weights_dims = std::vector<uint32_t> {INTERMEDIATE_SIZE, HIDDEN_SIZE};
  sin_cached_dims = std::vector<uint32_t> {MAX_SEQ_LEN, 32};
  cos_cached_dims = std::vector<uint32_t> {MAX_SEQ_LEN, 32};


  std::vector<uint64_t> decoder_weight_tensor_sizes {
    HIDDEN_SIZE, HIDDEN_SIZE, // LN
    HIDDEN_SIZE * HIDDEN_SIZE, HIDDEN_SIZE, // Q
    HIDDEN_SIZE * HIDDEN_SIZE, HIDDEN_SIZE, // K
    HIDDEN_SIZE * HIDDEN_SIZE, HIDDEN_SIZE, // V
    HIDDEN_SIZE * HIDDEN_SIZE, HIDDEN_SIZE, // Dense
    HIDDEN_SIZE * INTERMEDIATE_SIZE, INTERMEDIATE_SIZE, // fc1
    INTERMEDIATE_SIZE * HIDDEN_SIZE, HIDDEN_SIZE // fc2
  };

  std::vector<uint64_t> decoder_weight_offsets = {0};
  for (int i = 0; i < decoder_weight_tensor_sizes.size() - 1; i++) {
    uint64_t offset = decoder_weight_offsets.end()[-1];
    offset += decoder_weight_tensor_sizes[i];
    decoder_weight_offsets.push_back(offset);
  }

  
  uint64_t total_decoder_weight_tensor_size = 0;
  for (auto i : decoder_weight_tensor_sizes) {
    total_decoder_weight_tensor_size += i;
  }
  std::cout << "1 individual decoder weight size: " 
            << total_decoder_weight_tensor_size << "\n";
  // probably wanna put an assert statement with the macro

  // set outputs
  attn_output = (datatype*)UdoOutput;

  // remove later
  std::cout << "first elements of embedding: " <<
    half_to_float((ushort)hidden_states[0]) << "\n";
  printV("hidden_states_dims", hidden_states_dims);

  // CAREFUL FOR MEM-LEAK
  // b/c we don't wanna write to the input buffer
  datatype* const io_buff = (datatype*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(datatype));
  copyTensor_16f(hidden_states, io_buff, hidden_states_dims); // copy input into buffer
  hidden_states = io_buff;

   // remove later
  std::cout << "first elements of embedding: " <<
    half_to_float((ushort)hidden_states[0]) << "\n";

    std::cout << "-----LIST OF INITIAL DIMS------\n";
    printV("attention_mask", attention_mask_dims);
    printV("position_ids", position_ids);
    // printV("input_layernorm_weights_len", input_layernorm_weights_len);
    printV("fc1_weights", fc1_weights_dims);
    printV("fc2_weights", fc2_weights_dims);
    printV("q_proj_weights", q_proj_weights_dims);
    printV("k_proj_weights", k_proj_weights_dims);
    printV("v_proj_weights", v_proj_weights_dims);
    printV("dense_weights", dense_weights_dims);
    printV("sin_cached", sin_cached_dims);
    printV("cos_cached", cos_cached_dims);

    std::cout << "entering for-loop:";
 
  for (uint64_t i = 0; i < DECODERS; i++) {

    std::cout << "\n------------" << i << "------------\n";

    /* inputs */
    // data
    old_past_keys =   (datatype*)&Attn_And_Kv_In[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*i)];
    old_past_values = (datatype*)&Attn_And_Kv_In[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (2 + 2*i)];

    // im putting this here ofor convenience
    past_keys =   (datatype*)&UdoOutput[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*i)];
    past_values = (datatype*)&UdoOutput[((uint64_t)(4*4)+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (2 + 2*i)];

    // dims
    // may overflow
    init_dims_uint32_t(old_past_keys_dims, (uint8_t*)old_past_keys + (MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE), 4);
    init_dims_uint32_t(old_past_values_dims, (uint8_t*)old_past_values + (MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE), 4);
    if (old_past_keys_dims.end()[-1] == 0) { old_past_keys_dims = std::vector<uint32_t>(); }
    if (old_past_values_dims.end()[-1] == 0) { old_past_values_dims = std::vector<uint32_t>(); }
    /* Decoder Weights */
    // data
    input_layernorm_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[0]) * DATASIZE];
    input_layernorm_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[1]) * DATASIZE];
    q_proj_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[2]) * DATASIZE];
    q_proj_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[3]) * DATASIZE];
    k_proj_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[4]) * DATASIZE];
    k_proj_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[5]) * DATASIZE];
    v_proj_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[6]) * DATASIZE];
    v_proj_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[7]) * DATASIZE];
    dense_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[8]) * DATASIZE];
    dense_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[9]) * DATASIZE];
    fc1_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[10]) * DATASIZE];
    fc1_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[11]) * DATASIZE];
    fc2_weights = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[12]) * DATASIZE];
    fc2_bias = (datatype*)&Decoder_Weights[
      (i * DECODER_WEIGHT_SIZE + decoder_weight_offsets[13]) * DATASIZE];


    PhiDecoderLayer_16f_cpu(
      /* inputs */
      hidden_states, hidden_states_dims,
      attention_mask, attention_mask_dims,
      position_ids,
      old_past_keys, old_past_keys_dims,
      old_past_values, old_past_values_dims,
      /* Decoder Weights */
      input_layernorm_weights, input_layernorm_weights_len,
      input_layernorm_bias, decoder_eps,
      fc1_weights, fc1_weights_dims,
      fc1_bias,
      fc2_weights, fc2_weights_dims,
      fc2_bias,
      /* LARGE buffers */
      buff_1, buff_2, buff_3,
      buff_4, buff_5, buff_6,
      /* PhiAttention weights */
      q_proj_weights, q_proj_weights_dims,
      q_proj_bias,
      k_proj_weights, k_proj_weights_dims,
      k_proj_bias,
      v_proj_weights, v_proj_weights_dims,
      v_proj_bias,
  
      num_heads, head_dim, num_kv_heads,
      dense_weights, dense_weights_dims,
      dense_bias,
      /* init params */
      i,
      sin_cached, sin_cached_dims,
      cos_cached, cos_cached_dims,
      rotary_emb_dim,
      /* outputs */
      attn_output, attn_output_dims,
      past_keys, past_keys_dims,
      past_values, past_values_dims
    );

    // write kv dims to memory aswell
    uint32_t *kv_ptr;
    kv_ptr = (uint32_t*)((uint8_t*)past_keys + (MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE));
    for (int j = 0; j < past_keys_dims.size(); j++) { kv_ptr[j] = past_keys_dims[j]; }
    kv_ptr = (uint32_t*)((uint8_t*)past_values + (MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE));
    for (int j = 0; j < past_values_dims.size(); j++) { kv_ptr[j] = past_values_dims[j]; }

    // update input
    copyTensor_16f(attn_output, io_buff, attn_output_dims);
    hidden_states_dims = attn_output_dims;
  }

  /* Decoders Finished, output in UdoOutput buffer */
  int final_layernorm_weights_len = HIDDEN_SIZE;
  decoder_eps = 1e-5;
  std::vector<uint32_t> lm_head_weights_dims = {HIDDEN_SIZE, VOCAB_SIZE};

  // parse weights
  uint64_t offset = 0;
  datatype* final_layernorm_weights = (datatype*)(&LM_Head_Weights[0]);
  offset += HIDDEN_SIZE;
  datatype* final_layernorm_bias = (datatype*)(&LM_Head_Weights[offset*DATASIZE]);
  offset += HIDDEN_SIZE;
  datatype* lm_head_weights = (datatype*)(&LM_Head_Weights[offset*DATASIZE]);
  offset += HIDDEN_SIZE*VOCAB_SIZE;
  datatype* lm_head_bias = (datatype*)(&LM_Head_Weights[offset*DATASIZE]);

  uint32_t token_selected;

  LM_Head_16f(
    io_buff, hidden_states_dims,
    /* weights */
    final_layernorm_weights, final_layernorm_weights_len,
    final_layernorm_bias, decoder_eps,
    lm_head_weights, lm_head_weights_dims,
    lm_head_bias,
    /* buffers */
    buff_1, buff_2, buff_3,
    buff_4, buff_5,
    /* output */
    token_selected
  );

    std::cout << "finsihed LM head\n";

  // write the token as the first 4 bytes of the output buffer
  ((uint32_t*)UdoOutput)[0] = token_selected;

    std::cout << "freeing\n";

  free(buff_1);
  free(buff_2);
  free(buff_3);
  free(buff_4);
  free(buff_5);
  free(buff_6);
  free(io_buff);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 7, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

    /**
    * Add code here
    **/

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Decode"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 7, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace decode

CustomOpRegistration_t* register_DecodeCustomOp() {
  using namespace decode;
  static CustomOpRegistration_t DecodeRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &DecodeRegister;
}

REGISTER_OP(Decode, register_DecodeCustomOp);
