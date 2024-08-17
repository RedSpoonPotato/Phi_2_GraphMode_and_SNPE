#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <cmath>
#include "main_macros.h"

// #include <cfloat>
// #include <cstdint>

#ifdef ENABLE_FP16
    #include <arm_neon.h>
#endif

// comment out later
void func()
{

    // float16_t a = 1.0f;  // Duplicate 1.0f across the vector
    // float16_t b = 2.0f;  // Duplicate 2.0f across the vector
    // float16_t c = a + b;    // Perform FP16 addition

    // uint32_t x = 6;
    // float* y = (float*)&x;
    // std::cout << "y: " << *y << "\n";

    // // float16_t c = vadd_f16(a, b);    // Perform FP16 addition
    // std::cout << "size of fp16: " << sizeof(c) << "\n";
    // std::cout << "c: " << c << "\n";
    // unsigned short z = 16;
    // float16_t* x1 = (float16_t*)&z;
    // std::cout << "x1 casted as fp16: " << *x1 << "\n";
    // exit(0);
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
ushort float_to_half(float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint e = (b&0x7F800000)>>23; // exponent
    const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

// void fp16_to_fp32(ushort* in, float* out, const std::vector<uint32_t> dims) {
//     assert((void*)in != (void*)out);
//     size_t num_elem = dims.end()[-1];
//     for (int i = 0; i < dims.size()-1; i++) { num_elem *= dims[i]; }
//     for (size_t i = 0; i < num_elem; i++) {
//         out[i] = half_to_float(in[i]);
//     }
// }
// void fp32_to_fp16(float* in, ushort* out, const std::vector<uint32_t> dims) {
//     size_t num_elem = dims.end()[-1];
//     for (int i = 0; i < dims.size()-1; i++) { num_elem *= dims[i]; }
//     for (size_t i = 0; i < num_elem; i++) {
//         out[i] = float_to_half(in[i]);
//     }
// }

size_t reduceDims(const std::vector<size_t>& shape, int numDims = -1) {
    if (numDims == 0) {
        // reduce among none
        return 0;
    }
    else if (numDims < 0) {
        // case for reducing among all elements
        assert(numDims == -1);
        numDims = shape.size();
    }

    assert(shape.size() >= numDims);
    size_t numElem = shape.end()[-1];
    for (int i = 2; i < numDims+1; i++) {
        numElem *= shape.end()[-i];
    } 
    return numElem;
}

template <typename T>
size_t findNaN(T* tensor, const std::vector<size_t>& shape, std::string name="blank" ) {
    size_t totalElements = reduceDims(shape);
    size_t count = 0;
    for (size_t i = 0; i < totalElements; ++i) {
        if (std::isnan(static_cast<float>(tensor[i]))) {
            ++count;
            #ifdef FIX_MATH
                if (std::signbit(static_cast<float>(tensor[i]))) {
                    if (sizeof(T) == 4) { tensor[i] = static_cast<float>(FP32_NEG_INF); }
                    #ifdef ENABLE_FP16
                    else { tensor[i] = static_cast<float16_t>(FP16_NEG_INF); }
                    #endif
                }
                else {
                    if (sizeof(T) == 4) { tensor[i] = static_cast<float>(FP32_POS_INF); } 
                    #ifdef ENABLE_FP16
                    else { tensor[i] = static_cast<float16_t>(FP16_POS_INF); }
                    #endif
                }
            #endif
        }
    }
    std::cout << name << ": " << count << " NaNs\n";
    #ifdef FIX_MATH
    if (count != 0) {
        std::cout << ", BUT WE ARE FIXING AND IGNORING\n";
    }
    else {
        std::cout << "\n";
    }
    count = 0;
    #else 
    std::cout << "\n";
    #endif
    return count;
}

template <typename T>
size_t findInf(T* tensor, const std::vector<size_t>& shape, std::string name="blank" ) {
    size_t totalElements = reduceDims(shape);
    size_t count = 0;
    for (size_t i = 0; i < totalElements; ++i) {
        if (std::isinf(static_cast<float>(tensor[i]))) {
            ++count;
            #ifdef FIX_MATH
                if (std::signbit(static_cast<float>(tensor[i]))) {
                    if (sizeof(T) == 4) { tensor[i] = static_cast<float>(FP32_NEG_INF); }
                    #ifdef ENABLE_FP16
                    else { tensor[i] = static_cast<float16_t>(FP16_NEG_INF); }
                    #endif
                }
                else {
                    if (sizeof(T) == 4) { tensor[i] = static_cast<float>(FP32_POS_INF); } 
                    #ifdef ENABLE_FP16
                    else { tensor[i] = static_cast<float16_t>(FP16_POS_INF); }
                    #endif
                }
            #endif
        }
    }
    std::cout << name << ": " << count << " Infs";
    #ifdef FIX_MATH
    if (count != 0) {
        std::cout << ", BUT WE ARE FIXING AND IGNORING\n";
    }
    else {
        std::cout << "\n";
    }
    count = 0;
    #else 
    std::cout << "\n";
    #endif
    return count;
}

template <typename T>
size_t findNanAndInf(T* tensor, const std::vector<size_t>& shape, std::string name="blank" 
) {
    size_t count = 0;
    count += findNaN(tensor, shape, name);
    count += findInf(tensor, shape, name);
    return count;
}

void fp16_to_fp32(ushort* in, float* out, const std::vector<size_t> dims) {
    assert((void*)in != (void*)out);
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNanAndInf((float16_t*)in, dims, "fp16 before conversion") == 0);
    #endif
    size_t num_elem = reduceDims(dims);
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = half_to_float(in[i]);
    }
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNanAndInf((float*)out, dims, "fp32 after conversion") == 0);
    #endif
}
void fp32_to_fp16(float* in, ushort* out, const std::vector<size_t> dims) {
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNanAndInf((float*)in, dims, "fp32 before conversion") == 0);
    #endif
    size_t num_elem = reduceDims(dims);
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = float_to_half(in[i]);
    }
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNanAndInf((float16_t*)out, dims, "fp16 after conversion") == 0);
    #endif
}

template <typename T>
void printN(const std::string& str, const T* vec, const size_t N, bool quantize, bool enable_fp16=false) {
    std::cout << str;
    std::cout << ": [";
    for (size_t i = 0; i < N; i++) {
        if (quantize) {
            std::cout << (int)(vec[i]);
        }
        else if (enable_fp16) {
            std::cout << half_to_float((FP16)vec[i]);
        }
        else {
            std::cout << vec[i];
        }
        std::cout << ", ";
    }
    std::cout << "]\n";
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
void printTensorNoName(const T* tensor, const std::vector<size_t> shape, size_t depth = 0, size_t offset = 0) {
    size_t current_dim = shape[depth];
    if (depth == shape.size() - 1) {
        for (size_t i = 0; i < std::min<size_t>(3, current_dim); ++i) {
            std::cout << std::setw(4) << tensor[offset + i];
            if (i < std::min<size_t>(3, current_dim) - 1) std::cout << ", ";
        }
        if (current_dim > 6) std::cout << ", ..., ";
        for (size_t i = std::max<size_t>(3, current_dim - 3); i < current_dim; ++i) {
            std::cout << std::setw(4) << tensor[offset + i];
            if (i < current_dim - 1) std::cout << ", ";
        }
    } else {
        std::cout << "{";
        for (size_t i = 0; i < std::min<size_t>(3, current_dim); ++i) {
            printTensorNoName(tensor, shape, depth + 1, offset + i * shape[depth + 1]);
            if (i < std::min<size_t>(3, current_dim) - 1) std::cout << ", ";
        }
        if (current_dim > 6) std::cout << ", ..., ";
        for (size_t i = std::max<size_t>(3, current_dim - 3); i < current_dim; ++i) {
            printTensorNoName(tensor, shape, depth + 1, offset + i * shape[depth + 1]);
            if (i < current_dim - 1) std::cout << ", ";
        }
        std::cout << "}";
    }
}

#define PRINT_SIZE 3

template <typename T>
void printTensor2(const T* tensor, const std::vector<size_t>& shape) {
    size_t shape_size = 2;
    size_t offset = 1;
    for (size_t i = 1; i < shape.size(); i++) { offset *= shape[i]; }
    assert(shape.size() == shape_size);

    if (shape.end()[-shape_size] <= 2*PRINT_SIZE) {
        assert(shape.end()[-1] >= 2*PRINT_SIZE);
        for (size_t i = 0; i < shape.end()[-shape_size]; i++) {
            std::cout << "[";
            if (shape.end()[-1] <= 2*PRINT_SIZE) {
                for (size_t j = 0; j < shape.end()[-1]; j++ ) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            } else {
                // print first
                for (size_t j = 0; j < PRINT_SIZE; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
                std::cout << " ..., ";
                // print last
                for (size_t j = shape.end()[-shape_size + 1] - PRINT_SIZE; j < shape.end()[-shape_size + 1]; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            }
            std::cout << "],\n";
        }
    }
    else {
        assert(shape.end()[-shape_size] >= 2*PRINT_SIZE);
        std::cout << "[\n";
        
        for (size_t i = 0; i < PRINT_SIZE; i++) {
            std::cout << "[";
            if (shape.end()[-1] <= 2*PRINT_SIZE) {
                for (size_t j = 0; j < shape.end()[-1]; j++ ) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            }
            else {
                // print first
                for (size_t j = 0; j < PRINT_SIZE; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
                std::cout << " ..., ";
                // print last
                for (size_t j = shape.end()[-shape_size + 1] - PRINT_SIZE; j < shape.end()[-shape_size + 1]; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            }
            std::cout << "],\n";
        }

        std::cout << "\n...\n";

        for (size_t i = shape.end()[-shape_size] - PRINT_SIZE; i < shape.end()[-shape_size]; i++) {
            std::cout << "[";
            if (shape.end()[-1] <= 2*PRINT_SIZE) {
                for (size_t j = 0; j < shape.end()[-1]; j++ ) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            } else {
                // print first
                for (size_t j = 0; j < PRINT_SIZE; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
                std::cout << " ..., ";
                // print last
                for (size_t j = shape.end()[-shape_size + 1] - PRINT_SIZE; j < shape.end()[-shape_size + 1]; j++) {
                    std::cout << tensor[i*offset + j] << ",";
                }
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";
    }
}

template <typename T>
void printTensor(const std::string& name, const T* tensor, const std::vector<size_t>& shape) {
    printV(name, shape);
    assert(shape.size() >= 2);
    std::vector<size_t> shape_2d = {shape.end()[-2], shape.end()[-1]};
    if (sizeof(T) == 2) {
        std::cout << "case 1\n";
        #ifdef ENABLE_FP16
            printTensor2((float16_t*)tensor, shape_2d);
        #endif
    }
    else {
        std::cout << "case 2\n";
        printTensor2(tensor, shape_2d);
    }
}

template <typename T>
void printTensorColumn(
    const std::string& name, 
    const T* tensor, 
    const std::vector<size_t>& shape, 
    size_t col = 0
) {
    size_t counter = 0;
    size_t max_counter_val = 20;
    std::cout << name << "(column "  << std::to_string(col) << "): [";
    for (size_t i = 0; i < shape.end()[-2]; i++) {
        if (counter >= max_counter_val) { std::cout << "\n"; }
        std::cout << tensor[(shape.end()[-1] * i) + col];
        if (i < shape.end()[-2]-1) { std::cout << ", "; }
    }
    std::cout << "]\n";
}

template <typename T>
void printNumElem(
    const std::vector<T>& vec
) {
    size_t last_index = vec.size() - 1;
    size_t i;
    for (i = 0; i < vec.size(); i++) {
        if (vec[last_index - i] != 0) { break; }
    }
    std::cout << "num of elements in vec: " << vec.size() - i << "\n";
}


template <typename T>
void saveTensor(const std::string& path, const T* tensor, const std::vector<size_t>& shape) {
    std::ofstream outFile(path, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }
    // Save the shape
    size_t dimensions = shape.size();
    // outFile.write(reinterpret_cast<const char*>(&dimensions), sizeof(size_t));
    // outFile.write(reinterpret_cast<const char*>(shape.data()), dimensions * sizeof(size_t));
    // Calculate the total number of elements
    size_t totalElements = 1;
    for (size_t dim : shape) {
        totalElements *= dim;
    }
    // Save the tensor data
    outFile.write(reinterpret_cast<const char*>(tensor), totalElements * sizeof(T));
    outFile.close();
    if (!outFile.good()) {
        std::cerr << "Error occurred while writing to file: " << path << std::endl;
    }
}


// template <typename T>
// void printTensor_unfinished(const T* tensor, const std::vector<size_t>& shape, size_t depth, size_t tot_offset=0) {
//     size_t shape_size = 2;
//     size_t offset = 1;
//     for (size_t i = 1; i < shape.size(); i++) { offset *= shape[i]; }
//     assert(shape.size() == shape_size);

//     // base case
//     if (depth == shape.size()) {

//     }
//     else if (shape.end()[-shape_size] <= 2*PRINT_SIZE) {
//         for (size_t i = 0; i < shape.end()[-shape_size]; i++) {
//             std::cout << "[";
//             printTensor(tensor + tot_offset, shape, shape_size); // change
//             std::cout << "],\n";
//         }
//     }
//     else {
//         std::cout << "[\n";
//         for (size_t i = 0; i < PRINT_SIZE; i++) {
//             std::cout << "[";
//             printTensor(tensor + tot_offset, shape, shape_size); // change
//             std::cout << "],\n";
//         }
//         std::cout << "\n...\n";
//         for (size_t i = shape.end()[-shape_size] - PRINT_SIZE; i < shape.end()[-shape_size]; i++) {
//             std::cout << "[";
//             printTensor(tensor + tot_offset, shape, shape_size); // change
//             std::cout << "],\n";
//         }
//         std::cout << "]\n";
//     }
// }



// does not really work with every datatype
// template <typename T>
// void printT(
//     const std::string& str, const std::vector<uint32_t>& dims, const T* tensor, 
//     bool precise, size_t max_elem
//     ) {
//     // print dims
//     printV(str, dims);
//     // calculate size of each dimension
//     std::vector<uint32_t> dim_sizes = {dims.end()[-1]};
//     for (int i = dims.size()-2; i >= 0; i--) {
//         uint32_t offset = dims[i] * dim_sizes.end()[-1];
//         dim_sizes.push_back(offset);
//     }
//     // calculate total number of elements
//     size_t tot_elements = 1;
//     for (auto i : dims) { tot_elements *= i; }
//     // print everything
//     for (auto i : dims) { std::cout << "["; }
//     for (size_t i = 0; i < tot_elements; i++) {
//         if (i > max_elem) {break;}
//         for (auto j : dim_sizes) {
//             if (i % j == 0 && i != 0) {
//                 std::cout << "]\n[";
//             }
//         }
//         if (precise) { std::cout << std::setprecision(30) 
//             << half_to_float(static_cast<ushort>(tensor[i])) << ", "; }
//         else { std::cout << half_to_float(static_cast<ushort>(tensor[i]) << ", "; }
//     }
//     for (auto i : dims) { std::cout << "]"; }
//     std::cout << "\n";
// }


template <typename T>
void copyTensor(const T* ten1, T* out, const std::vector<size_t>& dims) {
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i];
    }
}


void mySoftmax(const float* tensor, float* out, const std::vector<size_t>& dims) {
    int rank = dims.size();
    float maxElt = std::numeric_limits<float>::lowest();
    float expSum;
    // std::cout << "max Elt: " << maxElt << "\n";
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
            #if defined(DEBUG) && defined(ENABLE_FP16)
                if (std::isnan(ei)) {
                    std::cout << "\nvalue: " << tensor[i*inner_dim_size + j] << ", ";
                    std::cout <<  "max: " << maxElt<< ", ";
                    std::cout <<  "diff: " << tensor[i*inner_dim_size + j] - maxElt << ", ";
                    std::cout <<  "exp(diff): " << ei << "\n";
                    exit(0);
                }
            #endif
            out[i*inner_dim_size + j] = ei;
            expSum += ei;
        }
        // normalize
        for (int j = 0; j < inner_dim_size; j++) {
            out[i*inner_dim_size + j] /= expSum;
        }
    }
}

// innermost softmax
void bufferedSoftmax(
    const std::vector<size_t>& buffered_shape,
    const std::vector<size_t>& unbuffered_shape,
    float* buff
) {
    assert(buffered_shape.size() == 4);
    assert(unbuffered_shape.size() == 4);
    // calculate offsets
    std::vector<size_t> buffered_offsets(buffered_shape.size());
    buffered_offsets.end()[-1] = 1;
    for (size_t i = buffered_shape.size() - 1; i > 0; i--) {
        // std::cout << "i: " << i << "\n";
        buffered_offsets[i-1] = buffered_offsets[i] * buffered_shape[i];
    }
    printV("softmax buffered_offsets", buffered_offsets);
    // apply softmax
    size_t offset;
    for (size_t i = 0; i < unbuffered_shape[0]; i++) {
    for (size_t j = 0; j < unbuffered_shape[1]; j++) {
    for (size_t k = 0; k < unbuffered_shape[2]; k++) {
        offset =    i*buffered_offsets[0] +
                    j*buffered_offsets[1] +
                    k*buffered_offsets[2];
        // if (j == 0) { 
            // std::cout << "offset: " << offset << ", inner_size: " << unbuffered_shape.end()[-1]  << "\n"; 
        // }
        mySoftmax(&buff[offset], &buff[offset], { unbuffered_shape.end()[-1] });
    }
    }
    }
}


void layernorm_1d_32f(
    const float* vec, const float* weight, const float* bias, float* out,
    const size_t vec_len, const float eps
) {
    // mean
    float mean = 0;
    for (size_t i = 0; i < vec_len; i++) { mean += vec[i]; }
    mean = mean / vec_len;
    // variance
    float variance = 0;
    for (size_t i = 0; i < vec_len; i++) { 
        variance += pow(vec[i] - mean, 2); 
    }
    variance = variance / vec_len;
    float variance_plus_eps_sprt = sqrt(variance + eps);
    // output
    for (size_t i = 0; i < vec_len; i++) { 
        out[i] = (vec[i] - mean) / variance_plus_eps_sprt; 
        out[i] = (out[i] * weight[i]) + bias[i];
    }
}

void layernorm_Nd_32f(
    const float* tensor, const float* weight, const float* bias, float* out,
    const std::vector<size_t>& tensor_dims,
    const float eps
) {
    int num_vectors = 1;
    size_t vec_len = tensor_dims.end()[-1];
    for (int i = 0; i < tensor_dims.size() - 1; i++) { num_vectors *= tensor_dims[i]; }
    for (int i = 0; i < num_vectors; i++) {
        layernorm_1d_32f(
            &tensor[i*vec_len], weight, bias, &out[i*vec_len],
            vec_len, eps);
    }
}

template <typename T>
size_t Argmax(const T* ptr, const size_t len) {
    size_t max_index = 0;
    T max = std::numeric_limits<T>::lowest();
    for (size_t i = 0; i < len; i++) {
        if (max < ptr[i]) {
            max = ptr[i];
            max_index = i;
        }
    }
    return max_index;
}

size_t fp16Argmax(const ushort* ptr, const size_t len) {
    size_t max_index = 0;
    float float_max = std::numeric_limits<float>::lowest();
    float float_val;
    for (size_t i = 0; i < len; i++) {
        float_val = half_to_float(ptr[i]);
        if (float_max < float_val) {
            float_max = float_val;
            max_index = i;
        }
    }
    return max_index;
}

// // not input-to-output safe
// void truncate_u8(
//     const uint8_t* input, const std::vector<size_t>& input_dims,
//     uint8_t* output, std::vector<size_t>& output_dims,
//     const std::vector<int>& dims_to_split,
//     const std::vector<int>& values,
//     const std::vector<int>& colon_lefts
// ) {
//     // assertions
//     const size_t rank = input_dims.size();
//     assert(rank <= 4); // assume 4d or less
//     assert(values.size() <= rank);
//     assert(dims_to_split.size() == values.size() == colon_lefts.size());
//     assert(input != output);
//     for (int i = 0; i < values.size(); i++) {
//         assert(values[i] < input_dims[dims_to_split[i]]);
//     }
//     // indice computation
//     int indice_start[] = {0,0,0,0};
//     int indice_end[]   = {0,0,0,0}; // including
//     for (int i = 0; i < rank; i++) { indice_end[i] = input_dims[i] - 1; }
//     for (int i = 0; i < values.size(); i++) {
//         if (colon_lefts[i]) { indice_end[dims_to_split[i]] = values[i] - 1; }
//         else                { indice_start[dims_to_split[i]] = values[i]; }
//     }

//     #ifdef DEBUG
//         for (int i=0;i<4;i++) {std::cout << indice_start[i] << " ";}
//         for (int i=0;i<4;i++) {std::cout << indice_end[i] << " ";}
//     #endif

//     // writing output
//     unsigned long long elements_written = 0;
//     unsigned long long dim_offsets[4] = {1, 1, 1, 1};
//     for (int i = rank - 2; i >= 0; i--) {
//         dim_offsets[i] = dim_offsets[i+1] * input_dims[i+1];
//     }

//     #ifdef DEBUG
//         std::cout << "\ttruncation(): about to enter quad for-loop\n";
//     #endif

//     for (int a = indice_start[0]; a <= indice_end[0]; a++) {
//         for (int b = indice_start[1]; b <= indice_end[1]; b++) {
//             for (int c = indice_start[2]; c <= indice_end[2]; c++) {
//                 for (int d = indice_start[3]; d <= indice_end[3]; d++) {
//                     output[elements_written] = input[
//                         d + 
//                         c*dim_offsets[2] +
//                         b*dim_offsets[1] +
//                         a*dim_offsets[0]];
//                     elements_written++;
//                 }
//             }
//         }
//     }
//     // writing output_dims
//     output_dims = std::vector<size_t>();
//     for (int i = 0; i < input_dims.size(); i++) {
//         output_dims.push_back(indice_end[i]+1 - indice_start[i]);
//     }
// }


// not input-to-output safe
template <typename T>
void truncate(
    const T* input, const std::vector<size_t>& input_dims,
    T* output, std::vector<size_t>& output_dims,
    const std::vector<int>& dims_to_split,
    const std::vector<int>& values,
    const std::vector<int>& colon_lefts
) {
    // assertions
    const size_t rank = input_dims.size();
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

    // for (int i=0;i<4;i++) {std::cout << indice_start[i] << " ";}
    // for (int i=0;i<4;i++) {std::cout << indice_end[i] << " ";}

    // writing output
    unsigned long long elements_written = 0;
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * input_dims[i+1];
    }
    // std::cout << "\ttruncation(): about to enter quad for-loop\n";
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
    output_dims = std::vector<size_t>();
    for (int i = 0; i < input_dims.size(); i++) {
        output_dims.push_back(indice_end[i]+1 - indice_start[i]);
    }
}

// void mul_u8(const uint8_t* ten1, const uint8_t* ten2, uint8_t* out, const std::vector<size_t>& dims) {
//     size_t num_elem = 1;
//     for (size_t i = 0; i < dims.size(); i++) { num_elem *= i; }
//     for (size_t i = 0; i < num_elem; i++) {
//         out[i] = ten1[i] * ten2[i];
//     }
// }

template <typename T>
void mul(const T* ten1, const T* ten2, T* out, const std::vector<size_t>& dims) {
    // std::cout << "check 1\n";
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    // std::cout << "\t\t multiplication num_elem: " << num_elem << "\n";
    // std::cout << "check 2\n";
    for (size_t i = 0; i < num_elem; i++) {
        // std::cout << "check 3\n";
        out[i] = ten1[i] * ten2[i];
        // std::cout << "check 4\n";
    }
    // std::cout << "check 5\n";
}

// void add_u8(const uint8_t* ten1, const uint8_t* ten2, uint8_t* out, const std::vector<size_t>& dims) {
//     size_t num_elem = 1;
//     for (size_t i = 0; i < dims.size(); i++) { num_elem *= i; }
//     for (size_t i = 0; i < num_elem; i++) {
//         out[i] = ten1[i] + ten2[i];
//     }
// }

template <typename T>
void add(const T* ten1, const T* ten2, T* out, const std::vector<size_t>& dims) {
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] + ten2[i];
    }
}

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
// void rotary_emb_u8(
//     const uint8_t* x, const std::vector<size_t>& x_dims,
//     const int seq_len,
//     const uint8_t* sin_cached, const std::vector<size_t>& sin_cached_dims,
//     const uint8_t* cos_cached, const std::vector<size_t>& cos_cached_dims,
//     uint8_t* sin, std::vector<size_t>& sin_dims,
//     uint8_t* cos, std::vector<size_t>& cos_dims
// ) {
//     assert(seq_len <= SIN_COS_MAX_SEQ_LEN); // TEMP solution
//     assert(sin_cached_dims.size() == cos_cached_dims.size());
//     for (int i = 0; i < sin_cached_dims.size(); i++) {
//         // idk if this needs to be true, but if it does not, change code below
//         assert(sin_cached_dims[i] == cos_cached_dims[i]);
//     }
//     std::vector<int> dims_to_split = {0};
//     std::vector<int> values = {seq_len};
//     std::vector<int> colon_lefts = {1};
//     std::cout << "\tCalling truncate()\n";
//     printV("x", x_dims);
//     std::cout << "\tseq_len in truncate: " << seq_len << "\n";
//     truncate_u8(
//         cos_cached, cos_cached_dims, cos, cos_dims, 
//         dims_to_split, // outer dim
//         values,
//         colon_lefts
//     );
//     truncate_u8(
//         sin_cached, sin_cached_dims, sin, sin_dims, 
//         dims_to_split, // outer dim
//         values,
//         colon_lefts
//     );
// }

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
template <typename T>
void rotary_emb(
    const T* x, const std::vector<size_t>& x_dims,
    const int seq_len,
    const T* sin_cached, const std::vector<size_t>& sin_cached_dims,
    const T* cos_cached, const std::vector<size_t>& cos_cached_dims,
    T* sin, std::vector<size_t>& sin_dims,
    T* cos, std::vector<size_t>& cos_dims
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
    // std::cout << "\tCalling truncate()\n";
    // printV("x", x_dims);
    // std::cout << "\tseq_len in truncate: " << seq_len << "\n";
    truncate(
        cos_cached, cos_cached_dims, cos, cos_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
    truncate(
        sin_cached, sin_cached_dims, sin, sin_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
}

// void gather_u8(
//     const uint8_t* x, const std::vector<size_t>& x_dims,
//     const std::vector<int>& indices,
//     uint8_t* out, std::vector<size_t>& out_dims
// ) {
//     // offset computation
//     int rank = x_dims.size();
//     assert(rank <= 4);
//     unsigned long long dim_offsets[4] = {1, 1, 1, 1};
//     for (int i = rank - 2; i >= 0; i--) {
//         dim_offsets[i] = dim_offsets[i+1] * x_dims[i+1];
//     }
//     unsigned long long offset = dim_offsets[0];
//     std::cout << "writing data\n";
//     // writing to out
//     for (int i = 0; i < indices.size(); i++) {
//         for (int j = 0; j < offset; j++) {
//             out[i*offset + j] = x[indices[i]*offset + j];
//         }
//     }
//     // writing out_dims
//     out_dims = std::vector<size_t>();
//     out_dims.push_back(indices.size());
//     for (int i = 1; i < rank; i++) {
//         out_dims.push_back(x_dims[i]);
//     }
// }

template <typename T>
void gather(
    const T* x, const std::vector<size_t>& x_dims,
    const std::vector<int>& indices,
    T* out, std::vector<size_t>& out_dims
) {
    // offset computation
    int rank = x_dims.size();
    assert(rank <= 4);
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = rank - 2; i >= 0; i--) {
        dim_offsets[i] = dim_offsets[i+1] * x_dims[i+1];
    }
    unsigned long long offset = dim_offsets[0];
    // std::cout << "writing data\n";
    // writing to out
    for (int i = 0; i < indices.size(); i++) {
        for (int j = 0; j < offset; j++) {
            out[i*offset + j] = x[indices[i]*offset + j];
        }
    }
    // writing out_dims
    out_dims = std::vector<size_t>();
    out_dims.push_back(indices.size());
    for (int i = 1; i < rank; i++) {
        out_dims.push_back(x_dims[i]);
    }
}

template <typename T>
void concat(
    const T* x1, const std::vector<size_t>& x1_dims,
    const T* x2, const std::vector<size_t>& x2_dims,
    const int axis,
    T* out, std::vector<size_t>& out_dims
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
    std::vector<size_t> x1_4d_dims = {1, 1, 1, 1};
    std::vector<size_t> x2_4d_dims = {1, 1, 1, 1};
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
    // std::cout << "\twriting x2 to out\n";
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

// void rotate_half_u8(
//     const uint8_t* x, const std::vector<size_t>& x_dims,
//     uint8_t* out, std::vector<size_t>& out_dims
// ) {
//     // internal buffers
//     // uint8_t x1[QUERY_STATES_BUFF_SIZE];
//     uint8_t* x1 = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
//     // uint8_t x2[QUERY_STATES_BUFF_SIZE];
//     uint8_t* x2 = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
//     std::vector<size_t> x1_dims, x2_dims;

//     int x1_len = x_dims.end()[-1] / 2;
//     std::vector<int> dims_to_split = {(int)x_dims.size()-1};
//     std::vector<int> values = {x1_len};
//     std::vector<int> colon_left = {1};
//     std::vector<int> colon_right = {0};
//     truncate_u8(
//         x, x_dims, x1, x1_dims, dims_to_split,
//         values, colon_left
//     );
//     truncate_u8(
//         x, x_dims, x2, x2_dims, dims_to_split,
//         values, colon_right
//     );
//     std::cout << "\t\tfinished both truncate()s in rotate_half()\n";
//     // negate x2
//     unsigned long long x2_size = 1;
//     for (auto i : x2_dims) { x2_size *= i; }
//     for (int i = 0; i < x2_size; i++) {x2[i] = -1 * x2[i];}
//     // concat
//     std::cout << "\t\tCalling concat()\n";
//     concat_u8(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
//     std::cout << "\t\tfreeing memory\n";
//     free(x1);
//     free(x2);
// }

template <typename T>
void rotate_half(
    const T* x, const std::vector<size_t>& x_dims,
    T* out, std::vector<size_t>& out_dims
) {
    // internal buffers
    // T x1[QUERY_STATES_BUFF_SIZE];
    T* x1 = (T*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(T));
    // T x2[QUERY_STATES_BUFF_SIZE];
    T* x2 = (T*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(T));
    std::vector<size_t> x1_dims, x2_dims;

    int x1_len = x_dims.end()[-1] / 2;
    std::vector<int> dims_to_split = {(int)x_dims.size()-1};
    std::vector<int> values = {x1_len};
    std::vector<int> colon_left = {1};
    std::vector<int> colon_right = {0};
    truncate(
        x, x_dims, x1, x1_dims, dims_to_split,
        values, colon_left
    );
    truncate(
        x, x_dims, x2, x2_dims, dims_to_split,
        values, colon_right
    );
    // std::cout << "\t\tfinished both truncate()s in rotate_half()\n";
    // negate x2
    unsigned long long x2_size = 1;
    for (auto i : x2_dims) { x2_size *= i; }
    for (int i = 0; i < x2_size; i++) {x2[i] = -1 * x2[i];}
    // concat
    // std::cout << "\t\tCalling concat()\n";
    // concat(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims); // i believe this is wrong
    concat(x2, x2_dims, x1, x1_dims, int(x1_dims.size()-1), out, out_dims);
    // std::cout << "\t\tfreeing memory\n";
    free(x1);
    free(x2);
}

// // SUGGESTION: unquantize the multiplication and addition inside
// void apply_rotary_pos_emb_u8(
//     const uint8_t* q, const std::vector<size_t>& q_dims,
//     const uint8_t* k, const std::vector<size_t>& k_dims,
//     const uint8_t* cos, const std::vector<size_t>& cos_dims,
//     const uint8_t* sin, const std::vector<size_t>& sin_dims,
//     const std::vector<int>& position_ids, const int unsqueeze_dim,
//     uint8_t* q_embed, std::vector<size_t>& q_embed_dims,
//     uint8_t* k_embed, std::vector<size_t>& k_embed_dims
// ) {
//     // buffers prbably dont need to be this big
//     // uint8_t cos_buff[SIN_COS_BUFF_SIZE];
//     uint8_t* cos_buff = (uint8_t*)malloc(SIN_COS_BUFF_SIZE * sizeof(uint8_t));
//     // uint8_t sin_buff[SIN_COS_BUFF_SIZE];
//     uint8_t* sin_buff = (uint8_t*)malloc(SIN_COS_BUFF_SIZE * sizeof(uint8_t));
//     std::vector<size_t> cos_buff_dims, sin_buff_dims;
//     std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
//     std::cout << "cos[0]: " << cos[0] << "\n";
//     printV("cos_dims", cos_dims);
//     for (auto i : position_ids) {std::cout << i << ", ";}
//     std::cout << "\n";
//     gather_u8(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
//     std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
//     gather_u8(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
//     std::cout << "unsqueeze_dim: " << unsqueeze_dim << "\n";
//     cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1);
//     sin_buff_dims.insert(sin_buff_dims.begin() + unsqueeze_dim, 1);
//     printV("cos_buff_dims", cos_buff_dims);
//     printV("q_dims", q_dims);
//     printV("k_dims", k_dims);
//     // computing embedding
//     // assume the last 2 dims are the same size for q and cos
//     assert(cos_buff_dims.end()[-1] == q_dims.end()[-1]);
//     assert(cos_buff_dims.end()[-1] == k_dims.end()[-1]);
//     // assert(q_dims.end()[-2] == cos_buff_dims.end()[-2] == k_dims.end()[-2]);
//     assert(q_dims.end()[-2] == k_dims.end()[-2]);
//     int q_rank = q_dims.size();
//     assert(q_rank <= 4);
//     unsigned long long q_dim_offsets[4] = {1, 1, 1, 1};
//     for (int i = q_rank - 2; i >= 0; i--) {
//         q_dim_offsets[i] = q_dim_offsets[i+1] * q_dims[i+1]; //ugly fix
//     }
//     assert(q_dims[0] == k_dims[0]); // we can adjust if this needs to be false
//     assert(q_dims[1] == k_dims[1]); // would need to create another collection of for loops
//     std::cout << "\tallocating interal buffers of k and q\n";
//     // uint8_t q_temp_buff[QUERY_STATES_BUFF_SIZE];
//     uint8_t* q_temp_buff = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
//     // uint8_t k_temp_buff[QUERY_STATES_BUFF_SIZE];
//     uint8_t* k_temp_buff = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
//     std::cout << "\tCalling Multiply Loop\n";
//     for (int i = 0; i < q_dims[0]; i++) {
//         for (int j = 0; j < q_dims[1]; j++) {
//             mul_u8(
//                 &(q[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff, 
//                 q_temp_buff, cos_buff_dims);
//             mul_u8(
//                 &(k[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff,
//                 k_temp_buff, cos_buff_dims);
//         }
//     }
//     std::cout << "\tCalling Rotate_Half\n";
//     rotate_half_u8(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
//     std::cout << "\tCalling Rotate_Half\n";
//     printV("k_dims", k_dims);
//     rotate_half_u8(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims
//     std::cout << "\tCalling Mutliply Loop\n";
//     for (int i = 0; i < q_dims[0]; i++) {
//         for (int j = 0; j < q_dims[1]; j++) {
//             mul_u8(
//                 &(q_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
//                 q_embed, sin_buff_dims);
//             mul_u8(
//                 &(k_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
//                 k_embed, sin_buff_dims);
//         }
//     }
//     std::cout << "\tCalling add_32f\n";
//     add_u8(q_embed, q_temp_buff, q_embed, q_embed_dims);
//     std::cout << "\tCalling add_32f\n";
//     add_u8(k_embed, k_temp_buff, k_embed, k_embed_dims);
//     //free
//     free(cos_buff);
//     free(sin_buff);
// }

// SUGGESTION: unquantize the multiplication and addition inside
template <typename T>
void apply_rotary_pos_emb(
    const T* q, const std::vector<size_t>& q_dims,
    const T* k, const std::vector<size_t>& k_dims,
    const T* cos, const std::vector<size_t>& cos_dims,
    const T* sin, const std::vector<size_t>& sin_dims,
    const std::vector<int>& position_ids, const int unsqueeze_dim,
    T* q_embed, std::vector<size_t>& q_embed_dims,
    T* k_embed, std::vector<size_t>& k_embed_dims
) {
    // buffers prbably dont need to be this big
    // T cos_buff[SIN_COS_BUFF_SIZE];
    T* cos_buff = (T*)malloc(SIN_COS_BUFF_SIZE * sizeof(T));
    // T sin_buff[SIN_COS_BUFF_SIZE];
    T* sin_buff = (T*)malloc(SIN_COS_BUFF_SIZE * sizeof(T));
    std::vector<size_t> cos_buff_dims, sin_buff_dims;
    // std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    // std::cout << "cos[0]: " << cos[0] << "\n";
    // printV("cos_dims", cos_dims);
    // printV("position ids", position_ids);
    // std::cout << "\n";
    gather(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
    // printTensor("cos after gather", cos_buff, cos_buff_dims);
    gather(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
    // std::cout << "unsqueeze_dim: " << unsqueeze_dim << "\n";
    { 
        // this is may be incorrect
        cos_buff_dims.insert(cos_buff_dims.begin() + 0, BATCH_SIZE);
        sin_buff_dims.insert(sin_buff_dims.begin() + 0, BATCH_SIZE);
    }
    cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1); // unsure if correct
    sin_buff_dims.insert(sin_buff_dims.begin() + unsqueeze_dim, 1); // unsure if correct
    // cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1);
    // cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1);
    // printV("cos_buff_dims", cos_buff_dims);
    // printV("q_dims", q_dims);
    // printV("k_dims", k_dims);
    // computing embedding
    // assume the last 2 dims are the same size for q and cos
    assert(cos_buff_dims.end()[-1] == q_dims.end()[-1]);
    assert(cos_buff_dims.end()[-1] == k_dims.end()[-1]);
    // assert(q_dims.end()[-2] == cos_buff_dims.end()[-2] == k_dims.end()[-2]);
    assert(q_dims.end()[-2] == k_dims.end()[-2]);
    int q_rank = q_dims.size();
    assert(q_rank <= 4);
    std::vector<size_t> q_dim_offsets = {1, 1, 1, 1};
    for (int i = q_rank - 2; i >= 0; i--) {
        q_dim_offsets[i] = q_dim_offsets[i+1] * q_dims[i+1]; //ugly fix
    }
    assert(q_dims[0] == k_dims[0]); // we can adjust if this needs to be false
    assert(q_dims[1] == k_dims[1]); // would need to create another collection of for loops
    // std::cout << "\tallocating interal buffers of k and q\n";
    // T q_temp_buff[QUERY_STATES_BUFF_SIZE];
    T* q_temp_buff = (T*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(T));
    // T k_temp_buff[QUERY_STATES_BUFF_SIZE];
    T* k_temp_buff = (T*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(T));
    // std::cout << "\tCalling Multiply Loop\n";
    // printV("cos_buff_dims", cos_buff_dims);
    // printV("q_dims:", q_dims);
    // printV("q_dim_offsets", q_dim_offsets);
    // printTensor("query right before multiplication", q, q_dims);
    // printTensor("cos_buff right before multiplication", cos_buff, cos_buff_dims);
    // printTensor("q * cos BEFORE", q_temp_buff, q_dims);

    #ifdef ENABLE_FP16
        #define ROTARY_TYPE float16_t
    #else
        #define ROTARY_TYPE T
    #endif

    std::cout << "first multiple\n";

    // remove all the do math stuff later**

    #ifdef DO_MATH
    for (size_t i = 0; i < q_dims[0]; i++) {
        for (size_t j = 0; j < q_dims[1]; j++) {
            mul(
                (ROTARY_TYPE*)&(q[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), (ROTARY_TYPE*)cos_buff, 
                (ROTARY_TYPE*)&(q_temp_buff[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff_dims);
            mul(
                (ROTARY_TYPE*)&(k[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), (ROTARY_TYPE*)cos_buff,
                (ROTARY_TYPE*)&(k_temp_buff[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff_dims);
        }
    }
    #endif

    std::cout << "done\n";
    rotate_half(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
    rotate_half(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims

    #ifdef DO_MATH
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul(
                (ROTARY_TYPE*)&(q_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), (ROTARY_TYPE*)sin_buff, 
                (ROTARY_TYPE*)&(q_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff_dims);
            mul(
                (ROTARY_TYPE*)&(k_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), (ROTARY_TYPE*)sin_buff, 
                (ROTARY_TYPE*)&(k_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff_dims);
        }
    }

    add((ROTARY_TYPE*)q_embed, (ROTARY_TYPE*)q_temp_buff, (ROTARY_TYPE*)q_embed, q_embed_dims);
    add((ROTARY_TYPE*)k_embed, (ROTARY_TYPE*)k_temp_buff, (ROTARY_TYPE*)k_embed, k_embed_dims);
    #endif

    //free
    free(cos_buff);
    free(sin_buff);
    free(q_temp_buff);
    free(k_temp_buff);
}

// void transpose_u8(
//     const uint8_t* tensor, uint8_t* out, const std::vector<size_t>& perm,
//     const std::vector<size_t>& tensor_dims, std::vector<size_t>& out_dims
// ) {
//     // not safe to do inplace
//     assert(tensor != out);
//     int rank = tensor_dims.size();
//     assert(rank == 4);
//     // set out_dims
//     out_dims.resize(4);
//     for (size_t i = 0; i < 4; i++) {
//         out_dims[i] = tensor_dims[perm[i]];
//     }
//     // code
//     int num_elements;
//     std::vector<int> old_indices(4);
//     std::vector<int> new_indices(4);
//     std::vector<int> dim_offsets = {1,1,1,1};
//     for (int i = rank - 2; i >= 0; i--) {
//         dim_offsets[i] = dim_offsets[i+1] * tensor_dims[i+1];
//     }
//     num_elements = dim_offsets[0];
//     // out_dim stuff
//     std::vector<int> out_dim_offsets = {1,1,1,1};
//     for (int i = rank - 2; i >= 0; i--) {
//         out_dim_offsets[i] = out_dim_offsets[i+1] * out_dims[i+1];
//     }
    
//     for (int i = 0; i < num_elements; i++) {
//         // get old indices
//         size_t index = i;
//         for (int j = 0; j < 4; j++) {
//             old_indices[j] = index / dim_offsets[j];
//             index = index % dim_offsets[j];
//         }
//         // get new indices
//         for (int j = 0; j < 4; j++) {
//             new_indices[j] = old_indices[perm[j]];
//         }
//         // write data
//         out[
//             new_indices[0] +
//             new_indices[1] * out_dim_offsets[1] +
//             new_indices[2] * out_dim_offsets[2] +
//             new_indices[3] * out_dim_offsets[3]
//         ] = tensor[i];
//     }
// }

template <typename T>
void transpose(
    const T* tensor, T* out, const std::vector<size_t>& perm,
    const std::vector<size_t>& tensor_dims, std::vector<size_t>& out_dims
) {
    // not safe to do inplace
    assert(tensor != out);
    int rank = tensor_dims.size();
    assert(rank == 4);
    // set out_dims
    out_dims.resize(4);
    for (size_t i = 0; i < 4; i++) {
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
        size_t index = i;
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

// NOTE: Make sure this works for
// NOTE: for safety, just made all the shapes 4d, then outside u can modify the shape after

// seq_len:     11 - 1
// tot_seq_len: 11 - 12

// since using a large buffer for as a temp buff, could optimize by removing
template <typename T>
void DynamicTruncationAndConcatentation(
    size_t seq_len,
    T* temp_buff, // buff_8
    T* query_states, // (1, 32, 11, 80) - (1, 32, 1, 80)
    T* key_states, // same as q
    T* value_states, // same as q
    // T* q_out, // (1, 32, 11, 80) - (1, 32, 1, 80)
    // T* k_out, // (1, 32, 12, 80) - (1, 32, 12, 80)
    // T* v_out, // (1, 32, 12, 80) - (1, 32, 12, 80)
    const T* sin_cached, // (11, 32) - (12, 32)
    const T* cos_cached,
    T* sin_buff,
    T* cos_buff,
    T* key_cache, // (1, 32, 0, 80)<basically 0> - (1, 32, 11, 80)
    T* value_cache, // same as key_cache
    T* query_rot_buff,
    T* query_pass_buff,
    T* key_rot_buff,
    T* key_pass_buff,
    std::vector<size_t>& query_shape,
    std::vector<size_t>& key_shape,
    std::vector<size_t>& value_shape,
    // std::vector<size_t>& q_out_shape,
    // std::vector<size_t>& k_out_shape,
    // std::vector<size_t>& v_out_shape,
    // std::vector<size_t>& sin_shape,
    // std::vector<size_t>& cos_shape,
    std::vector<size_t>& key_cache_shape,
    std::vector<size_t>& value_cache_shape,
    // std::vector<size_t>& query_rot_buff_dims,
    // std::vector<size_t>& query_pass_buff_dims,
    // std::vector<size_t>& key_rot_buff_dims,
    // std::vector<size_t>& key_pass_buff_dims,

    const int rotary_emb_dim,
    const std::vector<int>& position_ids
) {
    auto sin_shape           = std::vector<size_t>();
    auto cos_shape           = std::vector<size_t>();
    auto query_rot_buff_dims = std::vector<size_t>();
    auto query_pass_buff_dims = std::vector<size_t>();
    auto key_rot_buff_dims   = std::vector<size_t>();
    auto key_pass_buff_dims  = std::vector<size_t>();

    bool firstRunForDecoder = false;
    if (key_cache_shape.size() == 0) { firstRunForDecoder = true; }

    // printV("query_states_shape", query_shape);
    // printV("key_states_shape", key_shape);
    // printV("value_states_shape", value_shape);
    printV("key_cache_shape", key_cache_shape);
    printV("value_cache_shape", value_cache_shape);

    // reshape
    query_shape = {1, seq_len, 32, 80};
    key_shape   = {1, seq_len, 32, 80};
    value_shape = {1, seq_len, 32, 80};

    // printTensor("query after reshape", query_states, query_shape);
    // printTensor("key after reshape", key_states, key_shape);
    // printTensor("value after reshape", value_states, value_shape);
    
    // transpose
    std::vector<size_t> temp_shape;
    // query
    transpose(query_states, temp_buff, {0, 2, 1, 3}, query_shape, temp_shape);
    query_shape = temp_shape;
    copyTensor(temp_buff, query_states, query_shape);
    // key
    transpose(key_states, temp_buff, {0, 2, 1, 3}, key_shape, temp_shape);
    key_shape = temp_shape;
    copyTensor(temp_buff, key_states, key_shape);
    // value
    transpose(value_states, temp_buff, {0, 2, 1, 3}, value_shape, temp_shape);
    value_shape = temp_shape;
    copyTensor(temp_buff, value_states, value_shape);

    // implement kv stuff
    size_t kv_seq_len = key_shape.end()[-2];
    if (!firstRunForDecoder) {
        kv_seq_len += key_cache_shape.end()[-3]; // modified
        // kv_seq_len = tot_seq_len
    }

    // printTensor("cos before rotary_emb", cos_cached, {11, 32});
    // printTensor("sin before rotary_emb", sin_cached, {11, 32});

    std::vector<size_t> sin_cached_shape = {MAX_SEQ_LEN, 32};
    std::vector<size_t> cos_cached_shape = {MAX_SEQ_LEN, 32};

    // rotary emb
    rotary_emb(
        value_states, value_shape, kv_seq_len,
        sin_cached, sin_cached_shape, cos_cached, cos_cached_shape,
        sin_buff, sin_shape, cos_buff, cos_shape
    );

    // printTensor("cos after rotary_emb", cos_buff, cos_shape);
    // printTensor("sin after rotary_emb", sin_buff, sin_shape);

    // partial rotary embedding truncation
    truncate(
        query_states, query_shape,
        query_rot_buff, query_rot_buff_dims, 
        std::vector<int> {int(query_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
    );
    truncate(
        query_states, query_shape,
        query_pass_buff, query_pass_buff_dims, 
        std::vector<int> {int(query_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0}
    );
    truncate(
        key_states, key_shape,
        key_rot_buff, key_rot_buff_dims, 
        std::vector<int> {int(key_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
    );
    truncate(
        key_states, key_shape,
        key_pass_buff, key_pass_buff_dims, 
        std::vector<int> {int(key_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0}
    );


    // printTensor("query_pass fter truncations", query_pass_buff, query_pass_buff_dims);
    // printTensor("key_pass fter truncations", key_pass_buff, key_pass_buff_dims);
    

    // printTensor("query_rot_buff before apply_rotary_pos_emb", query_rot_buff, query_rot_buff_dims);
    // printTensor("key_rot_buff before apply_rotary_pos_emb", key_rot_buff, key_rot_buff_dims);


    // apply_rot_pos_emb
    apply_rotary_pos_emb(
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims,
        cos_buff, cos_shape, sin_buff, sin_shape, 
        position_ids, 1,
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims
    );

    // printTensor("query_rot_buff after apply_rotary_pos_emb", query_rot_buff, query_rot_buff_dims);
    // printTensor("key_rot_buff after apply_rotary_pos_emb", key_rot_buff, key_rot_buff_dims);

    // concat
    concat(
        query_rot_buff, query_rot_buff_dims,
        query_pass_buff, query_pass_buff_dims, 
        query_rot_buff_dims.size()-1, 
        query_states, query_shape
    );
    concat(
        key_rot_buff, key_rot_buff_dims,
        key_pass_buff, key_pass_buff_dims, 
        key_rot_buff_dims.size()-1, 
        key_states, key_shape
    );

    // kv caching
    if (firstRunForDecoder) {
        // transpose(
        //     key_states, key_cache, {0, 2, 1, 3}, 
        //     key_shape, key_cache_shape
        // );
        transpose(
            value_states, value_cache, {0, 2, 1, 3},
            value_shape, value_cache_shape
        );
        transpose(
            key_states, key_cache, {0, 2, 1, 3}, 
            key_shape, key_cache_shape
        );
        // to apply transpose on key_states
        transpose(
            key_cache, key_states, {0, 2, 3, 1}, 
            key_cache_shape, key_shape
        );
    }
    else {
        size_t offset = 1;
        for (const size_t& i : key_cache_shape) {
            offset *= i;
        }
        // key_states: (1, 32, 1, 80 ) --> (1, 1, 32, 80)
        // write to cache
        transpose(
            key_states, key_cache + offset, {0, 2, 1, 3}, 
            key_shape, key_cache_shape
        );
        transpose(
            value_states, value_cache + offset, {0, 2, 1, 3}, 
            value_shape, value_cache_shape
        );
        // cache: (1, 11, 32, 80) --> (1, 12, 32, 80)
        key_cache_shape.end()[-3] = kv_seq_len;
        value_cache_shape.end()[-3] = kv_seq_len;
        // write back to key buffer
        // (1, 12, 32, 80) --> (1, 32, 12, 80)
        // transpose(
        //     key_cache, key_states, {0, 2, 1, 3}, 
        //     key_cache_shape, key_shape
        // );
        transpose(
            value_cache, value_states, {0, 2, 1, 3}, 
            value_cache_shape, value_shape
        );

        // changing key_states transpose to optimize for next computation
        // {1, 12, 32, 80} --> {1, 32, 80, 12}
        transpose(
            key_cache, key_states, {0, 2, 3, 1}, 
            key_cache_shape, key_shape
        );
    }

    // extra steps b/c using P2_reshaped instead of P2_buffered

    // transpose(
    //     dwdw,
    // );

    std::cout << "\nfinal shapes for DynamicTruncationAndConcatentation():\n";
    printV("key_cache_shape", key_cache_shape);
    printV("value_cache_shape", value_cache_shape);

    printTensor("query_states", query_states, query_shape);
    printTensor("key_states transposed", key_states, key_shape);
    printTensor("value_states", value_states, value_shape);
}

// 
template <typename T>
void tensorPad(const T* in, const std::vector<size_t>& in_shape, const std::vector<std::vector<size_t>>& padding, const T& default_val, T* out, std::vector<size_t>& out_shape) {
    // Calculate the output shape based on input shape and padding
    out_shape.clear();
    for (size_t i = 0; i < in_shape.size(); ++i) {
        out_shape.push_back(in_shape[i] + padding[i][0] + padding[i][1]);
    }
    
    // Compute total number of elements in input and output tensors
    size_t in_size = 1;
    size_t out_size = 1;
    for (size_t dim : in_shape) in_size *= dim;
    for (size_t dim : out_shape) out_size *= dim;

    // Initialize the output tensor with the default value
    std::fill(out, out + out_size, default_val);

    // Compute strides for input and output tensors
    std::vector<size_t> in_strides(in_shape.size(), 1);
    std::vector<size_t> out_strides(out_shape.size(), 1);
    for (int i = in_shape.size() - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }
    for (int i = out_shape.size() - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Copy data from input tensor to output tensor with padding
    for (size_t idx = 0; idx < in_size; ++idx) {
        size_t in_idx = idx;
        size_t out_idx = 0;
        for (size_t i = 0; i < in_shape.size(); ++i) {
            size_t coord = in_idx / in_strides[i];
            in_idx %= in_strides[i];
            out_idx += (coord + padding[i][0]) * out_strides[i];
        }
        out[out_idx] = in[idx];
    }
}


// could change so that buffer_shape.size() != reshape_shape.size()
template <typename T>
void reshaped_to_buffered(
    const std::vector<size_t>& reshape_shape,
    const std::vector<size_t>& buffer_shape,
    const T& default_val,
    const T* in,
    T* temp_buff,
    T* out
) {
    // ensure shapes are compatible
    assert(reshape_shape.size() == buffer_shape.size());

    // set padding
    std::vector<std::vector<size_t>> padding(buffer_shape.size());
    for (size_t dim = 0; dim < buffer_shape.size(); dim++) {
        assert(buffer_shape[dim] >= reshape_shape[dim]);
        size_t right = buffer_shape[dim] - reshape_shape[dim];
        padding[dim] = {0, right};
    }

    // tensor pad to temp_buff
    T* directed_output = (in == out) ? temp_buff : out;
    std::vector<size_t> unnecessary_out_shape;
    tensorPad(in, reshape_shape, padding, default_val, directed_output, unnecessary_out_shape);

    // ensure shape is unnecessary_out_shape == buffer_shape
    for (size_t i = 0; i < unnecessary_out_shape.size(); i++) {
        assert(unnecessary_out_shape[i] == buffer_shape[i]);
    }

    // copy
    if (in == out) {
        copyTensor(directed_output, out, buffer_shape);
    }
}

// in-memory
template <typename T>
void buffered_to_reshaped(
    const std::vector<size_t>& buffer_shape,
    const std::vector<size_t>& reshape_shape,
    T* in
) {
    assert(buffer_shape.size() == reshape_shape.size());

    size_t buffer_size = 1;
    for (size_t dim : buffer_shape) {
        buffer_size *= dim;
    }

    size_t reshape_size = 1;
    for (size_t dim : reshape_shape) {
        reshape_size *= dim;
    }

    std::vector<size_t> buffer_strides(buffer_shape.size(), 1);
    for (int i = buffer_shape.size() - 2; i >= 0; --i) {
        buffer_strides[i] = buffer_strides[i + 1] * buffer_shape[i + 1];
    }

    std::vector<size_t> reshape_strides(reshape_shape.size(), 1);
    for (int i = reshape_shape.size() - 2; i >= 0; --i) {
        reshape_strides[i] = reshape_strides[i + 1] * reshape_shape[i + 1];
    }

    for (size_t i = 0; i < reshape_size; ++i) {
        size_t buffer_index = 0;
        size_t remaining_index = i;

        for (size_t j = 0; j < reshape_shape.size(); ++j) {
            size_t index_in_dim = remaining_index / reshape_strides[j];
            remaining_index %= reshape_strides[j];
            buffer_index += index_in_dim * buffer_strides[j];
        }

        in[i] = in[buffer_index];
    }
}

// can be done in memory
void NewGELU(
    const float* in,
    float* out,
    const std::vector<size_t>& shape
) {
    // 0.5 * input * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * tf.pow(input, 3.0))))
    // calculate total number of elements
    size_t total_elements = 1;
    for (const size_t& dim : shape) { total_elements *= dim; }
    for (size_t i = 0; i < total_elements; i++) {
        out[i] = 0.5f * in[i] * (1.0f + (float)tanh(sqrt(2.0 / M_PI) * (in[i] + 0.044715 * pow(in[i], 3.0))));
    }
}

// [upcast, divide by sqrt, apply mask], softmax, downcast
void upcastDivideSqrtApplyMask(
    QUANT_TYPE* p2,
    std::vector<size_t>& p2_shape
) {

}

// vertical expansion
void flatten_along_row(std::vector<size_t>& dims) {
  assert(dims.size() >= 2);
  long long int num_rows = dims.end()[-2]; // could cause issues for gpu
  long long int num_cols = dims.end()[-1];
  for (int i = 0; i < dims.size()-2; i++) { num_rows *= dims[i];}
  dims = std::vector<size_t>(2);
  dims[0] = num_rows;
  dims[1] = num_cols;
//   for (auto i : dims) {std::cout << i << " ";}
}

// horizontal expansion
void flatten_along_col(std::vector<size_t>& dims) {
  assert(dims.size() >= 2);
  long long int num_rows = dims.end()[-2];
  long long int num_cols = dims.end()[-1]; // could cause issues for gpu
  for (int i = 0; i < dims.size()-2; i++) { num_cols *= dims[i];}
  dims = std::vector<size_t>(2);
  dims[0] = num_rows;
  dims[1] = num_cols;
//   for (auto i : dims) {std::cout << i << " ";}
}

void flatten_to_3d(std::vector<size_t>& dims) {
    std::vector<size_t> temp_dims = {1,1,1};
    for (int i = 0; i < dims.size()-2; i++) { 
        temp_dims[0] *= dims[i];
    }
    temp_dims.end()[-1] = dims.end()[-1];
    temp_dims.end()[-2] = dims.end()[-2];
    dims = temp_dims;
}

// does not set output dims, use constrained version for that
void matmul_Nd_32f(const float* ten1, const float* ten2, float* out, 
                   std::vector<size_t> dims1, std::vector<size_t> dims2) {
    assert(dims1.end()[-1] == dims2.end()[-2]); // rule of matrix multiplication
    // use ints, as size_t's seem to segfault if they go negative
    printV("dims1", dims1);
    printV("dims2", dims2);
    // flatten
    flatten_to_3d(dims1);
    flatten_to_3d(dims2);
    std::cout << "dims after flattening:\n";
    printV("dims1", dims1);
    printV("dims2", dims2);
    size_t rows1 = dims1.end()[-2];
    size_t cols1 = dims1.end()[-1];
    size_t rows2 = dims2.end()[-2];
    size_t cols2 = dims2.end()[-1];
    assert(cols1 == rows2); // rule of matrix multiplication
    assert(dims1[0] == dims1[0]);
    assert(dims1.size() == 3);
    int offset = dims1.end()[-2] * dims2.end()[-1];
    for (size_t z = 0; z < dims1[0]; z++) {
        // 2d matmul algorithm
        for (size_t i = 0; i < rows1; ++i) {
            for (size_t j = 0; j < cols2; ++j) {
                float sum = 0.0;
                for (size_t k = 0; k < rows2; ++k) {
                    if (std::isinf(ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset])) {
                        std::cout << "ten1: " << ten1[i*cols1 + k + z*offset] << ", ";
                        std::cout << "ten2: " << ten2[k*cols2 + j + z*offset] << ", ";
                        std::cout << "ten1 + ten2: " << ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset] << "\n";
                    }
                    
                    if (std::isinf(sum + ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset])) {
                        std::cout << "sum before: " << sum << ", ";
                        std::cout << "ten1: " << ten1[i*cols1 + k + z*offset] << ", ";
                        std::cout << "ten2: " << ten2[k*cols2 + j + z*offset] << ", ";
                        std::cout << "ten1 + ten2: " << ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset] << "\n";
                        std::cout << "sum after: " << sum + ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset] << "\n";
                    }
                    sum += ten1[i*cols1 + k + z*offset] * ten2[k*cols2 + j + z*offset];
                }
                out[i*cols2 + j + z*offset] = sum;
            }
        }
    }
}

// makes certain assumptions to set output dimensions
void matmul_Nd_32f_constrained(const float* ten1, const float* ten2, float* out, 
                   std::vector<size_t> dims1, std::vector<size_t> dims2,
                   std::vector<size_t>& out_dims) {

    assert(dims1.end()[-1] == dims2.end()[-2]); // rule of matrix multiplication
    // set out_dims (assuming additional contraints below)
    assert(dims1.size() == dims2.size());
    int rank = dims1.size();
    out_dims = std::vector<size_t>();
    for (int i = 0; i < rank - 1; i++) { out_dims.push_back(dims1[i]); }
    out_dims.push_back(dims2.end()[-1]);
    // calling matmul
    matmul_Nd_32f(ten1, ten2, out, dims1, dims2);
}

#endif