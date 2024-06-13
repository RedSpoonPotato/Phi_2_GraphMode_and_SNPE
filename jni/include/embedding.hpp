#ifndef EMBEDDING_H_
#define EMBEDDING_H_

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>

#include "main_macros.h"
#include "snpe_exec_utils.h"

// template <typename T>
// void 

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
void read_embedding(
    std::string filename, const std::vector<uint32_t>& input_ids, 
    uint32_t rowSize, std::vector<std::vector<T>>& output)
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == NULL) {std::exit(5);}

    output = std::vector<std::vector<T>>(input_ids.size());
    std::vector<T> temp_arry(rowSize);

    for (int i = 0; i < input_ids.size(); i++) {
        if (input_ids[i] >= VOCAB_SIZE) {
            std::cerr << "Error: attempting to read from column " << input_ids[i] << 
                        " when the max size is " << VOCAB_SIZE << "\n";
        }
        fseek(fp, sizeof(T)*rowSize*input_ids[i], SEEK_SET);
        const size_t ret_code = fread(temp_arry.data(), sizeof(T), rowSize, fp); // read a row
        std::cout << "first element of row in fp32:" <<  half_to_float(((ushort*)temp_arry.data())[0]) << "\n";
        std::cout << "first element of row in fp32:" <<  ((float*)temp_arry.data())[0] << "\n";
        if (ret_code != rowSize) {
            std::cerr << "Error: number of elements read " << ret_code << " instead of " << rowSize <<
                        " for column "  << input_ids[i] << "\n";
        }
        output[i] = temp_arry; // probably not the most efficient
    }
    fclose(fp);
}

template <typename T>
void writeVecofVec(const std::vector<std::vector<T>>& table, T* out) {
    // assume all elements have same size
    size_t row_size = table[0].size();
    for (auto &row : table) { assert(row.size() == row_size); }
    // write data
    for (int i = 0; i < table.size(); i++) {
        for (int j = 0; j < row_size; j++) {
            out[j + row_size*i] = table[i][j];
        }
    }
}

template <typename T>
void writeEmbedding(
    std::string filename, const std::vector<uint32_t>& input_ids, 
    uint32_t rowSize,
    T* output)
{
    /* grab embedding slice from table in storage */
    std::vector<std::vector<T>> embedding_slice;
    read_embedding(filename, input_ids, rowSize, embedding_slice);
    assert(embedding_slice[0].size() == rowSize && rowSize == HIDDEN_SIZE);
    /* write data to output buffer */
    std::cout << "calling VecOfVec\n";
    writeVecofVec(embedding_slice, output);
}

template <typename T>
void writeEmbedding_old(
    std::string filename, const std::vector<uint32_t>& input_ids, 
    uint32_t rowSize,
    T* output)
{
    /* grab embedding slice from table in storage */
    std::vector<std::vector<T>> embedding_slice;
    read_embedding(filename, input_ids, rowSize, embedding_slice);
    assert(embedding_slice[0].size() == rowSize && rowSize == HIDDEN_SIZE);
    /* write data to output buffer */
    std::cout << "calling VecOfVec\n";
    writeVecofVec(embedding_slice, output);
    /* write output shape to output buffer */
    std::cout << "writing ptr\n";
    uint32_t* ptr = (uint32_t*)&output[MAX_SEQ_LEN * HIDDEN_SIZE];
    ptr[0] = 1; // should not be read by Decoder.cpp
    ptr[1] = 1;
    ptr[2] = input_ids.size();
    ptr[3] = HIDDEN_SIZE;
}

#endif