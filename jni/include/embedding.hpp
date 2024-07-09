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
        // std::cout << "first element of row in fp32:" <<  half_to_float(((ushort*)temp_arry.data())[0]) << "\n";
        // std::cout << "first element of row in fp32:" <<  ((float*)temp_arry.data())[0] << "\n";
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