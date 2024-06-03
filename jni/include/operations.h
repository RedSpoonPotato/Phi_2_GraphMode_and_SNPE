#ifndef OPERATIONS_H_
#define OPERATIONS_H_

#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <cmath>
#include "main_macros.h"


template <typename T>
void copyTensor(const T* ten1, T* out, const std::vector<size_t>& dims) {
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= dims[i]; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i];
    }
}

void layernorm_1d_32f(
    const float* vec, const float* weight, const float* bias, float* out,
    const size_t vec_len, const size_t weight_len, const float eps
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
    const std::vector<size_t>& tensor_dims, const int weight_len,
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

// not input-to-output safe
void truncate_u8(
    const uint8_t* input, const std::vector<size_t>& input_dims,
    uint8_t* output, std::vector<size_t>& output_dims,
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
    output_dims = std::vector<size_t>();
    for (int i = 0; i < input_dims.size(); i++) {
        output_dims.push_back(indice_end[i]+1 - indice_start[i]);
    }
}


void mul_u8(const uint8_t* ten1, const uint8_t* ten2, uint8_t* out, const std::vector<size_t>& dims) {
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] * ten2[i];
    }
}

void add_u8(const uint8_t* ten1, const uint8_t* ten2, uint8_t* out, const std::vector<size_t>& dims) {
    size_t num_elem = 1;
    for (size_t i = 0; i < dims.size(); i++) { num_elem *= i; }
    for (size_t i = 0; i < num_elem; i++) {
        out[i] = ten1[i] + ten2[i];
    }
}

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
void rotary_emb_u8(
    const uint8_t* x, const std::vector<size_t>& x_dims,
    const int seq_len,
    const uint8_t* sin_cached, const std::vector<size_t>& sin_cached_dims,
    const uint8_t* cos_cached, const std::vector<size_t>& cos_cached_dims,
    uint8_t* sin, std::vector<size_t>& sin_dims,
    uint8_t* cos, std::vector<size_t>& cos_dims
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
    truncate_u8(
        cos_cached, cos_cached_dims, cos, cos_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
    truncate_u8(
        sin_cached, sin_cached_dims, sin, sin_dims, 
        dims_to_split, // outer dim
        values,
        colon_lefts
    );
}

void gather_u8(
    const uint8_t* x, const std::vector<size_t>& x_dims,
    const std::vector<int>& indices,
    uint8_t* out, std::vector<size_t>& out_dims
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
    out_dims = std::vector<size_t>();
    out_dims.push_back(indices.size());
    for (int i = 1; i < rank; i++) {
        out_dims.push_back(x_dims[i]);
    }
}


void concat_u8(
    const uint8_t* x1, const std::vector<size_t>& x1_dims,
    const uint8_t* x2, const std::vector<size_t>& x2_dims,
    const int axis,
    uint8_t* out, std::vector<size_t>& out_dims
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

void rotate_half_u8(
    const uint8_t* x, const std::vector<size_t>& x_dims,
    uint8_t* out, std::vector<size_t>& out_dims
) {
    // internal buffers
    // uint8_t x1[QUERY_STATES_BUFF_SIZE];
    uint8_t* x1 = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
    // uint8_t x2[QUERY_STATES_BUFF_SIZE];
    uint8_t* x2 = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
    std::vector<size_t> x1_dims, x2_dims;

    int x1_len = x_dims.end()[-1] / 2;
    std::vector<int> dims_to_split = {(int)x_dims.size()-1};
    std::vector<int> values = {x1_len};
    std::vector<int> colon_left = {1};
    std::vector<int> colon_right = {0};
    truncate_u8(
        x, x_dims, x1, x1_dims, dims_to_split,
        values, colon_left
    );
    truncate_u8(
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
    concat_u8(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
    std::cout << "\t\tfreeing memory\n";
    free(x1);
    free(x2);
}

// SUGGESTION: unquantize the multiplication and addition inside
void apply_rotary_pos_emb_u8(
    const uint8_t* q, const std::vector<size_t>& q_dims,
    const uint8_t* k, const std::vector<size_t>& k_dims,
    const uint8_t* cos, const std::vector<size_t>& cos_dims,
    const uint8_t* sin, const std::vector<size_t>& sin_dims,
    const std::vector<int>& position_ids, const int unsqueeze_dim,
    uint8_t* q_embed, std::vector<size_t>& q_embed_dims,
    uint8_t* k_embed, std::vector<size_t>& k_embed_dims
) {
    // buffers prbably dont need to be this big
    // uint8_t cos_buff[SIN_COS_BUFF_SIZE];
    uint8_t* cos_buff = (uint8_t*)malloc(SIN_COS_BUFF_SIZE * sizeof(uint8_t));
    // uint8_t sin_buff[SIN_COS_BUFF_SIZE];
    uint8_t* sin_buff = (uint8_t*)malloc(SIN_COS_BUFF_SIZE * sizeof(uint8_t));
    std::vector<size_t> cos_buff_dims, sin_buff_dims;
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    std::cout << "cos[0]: " << cos[0] << "\n";
    printV("cos_dims", cos_dims);
    for (auto i : position_ids) {std::cout << i << ", ";}
    std::cout << "\n";
    gather_u8(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    gather_u8(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
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
    // uint8_t q_temp_buff[QUERY_STATES_BUFF_SIZE];
    uint8_t* q_temp_buff = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
    // uint8_t k_temp_buff[QUERY_STATES_BUFF_SIZE];
    uint8_t* k_temp_buff = (uint8_t*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(uint8_t));
    std::cout << "\tCalling Multiply Loop\n";
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_u8(
                &(q[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff, 
                q_temp_buff, cos_buff_dims);
            mul_u8(
                &(k[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), cos_buff,
                k_temp_buff, cos_buff_dims);
        }
    }
    std::cout << "\tCalling Rotate_Half\n";
    rotate_half_u8(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
    std::cout << "\tCalling Rotate_Half\n";
    printV("k_dims", k_dims);
    rotate_half_u8(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims
    std::cout << "\tCalling Mutliply Loop\n";
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_u8(
                &(q_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
                q_embed, sin_buff_dims);
            mul_u8(
                &(k_embed[i*q_dim_offsets[0] + j*q_dim_offsets[1]]), sin_buff, 
                k_embed, sin_buff_dims);
        }
    }
    std::cout << "\tCalling add_32f\n";
    add_u8(q_embed, q_temp_buff, q_embed, q_embed_dims);
    std::cout << "\tCalling add_32f\n";
    add_u8(k_embed, k_temp_buff, k_embed, k_embed_dims);
    //free
    free(cos_buff);
    free(sin_buff);
}


void transpose_u8(
    const uint8_t* tensor, uint8_t* out, const std::vector<size_t>& perm,
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

void DynamicTruncationAndConcatentation(
    uint8_t* query_states, // (1, 32, 11, 80) - (1, 32, 1, 80)
    uint8_t* key_states, // same as q
    uint8_t* value_states, // same as q
    // uint8_t* q_out, // (1, 32, 11, 80) - (1, 32, 1, 80)
    // uint8_t* k_out, // (1, 32, 12, 80) - (1, 32, 12, 80)
    // uint8_t* v_out, // (1, 32, 12, 80) - (1, 32, 12, 80)
    const uint8_t* sin_cached, // (11, 32) - (12, 32)
    const uint8_t* cos_cached,
    uint8_t* sin_buff,
    uint8_t* cos_buff,
    uint8_t* key_cache, // (1, 32, 0, 80)<basically 0> - (1, 32, 11, 80)
    uint8_t* value_cache, // same as key_cache
    uint8_t* query_rot_buff,
    uint8_t* query_pass_buff,
    uint8_t* key_rot_buff,
    uint8_t* key_pass_buff,
    std::vector<size_t>& query_shape,
    std::vector<size_t>& key_shape,
    std::vector<size_t>& value_shape,
    // std::vector<size_t>& q_out_shape,
    // std::vector<size_t>& k_out_shape,
    // std::vector<size_t>& v_out_shape,
    const std::vector<size_t>& sin_cached_shape,
    const std::vector<size_t>& cos_cached_shape,
    std::vector<size_t>& sin_shape,
    std::vector<size_t>& cos_shape,
    std::vector<size_t>& key_cache_shape,
    std::vector<size_t>& value_cache_shape,
    std::vector<size_t>& query_rot_buff_dims,
    std::vector<size_t>& query_pass_buff_dims,
    std::vector<size_t>& key_rot_buff_dims,
    std::vector<size_t>& key_pass_buff_dims,

    const int rotary_emb_dim,
    const std::vector<int>& position_ids
) {
    bool firstRunForDecoder = false;
    if (key_cache_shape.size() == 0) { firstRunForDecoder = true; }

    printV("query_states_shape", query_shape);
    printV("key_states_shape", key_shape);
    printV("value_states_shape", value_shape);
    printV("key_cache_shape", key_cache_shape);
    printV("value_cache_shape", value_cache_shape);

    // implement kv stuff
    size_t kv_seq_len = key_shape.end()[-2];
    if (!firstRunForDecoder) {
        kv_seq_len += key_cache_shape.end()[-3]; // modified
        // kv_seq_len = tot_seq_len
    }
        
    // rotary emb
    rotary_emb_u8(
        value_states, value_shape, kv_seq_len,
        sin_cached, sin_cached_shape, cos_cached, cos_cached_shape,
        sin_buff, sin_shape, cos_buff, cos_shape
    );

    // partial rotary embedding truncation
    truncate_u8(
        query_states, query_shape,
        query_rot_buff, query_rot_buff_dims, 
        std::vector<int> {int(query_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
    );
    truncate_u8(
        query_states, query_shape,
        query_pass_buff, query_pass_buff_dims, 
        std::vector<int> {int(query_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0}
    );
    truncate_u8(
        key_states, key_shape,
        key_rot_buff, key_rot_buff_dims, 
        std::vector<int> {int(key_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
    );
    truncate_u8(
        key_states, key_shape,
        key_pass_buff, key_pass_buff_dims, 
        std::vector<int> {int(key_shape.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0}
    );

    // apply_rot_pos_emb
    apply_rotary_pos_emb_u8(
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims,
        cos_buff, cos_shape, sin_buff, sin_shape, 
        position_ids, 1,
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims
    );

    // concat
    concat_u8(
        query_rot_buff, query_rot_buff_dims,
        query_pass_buff, query_pass_buff_dims, 
        query_rot_buff_dims.size()-1, 
        query_states, query_shape
    );
    concat_u8(
        key_rot_buff, key_rot_buff_dims,
        key_pass_buff, key_pass_buff_dims, 
        key_rot_buff_dims.size()-1, 
        key_states, key_shape
    );

    // kv caching
    if (firstRunForDecoder) {
        transpose_u8(
            key_states, key_cache, {0, 2, 1, 3}, 
            key_shape, key_cache_shape
        );
        transpose_u8(
            value_states, value_cache, {0, 2, 1, 3},
            value_shape, value_cache_shape
        );
    }
    else {
        size_t offset = 1;
        for (const size_t& i : key_cache_shape) {
            offset *= i;
        }
        // key_states: (1, 32, 1, 80 ) --> (1, 1, 32, 80)
        // write to cache
        transpose_u8(
            key_states, key_cache + offset, {0, 2, 1, 3}, 
            key_shape, key_cache_shape
        );
        transpose_u8(
            value_states, value_cache + offset, {0, 2, 1, 3}, 
            value_shape, value_cache_shape
        );
        // cache: (1, 11, 32, 80) --> (1, 12, 32, 80)
        key_cache_shape.end()[-3] = kv_seq_len;
        value_cache_shape.end()[-3] = kv_seq_len;
        // write back to key buffer
        // (1, 12, 32, 80) --> (1, 32, 12, 80)
        transpose_u8(
            key_cache, key_states, {0, 2, 1, 3}, 
            key_cache_shape, key_shape
        );
        transpose_u8(
            value_cache, value_states, {0, 2, 1, 3}, 
            value_cache_shape, value_shape
        );
    }
}

#endif