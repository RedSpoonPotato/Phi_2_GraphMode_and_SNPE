#include <vector>
#include <iostream>
#include <assert.h>
#include <cmath>

#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define FEATURE_SIZE 2560
#define QUERY_STATES_BUFF_SIZE BATCH_SIZE*MAX_SEQ_LEN*FEATURE_SIZE

/*
This is a repo of commmon operatiosns that need to be implemented for the phi-2 LLM. THey should be dynamic, but also efficient
Assume:
    -assume 32 bit floats for computation
    -Can use (uint32_t)
    -not sure if can use values (like char, etc.)

Remember that each tensor is effectively a array of void* (can cast as any value)

Data Layout (i think):
    -ex: (2,3,4) -> (batch, row_num, col_num)
        - [[1,2,3,4], [5,6,7,8], [9,10,11,12],
           [13,14,15,16], [17,18,19,20], [21,22,23,24]]
*/

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


// vertical expansion
void flatten_along_row(std::vector<uint32_t>& dims) {
  assert(dims.size() >= 2);
  long long int num_rows = dims.end()[-2]; // could cause issues for gpu
  long long int num_cols = dims.end()[-1];
  for (int i = 0; i < dims.size()-2; i++) { num_rows *= dims[i];}
  dims = std::vector<u_int32_t>(2);
  dims[0] = num_rows;
  dims[1] = num_cols;
  std::cout << "new dims: ";
  for (auto i : dims) {std::cout << i << " ";}
  std::cout << "\n";
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
  std::cout << "new dims: ";
  for (auto i : dims) {std::cout << i << " ";}
  std::cout << "\n";
}

void matmul_Nd_32f(const float* ten1, const float* ten2, float* out, 
                   std::vector<uint32_t> dims1, std::vector<uint32_t> dims2) {

    assert(dims1.end()[-1] == dims2.end()[-2]); // rule of matrix multiplication
    // use ints, as uint32_t's seem to segfault if they go negative
    // flatten
    flatten_along_row(dims1);
    flatten_along_col(dims2);
    u_int32_t rows1 = dims1.end()[-2];
    u_int32_t cols1 = dims1.end()[-1];
    u_int32_t rows2 = dims2.end()[-2];
    u_int32_t cols2 = dims2.end()[-1];
    assert(cols1 == rows2); // rule of matrix multiplication
    // 2d matmul algorithm
    for (uint32_t i = 0; i < rows1; ++i) {
        for (uint32_t j = 0; j < cols2; ++j) {
            float sum = 0.0;
            for (uint32_t k = 0; k < rows2; ++k) {
                sum += ten1[i*cols1 + k] * ten2[k*cols2 + j];
            }
            out[i*cols2 + j] = sum;
        }
    }
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
    const std::vector<uint32_t> tensor_dims, const int weight_len,
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

void PhiAttention(
    /* inputs */
    const float* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    const float* attention_mask, const std::vector<uint32_t>& attention_mask_dims,
    const float* position_ids, const std::vector<uint32_t>& position_ids_dims,
    const float* past_key_value_old, const std::vector<uint32_t>& past_key_value_old_dims,
    /* weights */
    const float* q_proj_weights, const std::vector<uint32_t>& q_proj_weights_dims,
    const float* q_proj_bias,
    const float* k_proj_weights, const std::vector<uint32_t>& k_proj_weights_dims,
    const float* k_proj_bias,
    const float* v_proj_weights, const std::vector<uint32_t>& v_proj_weights_dims,
    const float* v_proj_bias,
    const float* q_layernorm_weights, const int q_layernorm_weights_len,
    const float* q_layernorm_bias,
    const float* k_layernorm_weights, const int k_layernorm_weights_len,
    const float* k_layernorm_bias,
    const float eps,
    const int num_heads, const int head_dim, const int num_kv_heads,

    /* outputs */
    float* attn_output, std::vector<uint32_t>& attn_output_dims,
    float* past_key_value, std::vector<uint32_t>& past_key_value_dims
) {
    int bsz = hidden_states_dims[0];
    int q_len = hidden_states_dims[1];
    float query_states_buff[QUERY_STATES_BUFF_SIZE];
    float key_states_buff[QUERY_STATES_BUFF_SIZE];
    float value_states_buff[QUERY_STATES_BUFF_SIZE];

    std::vector<uint32_t> query_states_dims;
    std::vector<uint32_t> key_states_dims;
    std::vector<uint32_t> value_states_dims;

    linear_Nd_32f(
        hidden_states, q_proj_weights, q_proj_bias, query_states_buff,
        hidden_states_dims, q_proj_weights_dims, query_states_dims);
    linear_Nd_32f(
        hidden_states, k_proj_weights, k_proj_bias, key_states_buff,
        hidden_states_dims, k_proj_weights_dims, key_states_dims);
    linear_Nd_32f(
        hidden_states, v_proj_weights, v_proj_bias, value_states_buff,
        hidden_states_dims, v_proj_weights_dims, value_states_dims);
    
    // careful for using the same buffer as input & output
    layernorm_Nd_32f(
        query_states_buff, q_layernorm_weights, q_layernorm_bias, query_states_buff,
        query_states_dims, q_layernorm_weights_len, eps);
    layernorm_Nd_32f(
        key_states_buff, k_layernorm_weights, k_layernorm_bias, key_states_buff,
        key_states_dims, k_layernorm_weights_len, eps);
    
    // reshape
    query_states_dims.resize(4);
    query_states_dims[0] = bsz;
    query_states_dims[1] = q_len;
    query_states_dims[2] = num_heads;
    query_states_dims[3] = head_dim;
    key_states_dims.resize(4);
    key_states_dims[0] = bsz;
    key_states_dims[1] = q_len;
    key_states_dims[2] = num_kv_heads;
    key_states_dims[3] = head_dim;
    value_states_dims.resize(4);
    value_states_dims[0] = bsz;
    value_states_dims[1] = q_len;
    value_states_dims[2] = num_kv_heads;
    value_states_dims[3] = head_dim;

    // tranpose
    

    

}

int main() {

}
