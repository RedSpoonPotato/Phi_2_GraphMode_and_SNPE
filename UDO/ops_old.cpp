#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>

#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560
#define QUERY_STATES_BUFF_SIZE  BATCH_SIZE*MAX_SEQ_LEN*HIDDEN_SIZE

// partial_rotary_factor: 0.4
// head_dim = hidden_size // num_attention_heads i.e. (2560 / 32) = 80
// sin_cos(param:dim) = head_dim * partial_rotary_factor = 80 * .4 = 32
#define SIN_COS_DIM 32
#define SIN_COS_MAX_SEQ_LEN 2048 // a temporary solution
#define SIN_COS_BUFF_SIZE  SIN_COS_DIM*SIN_COS_MAX_SEQ_LEN

#define ATTN_WEIGHTS_SIZE MAX_SEQ_LEN*MAX_SEQ_LEN*32
 // this will probably scale to be bigger that QUERY_STATES_BUFF_SIZE

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

// can add tensors of different shapes assuming they are compatible
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

void transpose(
    const float* tensor, float* out, const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& tensor_dims, std::vector<uint32_t>& out_dims
) {
    // not safe to do inplace
    assert(tensor != out);
    assert(tensor_dims.size() == 4);
    // set out_dims
    out_dims.resize(4);
    for (uint32_t i = 0; i < 4; i++) {
        out_dims[i] = tensor_dims[perm[i]];
    }
    // code
    std::vector<int> old_indices(4);
    std::vector<int> offsets(4);
    offsets[0] = tensor_dims[0];
    offsets[1] = offsets[0] * tensor_dims[1];
    offsets[2] = offsets[1] * tensor_dims[2];
    offsets[3] = offsets[2] * tensor_dims[3];

    int num_elements = offsets[3];
    std::vector<int> new_indices(4);
    for (int i = 0; i < num_elements; i++) {
        // get old indices
        uint32_t index = i;
        for (int j = 0; j < 4; j++) {
            old_indices[j] = index % tensor_dims[j];
            index /= tensor_dims[j];
        }
        // get new indices
        for (int j = 0; j < 4; j++) {
            new_indices[j] = old_indices[perm[j]];
        }
        // write data
        uint32_t offset = 0;
        for (uint32_t j = 0; j < 4; j++) {
            offset = offset * out_dims[j] + new_indices[j];
        }
        out[offset] = tensor[i];
    }
}

// not input-to-output safe
void truncate(
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
    unsigned long long dim_offsets[4];
    dim_offsets[0] = 1;
    dim_offsets[1] = input_dims[0] * dim_offsets[0];
    dim_offsets[2] = input_dims[1] * dim_offsets[1];
    dim_offsets[3] = input_dims[2] * dim_offsets[2];
    for (int a = indice_start[0]; a <= indice_end[0]; a++) {
        for (int b = indice_start[1]; b <= indice_end[1]; b++) {
            for (int c = indice_start[2]; c <= indice_end[2]; c++) {
                for (int d = indice_start[3]; d <= indice_end[3]; d++) {
                    output[elements_written] = input[
                        d + 
                        c*dim_offsets[1] +
                        b*dim_offsets[2] +
                        a*dim_offsets[3]];
                    elements_written++;
                }
            }
        }
    }
    // writing output_dims
    for (int i = 0; i < input_dims.size(); i++) {
        output_dims[i] = indice_end[i] - indice_start[i];
    }
}

// IMPORTANT NOTE: IGNORING FOR NOW WHEN SEQ_LEN GETS BIG
void rotary_emb(
    const float* x, const std::vector<uint32_t>& x_dims,
    const int seq_len,
    const float* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const float* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    float* sin, std::vector<uint32_t>& sin_dims,
    float* cos, std::vector<uint32_t>& cos_dims
) {
    assert(seq_len <= SIN_COS_MAX_SEQ_LEN); // TEMP solution
    for (int i = 0; i < sin_cached_dims.size(); i++) {
        // idk if this needs to be true, but if it does not, change code below
        assert(sin_cached_dims[i] == cos_cached_dims[i]);
    }
    std::vector<int> dims_to_split = {0};
    std::vector<int> values = {seq_len};
    std::vector<int> colon_lefts = {1};
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

// cannot do in place
void gather(
    const float* x, const std::vector<uint32_t>& x_dims,
    const std::vector<int>& indices,
    float* out, std::vector<uint32_t>& out_dims
) {
    // offset computation
    int rank = x_dims.size();
    assert(rank <= 4);
    unsigned long long dim_offsets[4] = {1, 1, 1, 1};
    for (int i = 1; i < rank; i++) {
        dim_offsets[i] = dim_offsets[i-1] * x_dims[-1];
    }
    unsigned long long offset = dim_offsets[rank-1];
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

void concat(
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
    uint32_t x1_4d_dims[4]                = {1, 1, 1, 1};
    uint32_t x2_4d_dims[4]                = {1, 1, 1, 1};
    unsigned long long x1_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long x2_dim_offsets[4]  = {1, 1, 1, 1};
    unsigned long long out_dim_offsets[4] = {1, 1, 1, 1};
    for (int i = 1; i < rank; i++) {
        x1_4d_dims[i] = x1_dims[i];
        x2_4d_dims[i] = x2_dims[i];
        x1_dim_offsets[i]  = x1_dim_offsets[i-1]  * x1_dims[i-1];
        x2_dim_offsets[i]  = x2_dim_offsets[i-1]  * x2_dims[i-1];
        out_dim_offsets[i] = out_dim_offsets[i-1] * out_dims[i-1];
    }
    unsigned long long x1_tot_elems = x1_dim_offsets[rank-1] * x1_dims[rank-1];
    unsigned long long x2_tot_elems = x2_dim_offsets[rank-1] * x2_dims[rank-1];
    // writing x1 to out
    for (int a = 0; a < x1_4d_dims[0]; a++) {
        for (int b = 0; b < x1_4d_dims[1]; b++) {
            for (int c = 0; c < x1_4d_dims[2]; c++) {
                for (int d = 0; d < x1_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[1] + 
                        b*out_dim_offsets[2] +
                        a*out_dim_offsets[3]] 
                    = x1[
                        d + 
                        c*x1_dim_offsets[1] + 
                        b*x1_dim_offsets[2] +
                        a*x1_dim_offsets[3]];
                }
            }
        }
    }
    // writing x2 to out
    unsigned long long x2_write_offset = x1_dims[axis] * out_dim_offsets[axis];
    for (int a = 0; a < x2_4d_dims[0]; a++) {
        for (int b = 0; b < x2_4d_dims[1]; b++) {
            for (int c = 0; c < x2_4d_dims[2]; c++) {
                for (int d = 0; d < x2_4d_dims[3]; d++) {
                    out[
                        d + 
                        c*out_dim_offsets[1] + 
                        b*out_dim_offsets[2] +
                        a*out_dim_offsets[3] + 
                        x2_write_offset]
                    = x1[
                        d + 
                        c*x2_dim_offsets[1] + 
                        b*x2_dim_offsets[2] +
                        a*x2_dim_offsets[3]];
                }
            }
        }
    }
}

// uses an internal buffer for now (be careful if you remove)
// could optimize by writing directly to out rather than using truncate()
void rotate_half(
    const float* x, const std::vector<uint32_t>& x_dims,
    float* out, std::vector<uint32_t>& out_dims
) {
    // internal buffers
    float x1[QUERY_STATES_BUFF_SIZE];
    float x2[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> x1_dims, x2_dims;

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
    // negate x2
    unsigned long long x2_size = 1;
    for (auto i : x2_dims) { x2_size *= i; }
    for (int i = 0; i < x2_size; i++) {x2[i] = -1 * x2[i];}
    // concat
    concat(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
}


/* NOTE: b/c q and q_embed to same buffer in PhiAttetion, we will create a additional buffer
            so that the computation is correct, and we can mirror the python code. */
// are using an internal buffer
void apply_rotary_pos_emb(
    const float* q, const std::vector<uint32_t>& q_dims,
    const float* k, const std::vector<uint32_t>& k_dims,
    const float* cos, const std::vector<uint32_t>& cos_dims,
    const float* sin, const std::vector<uint32_t>& sin_dims,
    const std::vector<int>& position_ids, const int unsqueeze_dim,
    float* q_embed, std::vector<uint32_t>& q_embed_dims,
    float* k_embed, std::vector<uint32_t>& k_embed_dims
) {
    // buffers prbably dont need to be this big
    float cos_buff[SIN_COS_BUFF_SIZE];
    float sin_buff[SIN_COS_BUFF_SIZE];
    std::vector<uint32_t> cos_buff_dims, sin_buff_dims;
    gather(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
    gather(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
    cos_buff_dims.insert(cos_buff_dims.begin() + unsqueeze_dim, 1);
    sin_buff_dims.insert(sin_buff_dims.begin() + unsqueeze_dim, 1);
    // computing embedding
    // assume the last 2 dims are the same size for q and cos
    assert(q_dims.end()[-1] == cos_buff_dims.end()[-1] == k_dims.end()[-1]);
    assert(q_dims.end()[-2] == cos_buff_dims.end()[-2] == k_dims.end()[-2]);
    int q_rank = q_dims.size();
    assert(q_rank < 4);
    unsigned long long q_dim_offsets[4] = {1, 1, 1, 1};
    for (int i = 1; i < q_rank; i++) {
        q_dim_offsets[i] = q_dim_offsets[i-1] * q_dims[-1];
    }
    assert(q_dims[0] == k_dims[0]); // we can adjust if this needs to be false
    assert(q_dims[1] == k_dims[1]); // would need to create another collection of for loops
    float q_temp_buff[QUERY_STATES_BUFF_SIZE];
    float k_temp_buff[QUERY_STATES_BUFF_SIZE];
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_32f(
                &(q[i*q_dim_offsets[3] + j*q_dim_offsets[2]]), cos_buff, 
                q_temp_buff, cos_buff_dims);
            mul_32f(
                &(k[i*q_dim_offsets[3] + j*q_dim_offsets[2]]), cos_buff,
                k_temp_buff, cos_buff_dims);
        }
    }
    rotate_half(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
    rotate_half(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims
    for (int i = 0; i < q_dims[0]; i++) {
        for (int j = 0; j < q_dims[1]; j++) {
            mul_32f(
                &(q_embed[i*q_dim_offsets[3] + j*q_dim_offsets[2]]), sin_buff, 
                q_embed, sin_buff_dims);
            mul_32f(
                &(k_embed[i*q_dim_offsets[3] + j*q_dim_offsets[2]]), sin_buff, 
                k_embed, sin_buff_dims);
        }
    }
    add_32f(q_embed, q_temp_buff, q_embed, q_embed_dims);
    add_32f(k_embed, k_temp_buff, k_embed, k_embed_dims);
}

void copy(const float* ten1, float* out, const std::vector<uint32_t>& dims) {
    uint32_t num_elem = 1;
    for (uint32_t i = 0; i < dims.size(); i++) { num_elem *= i; }
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



void PhiAttention(
    /* inputs */
    const float* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    const float* attention_mask, const std::vector<uint32_t>& attention_mask_dims,
    const std::vector<int>& position_ids,
    const float* old_past_keys, const std::vector<uint32_t>& old_past_keys_dims,
    const float* old_past_values, const std::vector<uint32_t>& old_past_values_dims,
    /* weights */
    const float* q_proj_weights, const std::vector<uint32_t>& q_proj_weights_dims,
    const float* q_proj_bias,
    const float* k_proj_weights, const std::vector<uint32_t>& k_proj_weights_dims,
    const float* k_proj_bias,
    const float* v_proj_weights, const std::vector<uint32_t>& v_proj_weights_dims,
    const float* v_proj_bias,
    // const float* q_layernorm_weights, const int q_layernorm_weights_len,
    // const float* q_layernorm_bias,
    // const float* k_layernorm_weights, const int k_layernorm_weights_len,
    // const float* k_layernorm_bias,
    // const float eps,
    const int num_heads, const int head_dim, const int num_kv_heads,
    const float* dense_weights, const std::vector<uint32_t>& dense_weights_dims,
    const float* dense_bias,
    /* init params */
    const int layer_idx,
    const float* sin_cached, const std::vector<uint32_t>& sin_cached_dims,
    const float* cos_cached, const std::vector<uint32_t>& cos_cached_dims,
    int rotary_emb_dim,
    /* outputs */
    float* attn_output, std::vector<uint32_t>& attn_output_dims,
    float* past_keys, std::vector<uint32_t>& past_keys_dims,
    float* past_values, std::vector<uint32_t>& past_values_dims
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
    // aparently, we dont use them
    // layernorm_Nd_32f(
    //     query_states_buff, q_layernorm_weights, q_layernorm_bias, query_states_buff,
    //     query_states_dims, q_layernorm_weights_len, eps);
    // layernorm_Nd_32f(
    //     key_states_buff, k_layernorm_weights, k_layernorm_bias, key_states_buff,
    //     key_states_dims, k_layernorm_weights_len, eps);
    
    // reshape
    query_states_dims = std::vector<uint32_t>{
        (uint32_t)bsz, (uint32_t)q_len, (uint32_t)num_heads, (uint32_t)head_dim};
    key_states_dims = std::vector<uint32_t>{
        (uint32_t)bsz, (uint32_t)q_len, (uint32_t)num_kv_heads, (uint32_t)head_dim};
    value_states_dims = key_states_dims;

    // transpose
    // cant use same buffers (unless transpose() uses a temporary buffer)
    float query_states_buff_2[QUERY_STATES_BUFF_SIZE];
    float key_states_buff_2[QUERY_STATES_BUFF_SIZE];
    float value_states_buff_2[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> temp_dims;
    std::vector<uint32_t> perm = {0, 2, 1, 3};
    transpose(
        query_states_buff, query_states_buff_2, 
        perm, 
        query_states_dims, temp_dims);
    query_states_dims = temp_dims;
    transpose(
        key_states_buff, key_states_buff_2, 
        perm, 
        key_states_dims, temp_dims);
    key_states_dims = temp_dims;
    transpose(
        value_states_buff, value_states_buff_2, 
        perm, 
        value_states_dims, temp_dims);
    value_states_dims = temp_dims;
    
    // Cache
    // old_past_keys_dims: (seq_len, something), (seq_len, something)
    int kv_seq_len = key_states_dims.end()[-2];
    if (old_past_keys_dims.size() > 0) { // not first run
        kv_seq_len += old_past_keys_dims.end()[-2]; // seq_len MAKE SURE THIS IS RIGHT
    }

    // partial rotary embedding
    float sin_buff[SIN_COS_BUFF_SIZE];
    float cos_buff[SIN_COS_BUFF_SIZE];
    std::vector<uint32_t> sin_buff_dims;
    std::vector<uint32_t> cos_buff_dims;
    rotary_emb(
        value_states_buff_2, value_states_dims, kv_seq_len,
        sin_cached, sin_cached_dims, cos_cached, cos_cached_dims,
        sin_buff, sin_buff_dims, cos_buff, cos_buff_dims);
    //rotations
    float query_rot_buff[QUERY_STATES_BUFF_SIZE];
    float query_pass_buff[QUERY_STATES_BUFF_SIZE];
    float key_rot_buff[QUERY_STATES_BUFF_SIZE];
    float key_pass_buff[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> query_rot_buff_dims;
    std::vector<uint32_t> query_pass_buff_dims;
    std::vector<uint32_t> key_rot_buff_dims;
    std::vector<uint32_t> key_pass_buff_dims;
    truncate(
        query_states_buff_2, query_states_dims,
        query_rot_buff, query_rot_buff_dims, 
        std::vector<int> {int(query_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1}
        );
    truncate(
        query_states_buff_2, query_states_dims,
        query_pass_buff, query_pass_buff_dims, 
        std::vector<int> {int(query_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0});
    truncate(
        key_states_buff_2, key_states_dims,
        key_rot_buff, key_rot_buff_dims, 
        std::vector<int> {int(key_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {1});
    truncate(
        key_states_buff_2, key_states_dims,
        key_pass_buff, key_pass_buff_dims, 
        std::vector<int> {int(key_states_dims.size())-1},
        std::vector<int> {rotary_emb_dim},
        std::vector<int> {0});
    // applying rot_pos-emb
    // [batch_size, seq_length, num_heads, head_dim // partial_rotary_factor]
    apply_rotary_pos_emb(
    query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims,
    cos_buff, cos_buff_dims, sin_buff, sin_buff_dims, 
    position_ids, 1,
    query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims);
    
    // concatting rot and pass
    concat(
        query_rot_buff, query_rot_buff_dims,
        query_pass_buff, query_pass_buff_dims, 
        query_rot_buff_dims.size()-1, 
        query_states_buff, query_states_dims);
    concat(
    key_rot_buff, key_rot_buff_dims,
    key_pass_buff, key_pass_buff_dims, 
    key_rot_buff_dims.size()-1, 
    key_states_buff, key_states_dims);

    // updating cache
    // past_key_value_old: (seq_len, something), (seq_len, something)

    // COMMENT BACK IN
    // concat(
    //     old_past_keys, old_past_keys_dims, key_states_buff, key_states_dims,
    //     key_states_dims.size()-2, 
    //     past_keys, past_keys_dims);
    // concat(
    //     old_past_values, old_past_values_dims, value_states_buff_2, value_states_dims,
    //     value_states_dims.size()-2, 
    //     past_values, past_values_dims);
    // copy(past_keys, key_states_buff, past_keys_dims);
    // copy(past_values, value_states_buff, past_values_dims);

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
    transpose(
        key_states_buff, key_states_buff_2, 
        std::vector<uint32_t> {0, 1, 3, 2},
        key_states_dims, temp_dims
        );
    key_states_dims = temp_dims;
    // NOTE: not sure how big the buffer needs to be
    unsigned long long key_states_size = 1;
    for (auto i : key_states_dims) {key_states_size *= i;}
    assert(key_states_size < ATTN_WEIGHTS_SIZE);
    float attn_weights[ATTN_WEIGHTS_SIZE];
    std::vector<uint32_t> attn_weights_dims;
    matmul_Nd_32f_constrained(
        query_states_buff, key_states_buff_2, attn_weights,
        query_states_dims, key_states_dims, attn_weights_dims);
    
    // masking
    add_32f_general(
        attn_weights, attention_mask, attn_output,
        attn_weights_dims, attention_mask_dims, attn_output_dims);
    
    // softmax
    softmax(attn_output, attn_weights, attn_output_dims);
    attn_weights_dims = attn_output_dims;

    // matmul (to attn_output)
    matmul_Nd_32f_constrained(
        attn_weights, value_states_buff, attn_output,
        attn_weights_dims, value_states_dims, attn_output_dims);

    // tranpose (to attn_output)
    transpose(
        attn_output, attn_weights, 
        std::vector<uint32_t> {0, 2, 1, 3},
        attn_output_dims, attn_weights_dims);

    // reshape (attn_output)
    attn_weights_dims = std::vector<uint32_t> {
        (uint32_t)bsz, 
        (uint32_t)q_len, 
        HIDDEN_SIZE};
    
    // dense layer
    linear_Nd_32f(
        attn_weights, dense_weights, dense_bias, attn_output, 
        attn_weights_dims, dense_weights_dims, attn_output_dims);
}

#define q_LEN 11
int main() {
    /* inputs */
    float hidden_states[BATCH_SIZE * q_LEN * HIDDEN_SIZE];
    std::vector<uint32_t> hidden_states_dims {BATCH_SIZE, q_LEN, HIDDEN_SIZE};
    float attention_mask[BATCH_SIZE * q_LEN * q_LEN];
    std::vector<uint32_t> attention_mask_dims {BATCH_SIZE, 1, q_LEN, q_LEN};
    std::vector<int> position_ids;
    for (int i = 0; i < q_LEN; i++) { position_ids.push_back(i); }
    float *old_past_keys, *old_past_values;
    old_past_keys = old_past_values = NULL;
    std::vector<uint32_t> old_past_keys_dims, old_past_values_dims;

    /* weights */
    // projs (same for all)
    float proj_weights[HIDDEN_SIZE * HIDDEN_SIZE];
    float proj_bias[HIDDEN_SIZE];
    std::vector<uint32_t> proj_weights_dims = {HIDDEN_SIZE, HIDDEN_SIZE};
    // params
    const int num_heads = 32;
    const int head_dim = HIDDEN_SIZE / num_heads;
    const int num_kv_heads = 32;

    /* init params */
    float sin_cached[SIN_COS_BUFF_SIZE];
    std::vector<uint32_t> sin_cached_dims {MAX_SEQ_LEN, 32};
    float cos_cached[SIN_COS_BUFF_SIZE];
    std::vector<uint32_t> cos_cached_dims {MAX_SEQ_LEN, 32};
    int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)
    int layer_idx = 0;

    /* outputs */
    float attn_output[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> attn_output_dims;
    float past_keys[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> past_keys_dims;
    float past_values[QUERY_STATES_BUFF_SIZE];
    std::vector<uint32_t> past_values_dims;

        
PhiAttention(
    /* inputs */
    hidden_states,  hidden_states_dims,
    attention_mask, attention_mask_dims,
    position_ids,
     old_past_keys,  old_past_keys_dims,
     old_past_values,  old_past_values_dims,
    /* weights */
     proj_weights,  proj_weights_dims,
     proj_bias,
     proj_weights,  proj_weights_dims,
     proj_bias,
     proj_weights,  proj_weights_dims,
     proj_bias,
    num_heads, head_dim, num_kv_heads,
     proj_weights,  proj_weights_dims,
     proj_bias,
    /* init params */
    layer_idx,
     sin_cached,  sin_cached_dims,
     cos_cached,  cos_cached_dims,
    rotary_emb_dim,
    /* outputs */
    attn_output, attn_output_dims,
    past_keys, past_keys_dims,
    past_values, past_values_dims
);

}
