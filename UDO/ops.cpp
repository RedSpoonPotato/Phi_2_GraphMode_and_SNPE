#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560
#define QUERY_STATES_BUFF_SIZE  BATCH_SIZE*MAX_SEQ_LEN*HIDDEN_SIZE

#define INTERMEDIATE_SIZE 10240
#define INTERMEDIATE_STATES_BUFF_SIZE BATCH_SIZE*MAX_SEQ_LEN*INTERMEDIATE_SIZE

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

// vector print function
void printV(const std::string& str, const std::vector<uint32_t>& vec) {
    std::cout << str;
    std::cout << ": [";
    // for (auto i: vec) {std::cout << i << ", ";}
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i != vec.size()-1) { std::cout << ", ";}
    }
    std::cout << "]\n";
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
    int offset = dims1[1] * dims1[2];
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
void rotary_emb(
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
    // unsigned long long x1_tot_elems = x1_dim_offsets[0] * x1_dims[0];
    // unsigned long long x2_tot_elems = x2_dim_offsets[0] * x2_dims[0];
    // printV("\tout_dims", out_dims);
    // printV("\tx1_4d_dims", x1_4d_dims);
    // for (int i = 0; i < 4; i++) {
    //     std::cout << x1_dim_offsets[i] << " ";
    // }
    // std::cout << "out_dim_offsets:\n";
    // for (int i = 0; i < 4; i++) {
    //     std::cout << out_dim_offsets[i] << " ";
    // }
    // std::cout << "\twriting x1 to out\n";
    // writing x1 to out
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
void rotate_half(
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
    truncate(
        x, x_dims, x1, x1_dims, dims_to_split,
        values, colon_left
    );
    truncate(
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
    concat(x1, x1_dims, x2, x2_dims, int(x1_dims.size()-1), out, out_dims);
    std::cout << "\t\tfreeing memory\n";
    free(x1);
    free(x2);
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
    // float cos_buff[SIN_COS_BUFF_SIZE];
    float* cos_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    // float sin_buff[SIN_COS_BUFF_SIZE];
    float* sin_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> cos_buff_dims, sin_buff_dims;
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    gather(cos, cos_dims, position_ids, cos_buff, cos_buff_dims);
    std::cout << "\tCalling gather() with position_ids len: "<<position_ids.size()<<"\n";
    gather(sin, sin_dims, position_ids, sin_buff, sin_buff_dims);
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
    rotate_half(q, q_dims, q_embed, q_embed_dims); // this might cause problems with the outdims
    std::cout << "\tCalling Rotate_Half\n";
    printV("k_dims", k_dims);
    rotate_half(k, k_dims, k_embed, k_embed_dims); // rotate_half() intializes dims
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

void copyTensor(const float* ten1, float* out, const std::vector<uint32_t>& dims) {
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
    // float query_states_buff[QUERY_STATES_BUFF_SIZE];
    float* query_states_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float key_states_buff[QUERY_STATES_BUFF_SIZE];
    float* key_states_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float value_states_buff[QUERY_STATES_BUFF_SIZE];
    float* value_states_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));

    std::vector<uint32_t> query_states_dims;
    std::vector<uint32_t> key_states_dims;
    std::vector<uint32_t> value_states_dims;

    std::cout << "Calling Projection Linear Layers\n";

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
    printV("value_states", value_states_dims);

    // transpose
    // cant use same buffers (unless transpose() uses a temporary buffer)
    // float query_states_buff_2[QUERY_STATES_BUFF_SIZE];
    float* query_states_buff_2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float key_states_buff_2[QUERY_STATES_BUFF_SIZE];
    float* key_states_buff_2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float value_states_buff_2[QUERY_STATES_BUFF_SIZE];
    float* value_states_buff_2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> temp_dims;
    std::vector<uint32_t> perm = {0, 2, 1, 3};
    std::cout << "Calling tranpose() 3 times\n";
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
    printV("value_states", value_states_dims);
    // Cache
    // old_past_keys_dims: (seq_len, something), (seq_len, something)
    int kv_seq_len = key_states_dims.end()[-2];
    if (old_past_keys_dims.size() > 0) { // not first run
        kv_seq_len += old_past_keys_dims.end()[-2]; // seq_len MAKE SURE THIS IS RIGHT
    }
    std::cout << "kv_seq_len:" << kv_seq_len << "\n";

    // partial rotary embedding
    // float sin_buff[SIN_COS_BUFF_SIZE];
    float* sin_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    // float cos_buff[SIN_COS_BUFF_SIZE];
    float* cos_buff = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> sin_buff_dims;
    std::vector<uint32_t> cos_buff_dims;
    printV("value_states", value_states_dims);
    std::cout << "calling rot_emb()\n";
    rotary_emb(
        value_states_buff_2, value_states_dims, kv_seq_len,
        sin_cached, sin_cached_dims, cos_cached, cos_cached_dims,
        sin_buff, sin_buff_dims, cos_buff, cos_buff_dims);
    std::cout << "finished calling rot_emb()\n";
    printV("sin_buff_dims", sin_buff_dims);
    printV("cos_buff_dims", cos_buff_dims);
    //rotations
    // float query_rot_buff[QUERY_STATES_BUFF_SIZE];
    float* query_rot_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float query_pass_buff[QUERY_STATES_BUFF_SIZE];
    float* query_pass_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float key_rot_buff[QUERY_STATES_BUFF_SIZE];
    float* key_rot_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    // float key_pass_buff[QUERY_STATES_BUFF_SIZE];
    float* key_pass_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> query_rot_buff_dims;
    std::vector<uint32_t> query_pass_buff_dims;
    std::vector<uint32_t> key_rot_buff_dims;
    std::vector<uint32_t> key_pass_buff_dims;
    std::cout << "Calling truncate 4 times for rot and pass\n";
    printV("query_states_buff_2",query_states_dims);
    printV("key_states_buff_2",key_states_dims);
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
    printV("query_rot_buff_dims", query_rot_buff_dims);
    printV("query_pass_buff_dims", query_pass_buff_dims);
    // applying rot_pos-emb
    // [batch_size, seq_length, num_heads, head_dim // partial_rotary_factor]
    std::cout << "Calling apply_rotary_pos_emb()\n";
    apply_rotary_pos_emb(
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims,
        cos_buff, cos_buff_dims, sin_buff, sin_buff_dims, 
        position_ids, 1,
        query_rot_buff, query_rot_buff_dims, key_rot_buff, key_rot_buff_dims);
    printV("key_rot_buff_dims", key_rot_buff_dims);
    // concatting rot and pass
    std::cout << "Concatenating rot and pass\n";
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
    std::cout << "--Going to Cache--\n";
    printV("old_past_keys", old_past_keys_dims);
    printV("key_states_dims", key_states_dims);
    if (old_past_keys_dims.size() == 0) { 
        std::cout << "first run\n";
        // first run
        copyTensor(key_states_buff, past_keys, key_states_dims);
        past_keys_dims = key_states_dims;
        copyTensor(value_states_buff_2, past_values, value_states_dims);
        past_values_dims = value_states_dims;
    }
    else { 
        // not first run
        std::cout << "not first run\n";
        std::cout << "calling first concat\n";
        concat(
            old_past_keys, old_past_keys_dims, key_states_buff, key_states_dims,
            key_states_dims.size()-2, 
            past_keys, past_keys_dims);
        std::cout << "calling second concat\n";
        concat(
            old_past_values, old_past_values_dims, value_states_buff_2, value_states_dims,
            value_states_dims.size()-2, 
            past_values, past_values_dims);
        std::cout << "calling first copyTensor()\n";
        copyTensor(past_keys, key_states_buff, past_keys_dims);
        copyTensor(past_values, value_states_buff, past_values_dims);
    }

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
    // float attn_weights[ATTN_WEIGHTS_SIZE];
    float* attn_weights = (float*)malloc(ATTN_WEIGHTS_SIZE * sizeof(float));
    std::vector<uint32_t> attn_weights_dims;
    printV("query_states", query_states_dims);
    printV("key_states_tranposed", key_states_dims);
    std::cout << "Calling 1st matmul\n";
    matmul_Nd_32f_constrained(
        query_states_buff, key_states_buff_2, attn_weights,
        query_states_dims, key_states_dims, attn_weights_dims);
    printV("attn_weights", attn_weights_dims);

    // masking
    std::cout << "Calling Masking\n";
    add_32f_general(
        attn_weights, attention_mask, attn_output,
        attn_weights_dims, attention_mask_dims, attn_output_dims);
    printV("post masking attn_weights(attn_output)", attn_output_dims);
    
    // softmax
    std::cout << "Calling softmax\n";
    mySoftmax(attn_output, attn_weights, attn_output_dims);
    attn_weights_dims = attn_output_dims;
    printV("post softmaxing attn_weights", attn_weights_dims);

    // matmul (to attn_output)
    std::cout << "Calling 2nd matmul\n";
    matmul_Nd_32f_constrained(
        attn_weights, value_states_buff, attn_output,
        attn_weights_dims, value_states_dims, attn_output_dims);


    // tranpose (to attn_output)
    std::cout << "Calling transpose\n";
    transpose(
        attn_output, attn_weights, 
        std::vector<uint32_t> {0, 2, 1, 3},
        attn_output_dims, attn_weights_dims);

    // reshape (attn_output)
    std::cout << "Reshaping\n";
    attn_weights_dims = std::vector<uint32_t> {
        (uint32_t)bsz, 
        (uint32_t)q_len, 
        HIDDEN_SIZE};
    
    std::cout << "Calling Final Dense Layer\n";
    // dense layer
    linear_Nd_32f(
        attn_weights, dense_weights, dense_bias, attn_output, 
        attn_weights_dims, dense_weights_dims, attn_output_dims);
    
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

void NewGELU(
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

// currrently using an internal buffer
void PhiMLP(
    const float* input, const std::vector<uint32_t>& input_dims,
    float* output, std::vector<uint32_t>& output_dims,
    /* weights */
    const float* fc1_weights, const std::vector<uint32_t>& fc1_weights_dims,
    const float* fc1_bias,
    const float* fc2_weights, const std::vector<uint32_t>& fc2_weights_dims,
    const float* fc2_bias
) {
    // buffer to store larger intermediate size
    float* intermediate_buff = (float*)malloc(INTERMEDIATE_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> intermediate_buff_dims;
    // compute
    linear_Nd_32f(
        input, fc1_weights, fc1_bias, intermediate_buff, 
        input_dims, fc1_weights_dims, intermediate_buff_dims);
    NewGELU(intermediate_buff, intermediate_buff, intermediate_buff_dims);
    linear_Nd_32f(
        intermediate_buff, fc2_weights, fc2_bias, output, 
        intermediate_buff_dims, fc2_weights_dims, output_dims);
    // free
    free(intermediate_buff);
}

// NOTE: Could be optimized
// using an internal buffer for holding the residual
void PhiDecoderLayer(
    /* inputs */
    const float* hidden_states, const std::vector<uint32_t>& hidden_states_dims,
    const float* attention_mask, const std::vector<uint32_t>& attention_mask_dims,
    const std::vector<int>& position_ids,
    const float* old_past_keys, const std::vector<uint32_t>& old_past_keys_dims,
    const float* old_past_values, const std::vector<uint32_t>& old_past_values_dims,
    /* Decoder Weights */
    const float* input_layernorm_weights, const int input_layernorm_weights_len,
    const float* input_layernorm_bias, const float decoder_eps,
    const float* fc1_weights, const std::vector<uint32_t>& fc1_weights_dims,
    const float* fc1_bias,
    const float* fc2_weights, const std::vector<uint32_t>& fc2_weights_dims,
    const float* fc2_bias,
    /* Decoder Outputs: not needed */

    /* PhiAttention weights */
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
    // residual is contained within hidden_states

    // layernorm
    float* hidden_states_buff_2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> hidden_states_dims_2;
    layernorm_Nd_32f(
        hidden_states, input_layernorm_weights, input_layernorm_bias,
        hidden_states_buff_2, hidden_states_dims, input_layernorm_weights_len, decoder_eps);
    hidden_states_dims_2 = hidden_states_dims;


    // Phi Attention
    // look into implementing inplace addition so you dont have to allocate this much memory
    float* attn_output_buff_2 = (float*)malloc(ATTN_WEIGHTS_SIZE * sizeof(float));
    std::vector<uint32_t> attn_output_buff_2_dims;
    PhiAttention(
        /* inputs */
        hidden_states_buff_2,  hidden_states_dims_2, // could possibly be optmized
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
        num_heads, head_dim, num_kv_heads,
        dense_weights, dense_weights_dims,
        dense_bias,
        /* init params */
        layer_idx,
        sin_cached,  sin_cached_dims,
        cos_cached,  cos_cached_dims,
        rotary_emb_dim,
        /* outputs */
        attn_output_buff_2, attn_output_buff_2_dims,
        past_keys, past_keys_dims,
        past_values, past_values_dims
    );

    // MLP
    float* feed_forward_hidden_states = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> feed_forward_hidden_states_dims;
    PhiMLP(
        hidden_states_buff_2, hidden_states_dims_2, 
        feed_forward_hidden_states, feed_forward_hidden_states_dims,
        fc1_weights, fc1_weights_dims, fc1_bias, 
        fc2_weights, fc2_weights_dims, fc2_bias);
    
    // Large Addition (could optmizie by using an inplace addition for attn_outputs)
    // using hidden_states_buff_2 as a output buffer
    // hidden_states is the residual in this case
    add_32f_general(
        feed_forward_hidden_states, hidden_states, hidden_states_buff_2,
        feed_forward_hidden_states_dims, hidden_states_dims, hidden_states_dims_2);
    add_32f_general(
        hidden_states_buff_2, attn_output_buff_2, attn_output,
        hidden_states_dims_2, attn_output_buff_2_dims, attn_output_dims);
    
    // free
    free(hidden_states_buff_2);
    free(attn_output_buff_2);
    free(feed_forward_hidden_states);
}



#define q_LEN 11

// temp function, remove later
// int main() {

//     float* query_states_buff = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
//     std::vector<uint32_t> query_states_dims { 1, 32, q_LEN, 80};

//     float* key_states_buff_2 = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
//     std::vector<uint32_t> key_states_dims = {1, 32, 80, q_LEN};

//     float* attn_weights = (float*)malloc(ATTN_WEIGHTS_SIZE * sizeof(float));
//     std::vector<uint32_t> attn_weights_dims = {};

//     std::cout << "Calling matmul\n";
//     matmul_Nd_32f_constrained(
//         query_states_buff, key_states_buff_2, attn_weights,
//         query_states_dims, key_states_dims, attn_weights_dims);

//     std::cout << "Freeing memory\n";
//     free(query_states_buff);
//     free(key_states_buff_2);
//     free(attn_weights);
// }


int main() {
    std::cout << "--Intializing inputs\n";
    /* inputs */
    // float hidden_states[BATCH_SIZE * q_LEN * HIDDEN_SIZE];
    float* hidden_states = (float*)malloc(BATCH_SIZE * q_LEN * HIDDEN_SIZE * sizeof(float));
    std::vector<uint32_t> hidden_states_dims {BATCH_SIZE, q_LEN, HIDDEN_SIZE};
    // float attention_mask[BATCH_SIZE * q_LEN * q_LEN];
    float* attention_mask = (float*)malloc(BATCH_SIZE * q_LEN * q_LEN * sizeof(float));
    std::vector<uint32_t> attention_mask_dims {BATCH_SIZE, 1, q_LEN, q_LEN};
    std::vector<int> position_ids;
    for (int i = 0; i < q_LEN; i++) { position_ids.push_back(i); }
    // float *old_past_keys, *old_past_values;
    // old_past_keys = old_past_values = NULL;
    float *old_past_keys = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    float *old_past_values = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> old_past_keys_dims, old_past_values_dims;

    std::cout << "--Intializing weights\n";
    /* weights */
    // projs (same for all)
    // float proj_weights[HIDDEN_SIZE * HIDDEN_SIZE];
    float* proj_weights = (float*)malloc(HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    // float proj_bias[HIDDEN_SIZE];
    float* proj_bias = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    std::vector<uint32_t> proj_weights_dims = {HIDDEN_SIZE, HIDDEN_SIZE};
    // params
    const int num_heads = 32;
    const int head_dim = HIDDEN_SIZE / num_heads;
    const int num_kv_heads = 32;

    std::cout << "--Intializing params\n";
    /* init params */
    // float sin_cached[SIN_COS_BUFF_SIZE];
    float* sin_cached = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> sin_cached_dims {MAX_SEQ_LEN, 32};
    // float cos_cached[SIN_COS_BUFF_SIZE];
    float* cos_cached = (float*)malloc(SIN_COS_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> cos_cached_dims {MAX_SEQ_LEN, 32};
    int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)
    int layer_idx = 0;

    std::cout << "--Intializing outputs\n";
    /* outputs */
    // float attn_output[ATTN_WEIGHTS_SIZE];
    // has to be same size as attn_weights b/c we use it as a temp buffer
    float* attn_output = (float*)malloc(ATTN_WEIGHTS_SIZE * sizeof(float));
    std::vector<uint32_t> attn_output_dims;
    // float past_keys[QUERY_STATES_BUFF_SIZE];
    float* past_keys = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> past_keys_dims;
    // float past_values[QUERY_STATES_BUFF_SIZE];
    float* past_values = (float*)malloc(QUERY_STATES_BUFF_SIZE * sizeof(float));
    std::vector<uint32_t> past_values_dims;

    /* calling PhiAttention */
    std::cout << "Calling PhiAttention()\n";
    for (int i = 0; i < 2; i++) {
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
        // copy past_kvs into old_past_kvs
        copyTensor(past_keys, old_past_keys, past_keys_dims);
        old_past_keys_dims = past_keys_dims;
        copyTensor(past_values, old_past_values, past_values_dims);
        old_past_values_dims = past_values_dims;
        std::cout << "\n\nEND OF ITERATION: " << i << "\n\n";
        // update position ids (assuming 1 token input)
        position_ids = std::vector<int>{q_LEN + i};
        // update masking
        free(attention_mask);
        attention_mask = (float*)malloc(BATCH_SIZE * 1 * (i+1)+q_LEN * sizeof(float));
        for (int j = 0; j < ((i+1)+q_LEN); j++) {attention_mask[j] = 0.0f;}
        std::vector<uint32_t> attention_mask_dims {BATCH_SIZE, 1, 1, (uint32_t)(i+1)+q_LEN};
        
    }
    std::cout << "Exiting PhiAttention() and freeing memory\n";
    
    /* freeing */
    free(hidden_states);
    free(attention_mask);
    free(proj_weights);
    free(proj_bias);
    free(sin_cached);
    free(cos_cached);
    free(attn_output);
    free(past_keys);
    free(past_values);
    

    return 0;
}
