//==============================================================================
// Auto Generated Code for DecodePackage
//==============================================================================
#include <iostream>
#include <string>
#include <cstdlib>

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

// CHANGE THESE
#define DECODER_WEIGHT_SIZE 10
#define DECODER_WEIGHT_DIMS_SIZE 10
#define DECODER_INIT_PARAMS_SIZE 10


Qnn_ErrorHandle_t execute(CustomOp* operation) {

  auto m_Input  = operation->getInput(0);
  auto m_Outputs = operation->getOutput(0);

  const uint32_t rank = m_Outputs->rank;
  const size_t depth = m_Outputs->currentDimensions[rank - 1];

  std::cout << "depth: " << depth << "\n";

  // this appears to work
  float* out = (float*)m_Outputs->data;
  for (int i = 0; i < depth; i++) {
    out[i] = i+1;
  }


  std::cout << "4th input size: " << operation->getInput(4)->currentDimensions[rank-1];
  char* in = (char*)operation->getInput(4)->data;
  in[0] = 'K';

  ////////////////////////////////////////////////////////////

  /*
  assumed inputs (NOTE: THIS ORDER MAY HAVE OT CHANGE DRAMATICALLY) (all 16-bit except mask and pos_ids)

  SIZES IN MEMORY

  attn_and_kv_in[(4+(MAX_SEQ_LEN * HIDDEN_SIZE * DATASIZE)) * (1 + 2*DECODERS)]
  attention_mask[4 + 4*MAX_SEQ_LEN^2] // 32bit
  position_ids_1[4 + MAX_SEQ_LEN] // 32bit
  decoder_weights[DECODERS * DECODER_WEIGHT_SIZE * DATASIZE]
  decoder_weights_dims[DECODERS * DECODER_WEIGHT_DIMS_SIZE * DATASIZE]
  decoder_init_pararms[DECODERS * DECODER_INIT_PARAMS_SIZE * DATASIZE]
  */

  datatype* hidden_states;
  


  /* Buffers */
  float* buff_1 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  float* buff_2 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  float* buff_3 = (float*)malloc(LARGE_BUFFER_SIZE * sizeof(float));
  datatype* buff_4 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));
  datatype* buff_5 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));
  datatype* buff_6 = (datatype*)malloc(LARGE_BUFFER_SIZE * sizeof(datatype));


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
) 
  
  free(buff_1);
  free(buff_2);
  free(buff_3);
  free(buff_4);
  free(buff_5);
  free(buff_6);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
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

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 6, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
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
