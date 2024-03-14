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

  uint16_t;
  



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
