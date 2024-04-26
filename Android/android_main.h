#ifndef ANDROID_MAIN_H_
#define ANDROID_MAIN_H_

#include "snpe_tutorial_utils.h"
#include "snpe_exec_utils.h"
// #include "embedding.h"
#include "main_macros.h"
// #include "tokenizer.h"

enum class Free_Status {run, run_and_free, free};

/* 
runtime_modes: {
  CPU_FLOAT32  = 0,
  GPU_FLOAT32_16_HYBRID = 1,
  DSP_FIXED8_TF = 2,
  GPU_FLOAT16 = 3,
  AIP_FIXED8_TF = 5,
  CPU = CPU_FLOAT32,
  GPU = GPU_FLOAT32_16_HYBRID,
  DSP = DSP_FIXED8_TF,
  UNSET = -1
}
*/

std::string modelLaunch(
    const std::string& input_txt,
    const std::vector<zdl::DlSystem::Runtime_t>& runtime_modes,
    const std::string& srcDIR, 
    const std::vector<std::string>& inputList, 
    const std::vector<size_t>& first_model_input_sizes,
    const std::vector<size_t>& first_model_buffer_sizes,
    const std::vector<size_t>& first_model_output_buffer_sizes,
    const size_t& datasize,
    const std::string& embeddingFile,
    const std::string& dlcPath, 
    const std::vector<std::string>& outputNames,
    const uint32_t& NUM_ITERS,
    const std::string& udo_path,
    const bool use_udo,
    const bool firstRun,
    const std::string& outputDir,
    const Free_Status exitAndFree,
    const int debugReturnCode);

#endif