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
    const std::map<std::string, RuntimeParams>& runtime_params,
    const std::set<std::pair<std::string, std::string>>& ModelNameAndPaths, // abs paths
    const std::map<std::string, std::string>& otherPaths, // abs path of sin, cos, embeddingFIle
    const uint32_t max_iterations, 
    const uint8_t decoder_cache_size,
    const Free_Status exitAndFree,
    const int debugReturnCode,
    const uint32_t end_token_id,
    const bool use_end_token_id
);

#endif