#ifndef ANDROID_MAIN_H_
#define ANDROID_MAIN_H_

#include "snpe_tutorial_utils.h"
#include "snpe_exec_utils.h"
#include "embedding.h"
#include "main_macros.h"
#include "tokenizer.h"

enum class Free_Status {run, run_and_free, free};

std::string modelLaunch(
    const std::string& input_txt,
    const std::string& srcDIR, 
    const std::vector<std::string>& inputList, 
    const std::vector<size_t>& first_model_input_sizes,
    const std::string& embeddingFile,
    const std::string& dlcName, 
    const std::vector<std::string>& outputNames,
    const uint32_t& NUM_ITERS,
    const std::string& udo_path,
    bool use_udo,
    bool firstRun,
    Free_Status exitAndFree);

#endif
