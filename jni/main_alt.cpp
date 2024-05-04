#include "android_main.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {

    std::string input_txt = "this test does not matter";
    std::string output_str;

    std::string srcDIR = ".";
    std::vector<std::string> inputList = {
        "hidden_states_and_kv",
        "attention_mask",
        "position_ids_1",
        "decoder_weights_1",
        "lm_head_weights_1",
        "sin",
        "cos",
    };
    std::vector<size_t> first_model_input_sizes = {
        (1 + 2 * DECODERS) * (4*4 + (MAX_SEQ_LEN * HIDDEN_SIZE) * DATASIZE), 
        MASK_SIZE, 
        POS_IDS_SIZE, 
        TOTAL_DECODER_WEIGHT_SIZE,
        TOTAL_LM_WEIGHT_SIZE, 
        SIN_COS_TOTAL_SIZE, 
        SIN_COS_TOTAL_SIZE
    };
    std::string dlcName = "UnifiedPhiDecodersAndLogits.dlc";
    std::vector<std::string> outputNames = {"Output_1:0"};
    uint32_t NUM_ITERS = 1;
    std::string udo_path = "DecodePackage/libs/x86-64_linux_clang/libUdoDecodePackageReg.so";

    std::string embeddingFile = "embed_tokens.dat";
    bool use_udo = true;

    output_str = modelLaunch(
        input_txt,
        srcDIR, 
        inputList, 
        first_model_input_sizes,
        embeddingFile,
        dlcName, 
        outputNames,
        NUM_ITERS,
        udo_path,
        use_udo,
        true,
        Free_Status::run);

    // output_str = modelLaunch(
    //     input_txt,
    //     srcDIR, 
    //     inputList, 
    //     first_model_input_sizes,
    //     embeddingFile,
    //     dlcName, 
    //     outputNames,
    //     NUM_ITERS,
    //     udo_path,
    //     use_udo,
    //     false,
    //     Free_Status::run_and_free);

    #ifdef DEBUG
        std::cout << "checkpoint 3\n";
    #endif

    return 0;
}