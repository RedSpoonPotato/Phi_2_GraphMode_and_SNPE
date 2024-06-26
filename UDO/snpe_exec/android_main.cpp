/* 
    - Currently built for a single model, see test.cpp for an example of how to use multiple dlcs
    - this file will serve as the base for creating the .so for android
*/
// #include "snpe_tutorial_utils.h"
// #include "snpe_exec_utils.h"
// #include "embedding.h"
// #include "main_macros.h"
// #include "tokenizer.h"


#       ifdef ANDROID
            // LOGS ANDROID
#           include <android/log.h>
#           define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG,__VA_ARGS__)
#           define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , LOG_TAG,__VA_ARGS__)
#           define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , LOG_TAG,__VA_ARGS__)
#           define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , LOG_TAG,__VA_ARGS__)
#           define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , LOG_TAG,__VA_ARGS__)
#           define LOGSIMPLE(...)
#       else
            // LOGS NO ANDROID
#           include <stdio.h>
#           define LOGV(...) printf("  ");printf(__VA_ARGS__); printf("\t -  <%s> \n", LOG_TAG);
#           define LOGD(...) printf("  ");printf(__VA_ARGS__); printf("\t -  <%s> \n", LOG_TAG);
#           define LOGI(...) printf("  ");printf(__VA_ARGS__); printf("\t -  <%s> \n", LOG_TAG);
#           define LOGW(...) printf("  * Warning: "); printf(__VA_ARGS__); printf("\t -  <%s> \n", LOG_TAG);
#           define LOGE(...) printf("  *** Error:  ");printf(__VA_ARGS__); printf("\t -  <%s> \n", LOG_TAG);
#           define LOGSIMPLE(...) printf(" ");printf(__VA_ARGS__);
#       endif // ANDROID

#include "android_main.h"

#include <cassert>
#include <iostream>
#include <vector>

#define DEBUG


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
    Free_Status exitAndFree) {

    // change this later when you make multiple calls
    bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    /* grab cmd-line model information */
    static std::vector<ModelRunetime>* models = new std::vector<ModelRunetime>(1);

    (*models)[0].model = srcDIR  + "/" + dlcName;
    (*models)[0].inputs = inputList;
    (*models)[0].outputs = outputNames;

    #ifdef DEBUG
        print_model_runtimes(*models);
    #endif

    if (exitAndFree == Free_Status::free) {
        freeModels(models);
        return "";
    }

    if (firstRun) {
        /* load udo */
        if (use_udo) {
            int udo_load = Snpe_Util_AddOpPackage(udo_path.c_str());
            assert(udo_load == 1);
        }

        /* intialize runtimes */
        intialize_model_runtime(*models);

        /* allocate input buffer */
        allocate_model_input_buffers(*models, first_model_input_sizes, debug);

        /* intialize input buffer */
        intialize_input_buffers_custom(
            *models,
            srcDIR,
            first_model_input_sizes,
            debug);

        /* allocate output buffer */
        allocate_model_output_buffers(*models, first_model_input_sizes[0], debug);
    }

    /* execution stage */
    #ifdef DEBUG
        std::cout << "execution stage\n";
    #endif

    /* tokenizer encode */
    // "What is your favorite color?. Mine is red."
    std::vector<uint32_t> token_collection;
    tokenzie_generate(input_txt, token_collection);

    std::vector<uint32_t> tokens = token_collection;

    uint32_t tot_seq_len = tokens.size();
    uint32_t next_token;

    /* if kv_cache is supposed to be empty, dims should be [0,0,0,0] */
    if (kv_empty) {
        resetKV((datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
    }

    for (uint32_t iteration_num = 0; iteration_num < NUM_ITERS; iteration_num++) {

        #ifdef DEBUG
            printV("tokens", tokens);
        #endif

        /* embedding layer */
        writeEmbedding(
            srcDIR + "/" + embeddingFile, 
            tokens, 
            HIDDEN_SIZE, 
            (datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
        #ifdef DEBUG
            std::cout << "first elements of embedding: " <<
                half_to_float(((ushort*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data())[0]) << "\n";
            std::cout << "preparing inputs\n";
        #endif

        /* generate proper mask and position_ids */
        prepareInputs(
            (float*)((*models)[0].applicationInputBuffers["attention_mask:0"].data()),
            (int*)((*models)[0].applicationInputBuffers["position_ids_1:0"].data()),
            tot_seq_len, iteration_num);
        #ifdef DEBUG
            std::cout << "executing model\n";
        #endif

        /* call model */
        (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);

        /* write kv cache from out to in */
        #ifdef DEBUG
            std::cout << "calling copyKV\n";
        #endif
        copyKV(
            (datatype*)(*models)[0].applicationOutputBuffers["Output_1:0"].data(),
            (datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());

        /* grab next token */
        next_token = ((uint32_t*)(*models)[0].applicationOutputBuffers["Output_1:0"].data())[0];
        #ifdef DEBUG
            std::cout << "next token grabbed: " << next_token << "\n";
        #endif

        /* insert token */
        token_collection.push_back(next_token);
        tokens = std::vector<uint32_t> {next_token};

        tot_seq_len++;
        #ifdef DEBUG
            std::cout << "checkpoint 1\n";
        #endif
    }

    /* tokenizer decode */
    std::string output_txt;
    tokenize_decode(token_collection, output_txt);

    if (exitAndFree == Free_Status::run_and_free) {
        freeModels(models);
    }

    #ifdef DEBUG
        std::cout << "checkpoint 2\n";
    #endif

    return output_txt;
}

// int main() {

//     std::string input_txt = "this test does not matter";
//     std::string output_str;

//     std::string srcDIR = ".";
//     std::vector<std::string> inputList = {
//         "hidden_states_and_kv",
//         "attention_mask",
//         "position_ids_1",
//         "decoder_weights_1",
//         "lm_head_weights_1",
//         "sin",
//         "cos",
//     };
//     std::vector<size_t> first_model_input_sizes = {
//         (1 + 2 * DECODERS) * (4*4 + (MAX_SEQ_LEN * HIDDEN_SIZE) * DATASIZE), 
//         MASK_SIZE, 
//         POS_IDS_SIZE, 
//         TOTAL_DECODER_WEIGHT_SIZE,
//         TOTAL_LM_WEIGHT_SIZE, 
//         SIN_COS_TOTAL_SIZE, 
//         SIN_COS_TOTAL_SIZE
//     };
//     std::string dlcName = "UnifiedPhiDecodersAndLogits.dlc";
//     std::vector<std::string> outputNames = {"Output_1:0"};
//     uint32_t NUM_ITERS = 1;
//     std::string udo_path = "DecodePackage/libs/x86-64_linux_clang/libUdoDecodePackageReg.so";

//     output_str = modelLaunch(
//         input_txt,
//         srcDIR, 
//         inputList, 
//         first_model_input_sizes,
//         dlcName, 
//         outputNames,
//         NUM_ITERS,
//         udo_path,
//         true,
//         Free_Status::run);

//     output_str = modelLaunch(
//         input_txt,
//         srcDIR, 
//         inputList, 
//         first_model_input_sizes,
//         dlcName, 
//         outputNames,
//         NUM_ITERS,
//         udo_path,
//         false,
//         Free_Status::run_and_free);

//     #ifdef DEBUG
//         std::cout << "checkpoint 3\n";
//     #endif

//     return 0;
// }
