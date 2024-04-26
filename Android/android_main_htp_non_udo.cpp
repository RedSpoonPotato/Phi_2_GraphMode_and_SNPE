#include "android_main.h"
// #include "include/android_main.h"

#include <cassert>
#include <iostream>
#include <vector>



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
    const int debugReturnCode) {

    // change this later when you make multiple calls
    bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    /* grab cmd-line model information */
    static std::vector<ModelRunetime>* models = new std::vector<ModelRunetime>(1);

    (*models)[0].model = dlcPath;
    (*models)[0].inputs = inputList;
    (*models)[0].outputs = outputNames;

    #ifdef DEBUG
        print_model_runtimes(*models);
    #endif

    if (exitAndFree == Free_Status::free) {
        freeModels(models);
        return "";
    }

    if (debugReturnCode == 1) { return "1"; }

    if (firstRun) {
        /* load udo */
        if (use_udo) {
            int udo_load = Snpe_Util_AddOpPackage(udo_path.c_str());
            assert(udo_load == 1);
        }

        if (debugReturnCode == 2) { return "2"; }

        /* intialize runtimes */
        intialize_model_runtime(*models, runtime_modes);
        

        if (debugReturnCode == 3) { return "3"; }

        /* allocate input buffer */
        allocate_model_input_buffers(*models, first_model_buffer_sizes, debug, datasize);
        // return x;

        if (debugReturnCode == 4) { return "4"; }

        /* intialize input buffer */
        intialize_input_buffers_custom(
            *models,
            srcDIR,
            first_model_input_sizes,
            debug);

        if (debugReturnCode == 5) { return "5"; }

        /* allocate output buffer */
        allocate_model_output_buffers(*models, first_model_output_buffer_sizes, debug, datasize);
    }

    if (debugReturnCode == 6) { return "6"; }

    /* execution stage */
    #ifdef DEBUG
        std::cout << "execution stage\n";
    #endif

    //test
    // std::cout << "Calling setBuilderOptions AGAIN\n";
    // (*models)[0].snpe = setBuilderOptions((*models)[0].container, (*models)[0].runtime, true);
    // std::cout << "Done\n";


    /* tokenizer encode */
    // "What is your favorite color?. Mine is red."
    #ifdef LLM
    std::vector<uint32_t> token_collection;
    tokenize_generate(input_txt, token_collection);


    if (debugReturnCode == 7) { return "7"; }

    std::vector<uint32_t> tokens = token_collection;

    uint32_t tot_seq_len = tokens.size();
    uint32_t next_token;

    /* if kv_cache is supposed to be empty, dims should be [0,0,0,0] */
    if (kv_empty) {
        resetKV((datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
    }

    if (debugReturnCode == 8) { return "8"; }

    #endif

    for (uint32_t iteration_num = 0; iteration_num < NUM_ITERS; iteration_num++) {

        #ifdef LLM

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

            if (debugReturnCode == 9) { return "9"; }

            /* generate proper mask and position_ids */
            prepareInputs(
                (float*)((*models)[0].applicationInputBuffers["attention_mask:0"].data()),
                (int*)((*models)[0].applicationInputBuffers["position_ids_1:0"].data()),
                tot_seq_len, iteration_num);
            #ifdef DEBUG
                std::cout << "executing model\n";
            #endif

            if (debugReturnCode == 10) { return "10"; }

        #endif


        #ifdef DEBUG
            auto start = std::chrono::high_resolution_clock::now();
        #endif 

        /* call model */
        (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);

        #ifdef DEBUG
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "time to execute model " << (*models)[0].model << ": " << duration << "ms\n";
        #endif


        // remove later
        std::cout << "rebuilding and executing a 2nd time---------\n";
        std::unordered_map<std::string, std::vector<size_t>> new_map;
        new_map["vector:0"] = {3, 1, 1, int(1e3)};
        (*models)[0].snpe = setBuilderOptions((*models)[0].container, (*models)[0].runtime, true, new_map);
        std::string input_name = std::string((*((*models)[0].snpe->getInputTensorNames())).at(0));
        std::cout << "input_name: " << input_name << "\n";
        modifyUserBuffer((*models)[0].inputMap, (*models)[0].applicationInputBuffers,
                (*models)[0].input_user_buff_vec, (*models)[0].snpe, input_name.c_str(), datasize, 0);
        std::string output_name = std::string((*((*models)[0].snpe->getOutputTensorNames())).at(0));
        std::cout << "output_name: " << output_name << "\n";
        modifyUserBuffer((*models)[0].outputMap, (*models)[0].applicationOutputBuffers,
            (*models)[0].output_user_buff_vec, (*models)[0].snpe, output_name.c_str(), datasize, 0);
        std::cout << "executing\n";
        (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);
        std::cout << "finished-----------------------------\n";
        // end of remove

        if (debugReturnCode == 11) { return "11"; }

        #ifdef LLM

            /* write kv cache from out to in */
            #ifdef DEBUG
                std::cout << "calling copyKV\n";
            #endif
            copyKV(
                (datatype*)(*models)[0].applicationOutputBuffers["Output_1:0"].data(),
                (datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());

            if (debugReturnCode == 12) { return "12"; }

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

        #endif
    }

    #ifdef LLM
        /* tokenizer decode */
        std::string output_txt;
        tokenize_decode(token_collection, output_txt);
    #endif

    std::string output_txt = "uninitialized";

    if (exitAndFree == Free_Status::run_and_free) {
        #ifndef LLM
            std::cout << "saving data\n";
            saveBuffer((*models)[0], outputDir);
            std::cout << "success in saving\n";
            output_txt = "success!";
        #endif
        #ifdef DEBUG
            std::cout << "freeing...\n";
        #endif
        freeModels(models);
        #ifdef DEBUG
            std::cout << "done freeing\n";
        #endif
    }

    #ifdef DEBUG
        std::cout << "returning\n";
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
