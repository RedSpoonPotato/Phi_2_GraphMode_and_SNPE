/*
This is an newer copy of android main where we implement model_dict

This is built for running the phi-2 model with the htp
    - i.e. splittign the model into native parts
*/

#include "android_main.h"
// #include "include/android_main.h"
#include "tokenizer.hpp"
#include "embedding.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <map>

#define LLM


std::string modelLaunch(
    const std::string& input_txt,
    const std::map<std::string, zdl::DlSystem::Runtime_t>& runtime_modes,
    const std::string& srcDIR, 
    const std::vector<std::string>& inputList, // does not matter
    const std::map<std::string, std::map<std::string, std::string>>& inputNameToFileMaps,
    const std::map<std::string, std::vector<std::string>>& outputNames,
    const std::map<std::string, std::map<std::string, size_t>>& model_buffer_sizes,
    const std::map<std::string, std::map<std::string, size_t>>& model_output_buffer_sizes,
    const size_t& datasize,
    const bool isTFBuffer,
    const std::string& embeddingFile,
    const std::vector<std::string>& dlcPaths, 
    const uint32_t& NUM_ITERS,
    const std::string& udo_path,
    const bool use_udo,
    const std::string& outputDir,
    const Free_Status exitAndFree,
    const int debugReturnCode) {

    bool simple_exec = true;

    // change this later when you make multiple calls
    bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    /* grab cmd-line model information */
    // static std::vector<ModelRuntime>* models = new std::vector<ModelRuntime>(1);

    int num_models = dlcPaths.size();

    // static std::map<std::string, ModelRuntime>* models = new std::map<std::string, ModelRuntime>();

    /* sets each model with their inputs vector and inputNameToFile map */
    static std::map<std::string, ModelRuntime>* models = modelDictCreator(
        dlcPaths,
        inputNameToFileMaps,
        outputNames
        );

    /* memory buffers */
    const size_t max_seq_len = 2048;
    const size_t hidden_size = 2560;

    // for now, using un-qunatization using between buffers (does not have to be this way)
    // therefore, usiing 32-bit sized buffers rather than 8-bit
    static auto buff_1      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_2      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_3      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_4      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_5      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_3_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_4_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_5_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_6      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_7      =   std::vector<uint8_t>(max_seq_len * max_seq_len * 4);
    static auto buff_8      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_9      =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_10     =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);

    /* linking buffers */
    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        (*models)["P1_reshaped_" + i_str].applicationInputBuffers["residual:0"] = &buff_1;
        (*models)["P1_reshaped_" + i_str].applicationOutputBuffers["hidden_states:0"] = &buff_2;
        (*models)["P1_reshaped_" + i_str].applicationOutputBuffers["query_states:0"] = &buff_3;
        (*models)["P1_reshaped_" + i_str].applicationOutputBuffers["key_states:0"] = &buff_4;
        (*models)["P1_reshaped_" + i_str].applicationOutputBuffers["value_states:0"] = &buff_5;
        (*models)["P1_reshaped_" + i_str].applicationOutputBuffers["feed_forward_hidden_states:0"] = &buff_6;
    }

    (*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"] = &buff_3_1;
    (*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"] = &buff_4_1;
    (*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"] = &buff_7;
    (*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"] = &buff_8;

    (*models)["P2_not_first_reshaped"].applicationInputBuffers["query_states_0:0"] = &buff_3_1;
    (*models)["P2_not_first_reshaped"].applicationInputBuffers["key_states_0:0"] = &buff_4_1;
    (*models)["P2_not_first_reshaped"].applicationInputBuffers["attention_mask:0"] = &buff_7;
    (*models)["P2_not_first_reshaped"].applicationOutputBuffers["attn_weights:0"] = &buff_8;

    (*models)["P3_first_buffered"].applicationInputBuffers["value_states_0:0"] = &buff_5_1;
    (*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"] = &buff_8;
    (*models)["P3_first_buffered"].applicationOutputBuffers["attn_output:0"] = &buff_9;

    (*models)["P3_not_first_reshaped"].applicationInputBuffers["attn_weights_0:0"] = &buff_8;
    (*models)["P3_not_first_reshaped"].applicationInputBuffers["value_states_0:0"] = &buff_5_1;
    (*models)["P3_not_first_reshaped"].applicationOutputBuffers["attn_output:0"] = &buff_8;

    for (size_t i = 0; i < DECODERS; i++) {
        std::string i_str = std::to_string(i);
        (*models)["P4_reshaped_" + i_str].applicationInputBuffers["attn_weights:0"] = &buff_9;
        (*models)["P4_reshaped_" + i_str].applicationInputBuffers["feed_forward_hidden_states:0"] = &buff_6;
        (*models)["P4_reshaped_" + i_str].applicationInputBuffers["residual:0"] = &buff_1;
        (*models)["P4_reshaped_" + i_str].applicationOutputBuffers["decoder_output:0"] = &buff_10;
    }



    /* other buffers */
    static auto kv_cache    =   std::vector<std::vector<float>>(DECODERS);
    size_t kv_datasize = 4;
    for (auto& vec : kv_cache) { vec.resize(max_seq_len * hidden_size * kv_datasize * 2); }

    /*
        TIP: DONT USE VECTORS, use maps or indivdual variables instead
        NOTE: this is not "heavily" optimized (aka not enough reusing)

        P1
            -I 
                residual        (1)
            -O  
                hidden_states   (2) <DO NOT NEED AS OUTPUT, FIX LATER BY CREATING ALL THE DLC's AND Q_DLC's>
                query_states    (3)
                key_states      (4)
                value_states    (5)
                feed_forward_hidden_states (6)

        <Phi-Attention Stuff>
            - I: (3), (4), (5)
            - for now, assume that we must i/o must be diff
            - ignore itermediate buffers for now
            - O: (3-1), (4-1), (5-1)

        P2-1 <NOTE: gotta regenerate the dlc>
            -I
                query_states (3-1)
                key_states (4-1)
                attention_mask (7)
            -O
               attn_weights (8)
        
        P2-2 (no dlc, just implement ur own softmaxing)
            -I 
                attn_weights_in (8)
            -O
                attn_weights (8) do in memory
        
        P2_not_first_reshaped
            -I
                query_states_0 (3-1)
                key_states_0 (4-1)
                attention_mask (7)
            -O
                attn_weights (8)
        
        P3_first_buffered
            -I 
                value_states_0 (5-1)
                attn_weights (8)
            -O
                attn_output (9)
        
        P3_not_first_reshaped
            -I 
                attn_weights_0  (8)
                value_states_0  (5-1)
            -O
                attn_output (9)

        P3_not_first_buffered
            -I
                attn_weights (8)
                value_states (5-1)
            -O
                attn_output (9)
        
        P4_reshaped
            -I
                attn_output_0 (9)
                feed_forward_hidden_states (6)
                residual (1)
            -O
                decoder_output (10)
            
    */
        
    // fill in**


    

    #ifdef DEBUG
        // print_model_runtimes(*models);
    #endif

    if (exitAndFree == Free_Status::free) {
        freeModels(models);
        return "";
    }

    if (debugReturnCode == 1) { return "1"; }

    static bool intialize = true;

    // should only run once
    if (intialize) {
        intialize = false;

        /* load udo */
        if (use_udo) {
            int udo_load = Snpe_Util_AddOpPackage(udo_path.c_str());
            assert(udo_load == 1);
        }

        if (debugReturnCode == 2) { return "2"; }

        /* intialize runtimes */
        intialize_model_runtime(*models, runtime_modes);
        

        /* Verify IO naming */
        assert(verifyModelsIO(*models));
        

        if (debugReturnCode == 3) { return "3"; }

        /* allocate input buffer */
        // allocate_model_input_buffers(*models, model_buffer_sizes, datasize, isTFBuffer, 1);
 

        /* NOTE: define a function that lniks the runtimes to the proper buffers */
        // link buffers

        create_user_buffers(*models, datasize, isTFBuffer);

        if (debugReturnCode == 4) { return "4"; }

        /* intialize input buffer */
        #ifndef LLM
        intialize_input_buffers_custom(
            *models,
            srcDIR);
        #endif

        if (debugReturnCode == 5) { return "5"; }

        /* allocate output buffer */
        // allocate_model_output_buffers(*models, model_output_buffer_sizes, datasize, isTFBuffer, 2);

        zdl::SNPE::SNPEFactory::terminateLogging();
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
    // if (kv_empty) {
    //     resetKV((datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
    // }


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
                (float*)(*(*models)["P1_reshaped_0"].applicationInputBuffers["residual:0"]).data()); 

            if (debugReturnCode == 9) { return "9"; }

            /* generate proper mask and position_ids */
            // prepareInputs(
            //     (float*)((*models)[0].applicationInputBuffers["attention_mask:0"].data()),
            //     (int*)((*models)[0].applicationInputBuffers["position_ids_1:0"].data()),
            //     tot_seq_len, iteration_num);
            #ifdef DEBUG
                std::cout << "executing model\n";
            #endif

            if (debugReturnCode == 10) { return "10"; }

            /*
                **WARNING**
                NOTE: The reshaping functions below may not wokr properly b/c i may have to call create user buffer 
                    again before running
            */

            /* call model */
            if (simple_exec) {
                (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);
            }
            else {
                /* Need ALOT MORE WORK 
                    - processing inbetween buffers?
                    - other stuff
                */
                for (int i = 0; i < DECODERS; i++) {
                    std::string i_str = std::to_string(i);
                    execute(*models, "P1_reshaped_" + i_str);
                    /* implement processing */
                    func(
                        buff_3, buff_3_1, buff_4, buff_4_1, buff_5, buff_5_1,
                        
                        );

                    if (iteration_num == 0) {
                        execute(*models, "P2_1_first_buffered");
                        /* implement softmax*/ // P2_2
                        execute(*models, "P3_first_buffered");
                    }
                    else {
                        execute(*models, "P2_not_first_reshaped");
                        execute(*models, "P3_not_first_reshaped");
                    }
                    execute(*models, "P4_reshaped_" + i_str);
                }
            }

        #else
            auto start = std::chrono::high_resolution_clock::now();
            /* call model */
            (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "time to execute model " << (*models)[0].model << ": " << duration << "ms\n";
        #endif


        // remove later
        // std::cout << "rebuilding and executing a 2nd time---------\n";
        // std::unordered_map<std::string, std::vector<size_t>> new_map;
        // new_map["vector:0"] = {3, 1, 1, int(1e3)};
        // (*models)[0].snpe = setBuilderOptions((*models)[0].container, (*models)[0].runtime, true, new_map);
        // std::string input_name = std::string((*((*models)[0].snpe->getInputTensorNames())).at(0));
        // std::cout << "input_name: " << input_name << "\n";
        // modifyUserBuffer((*models)[0].inputMap, (*models)[0].applicationInputBuffers,
        //         (*models)[0].input_user_buff_vec, (*models)[0].snpe, input_name.c_str(), datasize, 0);
        // std::string output_name = std::string((*((*models)[0].snpe->getOutputTensorNames())).at(0));
        // std::cout << "output_name: " << output_name << "\n";
        // modifyUserBuffer((*models)[0].outputMap, (*models)[0].applicationOutputBuffers,
        //     (*models)[0].output_user_buff_vec, (*models)[0].snpe, output_name.c_str(), datasize, 0);
        // std::cout << "executing\n";
        // (*models)[0].snpe->execute((*models)[0].inputMap, (*models)[0].outputMap);
        // std::cout << "finished-----------------------------\n";
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

            /* reshape */
            // reshape stuff for the next run
            reshapeModels(*models, "P2_not_first_reshaped", 
                {
                    {"query_states_0:0", {1, 32, 80}},
                    {"key_states_0:0", {tot_seq_len, 32, 80}},
                    {"attention_mask:0", {tot_seq_len}}
                }, datasize);
            reshapeModels(*models, "P3_not_first_reshaped",
                {
                    {"attn_weights_0:0", {tot_seq_len, 32, 1}},
                    {"value_states_0:0", {tot_seq_len, 32, 80}}
                }, datasize);
            for (size_t i = 0; i < DECODERS; i++) {
                reshapeModels(*models, "P1_reshaped_" + std::to_string(i),
                    {{"residual:0", {1, hidden_size}}}, datasize);
                reshapeModels(*models, "P4_reshaped_" + std::to_string(i),
                {
                    {"attn_output_0:0", {1, hidden_size}},
                    {"feed_forward_hidden_states:0", {1, hidden_size}},
                    {"residual:0", {1, hidden_size}}
                }, datasize);
            }
            
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

            //testing
            // std::string outPath = "./output.raw";
            // std::ostringstream path;
            // path << outPath;
            // SaveUserBuffer(path.str(), (*models)[0].applicationOutputBuffers.at("matmul_out:0"));

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
