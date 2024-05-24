/*
This is an newer copy of android main where we implement model_dict

This is built for running the phi-2 model with the htp
    - i.e. splittign the model into native parts
*/

#include "android_main.h"
// #include "include/android_main.h"
#include "tokenizer.hpp"
#include "embedding.hpp"
#include "operations.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>

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
    const uint32_t& max_iterations,
    const std::string& udo_path,
    const bool use_udo,
    const std::string& outputDir,
    const Free_Status exitAndFree,
    const int debugReturnCode,
    const uint32_t end_token_id,
    const bool use_end_token_id) {

    bool simple_exec = false;

    static bool first_run = true;


    // change this later when you make multiple calls
    // bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    // mgiht also fail if max_iterations equals exactly MAX_SEQ_LEN (u should test)
    if (max_iterations > MAX_SEQ_LEN) {
        return "Error, max_iterations(" + std::to_string(max_iterations) + ") greater than MAX_SEQ_LEN("
         + std::to_string(MAX_SEQ_LEN) + ")\n";
    }

    /* grab cmd-line model information */
    // static std::vector<ModelRuntime>* models = new std::vector<ModelRuntime>(1);

    // int num_models = dlcPaths.size();

    // static std::map<std::string, ModelRuntime>* models = new std::map<std::string, ModelRuntime>();

    /* sets each model with their inputs vector and inputNameToFile map */
    static std::map<std::string, ModelRuntime>* models = modelDictCreator(
        dlcPaths,
        inputNameToFileMaps,
        outputNames
    );

    /* memory buffers */
    // const size_t max_seq_len = 2048;
    // const size_t hidden_size = 2560;

    // for now, using un-qunatization using between buffers (does not have to be this way)
    // therefore, using 32-bit sized buffers rather than 8-bit
    static auto buff_1      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_2      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_3      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_4      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_5      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    // static auto buff_3_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    // static auto buff_4_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    // static auto buff_5_1    =   std::vector<uint8_t>(max_seq_len * hidden_size * 4);
    static auto buff_6      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_7      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_8      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_9      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_10     =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);

    /* NOTE: REMEBER TO FILL THESE UP */
    static auto sin_cached  = std::vector<uint8_t>(SIN_COS_BUFF_SIZE);
    static auto cos_cached  = std::vector<uint8_t>(SIN_COS_BUFF_SIZE);
    
    // might be a way to optimize the 2 below (just use sin_cached)
    static auto sin_buff    = std::vector<uint8_t>(SIN_COS_BUFF_SIZE);
    static auto cos_buff    = std::vector<uint8_t>(SIN_COS_BUFF_SIZE);

    // static auto key_cache   = std::vector<uint8_t>(DECODERS * MAX_SEQ_LEN * HIDDEN_SIZE * sizeof(uint8_t));
    // static auto value_cache = std::vector<uint8_t>(DECODERS * MAX_SEQ_LEN * HIDDEN_SIZE * sizeof(uint8_t));

    // could optimize by ensuring the max size of each buffer is bounded to a lower size
    static auto query_rot_buff  = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE);
    static auto query_pass_buff = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE);
    static auto key_rot_buff    = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE);
    static auto key_pass_buff   = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE);

    static auto k_cache = std::vector<std::vector<uint8_t>>(DECODERS);
    static auto v_cache = std::vector<std::vector<uint8_t>>(DECODERS);
    for (auto& vec : k_cache) { vec.resize(MAX_SEQ_LEN * HIDDEN_SIZE); }
    for (auto& vec : v_cache) { vec.resize(MAX_SEQ_LEN * HIDDEN_SIZE); }

    /* shapes */
    auto query_shape         = std::vector<size_t>();
    auto key_shape           = std::vector<size_t>();
    auto value_shape         = std::vector<size_t>();
    auto sin_cached_shape    = std::vector<size_t>();
    auto cos_cached_shape    = std::vector<size_t>();
    auto sin_shape           = std::vector<size_t>();
    auto cos_shape           = std::vector<size_t>();
    auto query_rot_buff_dims = std::vector<size_t>();
    auto query_pass_buff_dims = std::vector<size_t>();
    auto key_rot_buff_dims   = std::vector<size_t>();
    auto key_pass_buff_dims  = std::vector<size_t>();
    auto key_cache_shape     = std::vector<size_t>();
    auto value_cache_shape   = std::vector<size_t>();

    const int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)
    std::vector<int> position_ids;


// REMEMBER: TO LATER ADD QUANTIZATION IN BETWWEN STEPS

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

        linkBuffers(models, buff_1, buff_2, buff_3, buff_4, buff_5, buff_6, buff_7, buff_8, buff_9, buff_10);
        loadAndQuantize(sin_cached, srcDIR + "/sin.bin");
        loadAndQuantize(cos_cached, srcDIR + "/cos.bin");

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
    for (int i = 0; i < token_collection.size(); i++) { position_ids.push_back(i); }


    if (debugReturnCode == 7) { return "7"; }

    std::vector<uint32_t> tokens = token_collection;

    uint32_t tot_seq_len = tokens.size();
    uint32_t next_token;

    query_shape = {1, 32, tot_seq_len, 80};
    key_shape   = {1, 32, tot_seq_len, 80};
    value_shape = {1, 32, tot_seq_len, 80};

    /* if kv_cache is supposed to be empty, dims should be [0,0,0,0] */
    // if (kv_empty) {
    //     resetKV((datatype*)(*models)[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
    // }


    if (debugReturnCode == 8) { return "8"; }

    #endif

    for (uint32_t iteration_num = 0; iteration_num < max_iterations; iteration_num++) {
        
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
                    // NEED TO SET THE SHAPES (MAKE THEM ALL 4D)
                    sin_cached_shape = {MAX_SEQ_LEN, 32};
                    cos_cached_shape = {MAX_SEQ_LEN, 32};

                    DynamicTruncationAndConcatentation(
                        buff_3.data(),
                        buff_4.data(),
                        buff_5.data(),
                        sin_cached.data(), // (11, 32) - (12, 32)
                        cos_cached.data(),
                        sin_buff.data(),
                        cos_buff.data(),
                        k_cache[i].data(), // (1, 32, 0, 80)<basically 0> - (1, 32, 11, 80)
                        v_cache[i].data(), // same as key_cache
                        query_rot_buff.data(),
                        query_pass_buff.data(),
                        key_rot_buff.data(),
                        key_pass_buff.data(),
                        query_shape, // set
                        key_shape, // set
                        value_shape, // set
                        sin_cached_shape, // set
                        cos_cached_shape, // set
                        sin_shape, // wil be set
                        cos_shape, // will be set
                        key_cache_shape, // intially it should not be set
                        value_cache_shape, // intially it should not be set
                        query_rot_buff_dims, // will be set
                        query_pass_buff_dims, // will be set
                        key_rot_buff_dims, // will be set
                        key_pass_buff_dims, // will be set
                        rotary_emb_dim, // set
                        position_ids); // set

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

            if (debugReturnCode == 12) { return "12"; }

            /* grab next token */
            // the line below is old, FIX IT
            // next_token = ((uint32_t*)(*models)[0].applicationOutputBuffers["Output_1:0"].data())[0];
            #ifdef DEBUG
                std::cout << "next token grabbed: " << next_token << "\n";
            #endif

            /* insert token */
            token_collection.push_back(next_token);
            tokens = std::vector<uint32_t> {next_token};

            tot_seq_len++;

            if (use_end_token_id && next_token == end_token_id) {
                break; 
            }

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
            // QUESTION: DOES THE CODE BELOW ONLY NEED TO RUN ONCE?
            for (size_t i = 0; i < DECODERS; i++) {
                reshapeModels(*models, "P1_reshaped_" + std::to_string(i),
                    {{"residual:0", {1, HIDDEN_SIZE}}}, datasize);
                reshapeModels(*models, "P4_reshaped_" + std::to_string(i),
                {
                    {"attn_output_0:0", {1, HIDDEN_SIZE}},
                    {"feed_forward_hidden_states:0", {1, HIDDEN_SIZE}},
                    {"residual:0", {1, HIDDEN_SIZE}}
                }, datasize);
            }
            position_ids.resize(1);
            position_ids[0] = tot_seq_len - 1;
            query_shape = {1, 32, 1, 80};
            key_shape   = {1, 32, 1, 80};
            value_shape = {1, 32, 1, 80};

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
