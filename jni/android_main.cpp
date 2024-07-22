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
#include "quant_params.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>

#define LLM

std::string modelLaunch(
    const std::string& input_txt,
    const std::map<std::string, zdl::DlSystem::Runtime_t>& runtime_modes,
    const size_t& datasize,
    const bool isTFBuffer,
    const std::set<std::pair<std::string, std::string>>& ModelNameAndPaths, // abs paths
    const std::map<std::string, std::string>& otherPaths, // abs path of sin, cos, embeddingFIle
    const uint32_t& max_iterations, 
    const Free_Status exitAndFree,
    const int debugReturnCode,
    const uint32_t end_token_id,
    const bool use_end_token_id
) {

    static bool first_run = true;

    // change this later when you make multiple calls
    // bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    // might also fail if max_iterations equals exactly MAX_SEQ_LEN (u should test)
    if (max_iterations > MAX_SEQ_LEN) {
        return "Error, max_iterations(" + std::to_string(max_iterations) + ") greater than MAX_SEQ_LEN("
         + std::to_string(MAX_SEQ_LEN) + ")\n";
    }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT A--\n";
    #endif

    /* sets each model with their inputs vector and inputNameToFile map */
    // static std::map<std::string, ModelRuntime>* models = modelDictCreator(ModelNameAndPaths);
    static auto models = std::map<std::string, ModelRuntime>(); // temp

    Tokenizer* tokenizer_ptr;

    // restore
    // if (exitAndFree == Free_Status::free) {
    //     freeModels(models);
    //     delete tokenizer_ptr;
    //     return "";
    // }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT B--\n";
    #endif

    /* memory buffers */

    // NOTE: Could optimize storing as fp16 instead, but would not have that much memory
    // layernorm buffers (stored as fp32)
    static auto layernorm_weights   = std::vector<std::vector<float>>(DECODERS);
    static auto layernorm_biases    = std::vector<std::vector<float>>(DECODERS);
    for (int i = 0; i < DECODERS; i++) { 
        layernorm_weights[i].resize(HIDDEN_SIZE);
        layernorm_biases[i].resize(HIDDEN_SIZE);
    }

    static auto final_layernorm_weight  = std::vector<float>(HIDDEN_SIZE);
    static auto final_layernorm_bias    = std::vector<float>(HIDDEN_SIZE);

    const int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)

    // NOTE: CHANGE the sizes (4 or 1) to a variable that may be changed to 2 (fp16) if android c++ is capable
    size_t quant_size = 4; // **NOTE: CHANGE THIS BACK TO 1 WHEN GOING TO ANDROID
    size_t float_size = 4; // LOOK INTO CHANGING THIS TO SAVE MEMORY (change to 2)
    static auto buff_1      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size, 0);
    // static auto buff_2      =   std::vector<uint8_t>(MAX_SEQ_LEN * INTERMEDIATE_SIZE * float_size);
    static auto buff_3      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4, 0);
    static auto buff_4      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4, 0);
    static auto buff_5      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size, 0);
    static auto buff_6      =   std::vector<uint8_t>(MAX_SEQ_LEN * INTERMEDIATE_SIZE * quant_size, 0);
    static auto buff_7      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4, 0);
    static auto buff_8      =   std::vector<uint8_t>(rotary_emb_dim *  MAX_SEQ_LEN * HIDDEN_SIZE * quant_size, 0);
    // static auto buff_9      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size);
    // static auto buff_10     =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size);

    // explicit references
    auto mask_ptr = (UNQUANT_TYPE*)buff_7.data();

    /* NOTE: REMEBER TO FILL THESE UP */
    static auto sin_cached  = std::vector<uint8_t>(SIN_COS_BUFF_SIZE * quant_size);
    static auto cos_cached  = std::vector<uint8_t>(SIN_COS_BUFF_SIZE * quant_size);
    
    // might be a way to optimize the 2 below (just use sin_cached)
    static auto sin_buff    = std::vector<uint8_t>(SIN_COS_BUFF_SIZE * quant_size);
    static auto cos_buff    = std::vector<uint8_t>(SIN_COS_BUFF_SIZE * quant_size);

    // could optimize by ensuring the max size of each buffer is bounded to a lower size
    static auto query_rot_buff  = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size);
    static auto query_pass_buff = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size);
    static auto key_rot_buff    = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size);
    static auto key_pass_buff   = std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size);

    static auto k_cache = std::vector<std::vector<uint8_t>>(DECODERS);
    static auto v_cache = std::vector<std::vector<uint8_t>>(DECODERS);
    for (auto& vec : k_cache) { vec.resize(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size); }
    for (auto& vec : v_cache) { vec.resize(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size); }

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
    auto key_cache_shape     = std::vector<std::vector<size_t>>(DECODERS);
    auto value_cache_shape   = std::vector<std::vector<size_t>>(DECODERS);

    // not for DynamicTruncation()
    auto residual_shape         = std::vector<size_t>();
    auto attn_weights_shape     = std::vector<size_t>();
    auto attn_output_shape      = std::vector<size_t>();
    auto decoder_output_shape   = std::vector<size_t>();
    auto mask_shape             = std::vector<size_t>();

    {
        // exit(0);
        // remove later
            // size_t dum = 0;
            // int* c = (int*)malloc(sizeof(int) * 500000000);
            // for (size_t i = 0; i < 100000000; i++) {
            //     dum++;
            // }
            // std::cout << "dum: " << dum << "\n";
            // free(c);
            // exit(0);
    }

    // quantization params
    std::vector<std::map<std::string, quantParams>> decoderQuantParams = quantizationParams();

    std::vector<int> position_ids;

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT C--\n";        
    #endif

    bool quantize = (quant_size == 1);
    static bool intialize = true;

    // should only run once
    if (intialize) {
        intialize = false;

        // temp
        modelDictCreatorStatic(models, ModelNameAndPaths);

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT D--\n";
        #endif

        tokenizer_ptr = new Tokenizer(
            otherPaths.at("token_vocab"),
            otherPaths.at("token_merges")
        );

        // for DECODERS=3 unqunatized, ~1GB

        /* intialize runtimes */
        // intialize_model_runtime(*models, runtime_modes);
        intialize_model_runtime(models, runtime_modes); // temp

        // for DECODERS=3 unqunatized, ~2.412GB

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT E--\n";
        #endif

        // linkBuffers(models, buff_1, buff_3, buff_4, buff_5, buff_6, buff_7, buff_8);
        linkBuffers(&models, buff_1, buff_3, buff_4, buff_5, buff_6, buff_7, buff_8); // temp

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT F--\n";
        #endif

        // might need to be changed to account for some models that will and won't be qunatized
        // maybe shouild attach a quantization flag to each modelRuneitme
        // create_user_buffers(*models, datasize, isTFBuffer);
        create_user_buffers(models, datasize, isTFBuffer); // temp (restore for testing later)

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT G1--\n";
        #endif
        // put these back in later
        loadAndQuantize(sin_cached, otherPaths.at("sin"), quantize);
        loadAndQuantize(cos_cached, otherPaths.at("cos"), quantize);

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT G2--\n";
        #endif

        loadLayerNorms(layernorm_weights, layernorm_biases, otherPaths, final_layernorm_weight, final_layernorm_bias);

        zdl::SNPE::SNPEFactory::terminateLogging();
    }

    /* execution stage */
    #ifdef DEBUG
        std::cout << "execution stage\n";
    #endif

    /* tokenizer encode */
    // "What is your favorite color?. Mine is red."

    std::vector<uint32_t> tot_token_seq;
    tot_token_seq = tokenizer_ptr->generate(input_txt);

    // {
    //     // REMOVE LATER
    //     // tot_token_seq = std::vector<uint32_t>{(uint32_t)5, (uint32_t)5};
    //     tot_token_seq = std::vector<uint32_t>();
    //     for (int i =0; i < 42; i++) {
    //         tot_token_seq.push_back((uint32_t)5);
    //     }
    // }

    for (int i = 0; i < tot_token_seq.size(); i++) { position_ids.push_back(i); }

    #ifdef DEBUG
        std::cout << "tokens: ";
        for (const auto i : tot_token_seq) {std::cout << i << ", ";}
        std::cout << "\n";
        std::cout << "\t\t\t--CHECKPOINT H--\n";
    #endif

    std::vector<uint32_t> token_seq = tot_token_seq; // will be size 11 on first run, 1 on runs after

    uint32_t next_token;

    //  these are intially the same
    size_t tot_seq_len  = tot_token_seq.size();
    size_t seq_len      = token_seq.size();

    /* NEED TO IMPLEMENT INITIAL RESHAPING */
    std::cout << "calling reshapeStuff\n";

    // reshapeModels(*models, "P2_reshaped",
    // {
    //     {"query_states:0", {32, 1, 80}},
    //     {"key_states:0", {32, 13, 80}}
    // }, sizeof(UNQUANT_TYPE));
    // execute(*models, "P2_reshaped", quantize);

    // reshapeModels(*models, "P3_reshaped",
    // {
    //     {"attn_weights:0", {32, 1, 13}},
    //     {"value_states:0", {32, 13, 80}}
    // }, sizeof(QUANT_TYPE));
    // execute(*models, "P3_reshaped", false);

    
    // reshapeModels(*models, "P3_not_first_reshaped",
    // {
    //     {"attn_weights:0", {32, seq_len, seq_len}},
    //     {"value_states:0", {32, seq_len, 80}}
    // }, sizeof(QUANT_TYPE));
    // std::cout << "running\n";
    // execute(*models, "P3_not_first_reshaped", false);
    
    // reshapeModels(*models, "P3_first_buffered",
    // {
    //     {"attn_weights:0", {32, 2, seq_len}},
    //     {"value_states:0", {32, seq_len, 80}}
    // }, sizeof(QUANT_TYPE));
    // std::cout << "running\n";
    // execute(*models, "P3_first_buffered", false);

    std::cout << "done\n";
    // exit(0);

            {
            // exit(0);
            // remove later
                // size_t dum = 0;
                // int* c = (int*)malloc(sizeof(int) * 500000000);
                // for (size_t i = 0; i < 100000000; i++) {
                //     dum++;
                // }
                // std::cout << "dum: " << dum << "\n";
                // free(c);
                // exit(0);
        }

    std::cout << "\n\n CALLING reshapeInitial\n\n";
    {
            // remove
            // exit(0);
            CLOCK_INIT
            std::cout << "resetting\n";

            // auto it = models->find("P1_Q_reshaped_layer_1");
            // assert(it != models->end());
            // models->erase(it);

            // models["P1_Q_reshaped_layer_1"].container.reset();

            // std::string path_name = "./fp16_test/model_split/dlc/model_P1_Q_reshaped_layer_0.dlc";
            // models["Final_LM_Head"].container.reset();
            // models["Final_LM_Head"].container = loadContainerFromFile(path_name);

            // exit(0);

            stall();

            std::cout << "reshaping\n";

            /// 
            size_t seq_len = 900;

            // exit(0);

            reshapeModels(models, "P1_QKV_reshaped_no_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, HIDDEN_SIZE}}
            }, sizeof(QUANT_TYPE));
            reshapeModels(models, "P2_reshaped",
            {
                {"query_states:0", {32, seq_len, 80}},
                {"key_states:0", {32, seq_len, 80}}
            }, sizeof(QUANT_TYPE)
            );
            reshapeModels(models, "P3_reshaped",
                {
                    {"attn_weights:0", {32, seq_len, seq_len}},
                    {"value_states:0", {32, seq_len, 80}}
                }, sizeof(QUANT_TYPE)
            );
            reshapeModels(models, "FC1_reshaped_no_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, INTERMEDIATE_SIZE}}
            }, sizeof(QUANT_TYPE));
            reshapeModels(models, "FC2_reshaped_no_bias",
            {
                {"gelu_out:0", {1, seq_len, INTERMEDIATE_SIZE}},
                {"weights:0", {1, INTERMEDIATE_SIZE, HIDDEN_SIZE}}
            }, sizeof(QUANT_TYPE));
            reshapeModels(models, "FinalLMHead_reshaped_no_bias",
            {
                {"final_input:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, VOCAB_SIZE}}
            }, sizeof(QUANT_TYPE));
            stall();

            // exit(0);

            execute(models, "P2_reshaped", false);
            execute(models, "P3_reshaped", false);
            execute(models, "FC1_reshaped_no_bias", false);
            execute(models, "FC2_reshaped_no_bias", false);
            execute(models, "FinalLMHead_reshaped_no_bias", false);
            execute(models, "P1_QKV_reshaped_no_bias", false);


            stall();

            exit(0);
            /// 
            
            // create buffer construct
            // zdl::DlSystem::UserMemoryMap map;
            // static auto temp_buff = std::vector<uint8_t>(rotary_emb_dim *  MAX_SEQ_LEN * HIDDEN_SIZE * 4, 0);

            // stall();
            // exit(0);
            
            // create_user_buffers_with_memory(
            //     models,
            //     map,
            //     temp_buff,
            //     datasize,
            //     isTFBuffer,
            //     "P2_reshaped"
            // );
            // models["P2_reshaped"].snpe->registerMemoryMappedBuffers(map);


            CLOCK_START
            
            // create_user_buffers_with_memory(
            //     models,
            //     map,
            //     temp_buff,
            //     datasize,
            //     isTFBuffer,
            //     "P3_reshaped"
            // );

            CLOCK_END

            // models["P3_reshaped"].snpe->registerMemoryMappedBuffers(map);

            // create_user_buffers(models, datasize, isTFBuffer, "P2_reshaped");
            
            stall();
            // exit(0);

            // auto temp_buff_ptr = &temp_buff;
            // auto map_ptr = &map;

            // temp_buff_ptr = nullptr;
            // map_ptr = nullptr;

            // reshapeModels(models, "P2_reshaped",
            // {
            //     {"query_states:0", {32, 1500, 80}},
            //     {"key_states:0", {32, 1500, 80}}
            // }, sizeof(QUANT_TYPE),
            // temp_buff_ptr,
            // map_ptr
            // );

            // stall();

            

            CLOCK_START
            size_t dim_size  = 1500;
            
            reshapeModels(models, "P3_reshaped",
                {
                    {"attn_weights:0", {32, dim_size, dim_size}},
                    {"value_states:0", {32, dim_size, 80}}
                }, sizeof(QUANT_TYPE)
            );
            CLOCK_END
            stall();
            // exit(0);
            // execute(models, "P3_reshaped", false);
            stall();

            // exit(0);

            dim_size = 800;
            CLOCK_START
            reshapeModels(models, "P3_reshaped",
                {
                    {"attn_weights:0", {32, dim_size, dim_size}},
                    {"value_states:0", {32, dim_size, 80}}
                }, sizeof(QUANT_TYPE)
            );
            CLOCK_END
            stall();
            exit(0);
            execute(models, "P3_reshaped", false);



            std::cout << "done reshaping\n";

            stall();
            exit(0);

            execute(models, "P2_reshaped", false);

            stall();
            exit(0);
            

            // execute(models, "P1_Q_reshaped_layer_0", false);
            // execute(models, "P1_Q_reshaped_layer_1", false);
            // execute(models, "P1_Q_reshaped_layer_2", false);

            // execute(models, "P1_K_reshaped_layer_0", false);
            // execute(models, "P1_K_reshaped_layer_1", false);
            // execute(models, "P1_K_reshaped_layer_2", false);

            // execute(models, "P1_V_reshaped_layer_0", false);
            // execute(models, "P1_V_reshaped_layer_1", false);
            // execute(models, "P1_V_reshaped_layer_2", false);

            // execute(models, "P1_FC1_reshaped_layer_0", false);
            // execute(models, "P1_FC1_reshaped_layer_1", false);
            // execute(models, "P1_FC1_reshaped_layer_2", false);

            // stall();

            // std::cout << "reshaping\n";
            // reshapeModels(models, "MatmulTest",
            // {
            //     {"query_states:0", {32, 1500, 80}},
            //     {"key_states:0", {32, 80, 1500}}
            // }, sizeof(QUANT_TYPE));
            // std::cout << "done reshaping\n";


            // stall();

            // execute(models, "MatmulTest", false);
            // execute(models, "MatmulTest", false);
            // execute(models, "MatmulTest", false);

            // execute(models, "P3_reshaped", false);
            // execute(models, "P3_reshaped", false);
            // execute(models, "P3_reshaped", false);

            // execute(models, "P4_1_reshaped_layer_0", false);
            // execute(models, "P4_1_reshaped_layer_1", false);
            // execute(models, "P4_1_reshaped_layer_2", false);

            // execute(models, "P4_2_reshaped", false);
            // execute(models, "P4_2_reshaped", false);
            // execute(models, "P4_2_reshaped", false);


            // models->at("P1_Q_reshaped_layer_1").container.reset();
            // models->at("P1_Q_reshaped_layer_1").container.release();
            // auto ptr = models->at("P1_Q_reshaped_layer_1").container.release();
            // delete ptr;
            // models->at("P1_Q_reshaped_layer_1").snpe.reset();
            std::cout << "finished\n";
            // exit(0);
            // execute(*models, "P1_Q_reshaped_layer_1", quantize);
            // execute(*models, "P2_reshaped", false);


            // stall();

            // execute(models, "Final_LM_Head", false);

            stall();

            

            // execute(models, "Final_LM_Head", false);


            // stall();


            exit(0);

    }

    // temp
    /*

    reshapeInitial(models, seq_len, tot_seq_len, sizeof(QUANT_TYPE), sizeof(UNQUANT_TYPE)); // restore
    // reshapeInitial(models, seq_len+1, tot_seq_len+1, sizeof(QUANT_TYPE), sizeof(UNQUANT_TYPE)); // remove later
    // exit(0);
    std::cout << "\n\n FINISHED CALLING reshapeIntial\n\n";

            {
            // exit(0);
            // remove later
                // size_t dum = 0;
                // int* c = (int*)malloc(sizeof(int) * 500000000);
                // for (size_t i = 0; i < 100000000; i++) {
                //     dum++;
                // }
                // std::cout << "dum: " << dum << "\n";
                // free(c);
                // exit(0);
        }

    for (uint32_t iteration_num = 0; iteration_num < max_iterations; iteration_num++) {

        #ifdef DEBUG
            printV("token_seq", token_seq);
        #endif

        // embedding layer
        writeEmbedding( 
            otherPaths.at("embedding"),
            token_seq, 
            HIDDEN_SIZE, 
            (uint16_t*)buff_8.data() // using an intermediate buffer
        ); 
        // copy convert 16 bit data to 32 bit inside of the buffer
        assert(float_size == 4);
        fp16_to_fp32(
            (ushort*)buff_8.data(), // using an intermediate buffer
            (float*)buff_1.data(), 
            {(uint32_t)seq_len, HIDDEN_SIZE}
        );

        #ifdef DEBUG
            printN("embedding 32 bit", (float*)buff_1.data(), N_PRINT, false);
        #endif

        // set decoder_output as input to layernorm
        residual_shape = {1, seq_len, HIDDEN_SIZE};

        // generate mask
        assert(float_size == 4); // otherwise chagne the casting of the mask pointer
        std::cout << "calling mask\n";
        mask_shape = prepareMask(
            mask_ptr,
            tot_seq_len, 
            iteration_num,
            false
            // (float*)buff_8.data() // intermediate buffer
        );

        // {
        //     // remove later
        //     printTensorColumn(
        //         "attention_mask columns",
        //         (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"]->data(),
        //         {seq_len, seq_len},
        //         2
        //     );
        // }

        #ifdef DEBUG
            // printN(
            //     "prepared Mask", 
            //     (float*)((*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"]->data()),
            //     N_PRINT,
            //     false
            // );
            std::cout << "executing model\n";
        #endif

        // call model
        for (int i = 0; i < DECODERS; i++) {
            std::string i_str = std::to_string(i);

            printN("first row of residual: ", (float*)buff_1.data(), 10, false);
            printN("first row of output buffer before layernorm: ", (float*)buff_1.data(), 10, false);
            printN("second row of residual: ", (float*)buff_1.data() + HIDDEN_SIZE, 10, false);
            printN("second row of output buffer before layernorm: ", (float*)buff_1.data() + HIDDEN_SIZE, 10, false);

            // use decoder_output as a input to layernorm
            layernorm_Nd_32f(
                (float*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), // buff_1
                layernorm_weights[i].data(), 
                layernorm_biases[i].data(), 
                (float*)(*models)["P1_Q_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(), // buff_8
                residual_shape,
                1e-5
            );

            // {
            //     // remove later
            //     if (iteration_num == 1 && i == 0) {
            //         loadFileAndDontResize(
            //             *(*models)["P1_Q_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"],
            //             "/home/kernal1/QM_Sandbox/htp/fp16_test/model_split/layer_norm_2nd_iteration.bin"
            //         );
            //     }
            // }

            printN("first row of output buffer after layernorm: ", (float*)buff_8.data(), 10, false);
            printN("second row of output buffer after layernorm: ", (float*)buff_8.data() + HIDDEN_SIZE, 10, false);

            #ifdef DEBUG
            // printTensor(
            //     "residual", 
            //     (float*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), 
            //     residual_shape
            // );
            // printTensor(
            //     "LayerNorm #" + i_str, 
            //     (float*)(*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
            //     residual_shape
            // );
            #endif

            if (quantize) {
                // need to quantize (not sure if fixed or unfixed)
                FloatToTfN(
                    (*models)["P1_Q_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                    decoderQuantParams[i].at("hidden_states"),
                    false,
                    (float*)(*models)["P1_Q_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                    seq_len * HIDDEN_SIZE,
                    8
                );
            
                // remove later, testing quantizing
                // {
                //     TfNToFloat(
                //         (float*)buff_6.data(),
                //         (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                //         decoderQuantParams[i].at("hidden_states"),
                //         seq_len * HIDDEN_SIZE,
                //         8
                //     );

                //     printN("hidden_states after quantization test (should match prev)", (float*)buff_6.data(), N_PRINT, true);

                //     FloatToTfN(
                //         (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                //         decoderQuantParams[i].at("hidden_states"),
                //         false,
                //         (float*)buff_6.data(),
                //         seq_len * HIDDEN_SIZE,
                //         8
                //     );
                // }
            }

            {
                // remove later
                // printTensorColumn(
                //     "\nquery_states columns befoe running P1_1",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );
                // exit(0);

                // printTensorColumn(
                //         "\nvalue_states columns before P1_1 execution",
                //         (float*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                // );
            }



            // execute(*models, "P1_1_reshaped_layer_" + i_str, quantize); //put back in
            execute(*models, "P1_Q_reshaped_layer_" + i_str, quantize);
            execute(*models, "P1_K_reshaped_layer_" + i_str, quantize);
            execute(*models, "P1_V_reshaped_layer_" + i_str, quantize);
            execute(*models, "P1_FC1_reshaped_layer_" + i_str, quantize);


            {
                // // rremove later
                // printTensorColumn(
                //         "\nvalue_states columns after P1_1 execution",
                //         (float*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                // );
                //     printTensorColumn(
                //         "\nquery_states columns after P1_1 executio",
                //         (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                //     );
                //     printTensorColumn(
                //         "\nkey_states columns after P1_1 execution",
                //         (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                //     );
            }

            if (quantize) {
                // need to dequantize (not sure if fixed or unfixed)
                TfNToFloat(
                    (float*)buff_8.data(),
                    (*models)["P1_FC1_reshaped_layer_" + i_str].applicationOutputBuffers["fc1_out:0"]->data(),
                    decoderQuantParams[i].at("fc1_out"),
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
            }
            else {
                // this a test for seeing if gelu works in memory.
                // You can remove this block and replace it with letting gelu input point to buff_6 if unquantized instead
                copyTensor((UNQUANT_TYPE*)buff_6.data(), (UNQUANT_TYPE*)buff_8.data(), {seq_len, INTERMEDIATE_SIZE});
            }

            printTensor(
                "gelu in (buff_6 before)",
                (float*)buff_6.data(),
                {seq_len, INTERMEDIATE_SIZE}
            );
            
            // execute(*models, "gelu", false); // disabled b/c gelu dlc cannot do in-memory
            // in memory
            NewGELU(
                (UNQUANT_TYPE*)(*models)["P1_2_reshaped_layer_" + i_str].applicationInputBuffers["gelu_fc1_out:0"]->data(),
                (UNQUANT_TYPE*)(*models)["P1_2_reshaped_layer_" + i_str].applicationInputBuffers["gelu_fc1_out:0"]->data(),
                {seq_len, INTERMEDIATE_SIZE}
            );

            {
                printTensor(
                    "gelu out (buff_8 after)",
                    (float*)buff_8.data(),
                    {seq_len, INTERMEDIATE_SIZE}
                );
            }


            if (quantize) {
                // need to quantize (not sure if fixed or unfixed)
                FloatToTfN(
                    buff_8.data(),
                    decoderQuantParams[i].at("gelu_fc1_out"),
                    true,
                    (float*)buff_8.data(),
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
            }
            execute(*models, "P1_2_reshaped_layer_" + i_str, quantize);

            // implement processing
            // NEED TO SET THE SHAPES (MAKE THEM ALL 4D)
            sin_cached_shape = {MAX_SEQ_LEN, 32};
            cos_cached_shape = {MAX_SEQ_LEN, 32};

            
            {
                // remove later
                // printTensorColumn(
                //     "\nquery_states columns before DynTruncation()",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );
                // printTensorColumn(
                //     "\nkey_states columns before DynTruncation()",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );

                // printTensorColumn(
                //         "\nvalue_states columns before DynTruncation",
                //         (float*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                // );
            }


            std::cout << "\t\t\tCalling DynamicTruncationAndConcatentation() #" << i_str << "\n";
            // insure buff_8 is not being used at this point
            DynamicTruncationAndConcatentation(
                seq_len,
                (QUANT_TYPE*)buff_8.data(), // temp_buff
                (QUANT_TYPE*)buff_3.data(), // query
                (QUANT_TYPE*)buff_4.data(), // key
                (QUANT_TYPE*)buff_5.data(), // value
                (QUANT_TYPE*)sin_cached.data(), // (11, 32) - (12, 32)
                (QUANT_TYPE*)cos_cached.data(),
                (QUANT_TYPE*)sin_buff.data(),
                (QUANT_TYPE*)cos_buff.data(),
                (QUANT_TYPE*)k_cache[i].data(), // (1, 32, 0, 80)<basically 0> - (1, 32, 11, 80)
                (QUANT_TYPE*)v_cache[i].data(), // same as key_cache
                (QUANT_TYPE*)query_rot_buff.data(),
                (QUANT_TYPE*)query_pass_buff.data(),
                (QUANT_TYPE*)key_rot_buff.data(),
                (QUANT_TYPE*)key_pass_buff.data(),
                query_shape, // set
                key_shape, // set
                value_shape, // set
                sin_cached_shape, // set
                cos_cached_shape, // set
                sin_shape, // wil be set
                cos_shape, // will be set
                key_cache_shape[i], // intially it should not be set
                value_cache_shape[i], // intially it should not be set
                query_rot_buff_dims, // will be set
                query_pass_buff_dims, // will be set
                key_rot_buff_dims, // will be set
                key_pass_buff_dims, // will be set
                rotary_emb_dim, // set
                position_ids // set
            );

            {
                
                // remove later
                // printTensorColumn(
                //     "\nquery_states columns after DynTruncation()",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );
                // printTensorColumn(
                //     "\nkey_states columns after DynTruncation()",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["key_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );

                // printTensorColumn(
                //         "\nvalue_states columns after DynTruncation",
                //         (float*)(*models)["P3_first_buffered"].applicationInputBuffers["value_states:0"]->data(),
                //         {32, MAX_SEQ_LEN, 80}
                // );
                // exit(0);
            }

            if (quantize) {
                // UnQuantize (query_states)(buff_3) (buff_8 as intermediate)
                TfNToFloat(
                    (float*)buff_8.data(),
                    (*models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
                    decoderQuantParams[i].at("query_states"),
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
                copyTensor(
                    (float*)buff_8.data(), 
                    (float*)(*models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
                    {seq_len, INTERMEDIATE_SIZE}
                );

                // UnQuantize (key_states)(buff_4) (buff_8 as intermediate)
                TfNToFloat(
                    (float*)buff_8.data(),
                    (*models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
                    decoderQuantParams[i].at("query_states"),
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
                copyTensor(
                    (float*)buff_8.data(), 
                    (float*)(*models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
                    {seq_len, INTERMEDIATE_SIZE}
                );
            }

            std::cout << "\t\t\tFinished DynamicTruncationAndConcatentation() #" << i_str << "\n";

            // reshape_to_buff() {note: could optimize this by doing this before quantization}
            // reshapeToBufferedBeforeP2first(
            //     seq_len, 
            //     tot_seq_len,
            //     buff_8.data(),
            //     models,
            //     (UNQUANT_TYPE)1,
            //     (QUANT_TYPE)1
            // );
            execute(*models, "P2_reshaped", false);

            {
                // remove later
                // seems ok
                // bufferedToReshapedAfterP2first(
                //     seq_len, 
                //     models, 
                //     (QUANT_TYPE)2, 
                //     "P2_1 output (attn_weights before softmaxing)"
                // ); // i dont think we need
                // printTensorColumn(
                //     "value_states after buffering",
                //     (float*)(*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["value_states:0"]->data(),
                //     {32, MAX_SEQ_LEN, 80}
                // );
                // printTensorColumn(
                //     "attn_weights BEFORE SOFTMAX columns",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN}
                // );
                // printTensorColumn(
                //     "attn_weights BEFORE SOFTMAX columns",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     1
                // );
                // printTensorColumn(
                //     "attn_weights BEFORE SOFTMAX columns",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     2
                // );
                // printTensorColumn(
                //     "attn_weights BEFORE SOFTMAX columns",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     MAX_SEQ_LEN-1
                // );
                // exit(0);
            }

            attn_weights_shape = {1, 32, seq_len, tot_seq_len};
            // bufferedSoftmax(
            //     {1, 32, MAX_SEQ_LEN, MAX_SEQ_LEN},
            //     attn_weights_shape,
            //     (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data()
            // );

            printTensor(
                "P2 attn_weights", 
                (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(), 
                attn_weights_shape
            );

            // apply masking and softmaxing for non-buffered
            size_t mask_offset = reduceDims(mask_shape);
            assert(mask_offset == reduceDims(attn_weights_shape, 2));
            for (size_t j = 0; j < attn_weights_shape.end()[-3]; j++) {
                add(
                    mask_ptr, 
                    (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data() + j*mask_offset,
                    (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data() + j*mask_offset,
                    mask_shape
                );
            }
            mySoftmax(
                (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(),
                (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(),
                attn_weights_shape
            );

            printTensor(
                "P2 attn_weights after masking and softmaxing", 
                (float*)(*models)["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(), 
                attn_weights_shape
            );

            //remove later
            {
                // bufferedToReshapedAfterP2first(seq_len, models, (QUANT_TYPE)2, "P2_1 first after softmax");

                // printTensorColumn(
                //     "attn_weights",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN}
                // );
                // printTensorColumn(
                //     "attn_weights",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     1
                // );
                // printTensorColumn(
                //     "attn_weights",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     2
                // );
                // printTensorColumn(
                //     "attn_weights",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     MAX_SEQ_LEN-1
                // );

                // printTensor(
                //     "attn_weights",
                //     (float*)(*models)["P3_first_buffered"].applicationInputBuffers["attn_weights:0"]->data(),
                //     {32, MAX_SEQ_LEN, MAX_SEQ_LEN}
                // );

                // printTensorColumn(
                //     "attention_mask ",
                //     (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"]->data(),
                //     {MAX_SEQ_LEN, MAX_SEQ_LEN},
                //     7
                // );

                // exit(0);
            }

            if (quant_size) {
                // qunatize
                // attn_weights (buff_8)
            }
            // execute(*models, "P3_first_buffered", quantize);
            execute(*models, "P3_reshaped", quantize);
            attn_output_shape = {1, seq_len, HIDDEN_SIZE};

            // the 2 is a dummy val for the template
            // bufferedToReshapeBeforeP4(seq_len, i_str, models, (QUANT_TYPE)2);

                {
                    // remove later
                    // exit(0);
                }

            execute(*models, "P4_1_reshaped_layer_" + i_str, quantize);
            if (quantize) {
                // unquantize
            }
            else {
                copyTensor(
                    (UNQUANT_TYPE*)(*models)["P1_2_reshaped_layer_" + i_str].applicationOutputBuffers["feed_forward_hidden_states:0"]->data(),
                    (UNQUANT_TYPE*)(*models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(),
                    {seq_len , HIDDEN_SIZE}
                );
            }
            execute(*models, "P4_2_reshaped", false);

            decoder_output_shape = {1, seq_len, HIDDEN_SIZE};

            // copy data from decoder_out to residual buffer if we are not on the last decoder layer
            if (i != DECODERS-1) { 
                std::cout << "\nCOPYING DATA to residual\n";
                residual_shape = decoder_output_shape;
                assert(float_size == 4); // if this is false, think about what dimensions represent before changing
                copyTensor(
                    (UNQUANT_TYPE*)(*models)["P4_2_reshaped"].applicationOutputBuffers["decoder_output:0"]->data(), 
                    (UNQUANT_TYPE*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), 
                    decoder_output_shape
                );
            }

            {
                // remove later
                // exit(0);
            }
        }

        // remove later
        // exit(0);

        // write kv cache from out to in
        #ifdef DEBUG
            std::cout << "calling copyKV\n";
        #endif

        // need to add rest of model (outside of decoder layers)
        if (quantize) {
            // qunatize buff_3 in memory
        }
        // check to make sure this works in memory
        layernorm_Nd_32f(
            (float*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(),
            final_layernorm_weight.data(),
            final_layernorm_bias.data(),
            (float*)(*models)["Final_LM_Head"].applicationInputBuffers["final_input:0"]->data(),
            residual_shape,
            1e-5
        );
        
        execute(*models, "Final_LM_Head", quantize);
        // exit(0);

        // grab next token
        // the line below is old, FIX IT
        next_token = Argmax(
            (VOCAB_SIZE * (seq_len-1)) + (QUANT_TYPE*)(*models)["Final_LM_Head"].applicationOutputBuffers["final_output:0"]->data(),
            VOCAB_SIZE
        );

        #ifdef DEBUG
            std::cout << "next token grabbed: " << next_token << "\n";
            std::cout << "token translated: " << tokenizer_ptr->decode({next_token}) << "\n";
        #endif

        // insert token 
        tot_token_seq.push_back(next_token);
        token_seq = std::vector<uint32_t> {next_token};

        tot_seq_len  = tot_token_seq.size();
        seq_len      = token_seq.size();

        position_ids = {(int)tot_seq_len - 1};

        if (use_end_token_id && next_token == end_token_id) {
            break; 
        }

        // reshape
        // reshape stuff for the next run

        std::cout << "\t\t\tTOT_SEQ_LEN: " << tot_seq_len << "\n";
        // exit(0);

        if (iteration_num != max_iterations-1) {
            reshapeStuff(models, iteration_num, tot_seq_len, sizeof(QUANT_TYPE), sizeof(UNQUANT_TYPE));
        }
    }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT I--\n";
    #endif
    
    // tokenizer decode
    std::string output_txt;
    output_txt = tokenizer_ptr->decode(tot_token_seq);
    
    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT J--\n";
    #endif


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
        delete tokenizer_ptr;
        #ifdef DEBUG
            std::cout << "done freeing\n";
        #endif
    }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT K--\n";
    #endif

    printV("tot_token_seq", tot_token_seq);

    #ifdef DEBUG
        std::cout << "returning\n";
    #endif

    return output_txt;
    */ // temp
    return "BEANS\n";
}
