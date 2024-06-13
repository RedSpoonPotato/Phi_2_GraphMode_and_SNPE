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

#define N_PRINT 5 // legnth of elements to print

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
    static std::map<std::string, ModelRuntime>* models = modelDictCreator(ModelNameAndPaths);

    Tokenizer* tokenizer_ptr;

    if (exitAndFree == Free_Status::free) {
        freeModels(models);
        delete tokenizer_ptr;
        return "";
    }

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

    const int rotary_emb_dim = 32; // 80 * .4 (self.head_dim * partial_rot_fact)

    // NOTE: CHANGE the sizes (4 or 1) to a variable that may be changed to 2 (fp16) if android c++ is capable
    size_t quant_size = 4; // **NOTE: CHANGE THIS BACK TO 1 WHEN GOING TO ANDROID
    size_t float_size = 4; // LOOK INTO CHANGING THIS TO SAVE MEMORY (change to 2)
    static auto buff_1      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size);
    // static auto buff_2      =   std::vector<uint8_t>(MAX_SEQ_LEN * INTERMEDIATE_SIZE * float_size);
    static auto buff_3      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4); // needs to be fp32
    static auto buff_4      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_5      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * quant_size);
    static auto buff_6      =   std::vector<uint8_t>(MAX_SEQ_LEN * INTERMEDIATE_SIZE * quant_size); // used to be float_size
    static auto buff_7      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    static auto buff_8      =   std::vector<uint8_t>(rotary_emb_dim *  MAX_SEQ_LEN * HIDDEN_SIZE * 4);
    // static auto buff_9      =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size);
    // static auto buff_10     =   std::vector<uint8_t>(MAX_SEQ_LEN * HIDDEN_SIZE * float_size);

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

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT D--\n";
        #endif

        tokenizer_ptr = new Tokenizer(
            otherPaths.at("token_vocab"), 
            otherPaths.at("token_merges")
        );

        /* intialize runtimes */
        intialize_model_runtime(*models, runtime_modes);

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT E--\n";
        #endif

        linkBuffers(models, buff_1, buff_3, buff_4, buff_5, buff_6, buff_7, buff_8);

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT F--\n";
        #endif

        create_user_buffers(*models, datasize, isTFBuffer);

        #ifdef DEBUG
            std::cout << "\t\t\t--CHECKPOINT G--\n";
        #endif
        // put these back in later
        loadAndQuantize(sin_cached, otherPaths.at("sin"));
        loadAndQuantize(cos_cached, otherPaths.at("cos"));

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
    for (int i = 0; i < tot_token_seq.size(); i++) { position_ids.push_back(i); }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT H--\n";
    #endif

    std::vector<uint32_t> token_seq = tot_token_seq; // will be size 11 on first run, 1 on runs after

    uint32_t next_token;
    
    size_t tot_seq_len  = tot_token_seq.size();
    size_t seq_len      = token_seq.size();

    for (uint32_t iteration_num = 0; iteration_num < max_iterations; iteration_num++) {

        #ifdef DEBUG
            printV("token_seq", token_seq);
        #endif

        /* embedding layer */
        writeEmbedding( 
            otherPaths.at("embedding"),
            token_seq, 
            HIDDEN_SIZE, 
            (uint16_t*)buff_3.data() // using an intermediate buffer
        ); 
        // copy convert 16 bit data to 32 bit inside of the buffer
        assert(float_size == 4);
        fp16_to_fp32(
            (ushort*)buff_3.data(), // using an intermediate buffer
            (float*)buff_1.data(), 
            {(uint32_t)seq_len, HIDDEN_SIZE}
        );

        #ifdef DEBUG
            printN("embedding 32 bit", (float*)buff_1.data(), N_PRINT);
        #endif

        // set decoder_output as input to layernorm
        residual_shape = {1, seq_len, HIDDEN_SIZE};

        /* generate mask */
        assert(float_size == 4); // otherwise chagne the casting of the mask pointer
        prepareMask(
            (float*)((*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"]->data()),
            tot_seq_len, 
            iteration_num
        );
        #ifdef DEBUG
            printN(
                "prepared Mask", 
                (float*)((*models)["P2_1_first_buffered"].applicationInputBuffers["attention_mask:0"]->data()),
                N_PRINT
            );
            std::cout << "executing model\n";
        #endif

        /* call model */
        /* Need ALOT MORE WORK 
            - processing inbetween buffers?
            - other stuff
        */
        for (int i = 0; i < DECODERS; i++) {
            std::string i_str = std::to_string(i);

            // use decoder_output as a inut to layernorm
            layernorm_Nd_32f(
                (float*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), // buff_1
                layernorm_weights[i].data(), 
                layernorm_biases[i].data(), 
                (float*)(*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(), // buff_8
                residual_shape, 
                HIDDEN_SIZE,
                1e-5
            );
            #ifdef DEBUG
                printN(
                    "Decoder Layer " + i_str, 
                    (float*)(*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                    N_PRINT
                );
            std::cout << "executing model\n";
            #endif

            if (quantize) {
                // need to quantize (not sure if fixed or unfixed)
                FloatToTfN(
                    (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                    decoderQuantParams[i]["hidden_states"],
                    false,
                    (float*)(*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                    seq_len * HIDDEN_SIZE,
                    8
                );
            
                // remove later, testing quantizing
                {
                    TfNToFloat(
                        (float*)buff_6.data(),
                        (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                        decoderQuantParams[i]["hidden_states"],
                        seq_len * HIDDEN_SIZE,
                        8
                    );

                    printN("hidden_states after quantization test (should match prev)", (float*)buff_6.data(), N_PRINT);

                    FloatToTfN(
                        (*models)["P1_1_reshaped_layer_" + i_str].applicationInputBuffers["hidden_states:0"]->data(),
                        decoderQuantParams[i]["hidden_states"],
                        false,
                        (float*)buff_6.data(),
                        seq_len * HIDDEN_SIZE,
                        8
                    );
                }
            }

            #ifdef DEBUG
                executeDebug(*models, "P1_1_reshaped_layer_" + i_str, N_PRINT);
                if (quantize) {
                    // need to dequantize (not sure if fixed or unfixed)
                    TfNToFloat(
                        (float*)buff_8.data(),
                        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["fc1_out:0"]->data(),
                        decoderQuantParams[i]["fc1_out"],
                        seq_len * INTERMEDIATE_SIZE,
                        8
                    );
                }
                executeDebug(*models, "gelu", N_PRINT);
                if (quantize) {
                    // need to quantize (not sure if fixed or unfixed)
                    FloatToTfN(
                        buff_8.data(),
                        decoderQuantParams[i]["gelu_fc1_out"],
                        true,
                        (float*)buff_8.data(),
                        seq_len * INTERMEDIATE_SIZE,
                        8
                    );
                }
                executeDebug(*models, "P1_2_reshaped_layer_" + i_str, N_PRINT);
            #else            
                execute(*models, "P1_1_reshaped_layer_" + i_str);
                if (quantize) {
                    // need to dequantize (not sure if fixed or unfixed)
                    TfNToFloat(
                        (float*)buff_8.data(),
                        (*models)["P1_1_reshaped_layer_" + i_str].applicationOutputBuffers["fc1_out:0"]->data(),
                        decoderQuantParams[i]["fc1_out"],
                        seq_len * INTERMEDIATE_SIZE,
                        8
                    );
                }
                execute(*models, "gelu");
                if (quantize) {
                    // need to quantize (not sure if fixed or unfixed)
                    FloatToTfN(
                        buff_8.data(),
                        decoderQuantParams[i]["gelu_fc1_out"],
                        true,
                        (float*)buff_8.data(),
                        seq_len * INTERMEDIATE_SIZE,
                        8
                    );
                }
                execute(*models, "P1_2_reshaped_layer_" + i_str);
            #endif
            
            query_shape = {1, 32, seq_len, 80};
            key_shape   = {1, 32, seq_len, 80};
            value_shape = {1, 32, seq_len, 80};

            /* implement processing */
            // NEED TO SET THE SHAPES (MAKE THEM ALL 4D)
            sin_cached_shape = {MAX_SEQ_LEN, 32};
            cos_cached_shape = {MAX_SEQ_LEN, 32};

            std::cout << "\t\t\tCalling DynamicTruncationAndConcatentation() #" << i_str << "\n";

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
                key_cache_shape[i], // intially it should not be set
                value_cache_shape[i], // intially it should not be set
                query_rot_buff_dims, // will be set
                query_pass_buff_dims, // will be set
                key_rot_buff_dims, // will be set
                key_pass_buff_dims, // will be set
                rotary_emb_dim, // set
                position_ids // set
            );


            if (quantize) {
                // UnQuantize (query_states)(buff_3) (buff_8 as intermediate)
                TfNToFloat(
                    (float*)buff_8.data(),
                    (*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                    decoderQuantParams[i]["query_states"],
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
                copyTensor(
                    (float*)buff_8.data(), 
                    (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                    {seq_len, INTERMEDIATE_SIZE}
                );

                // UnQuantize (key_states)(buff_4) (buff_8 as intermediate)
                TfNToFloat(
                    (float*)buff_8.data(),
                    (*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                    decoderQuantParams[i]["query_states"],
                    seq_len * INTERMEDIATE_SIZE,
                    8
                );
                copyTensor(
                    (float*)buff_8.data(), 
                    (float*)(*models)["P2_1_first_buffered"].applicationInputBuffers["query_states:0"]->data(),
                    {seq_len, INTERMEDIATE_SIZE}
                );
            }

            std::cout << "\t\t\tFinished DynamicTruncationAndConcatentation() #" << i_str << "\n";

            if (iteration_num == 0) {

                #ifdef DEBUG
                    executeDebug(*models, "P2_1_first_buffered", N_PRINT);
                    attn_weights_shape = {1, 32, seq_len, seq_len};
                    mySoftmax(
                        (float*)(*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"]->data(),
                        (float*)(*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"]->data(),
                        attn_weights_shape
                    );
                    if (quant_size) {
                        // quantize
                        // attn_weights (buff_8)
                    }
                    executeDebug(*models, "P3_first_buffered", N_PRINT);
                    attn_output_shape = {1, seq_len, HIDDEN_SIZE};
                #else
                    execute(*models, "P2_1_first_buffered");
                    attn_weights_shape = {1, 32, seq_len, seq_len};
                    mySoftmax(
                        (float*)(*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"]->data(),
                        (float*)(*models)["P2_1_first_buffered"].applicationOutputBuffers["attn_weights:0"]->data(),
                        attn_weights_shape
                    );
                    if (quant_size) {
                        // qunatize
                        // attn_weights (buff_8)
                    }
                    execute(*models, "P3_first_buffered");
                    attn_output_shape = {1, seq_len, HIDDEN_SIZE};
                #endif
            }
            else {
                #ifdef DEBUG
                    executeDebug(*models, "P2_not_first_reshaped", N_PRINT);
                    if (quantize) {
                        // qunatize
                        // attn_weights (buff_8)
                    }
                    executeDebug(*models, "P3_not_first_buffered", N_PRINT);
                #else
                    execute(*models, "P2_not_first_reshaped");
                    if (quantize) {
                         // qunatize
                        // attn_weights (buff_8)
                    }
                    execute(*models, "P3_not_first_buffered");
                #endif
            }

            #ifdef DEBUG
                executeDebug(*models, "P4_1_reshaped_layer_" + i_str, N_PRINT);
                if (quantize) {
                    // unquantize
                    
                }
                executeDebug(*models, "P4_2_reshaped", N_PRINT);
            #else
                execute(*models, "P4_1_reshaped_layer_" + i_str);
                if (quantize) {
                    // unquantize
                }
                execute(*models, "P4_2_reshaped");
            #endif

            decoder_output_shape = {1, seq_len, HIDDEN_SIZE};

            /* copy data from decoder_out to residual buffer if we are not on the last decoder layer*/
            if (i != DECODERS-1) { 
                residual_shape = decoder_output_shape;
                assert(float_size == 4); // if this is false, think about what dimensions represent before changing
                copyTensor(
                    (float*)(*models)["P4_2_reshaped"].applicationOutputBuffers["decoder_output:0"]->data(), 
                    (float*)(*models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), 
                    decoder_output_shape
                );
            }
        }

        /* write kv cache from out to in */
        #ifdef DEBUG
            std::cout << "calling copyKV\n";
        #endif

        // need to add rest of model (outside of decoder layers)

        /* grab next token */
        // the line below is old, FIX IT
        // next_token = ((uint32_t*)(*models)[0].applicationOutputBuffers["Output_1:0"].data())[0];
        #ifdef DEBUG
            std::cout << "next token grabbed: " << next_token << "\n";
        #endif

        /* insert token */
        tot_token_seq.push_back(next_token);
        token_seq = std::vector<uint32_t> {next_token};

        tot_seq_len  = tot_token_seq.size();
        seq_len      = token_seq.size();

        if (use_end_token_id && next_token == end_token_id) {
            break; 
        }

        /* reshape */
        // reshape stuff for the next run

        std::cout << "\t\t\tTOT_SEQ_LEN: " << tot_seq_len << "\n";

        reshapeStuff(models, iteration_num, datasize, tot_seq_len);

        position_ids = {(int)tot_seq_len - 1};
    }

    #ifdef DEBUG
        std::cout << "\t\t\t--CHECKPOINT I--\n";
    #endif
    
    /* tokenizer decode */
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

    #ifdef DEBUG
        std::cout << "returning\n";
    #endif

    return output_txt;
}
