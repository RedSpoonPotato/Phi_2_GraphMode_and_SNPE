#ifndef QUANTIZE_H_
#define QUANTIZE_H_

#include "snpe_exec_utils.h"

void quantize_1(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    size_t seq_len,
    size_t i
) {
    FloatToTfN(
        models["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        decoderQuantParams[i].at("decoder_layernorm"),
        false,
        (float*)(models)["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        seq_len * HIDDEN_SIZE,
        8
    );
}

void fp16_cast_1(
    std::map<std::string, ModelRuntime>& models,
    std::vector<size_t>& residual_shape
) {
    fp32_to_fp16(
        (float*)(models)["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        (FP16*)models["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        residual_shape
    );
}

void quantize_2(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<uint8_t>& buff_8,
    size_t seq_len,
    size_t i
) {
    FloatToTfN(
        buff_8.data(),
        decoderQuantParams[i].at("gelu_out"),
        true,
        (float*)buff_8.data(),
        seq_len * INTERMEDIATE_SIZE,
        8
    );
}

void fp16_cast_2(
    std::vector<uint8_t>& buff_8,
    std::vector<size_t>& fc1_out_shape
) {
    fp32_to_fp16(
        (float*)buff_8.data(),
        (FP16*)(float*)buff_8.data(),
        fc1_out_shape
    );
}

void quantize_3(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    float* attn_weight_buff,
    size_t numElem,
    size_t i
) {
    FloatToTfN(
        models["P3_reshaped"].applicationInputBuffers["attn_weights:0"]->data(),
        decoderQuantParams[i].at("p3_attn_weights"),
        true,
        attn_weight_buff,
        numElem,
        8
    );
}

void fp16_cast_3(
    std::map<std::string, ModelRuntime>& models,
    float* attn_weight_buff,
    std::vector<size_t>& attn_weights_shape
) {
    fp32_to_fp16(
        (float*)attn_weight_buff,
        (FP16*)models["P3_reshaped"].applicationInputBuffers["attn_weights:0"]->data(),
        attn_weights_shape
    );
}

void fp16_cast_4(
    std::map<std::string, ModelRuntime>& models,
    std::vector<size_t>& decoder_output_shape
) {
    fp32_to_fp16(
        (float*)(models)["Final_LM_Head"].applicationInputBuffers["final_input:0"]->data(),
        (FP16*)(models)["Final_LM_Head"].applicationInputBuffers["final_input:0"]->data(),
        decoder_output_shape
    );
}

void unquantize_1(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<uint8_t>& buff_8,
    size_t seq_len,
    size_t i
) {
    TfNToFloat(
        (float*)buff_8.data(),
        (models)["FC1_reshaped_with_bias"].applicationOutputBuffers["fc1_out:0"]->data(),
        decoderQuantParams[i].at("fc1_out"),
        seq_len * INTERMEDIATE_SIZE,
        8
    );
}

void fp32_cast_1(
    std::map<std::string, ModelRuntime>& models,
    std::vector<size_t>& fc1_out_shape,
    std::vector<uint8_t>& buff_8
) {
    fp16_to_fp32(
        (FP16*)(models)["FC1_reshaped_with_bias"].applicationOutputBuffers["fc1_out:0"]->data(),
        (float*)buff_8.data(),
        fc1_out_shape
    );
}

void unquantize_2(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<uint8_t>& buff_8,
    size_t seq_len,
    size_t i,
    std::vector<size_t>& query_shape,
    std::vector<size_t>& key_shape
) {
    // UnQuantize (query_states)(buff_3) (buff_8 as intermediate)
    TfNToFloat(
        (float*)buff_8.data(),
        (models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        decoderQuantParams[i].at("query_states"),
        reduceDims(query_shape),
        8
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        query_shape
    );
    // UnQuantize (key_states)(buff_4) (buff_8 as intermediate)
    TfNToFloat(
        (float*)buff_8.data(),
        (models)["P2_reshaped"].applicationInputBuffers["key_states:0"]->data(),
        decoderQuantParams[i].at("key_states"),
        reduceDims(key_shape),
        8
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["key_states:0"]->data(),
        key_shape
    );
}

void fp32_cast_2(
    std::map<std::string, ModelRuntime>& models,
    std::vector<uint8_t>& buff_8,
    std::vector<size_t>& query_shape,
    std::vector<size_t>& key_shape
) {
    fp16_to_fp32(
        (FP16*)(models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        (float*)buff_8.data(),
        query_shape
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        query_shape
    );
    fp16_to_fp32(
        (FP16*)(models)["P2_reshaped"].applicationInputBuffers["key_states:0"]->data(),
        (float*)buff_8.data(),
        key_shape
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["key_states:0"]->data(),
        key_shape
    );
}

void unquantize_3(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    size_t seq_len,
    size_t i
) {
    TfNToFloat(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(), // buff_8
        models["P4_1_reshaped_with_bias"].applicationOutputBuffers["dense_out:0"]->data(),
        decoderQuantParams[i].at("p4_1_out"),
        seq_len * HIDDEN_SIZE,
        8
    );
        // (float*)(models)["P4_2_reshaped"].applicationInputBuffers["p4_1_out:0"]->data(),
    copyTensor(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(), // buff_8
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["p4_1_out:0"]->data(),
        {seq_len, HIDDEN_SIZE}
    );

    TfNToFloat(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(),
        models["FC2_reshaped_with_bias"].applicationOutputBuffers["fc2_out:0"]->data(),
        decoderQuantParams[i].at("p4_ff"),
        seq_len * HIDDEN_SIZE,
        8
    );
}

void fp32_cast_3(
    std::map<std::string, ModelRuntime>& models,
    std::vector<size_t>& residual_shape
) {
    fp16_to_fp32(
        (FP16*)models["P4_1_reshaped_with_bias"].applicationOutputBuffers["dense_out:0"]->data(),
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(), // buff_8
        residual_shape
    );
    copyTensor(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(), // buff_8
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["p4_1_out:0"]->data(),
        residual_shape
    );
    fp16_to_fp32(
        (FP16*)models["FC2_reshaped_with_bias"].applicationOutputBuffers["fc2_out:0"]->data(),
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(),
        residual_shape
    );
}

// could optmize by removing temp_buff
void maskBufferDivideSoftmax(
    std::map<std::string, ModelRuntime>& models,
    const std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<size_t>& attn_weights_shape,
    std::vector<size_t>& mask_shape,
    float * mask_ptr,
    bool quantize,
    size_t float_size,
    size_t i,
    bool enable_fp16
) {
    size_t numElem = reduceDims(attn_weights_shape);
    std::vector<UNQUANT_TYPE> temp_buff;
    float* attn_weight_buff;
 
    attn_weight_buff = (UNQUANT_TYPE*)models["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data();

    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNaN(attn_weight_buff, attn_weights_shape, "before dividing") == 0);
        assert(findInf(attn_weight_buff, attn_weights_shape, "before dividing") == 0);
    #endif

    // divide
    float sqrt_head_dim = sqrt(HEAD_DIM);
    for (size_t j = 0; j < numElem; j++) {
        attn_weight_buff[j] /= sqrt_head_dim;
    }
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNaN(attn_weight_buff, attn_weights_shape, "after dividing") == 0);
        assert(findInf(attn_weight_buff, attn_weights_shape, "after dividing") == 0);
    #endif

    #ifdef DEBUG
        printTensor(
            "attn_weights <P2_1> after dividing", 
            (float*)models["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(),
            attn_weights_shape
        );
        printTensor("mask", mask_ptr, mask_shape);
    #endif

    // masking
    #ifdef DEBUG
        std::cout << "masking\n";
    #endif
    size_t mask_offset = reduceDims(mask_shape);
    for (size_t j = 0; j < attn_weights_shape.end()[-3]; j++) {
        add(
            mask_ptr, 
            attn_weight_buff + j*mask_offset,
            attn_weight_buff + j*mask_offset,
            mask_shape
        );
    }
    #if defined(DEBUG) && defined(ENABLE_FP16)
        assert(findNaN(attn_weight_buff, attn_weights_shape, "after masking") == 0);
        assert(findInf(attn_weight_buff, attn_weights_shape, "after masking") == 0);
    #endif

    // softmaxing
    #ifdef DEBUG
        std::cout << "softmaxxing\n";
    #endif
    mySoftmax(
        attn_weight_buff,
        attn_weight_buff,
        attn_weights_shape
    );

    // downcast
    if (quantize) {
        quantize_3(models, decoderQuantParams, attn_weight_buff, numElem, i);
    }
    else if (enable_fp16) {
        fp16_cast_3(models, attn_weight_buff, attn_weights_shape);
    }
}

void copy_FF_Hidden_States(
    std::map<std::string, ModelRuntime>& models,
    const std::string& i_str,
    size_t seq_len
) {
    copyTensor(
        (UNQUANT_TYPE*)(models)["FC2_reshaped_with_bias"].applicationOutputBuffers["fc2_out:0"]->data(),
        (UNQUANT_TYPE*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(),
        {seq_len , HIDDEN_SIZE}
    );
}

void copyResidual(
    std::map<std::string, ModelRuntime>& models,
    std::vector<size_t>& decoder_output_shape
) {
    copyTensor(
        (UNQUANT_TYPE*)(models)["P4_2_reshaped"].applicationOutputBuffers["decoder_output:0"]->data(), 
        (UNQUANT_TYPE*)(models)["P4_2_reshaped"].applicationInputBuffers["residual:0"]->data(), 
        decoder_output_shape
    );
}

#ifdef DEBUG_2
void modelTesting(
    std::map<std::string, ModelRuntime>& models
) {
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

            stall();
            size_t seq_len = 15;
            size_t tot_seq_len = 15;
            reshapeModels(models, "P2_reshaped",
            {
                {"query_states:0", {32, seq_len, 80}},
                {"key_states:0", {32, 80, tot_seq_len}}
            });
            reshapeModels(models, "P3_reshaped",
            {
                {"attn_weights:0", {32, seq_len, tot_seq_len}},
                {"value_states:0", {32, tot_seq_len, 80}}
            });
            reshapeModels(models, "P1_Q_reshaped_with_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, HIDDEN_SIZE}},
                {"bias:0", {1, HIDDEN_SIZE}}
            });
            reshapeModels(models, "FC1_reshaped_with_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, HIDDEN_SIZE}},
                {"bias:0", {1, HIDDEN_SIZE}}
            });
            reshapeModels(models, "P4_2_reshaped",
            {
                {"p4_1_out:0", {2, HIDDEN_SIZE}},
                {"feed_forward_hidden_states:0", {2, HIDDEN_SIZE}},
                {"residual:0", {2, HIDDEN_SIZE}}
            });
            stall();
            // exit(0);
            execute(models, "P3_reshaped", false);
            stall();
            execute(models, "P2_reshaped", false);
            execute(models, "P1_Q_reshaped_with_bias", false);
            stall();
            exit(0);

            stall();

            std::cout << "reshaping\n";

            /// 
            // size_t seq_len = 20;
            // size_t tot_seq_len = 20;
            // exit(0);

            reshapeModels(models, "P1_QKV_reshaped_no_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, HIDDEN_SIZE}}
            });
            reshapeModels(models, "P2_reshaped",
            {
                {"query_states:0", {32, seq_len, 80}},
                {"key_states:0", {32, 80, tot_seq_len}}
            });
            reshapeModels(models, "P3_reshaped",
                {
                    {"attn_weights:0", {32, seq_len, tot_seq_len}},
                    {"value_states:0", {32, tot_seq_len, 80}}
                });
            reshapeModels(models, "FC1_reshaped_no_bias",
            {
                {"hidden_states:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, INTERMEDIATE_SIZE}}
            });
            reshapeModels(models, "FC2_reshaped_no_bias",
            {
                {"gelu_out:0", {1, seq_len, INTERMEDIATE_SIZE}},
                {"weights:0", {1, INTERMEDIATE_SIZE, HIDDEN_SIZE}}
            });

            reshapeModels(models, "FinalLMHead_reshaped_no_bias",
            {
                {"final_input:0", {1, seq_len, HIDDEN_SIZE}},
                {"weights:0", {1, HIDDEN_SIZE, VOCAB_SIZE}}
            });
            stall();


            for (size_t i = 0; i < 1; i++) {
                execute(models, "P1_QKV_reshaped_no_bias", false);
                execute(models, "P2_reshaped", false);
                execute(models, "P3_reshaped", false);
                execute(models, "FC1_reshaped_no_bias", false);
                execute(models, "FC2_reshaped_no_bias", false);
            }
            
            execute(models, "FinalLMHead_reshaped_no_bias", false);
            // execute(models, "Final_LM_Head", false);

            stall();

            execute(models, "FinalLMHead_reshaped_no_bias", false);
            // execute(models, "Final_LM_Head", false);

            stall();

            // free the weight buffer
            models["FinalLMHead_reshaped_no_bias"].snpe.reset();

            stall();

            reshapeModels(models, "Final_LM_Head",
            {
                {"final_input:0", {1, HIDDEN_SIZE}},
            });

            stall();

            execute(models, "Final_LM_Head", false);

            stall();

            exit(0);
}
#endif

#endif