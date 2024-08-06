#ifndef QUANTIZE_H_
#define QUANTIZE_H_

#include "snpe_exec_utils.h"

void quantize_1(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    size_t seq_len,
    size_t i
) {
    FloatToTfN(
        models["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        decoderQuantParams[i].at("hidden_states"),
        false,
        (float*)(models)["P1_Q_reshaped_with_bias"].applicationInputBuffers["hidden_states:0"]->data(),
        seq_len * HIDDEN_SIZE,
        8
    );
}

void quantize_2(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<uint8_t>& buff_8,
    size_t seq_len,
    size_t i
) {
    FloatToTfN(
        buff_8.data(),
        decoderQuantParams[i].at("gelu_fc1_out"),
        true,
        (float*)buff_8.data(),
        seq_len * INTERMEDIATE_SIZE,
        8
    );
}


void quantize_4(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams
) {

}

void quantize_5(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams
) {

}

void quantize_6(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams
) {

}

void unquantize_1(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
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

void unquantize_2(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<uint8_t>& buff_8,
    size_t seq_len,
    size_t i
) {
    // UnQuantize (query_states)(buff_3) (buff_8 as intermediate)
    TfNToFloat(
        (float*)buff_8.data(),
        (models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        decoderQuantParams[i].at("query_states"),
        seq_len * INTERMEDIATE_SIZE,
        8
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        {seq_len, INTERMEDIATE_SIZE}
    );
    // UnQuantize (key_states)(buff_4) (buff_8 as intermediate)
    TfNToFloat(
        (float*)buff_8.data(),
        (models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        decoderQuantParams[i].at("query_states"),
        seq_len * INTERMEDIATE_SIZE,
        8
    );
    copyTensor(
        (float*)buff_8.data(), 
        (float*)(models)["P2_reshaped"].applicationInputBuffers["query_states:0"]->data(),
        {seq_len, INTERMEDIATE_SIZE}
    );
}

void unquantize_3(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<float>& temp_buff,
    size_t numElem,
    size_t i
) {
    TfNToFloat(
        temp_buff.data(),
        models["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data(),
        decoderQuantParams[i].at("p2_attn_weights"),
        numElem,
        8
    );
}

void unquantize_4(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    size_t seq_len,
    size_t i
) {
    TfNToFloat(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["feed_forward_hidden_states:0"]->data(),
        models["FC2_reshaped_with_bias"].applicationOutputBuffers["fc2_out:0"]->data(),
        decoderQuantParams[i].at("p4_ff"),
        seq_len * HIDDEN_SIZE,
        8
    );
    TfNToFloat(
        (float*)(models)["P4_2_reshaped"].applicationInputBuffers["p4_1_out:0"]->data(),
        models["P4_1_reshaped_with_bias"].applicationOutputBuffers["dense_out:0"]->data(),
        decoderQuantParams[i].at("p4_1_out"),
        seq_len * HIDDEN_SIZE,
        8
    );
}

void unquantize_5(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams
) {

}

void unquantize_6(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams
) {

}

// could optmize by removing temp_buff
void maskBufferDivideSoftmax(
    std::map<std::string, ModelRuntime>& models,
    std::vector<std::map<std::string, quantParams>>& decoderQuantParams,
    std::vector<size_t>& attn_weights_shape,
    std::vector<size_t>& mask_shape,
    float * mask_ptr,
    bool quantize,
    size_t float_size,
    size_t i
) {
    size_t numElem = reduceDims(attn_weights_shape);
    std::vector<UNQUANT_TYPE> temp_buff;
    float* attn_weight_buff;
    if (quantize) {
        // upcast
        temp_buff.resize(numElem * float_size);
        unquantize_3(models, decoderQuantParams, temp_buff, numElem, i);
        attn_weight_buff = temp_buff.data();
    }
    else {
        attn_weight_buff = (UNQUANT_TYPE*)models["P2_reshaped"].applicationOutputBuffers["attn_weights:0"]->data();
    }
    // divide
    float sqrt_head_dim = sqrt(HEAD_DIM);
    for (size_t j = 0; j < numElem; j++) {
        attn_weight_buff[j] /= sqrt_head_dim;
    }
    // masking
    size_t mask_offset = reduceDims(mask_shape);
    for (size_t j = 0; j < attn_weights_shape.end()[-3]; j++) {
        add(
            mask_ptr, 
            attn_weight_buff + j*mask_offset,
            attn_weight_buff + j*mask_offset,
            mask_shape
        );
    }
    // softmaxing
    mySoftmax(
        attn_weight_buff,
        attn_weight_buff,
        attn_weights_shape
    );

    if (quantize) {
        // downcast
        FloatToTfN(
            models["P3_reshaped"].applicationInputBuffers["attn_weights"]->data(),
            decoderQuantParams[i].at("p3_attn_weights"),
            true,
            attn_weight_buff,
            numElem,
            8
        );
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

#endif