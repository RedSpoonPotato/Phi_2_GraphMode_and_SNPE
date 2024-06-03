set -e

export SEQ_LEN=11
export TOT_SEQ_LEN=11
export HIDDEN_SIZE=2560
export INTERMEDIATE_SIZE=10240
export MAX_SEQ_LEN=2048
export DECODERS_LAYERS=3 # change back later

# doing the dlcs without weights

# snpe-tensorflow-to-dlc \
#     --input_network model_gelu \
#     --input_dim input "$SEQ_LEN, $INTERMEDIATE_SIZE" \
#     --out_node gelu_out  \
#     --output_path ../dlc/model_gelu.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P2_1_first_buffered \
#     --input_dim query_states "32, $MAX_SEQ_LEN, 80" \
#     --input_dim key_states "32, $MAX_SEQ_LEN, 80" \
#     --input_dim attention_mask "$MAX_SEQ_LEN, $MAX_SEQ_LEN" \
#     --out_node attn_weights  \
#     --output_path ../dlc/model_P2_1_first_buffered.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P2_not_first_reshaped \
#     --input_dim query_states_0 "$SEQ_LEN, 32, 80" \
#     --input_dim key_states_0 "$TOT_SEQ_LEN, 32, 80" \
#     --input_dim attention_mask "$TOT_SEQ_LEN" \
#     --out_node attn_weights  \
#     --output_path ../dlc/model_P2_not_first_reshaped.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P3_first_buffered \
#     --input_dim value_states_0 "$MAX_SEQ_LEN, 32, 80" \
#     --input_dim attn_weights "32, $MAX_SEQ_LEN, $MAX_SEQ_LEN" \
#     --out_node attn_output \
#     --output_path ../dlc/model_P3_first_buffered.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P3_not_first_reshaped \
#     --input_dim attn_weights_0 "$TOT_SEQ_LEN, 32, 1" \
#     --input_dim value_states_0 "$TOT_SEQ_LEN, 32, 80" \
#     --out_node attn_output \
#     --output_path ../dlc/model_P3_not_first_reshaped.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P3_not_first_buffered \
#     --input_dim attn_weights "32, 1, $MAX_SEQ_LEN" \
#     --input_dim value_states "32, $MAX_SEQ_LEN, 80" \
#     --out_node attn_output \
#     --output_path ../dlc/model_P3_not_first_buffered.dlc

# snpe-tensorflow-to-dlc \
#     --input_network model_P4_2_reshaped \
#     --input_dim p4_1_out "$SEQ_LEN, $HIDDEN_SIZE" \
#     --input_dim feed_forward_hidden_states "$SEQ_LEN, $HIDDEN_SIZE" \
#     --input_dim residual "$SEQ_LEN, $HIDDEN_SIZE" \
#     --out_node decoder_output \
#     --output_path ../dlc/model_P4_2_reshaped.dlc

###################################################################################
# quantizing dlcs without weights (old, update later)

# snpe-dlc-quantize \
#     --input_dlc="../dlc/model_P2_1_first_buffered.dlc" \
#     --input_list="P2_1_first_buffered.txt" \
#     --htp_socs=sm8450 \
#     --output_dlc="../q_dlc/q_model_P2_1_first_buffered.dlc"

# snpe-dlc-quantize \
#     --input_dlc="../dlc/model_P2_1_not_first_reshaped.dlc" \
#     --input_list="P2_1_not_first_reshaped.txt" \
#     --htp_socs=sm8450 \
#     --output_dlc="../q_dlc/q_model_P2_1_not_first_reshaped.dlc"

# snpe-dlc-quantize \
#     --input_dlc="../dlc/model_P3_first_buffered.dlc" \
#     --input_list="P3_first_buffered.txt" \
#     --htp_socs=sm8450 \
#     --output_dlc="../q_dlc/q_model_P3_first_buffered.dlc"

# snpe-dlc-quantize \
#     --input_dlc="../dlc/model_P3_not_first_reshaped.dlc" \
#     --input_list="P3_not_first_reshaped.txt" \
#     --htp_socs=sm8450 \
#     --output_dlc="../q_dlc/q_model_P3_not_first_reshaped.dlc"

# snpe-dlc-quantize \
#     --input_dlc="../dlc/model_P3_not_first_buffered.dlc" \
#     --input_list="P3_not_first_buffered.txt" \
#     --htp_socs=sm8450 \
#     --output_dlc="../q_dlc/q_model_P3_not_first_buffered.dlc"

#####################################################################################
# dlcs with weights
start=0
end=2

for i in $(seq $start $end); do
    echo "Index: ${i}"

    # snpe-tensorflow-to-dlc \
    #     --input_network model_P1_1_reshaped_layer_${i} \
    #     --input_dim hidden_states "$SEQ_LEN, $HIDDEN_SIZE" \
    #     --out_node query_states  \
    #     --out_node key_states  \
    #     --out_node value_states  \
    #     --out_node fc1_out  \
    #     --output_path ../dlc/model_P1_1_reshaped_layer_${i}.dlc

    # snpe-tensorflow-to-dlc \
    #     --input_network model_P1_2_reshaped_layer_${i} \
    #     --input_dim gelu_fc1_out "$SEQ_LEN, $INTERMEDIATE_SIZE" \
    #     --out_node feed_forward_hidden_states  \
    #     --output_path ../dlc/model_P1_2_reshaped_layer_${i}.dlc

    # snpe-tensorflow-to-dlc \
    #     --input_network model_P4_1_reshaped_layer_${i} \
    #     --input_dim p3_out "$SEQ_LEN, $HIDDEN_SIZE" \
    #     --out_node p4_1_out  \
    #     --output_path ../dlc/model_P4_1_reshaped_layer_${i}.dlc

    # quantizing (old, update and run later)
    # snpe-dlc-quantize \
    # --input_dlc="../dlc/model_P1_reshaped_layer_$i.dlc" \
    # --input_list="P1_reshaped.txt" \
    # --htp_socs=sm8450 \
    # --output_dlc="../q_dlc/q_model_P1_reshaped_layer_$i.dlc"

    # snpe-dlc-quantize \
    # --input_dlc="../dlc/model_P4_reshaped_layer_$i.dlc" \
    # --input_list="P4_reshaped.txt" \
    # --htp_socs=sm8450 \
    # --output_dlc="../q_dlc/q_model_P4_reshaped_layer_$i.dlc"

done

# snpe-tensorflow-to-dlc \
#     --input_network model_P1_reshaped \
#     --input_dim residual "$SEQ_LEN, $HIDDEN_SIZE" \
#     --out_node hidden_states  \
#     --out_node query_states  \
#     --out_node key_states  \
#     --out_node value_states  \
#     --out_node feed_forward_hidden_states  \
#     --output_path ../dlc/model_P1_reshaped.dlc




#######################################################################################################
# TESTING

# # this is for testing, you can delete later
# snpe-tensorflow-to-dlc \
#     --input_network model_P3_not_first_reshaped_test \
#     --input_dim attn_weights_0 "1, 32, 2" \
#     --input_dim value_states_0 "1, 32, 80" \
#     --out_node attn_output \
#     --output_path ../dlc/model_P3_not_first_reshaped_test.dlc

# # this is for testing, you can delete later
# export SMALL_SIZE=17

    # snpe-tensorflow-to-dlc \
    #     --input_network model_P1_reshaped_test \
    #     --input_dim residual "1, $SMALL_SIZE" \
    #     --out_node hidden_states  \
    #     --out_node query_states  \
    #     --out_node key_states  \
    #     --out_node value_states  \
    #     --out_node feed_forward_hidden_states  \
    #     --output_path ../dlc/model_P1_reshaped_test.dlc


    # snpe-tensorflow-to-dlc \
    #     --input_network model_P1_reshaped_test \
    #     --input_dim residual "1, $SMALL_SIZE" \
    #     --out_node query_states  \
    #     --out_node key_states  \
    #     --out_node value_states  \
    #     --out_node feed_forward_hidden_states  \
    #     --output_path ../dlc/model_P1_reshaped_test.dlc
########################################################################################################
