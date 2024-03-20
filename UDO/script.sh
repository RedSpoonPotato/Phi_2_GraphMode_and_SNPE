# cp ./tf_ops/Decode.so .

# python Unified_Phi_Model.py

echo "------------------------------------"

export MAX_SEQ_LEN=2048
export HIDDEN_SIZE=2560
export DECODERS=4

# all 3 below's tensor size is actually double
export DECODER_WEIGHT_SIZE=39335680
export LM_WEIGHTS_SIZE=65564160
export SIN_COS_SIZE=32768


# these tensors's sizes include size of the dims (4d)
export HIDDEN_STATES_AND_KV_SIZE=$((2621444*$((1+2*$DECODERS))))
export MASK_SIZE=$((4+$MAX_SEQ_LEN*$MAX_SEQ_LEN))
export POS_IDS_SIZE=$((4+$MAX_SEQ_LEN))
export TOTAL_DECODER_WEIGHT_SIZE=$(($DECODERS*$DECODER_WEIGHT_SIZE))



echo $HIDDEN_STATES_AND_KV_SIZE
echo $MASK_SIZE
echo $POS_IDS_SIZE

echo $TOTAL_DECODER_WEIGHT_SIZE
echo $LM_WEIGHTS_SIZE
echo $SIN_COS_SIZE


# rm -r DecodePackage # CAREFUL
# snpe-udo-package-generator \
#     -p Decode.json \
#     -o ./

# echo "ee$HIDDEN_SIZE, $MAX_SEQ_LEN"

# snpe-tensorflow-to-dlc \
#     --input_network UnifiedPhiDecodersAndLogits_tf_model \
#     --input_dim hidden_states_and_kv "1, 1, 1, $HIDDEN_STATES_AND_KV_SIZE"  \
#     --input_dim attention_mask "1, 1, 1, $MASK_SIZE" \
#     --input_dim position_ids_1 "1, 1, 1, $POS_IDS_SIZE" \
#     --input_dim decoder_weights_1 "1, 1, 1, $TOTAL_DECODER_WEIGHT_SIZE" \
#     --input_dim lm_head_weights_1 "1, 1, 1, $LM_WEIGHTS_SIZE" \
#     --input_dim sin "1, 1, 1, $SIN_COS_SIZE" \
#     --input_dim cos "1, 1, 1, $SIN_COS_SIZE" \
#     --out_node "Output_1" \
#     --output_path UnifiedPhiDecodersAndLogits.dlc \
#     --udo Decode.json



# cp ./UnifiedPhiDecodersAndLogits.dlc /home/kernal1/QM_Sandbox/SNPE_sandbox/dyn_linear/.


# cp ./inputs/inputs.zip /home/kernal1/QM_Sandbox/SNPE_sandbox/dyn_linear/.

# # # building
cd ./DecodePackage
make
cd ../

zip -r package.zip DecodePackage/
cp package.zip /home/kernal1/QM_Sandbox/SNPE_sandbox/dyn_linear/.


# # # need to run only once per session
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kernal1/QM_Sandbox/Phi_2/desktop/udo/unified/DecodePackage/libs/x86-64_linux_clang/

# snpe-net-run --container UnifiedPhiDecodersAndLogits.dlc \
#     --input_list inputs.txt \
#     --udo_package_path DecodePackage/libs/x86-64_linux_clang/libUdoDecodePackageReg.so
