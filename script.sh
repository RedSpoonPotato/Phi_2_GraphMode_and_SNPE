# make
# cp obj/local/x86_64-linux-clang/libmain.so .

# cp obj/local/x86_64-linux-clang/main .

export ANDROID_NDK_ROOT="/home/kernal1/android-ndk-r19c/build/"
export PATH=$ANDROID_NDK_ROOT:$PATH

# export NDK_PROJECT_PATH=$NDK_PROJECT_PATH:/home/kernal1/QM_Sandbox/Phi_2/desktop/udo/jni/

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kernal1/QM_Sandbox/Phi_2/desktop/udo/jni/DecodePackage/libs/x86-64_linux_clang/

# for linking with libmain.so
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kernal1/QM_Sandbox/htp



# ./main -m UnifiedPhiDecodersAndLogits.dlc -o Output_1 \
#     -i hidden_states_and_kv.dat -i attention_mask.dat -i position_ids_1.dat -i decoder_weights_1.dat -i lm_head_weights_1.dat \
#     -i sin.dat -i cos.dat \
#     -N 1
