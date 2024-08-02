set -e

# for linking with libmain.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kernal1/QM_Sandbox/htp/jni/obj/local/x86_64-linux-clang

#cd jni
#make clean
#make
#cd ../
#make clean
#make

# restore this
# ./jni/load_test cpu32 2 ./fp16_test/model_split/data/ ./fp16_test/model_split/dlc/ model_ \
#     { \
    # -m P1_QKV_reshaped_no_bias cpu32 \
    # -m P2_reshaped cpu32 \
    # -m P3_reshaped cpu32 \
    # -m FC1_reshaped_no_bias cpu32 \
    # -m FC2_reshaped_no_bias cpu32 \
    # -m FinalLMHead_reshaped_no_bias cpu32
#     }

# remove this
valgrind --tool=massif --threshold=0.1 ./jni/load_test 2 ./fp16_test/model_split \
    -m P1_QKV_reshaped_no_bias cpu32 \
    -m P2_reshaped cpu32 \
    -m P3_reshaped cpu32 \
    -m FC1_reshaped_no_bias cpu32 \
    -m FC2_reshaped_no_bias cpu32 \
    -m Final_LM_Head cpu32
    # -m FinalLMHead_reshaped_no_bias cpu32

######################################################################
# OLD STUFF
# would need to change load_test.cpp to run this
# ./jni/load_test cpu32 ./model_generation/matmul_reshape.dlc 100

# ./jni/load_test dsp16 ./model_generation/matmul.dlc 1000

# python test.py
# ./jni/load_test cpu32 ./model_generation/matmul_2.dlc 10000

