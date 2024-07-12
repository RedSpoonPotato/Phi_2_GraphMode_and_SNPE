set -e

# for linking with libmain.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kernal1/QM_Sandbox/htp/jni/obj/local/x86_64-linux-clang

cd jni
make clean
make
cd ../
make clean
make


######################################################################
# OLD STUFF
# would need to change load_test.cpp to run this
# ./jni/load_test cpu32 ./model_generation/matmul_reshape.dlc 100

# ./jni/load_test dsp16 ./model_generation/matmul.dlc 1000

# python test.py
# ./jni/load_test cpu32 ./model_generation/matmul_2.dlc 10000

