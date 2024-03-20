TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared decoder.cpp -o Decode.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2



# python modelgen.py