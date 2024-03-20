import tensorflow as tf

# import all custom TF operations
# decoder_module = tf.load_op_library('./tf_ops/Decode.so') 
# this might be a problem if its in a different directory
decoder_module = tf.load_op_library('./Decode.so')

# hyper params
MAX_SEQ_LEN = 2048
HIDDEN_SIZE = 2560
VOCAB_SIZE = 51200

DECODERS = 4
DECODER_WEIGHT_SIZE = 78671360
LM_WEIGHTS_SIZE = 131128320

SIN_COS_SIZE = 32 * MAX_SEQ_LEN

# LM: ln_w: 2560, ln_b: 2560, lm_head_w: (2560, 51200), lm_head_b: (51200)

# for now, ignoring the first udo (embedding, mask generation, etc...)

# if these fail, try making DECODERS even, that MAY solve the issue

assert(((MAX_SEQ_LEN*HIDDEN_SIZE)//2)*2 == MAX_SEQ_LEN*HIDDEN_SIZE)
assert(((DECODERS*DECODER_WEIGHT_SIZE) // 2) * 2 == DECODERS*DECODER_WEIGHT_SIZE)
assert((LM_WEIGHTS_SIZE // 2) * 2 == LM_WEIGHTS_SIZE)

print((4+((MAX_SEQ_LEN*HIDDEN_SIZE)//2)))

class UnifiedPhiDecodersAndLogits(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 1, 1, 
        (4+((MAX_SEQ_LEN*HIDDEN_SIZE)//2)) * (1 + 2*DECODERS) ], dtype=tf.float32), # hidden_states_and_kv
        tf.TensorSpec(shape=[1, 1, 1, 4 + MAX_SEQ_LEN * MAX_SEQ_LEN], dtype=tf.float32), # mask
        tf.TensorSpec(shape=[1, 1, 1, 4 + MAX_SEQ_LEN]              , dtype=tf.float32), # position_ids
        tf.TensorSpec(shape=[1, 1, 1, (DECODERS*DECODER_WEIGHT_SIZE) // 2], dtype=tf.float32), # decode_weights
        tf.TensorSpec(shape=[1, 1, 1, LM_WEIGHTS_SIZE // 2], dtype=tf.float32), # lm_weights
        tf.TensorSpec(shape=[1, 1, 1, SIN_COS_SIZE // 2], dtype=tf.float32), # sin
        tf.TensorSpec(shape=[1, 1, 1, SIN_COS_SIZE // 2], dtype=tf.float32), # cos
        ] 
        )
    def __call__( 
        self, 
        hidden_states_and_kv,
        attention_mask,
        position_ids_1, 
        decoder_weights_1,
        lm_head_weights_1,
        sin,
        cos
        ):

#         # x1 = hidden_states + 1
#         # x2 = hidden_states + 2  
#         # x3 = hidden_states + 3
        # reuired inorder to allow udo to take in the same tensor for differnt arguements
#         x1 = hidden_states + 0
#         x2 = hidden_states + 0
#         x3 = hidden_states + 0
        
        # x2 = kv_0 + 0
        # x3 = hidden_states + 0
        # x1 = tf.concat([x2, x3], axis=-1) + 0
        # x1 = kv_0 + 1
        # x2 = hidden_states + 2

        out = decoder_module.Decode(
            attn_and_kv_in=hidden_states_and_kv,
            mask=attention_mask,
            position_ids=position_ids_1,
            decoder_weights=decoder_weights_1,
            lm_head_weights=lm_head_weights_1,
            sin_cached=sin,
            cos_cached=cos
        )

#         # _attn_output_dims += 1

        return {
            # "PhiDecoderLogitOut": _attn_output,
            # "PhiDecoderLogitOutDims": _attn_output_dims
            "Output_1": out
        }

#         # return {
#         #     "PhiDecoderLogitOut": _attn_output,
#         #     "PhiDecoderLogitOutDims": _attn_output_dims,
#         #     "PhiDecoderLogitKv": _kv_out,
#         #     "PhiDecoderLogitKvDims": _kv_out_dims
#         # }


model = UnifiedPhiDecodersAndLogits()

# saving model
# tf.saved_model.save(model, "UnifiedPhiDecodersAndLogits_tf_model")


########################################################################
# inputs:
# a = model(
#         tf.random.uniform([1, 1, 1, 4 + MAX_SEQ_LEN * HIDDEN_SIZE], dtype=tf.float32), # hidden_states
#         tf.random.uniform([1, 1, 1, 2 * (4 + MAX_SEQ_LEN * HIDDEN_SIZE)], dtype=tf.float32), # kv_0
#         tf.random.uniform([1, 1, 1, 4 + MAX_SEQ_LEN * MAX_SEQ_LEN], dtype=tf.float32), # mask
#         tf.random.uniform([1, 1, 1, 4 + MAX_SEQ_LEN]              , dtype=tf.float32), # position_ids
#         tf.random.uniform([1, 1, 1, DECODERS * DECODER_WEIGHT_SIZE], dtype=tf.float32), # decode_weights
#         tf.random.uniform([1, 1, 1, DECODERS * DECODER_WEIGHT_DIMS_SIZE], dtype=tf.float32), # decode_weights_dims
#         tf.random.uniform([1, 1, 1, DECODERS * DECODER_INIT_PARAMS_SIZE], dtype=tf.float32), # init_params
    # )

# print(a)

#############################################################################

# generating input files
import os
import random
import string

def write_random_data(num_elements, file_name):
    print("num of elements: ", num_elements)
    num_bytes = 4 * num_elements
    random_data = ''.join(random.choices(string.ascii_letters + string.digits, k=num_bytes))
    with open(file_name, 'w') as file:
        file.write(random_data)


# write_random_data(4 + MAX_SEQ_LEN * MAX_SEQ_LEN,        "inputs/attention_mask.dat") # mask
write_random_data(4 + MAX_SEQ_LEN,                      "inputs/position_ids_1.dat")  # position_ids


# write_random_data((4+((MAX_SEQ_LEN*HIDDEN_SIZE)//2)) * (1 + 2*DECODERS),
#                    "inputs/hidden_states_and_kv.dat")
        

import struct
# def write_float32_to_file(file_path, byte_address, value):
#     # Open the file in binary read/write mode
#     with open(file_path, 'rb+') as file:
#         # Move to the specified byte address
#         file.seek(byte_address)
#         # Convert the float32 number to binary data
#         float_bytes = struct.pack('f', value)
#         # Write the binary data to the specified location
#         file.write(float_bytes)

def write_int32_to_file(file_path, byte_address, value):
    offset = byte_address
    data = struct.pack('<i', value)
    with open(file_path, 'rb+') as f:
        f.seek(offset)
        f.write(data)

def write_uint32_t_to_file(file_path, byte_address, value):
    offset = byte_address
    data = struct.pack('<I', value)
    with open(file_path, 'rb+') as f:
        f.seek(offset)
        f.write(data)



offset = 2 * MAX_SEQ_LEN * HIDDEN_SIZE
# write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", offset, 1)
# write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", offset+4, 1)
# write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", offset+8, 11)
# write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", offset+12, HIDDEN_SIZE)

# full_offset = (16 + offset)
# addr = 0
# decoder_offset = 0
# for j in range(DECODERS):
#     decoder_offset = j * 2*full_offset + full_offset
#     for i in range(2):
#         addr = (i)*full_offset + offset + decoder_offset
#         write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", addr, 0)
#         write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", addr+4, 0)
#         write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", addr+8, 0)
#         write_uint32_t_to_file("inputs/hidden_states_and_kv.dat", addr+12, 0)


# offset = 4 * MAX_SEQ_LEN * MAX_SEQ_LEN
# write_uint32_t_to_file("inputs/attention_mask.dat", offset, 1)
# write_uint32_t_to_file("inputs/attention_mask.dat", offset+4, 1)
# write_uint32_t_to_file("inputs/attention_mask.dat", offset+8, 11)
# write_uint32_t_to_file("inputs/attention_mask.dat", offset+12, 11)

for i in range(11):
    write_int32_to_file("inputs/position_ids_1.dat", 4*i, i)
write_int32_to_file("inputs/position_ids_1.dat", 4*MAX_SEQ_LEN, 1)
write_int32_to_file("inputs/position_ids_1.dat", 4*MAX_SEQ_LEN + 4, 1)
write_int32_to_file("inputs/position_ids_1.dat", 4*MAX_SEQ_LEN + 8, 1)
write_int32_to_file("inputs/position_ids_1.dat", 4*MAX_SEQ_LEN + 12, 11)


##########
# test
# write_random_data(4, "test.dat")
# write_uint32_t_to_file("test.dat", 0, 1)
# write_uint32_t_to_file("test.dat", 4, 0)
# write_uint32_t_to_file("test.dat", 8, 11)



