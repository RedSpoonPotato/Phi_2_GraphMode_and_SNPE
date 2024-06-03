import tensorflow as tf
import math
import bert
import my_phi_gm

TOT_SEQ_LEN = 11 # will be 12 on 2nd+ iterations
SEQ_LEN = 11 # will be 1 on 2nd+ iterations
MAX_SEQ_LEN = 2048
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 10240
NUM_HEADS = 32
LN_EPS = 1e-5
one = 1
    
class PhiDecodeP1_1_reshaped_new_quant(tf.Module):
    def __init__(self, config, params, name=None):
        super().__init__(name)
        # self.input_layernorm = bert.LayerNorm(params['layernorm_weight'], params['layernorm_bias'], eps=config.layer_norm_eps)
        self.q_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['q_proj_weight'], params['q_proj_bias'])
        self.k_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['k_proj_weight'], params['k_proj_bias'])
        self.v_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['v_proj_weight'], params['v_proj_bias'])
        self.fc1 = bert.Dense_v2(config.hidden_size, config.intermediate_size,
                            params['mlp_fc1_weight'], params['mlp_fc1_bias'])
    @tf.function(input_signature=[tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)])
    def __call__(self, hidden_states):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        fc1_out = self.fc1(hidden_states)
        return {
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "fc1_out" : fc1_out
        }

# implement this this as C++ code, you could implement this as its own dlc later
class NewGELU(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(input_signature=[tf.TensorSpec(shape=[SEQ_LEN, INTERMEDIATE_SIZE], dtype=tf.float32)])
    def __call__(self, input):
        gelu_out = 0.5 * input * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * tf.pow(input, 3.0))))
        return {
            "gelu_out": gelu_out
        }

# need to come back to this
class PhiDecodeP1_2_reshaped_new_quant(tf.Module):
    def __init__(self, config, params, name=None):
        super().__init__(name)
        self.config = config
        self.fc2 = bert.Dense_v2(config.intermediate_size, config.hidden_size, 
                                 params['mlp_fc2_weight'], params['mlp_fc2_bias'])
    @tf.function(input_signature=[tf.TensorSpec(shape=[SEQ_LEN, INTERMEDIATE_SIZE], dtype=tf.float32)])
    def __call__(self, gelu_fc1_out):
        feed_forward_hidden_states = self.fc2(gelu_fc1_out)
        return {
            "feed_forward_hidden_states": feed_forward_hidden_states
        }

""" 
for PhiDecodeP1:
    query_states: (1, 32, SEQ_LEN, 80)
"""


# (1, 32, 11, 80)

# not sure if this transpose will work
    # could test
    # if it does not work, could do buffered


# mask shape: 1st: (11, 11), 2nd: (1,12) could make 2 differnt dlcs for this purpose

class PhiDecodeP2_1_first_buffered_unquant_fp32(tf.Module):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.hidden_size = HIDDEN_SIZE
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[32, MAX_SEQ_LEN, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[32, MAX_SEQ_LEN, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[MAX_SEQ_LEN, MAX_SEQ_LEN], dtype=tf.float32),
                             ])
    def __call__(self, 
                 query_states,
                 key_states, 
                 attention_mask):
        attn_weights = tf.matmul(
            tf.cast(query_states, dtype=tf.float32), 
            tf.transpose(tf.cast(key_states, dtype=tf.float32), perm=(0, 2, 1))
        ) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        # attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        return {
            "attn_weights": attn_weights
        }
    
"""
Issue:
    For softmaxing you have 2 options (assuming using buffered on the prior step)
    b/c of the fixed buffer size
    - implement ur own dynamic code for softmaxing (simplest)
    - maybe could use transposing and rehaping to do it on you own
    - use a UDO inwhich u use GPU UDO
"""

# NOTICE
""" unfinished, use ur own softmax code instead """
class PhiDecodeP2_2_first_buffered(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(input_signature=[tf.TensorSpec(shape=32, )])
    def __call__(self, attn_weights_in):
        attn_weights = tf.nn.softmax(attn_weights_in, axis=-1)
        return {
            "attn_weights": attn_weights
        }

# COULD OPTMIZE THIS BY MERGING THE KEY_STATES TRANSPOSE
class PhiDecodeP2_not_first_reshaped_unquant_fp32(tf.Module):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.hidden_size = HIDDEN_SIZE
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[SEQ_LEN, 32, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[TOT_SEQ_LEN, 32, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[TOT_SEQ_LEN], dtype=tf.float32),
                  # mask shape: 1st: (11, 11), 2nd: (1,12) could make 2 differnt dlcs for this purpose
                             ])
    def __call__(self, 
                 query_states_0, 
                 key_states_0, 
                 attention_mask):
        query_states = tf.transpose(query_states_0, perm=(1,0,2)) # {32, 1, 80}
        key_states = tf.transpose(key_states_0, perm=(1,0,2)) # {32, 11, 80}
        attn_weights = tf.matmul(
            tf.cast(query_states, dtype=tf.float32), 
            tf.transpose(key_states, perm=(0, 2, 1)) # {32, 80, 11}
        ) / math.sqrt(self.head_dim) # {32, 1, 11}
        attn_weights = attn_weights + tf.reshape(attention_mask, [1, TOT_SEQ_LEN])
        # upcast attention to fp32
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        return {
            "attn_weights": attn_weights
        }


"""
Proposal: 
    Make a buffered version for the first inference run,
    Use another dlc for the 2nd infernce run
        -A reshaped version
        -Or a buffered verison
            - dont have to do as many reshapes in this one I think
"""

class PhiDecodeP3_first_buffered_quant(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[MAX_SEQ_LEN, 32, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[32, MAX_SEQ_LEN, MAX_SEQ_LEN], dtype=tf.float32),
            ])
    def __call__(self, value_states_0, attn_weights):
        value_states = tf.transpose(value_states_0, perm=(1, 0, 2))
        attn_output = tf.matmul(attn_weights, value_states)
        attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        attn_output = tf.reshape(attn_output, (MAX_SEQ_LEN, HIDDEN_SIZE))
        return {
            "attn_output": attn_output
        }

class PhiDecodeP3_not_first_reshaped_quant(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[TOT_SEQ_LEN, 32, one], dtype=tf.float32),
                tf.TensorSpec(shape=[TOT_SEQ_LEN, 32, 80], dtype=tf.float32),
            ])
    def __call__(self, attn_weights_0, value_states_0):
        attn_weights = tf.transpose(attn_weights_0, perm=(1, 2, 0)) # {32, 1, 11}
        value_states = tf.transpose(value_states_0, perm=(1, 0, 2)) # {32, 11, 80}
        attn_output = tf.matmul(attn_weights, value_states) # {32, 1, 80}
        attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        attn_output = tf.reshape(attn_output, (one, HIDDEN_SIZE))
        return {
            "attn_output": attn_output
        }


class PhiDecodeP3_not_first_buffered_quant(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[32, one, MAX_SEQ_LEN], dtype=tf.float32),
                tf.TensorSpec(shape=[32, MAX_SEQ_LEN, 80], dtype=tf.float32),
            ])
    def __call__(self, attn_weights, value_states):
        attn_output = tf.matmul(attn_weights, value_states)
        attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        attn_output = tf.reshape(attn_output, (one, HIDDEN_SIZE))
        return {
            "attn_output": attn_output
        }

class PhiDecodeP4_1_reshaped_quant(tf.Module):
    def __init__(self, params, config, name=None):
        super().__init__(name)
        self.hidden_size = HIDDEN_SIZE
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dense = bert.Dense_v2(self.num_heads * self.head_dim, self.hidden_size, params['dense_weight'], params['dense_bias'])
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32),
            ])
    def __call__(self, p3_out):
        p4_1_out = self.dense(p3_out)
        return {
            "p4_1_out": p4_1_out
        }

class PhiDecodeP4_2_reshaped_unquant(tf.Module):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.hidden_size = HIDDEN_SIZE
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32),
                tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32),
                tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32),
            ])
    def __call__(self, p4_1_out, feed_forward_hidden_states, residual):
        decoder_output = p4_1_out + feed_forward_hidden_states + residual
        return {
            "decoder_output": decoder_output
        }
    

#########################################################################################
# TESTING



# this is merely for testing, can remove later
SMALL_SIZE = 17
MEDIUM_SIZE = 23
class PhiDecodeP1_reshaped_test(tf.Module):
    def __init__(self, config, params, name=None):
        super().__init__(name)
        self.input_layernorm = bert.LayerNorm(params['layernorm_weight'], params['layernorm_bias'], eps=config.layer_norm_eps)
        self.q_proj = bert.Dense_v2(SMALL_SIZE, SMALL_SIZE, params['q_proj_weight'], params['q_proj_bias'])
        self.k_proj = bert.Dense_v2(SMALL_SIZE, SMALL_SIZE, params['k_proj_weight'], params['k_proj_bias'])
        self.v_proj = bert.Dense_v2(SMALL_SIZE, SMALL_SIZE, params['v_proj_weight'], params['v_proj_bias'])
        self.mlp = my_phi_gm.PhiMLP(config, params)
    # @tf.function(input_signature=[tf.TensorSpec(shape=[SEQ_LEN, SMALL_SIZE], dtype=tf.float32)])
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, SMALL_SIZE], dtype=tf.float32)])
    def __call__(self, residual):
        hidden_states = self.input_layernorm(residual)

        # remove later
        query_states = self.q_proj(residual)
        key_states = self.k_proj(residual)
        value_states = self.v_proj(residual)
        feed_forward_hidden_states = self.mlp(residual)
        # q: is it b/c we have a intermediate buff that it faisl to reshape
        # no b/c if we set hidden_states to the sole output it still fails

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        # feed_forward_hidden_states = self.mlp(hidden_states)
        return {
            "hidden_states": hidden_states,
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "feed_forward_hidden_states": feed_forward_hidden_states
        }

    
class PhiDecodeP3_not_first_reshaped_test(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, 32, 2], dtype=tf.float32),
                tf.TensorSpec(shape=[1, 32, 80], dtype=tf.float32),
            ])
    def __call__(self, attn_weights_0, value_states_0):
        # two different ways
        # attn_weights = tf.transpose(tf.expand_dims(attn_weights_0, axis=2), perm=(1, 2, 0))
        # attn_weights = tf.expand_dims(tf.transpose(attn_weights_0, perm=(1, 0)), axis=1)

        # value_states = tf.transpose(value_states_0, perm=(1, 0, 2))

        # current


        attn_weights = tf.transpose(attn_weights_0, perm=(1, 2, 0)) # {32, 2, 1}
        value_states = tf.transpose(value_states_0, perm=(1, 0, 2)) # {32, 1, 80}
        attn_output = attn_weights + value_states
        # attn_output = tf.matmul(attn_weights, value_states) # {32, 1, 80}
        # attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        # attn_output = tf.reshape(attn_output, (one, HIDDEN_SIZE))


        # good
        # tf.TensorSpec(shape=[1, 32, 2], dtype=tf.float32),
        # tf.TensorSpec(shape=[1, 32, 80], dtype=tf.float32),
        # attn_output = tf.matmul(tf.transpose(attn_weights_0, perm=[0, 2, 1]), value_states_0)

        # good
        # tf.TensorSpec(shape=[1, TOT_SEQ_LEN, 32, 1], dtype=tf.float32),
        # tf.TensorSpec(shape=[1, TOT_SEQ_LEN, 32, 1], dtype=tf.float32),
        # value_states = tf.transpose(value_states_0, perm=(0, 1, 3, 2))
        # attn_output = tf.matmul(attn_weights_0, value_states)


        # attn_output = tf.transpose(attn_output, perm=(1, 0, 2))
        # attn_output = tf.reshape(attn_output, (one, HIDDEN_SIZE))

        # bad line
        # attn_output = tf.expand_dims(attn_weights_0, axis=2) + value_states_0

        # attn_output = attn_weights_0 + value_states_0

        return {
            "attn_output": attn_output
        }




#########################################################################################

# OLD STUFF


# could create a buffered verion if u want, time to really find out
class PhiDecodeP1_reshaped(tf.Module):
    def __init__(self, config, params, name=None):
        super().__init__(name)
        self.input_layernorm = bert.LayerNorm(params['layernorm_weight'], params['layernorm_bias'], eps=config.layer_norm_eps)
        self.q_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['q_proj_weight'], params['q_proj_bias'])
        self.k_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['k_proj_weight'], params['k_proj_bias'])
        self.v_proj = bert.Dense_v2(HIDDEN_SIZE, HIDDEN_SIZE, params['v_proj_weight'], params['v_proj_bias'])
        self.mlp = my_phi_gm.PhiMLP(config, params)
    @tf.function(input_signature=[tf.TensorSpec(shape=[SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)])
    def __call__(self, residual):
        hidden_states = self.input_layernorm(residual)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        return {
            "hidden_states": hidden_states,
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "feed_forward_hidden_states": feed_forward_hidden_states
        }

class PhiDecodeP3_1_old(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[32, SEQ_LEN, TOT_SEQ_LEN], dtype=tf.float32),
                tf.TensorSpec(shape=[TOT_SEQ_LEN, 32, 80], dtype=tf.float32),
            ])
    def __call__(self, attn_weights, value_states_0):
        value_states = tf.transpose(value_states_0, perm=(1,0,2))
        attn_output = tf.matmul(attn_weights, value_states)
        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (SEQ_LEN, HIDDEN_SIZE))


class PhiDecodeP2_first_old(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.head_dim # ??
    @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[SEQ_LEN, 32, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[TOT_SEQ_LEN, 32, 80], dtype=tf.float32),
                tf.TensorSpec(shape=[SEQ_LEN, SEQ_LEN], dtype=tf.float32),
                  # mask shape: 1st: (11, 11), 2nd: (1,12) could make 2 differnt dlcs for this purpose
                             ])
    def __call__(self, 
                 query_states_0, 
                 key_states_0, 
                 attention_mask):
        query_states = tf.transpose(query_states_0, perm=(1,0,2))
        key_states = tf.transpose(key_states_0, perm=(1,0,2))
        attn_weights = tf.matmul(
            tf.cast(query_states, dtype=tf.float32), 
            tf.transpose(tf.cast(key_states, dtype=tf.float32), perm=(0, 1, 3, 2))
        ) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        # upcast attention to fp32
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        return 