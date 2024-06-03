# %%
import tensorflow as tf
import numpy as np

# %%
import numpy as np

dtype = tf.float16 # subject to change
BASE_DIR = "/home/kernal1/QM_Sandbox/Phi_2/phi_local/weights/"
# DECODER_LAYERS = 3 # PUT BACK IN
DECODER_LAYERS = 1  # REMOVE
params = {}
params['embed_tokens'] = tf.constant(np.load(BASE_DIR + 'embed_tokens.npy'), dtype=dtype)
params['decoder_layers'] = []

for i in range(DECODER_LAYERS):
  layer_params = {}
  layer_params['layernorm_weight'] = 'layernorm_weight.npy'
  layer_params['layernorm_bias'] = 'layernorm_bias'
  layer_params['q_proj_weight'] = 'q_proj_weight'
  layer_params['q_proj_bias'] = 'q_proj_bias'
  layer_params['k_proj_weight'] = 'k_proj_weight'
  layer_params['k_proj_bias'] = 'k_proj_bias'
  layer_params['v_proj_weight'] = 'v_proj_weight'
  layer_params['v_proj_bias'] = 'v_proj_bias'
  layer_params['dense_weight'] = 'dense_weight'
  layer_params['dense_bias'] = 'dense_bias'
  layer_params['mlp_fc1_weight'] = 'mlp_fc1_weight'
  layer_params['mlp_fc1_bias'] = 'mlp_fc1_bias'
  layer_params['mlp_fc2_weight'] = 'mlp_fc2_weight'
  layer_params['mlp_fc2_bias'] = 'mlp_fc2_bias'
  for key in layer_params:
    layer_params[key] = tf.constant(np.load(BASE_DIR + str(i) + '_' + key + '.npy'), dtype=dtype)
    if "weight" in key and "layernorm" not in key:
      layer_params[key] = tf.transpose(layer_params[key], perm=[1,0])
    print(key, ": ", layer_params[key].shape) # remove later
  params['decoder_layers'].append(layer_params)

params['final_layernorm_weight'] = tf.constant(np.load(BASE_DIR + 'final_layernorm_weight.npy'), dtype=dtype)
params['final_layernorm_bias'] = tf.constant(np.load(BASE_DIR + 'final_layernorm_bias.npy'), dtype=dtype)

params['lm_head_weight'] = tf.transpose(tf.constant(np.load(BASE_DIR + 'lm_head_weight.npy'), dtype=dtype), perm=[1,0])
params['lm_head_bias'] = tf.constant(np.load(BASE_DIR + 'lm_head_bias.npy'), dtype=dtype)

# %%
class PhiConfig():
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=DECODER_LAYERS, # modified,
        num_attention_heads=32,
        num_key_value_heads=32,
        resid_pdrop=0.0, # the model bt default has 0.1 on colab, but we dont have randomness
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="gelu_new",
        max_position_embeddings=2048,
        initializer_range=0.00,
        layer_norm_eps=1e-5,
        use_cache=True,                  # Modifed
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.4,
        qk_layernorm=False,
        bos_token_id=50256,
        eos_token_id=50256,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm
        self.pad_token_id = None
        self._attn_implementation = 'eager'

# %%
# upcasting from fp16 to fp32
# for d in params:
#     print(d)
#     print(type(params[d]))

# %%
# upcasting from fp16 to fp32
for key in params:
    if (key != 'decoder_layers'):
        params[key] = tf.cast(params[key], dtype=tf.float32)

for weights in params['decoder_layers']:
    for key in weights:
        weights[key] = tf.cast(weights[key], dtype=tf.float32)

# %%
config = PhiConfig(use_cache=True)
import importlib

import split_defs

# %%
def saveTensor(tensors:list, base_name:str, dir='.'):
    for i in range(len(tensors)):
        tensors[i].numpy().tofile(dir + "/" + base_name + "_" + str(i) + ".bin")

# %%
# testing each part
importlib.reload(split_defs)
from split_defs import *


# for i in range(DECODER_LAYERS):
for i in range(1):
	model_P1_reshaped = PhiDecodeP1_reshaped(config, params['decoder_layers'][i])
	# tf.saved_model.save(model_P1_reshaped, "tf/model_P1_reshaped_layer_"+ str(i))
	i1 = tf.random.uniform([SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)
	# saveTensor([i1], "P1_reshaped_layer" + str(i), "tf")
	# model_P1_reshaped(i1)

	model_P2_1_first_buffered = PhiDecodeP2_1_first_buffered(config)
	# tf.saved_model.save(model_P2_1_first_buffered, "tf/model_P2_1_first_buffered")
	i1 = tf.random.uniform([32, MAX_SEQ_LEN, 80], dtype=tf.float32)
	i2 = tf.random.uniform([32, MAX_SEQ_LEN, 80], dtype=tf.float32)
	i3 = tf.random.uniform([MAX_SEQ_LEN, MAX_SEQ_LEN], dtype=tf.float32)
	# saveTensor([i1, i2, i3], "P2_1_first_buffered", "tf")
	# model_P2_1_first_buffered(i1, i2, i3)
	
	model_P2_1_not_first_reshaped = PhiDecodeP2_not_first_reshaped(config)
	# tf.saved_model.save(model_P2_1_not_first_reshaped, "tf/model_P2_1_not_first_reshaped")
	i1 = tf.random.uniform([SEQ_LEN, 32, 80], dtype=tf.float32)
	i2 = tf.random.uniform([TOT_SEQ_LEN, 32, 80], dtype=tf.float32)
	i3 = tf.random.uniform([TOT_SEQ_LEN], dtype=tf.float32)
	# saveTensor([i1, i2, i3], "P2_1_not_first_reshaped", "tf")
	# model_P2_1_not_first_reshaped(i1, i2, i3)
	
	model_P3_first_buffered = PhiDecodeP3_first_buffered()
	# tf.saved_model.save(model_P3_first_buffered, "tf/model_P3_first_buffered")
  # i1 = tf.random.uniform([MAX_SEQ_LEN, 32, 80], dtype=tf.float32)
	# i2 = tf.random.uniform([32, MAX_SEQ_LEN, MAX_SEQ_LEN], dtype=tf.float32)
	# saveTensor([i1, i2], "P3_first_buffered", "tf")
	model_P3_first_buffered(i1, i2)
	
	model_P3_not_first_reshaped = PhiDecodeP3_not_first_reshaped()
	# tf.saved_model.save(model_P3_not_first_reshaped, "tf/model_P3_not_first_reshaped")
	i1 = tf.random.uniform([TOT_SEQ_LEN, 32, one], dtype=tf.float32)
	i2 = tf.random.uniform([TOT_SEQ_LEN, 32, 80], dtype=tf.float32)
	# saveTensor([i1, i2], "P3_not_first_reshaped", "tf")
	# model_P3_not_first_reshaped(i1, i2)['attn_output'].shape
	
	model_P3_not_first_buffered = PhiDecodeP3_not_first_buffered()
	# tf.saved_model.save(model_P3_not_first_buffered, "tf/model_P3_not_first_buffered")
	i1 = tf.random.uniform([32, one, MAX_SEQ_LEN], dtype=tf.float32)
	i2 = tf.random.uniform([32, MAX_SEQ_LEN, 80], dtype=tf.float32)
	# saveTensor([i1, i2], "P3_not_first_buffered", "tf")
	# model_P3_not_first_buffered(i1, i2)['attn_output'].shape
	
	model_P4_reshaped = PhiDecodeP4_reshaped(params['decoder_layers'][i], config)
	# tf.saved_model.save(model_P4_reshaped, "tf/model_P4_reshaped_layer_"+ str(i))
	i1 = tf.random.uniform([SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)
	i2 = tf.random.uniform([SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)
	i3 = tf.random.uniform([SEQ_LEN, HIDDEN_SIZE], dtype=tf.float32)
	# saveTensor([i1, i2, i3], "P4_reshaped", "tf")
	# model_P4_reshaped(i1, i2, i3)

# %%



