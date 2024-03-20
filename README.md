The two directories of this repo are UDO and GM.

GM: For showing off the tensorflow graphmode code. The Phi-2 model code is found within my_phi_gm.py. 
File guide:
  bert.py: contains operations for Bert, some of those are used in Phi
  cache.py: defines handling cached KV tensors
  masking_utils_gm.py: handles masking
  my_phi_gm.py: defines model architecture
  my_stuff.py: not used in phi-2 model, contains isolated tokenizer code
  workspace.ipynb: sample of running the graph mode model
