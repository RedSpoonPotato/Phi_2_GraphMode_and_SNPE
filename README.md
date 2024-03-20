The two directories of this repo are UDO and GM.

## GM: For showing off the tensorflow graphmode code. The Phi-2 model code is found within my_phi_gm.py. 

File guide:

  bert.py: contains operations for Bert, some of those are used in Phi
  
  cache.py: defines handling cached KV tensors

  masking_utils_gm.py: handles masking
  
  my_phi_gm.py: defines model architecture
  
  my_stuff.py: not used in phi-2 model, contains isolated tokenizer code
  
  workspace.ipynb: sample of running the graph mode model
  
## UDO: For showing off custom written C++ code that is inserted within a UDO

File guide:

  Decode.cpp: Defines UDO, 16-bit cpu operations, input parsing
  
  Decode.json: SNPE UDO config file
  
  Unified_Phi_Model.py: Python code for generated Tensorflow model with custom tensorflow opaertions which will then be converted into a DLC.
  
  ops.cpp: standalone C++ file that can be compiled and run on its own to demonstrate the ability for the Decoder code to run (Does have bugs, see Decode.cpp for latest verison)
  
  test.cpp: C++ file for using SNPE-api to launch entire, however it currently does segfault.
