# CXX := 

# I AM ADDING THE "-g", can remove later
CXXFLAGS += -std=c++11 -fPIC -march=x86-64 -g

# Include paths
INCLUDES += -I ./
INCLUDES += -I $(SNPE_ROOT)/include/zdl -I $(SRC_DIR)/include/ -I $(SNPE_ROOT)/include/SNPE

# Specify the paths to the libraries
LDFLAGS  += -L $(SNPE_ROOT)/lib/x86_64-linux-clang -L $(SRC_DIR) -L ./jni/obj/local/x86_64-linux-clang

# Specify the link libraries
LLIBS    += -lSNPE -lmain -lboost_regex -lboost_system -lboost_filesystem

# Specify the target
PROGRAM  := load_test
SRC_DIR  := ./jni
OBJ_DIR  := $(SRC_DIR)/obj

# SRC := htp_main_no_udo.cpp
 

default: all
all: $(SRC_DIR)/$(PROGRAM)

# $(SRC_DIR)/android_main.h

$(OBJ_DIR)/$(PROGRAM).o : $(SRC_DIR)/$(PROGRAM).cpp | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(SRC_DIR)/$(PROGRAM) : $(OBJ_DIR)/$(PROGRAM).o
	$(CXX) $(LDFLAGS) $^ $(LLIBS) -o $@

clean:
	-rm -f $(OBJ_DIR)/$(PROGRAM).o
# -rm -f $(SRC_DIR)/$(PROGRAM) # careful with this

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: default clean