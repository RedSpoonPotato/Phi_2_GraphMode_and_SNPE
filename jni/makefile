# Specify the compiler flags
# CXX ?= g++
# CXX = clang++-14

# I AM ADDING THE "-g", can remove later
CXXFLAGS += -std=c++11 -fPIC -march=x86-64 -g
# CXXFLAGS += -std=c++20 -fPIC -march=x86-64 -g


# Include paths
INCLUDES += -I ./
INCLUDES += -I $(SNPE_ROOT)/include/zdl -I include/ -I $(SNPE_ROOT)/include/SNPE

# Specify the paths to the libraries
LDFLAGS  += -shared -L $(SNPE_ROOT)/lib/x86_64-linux-clang

# Specify the link libraries
LLIBS    += -lSNPE -lboost_regex -lboost_system -lboost_filesystem

# Specify the target
SO_FILE := libmain.so
SRC_DIR  := ./
OBJ_DIR  := obj/local/x86_64-linux-clang

EXCLUDE_FILES := test.cpp main.cpp load_test_old.cpp main_alt.cpp htp_main_no_udo.cpp load_test.cpp android_main_htp_non_udo.cpp

# goal is to compile: android_main.cpp

SRC := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDE_FILES)), $(wildcard $(SRC_DIR)/*.cpp))
# Generate the output names by substituting the object dir for the source dir
OBJS     := $(subst $(SRC_DIR),$(OBJ_DIR),$(subst .cpp,.o,$(SRC)))

default: all
all: $(OBJ_DIR)/$(SO_FILE)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@

$(OBJ_DIR)/$(SO_FILE): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LLIBS) -o $@

clean:
	-rm -f $(OBJS)
	-rm -f $(SO_FILE)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

.PHONY: default clean