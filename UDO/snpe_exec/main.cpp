/* 
    - Recreatation of test.cpp using more levels of abstraction to keep everything neater 
    - Currently built for a single model, see test.cpp for an example of how to use multiple dlcs
*/


#include "snpe_tutorial_utils.h"
#include "snpe_exec_utils.h"
#include "embedding.h"
#include "main_macros.h"

#include <cassert>
#include <iostream>
#include <vector>

#define DEBUG

int main(int argc, char* argv[]) {

    // change this later when you make multiple calls
    bool kv_empty = true;

    /* set debug flag */
    bool debug = false;
    #ifdef DEBUG
        debug = true;
    #endif

    /* grab cmd-line model information */
    std::vector<ModelRunetime> models;
    parse_argv_models(models, argc, argv);
    #ifdef DEBUG
        print_model_runtimes(models);
    #endif

    /* grab cmd-line other information */
    uint32_t NUM_ITERS = 0;
    parse_argv_other(argc, argv, NUM_ITERS);
    #ifdef DEBUG
        std::cout << "NUM_ITERS: " << NUM_ITERS << "\n";
    #endif

    /* load udo */
    std::string udo_name = "DecodePackage/libs/x86-64_linux_clang/libUdoDecodePackageReg.so";
    int udo_load = Snpe_Util_AddOpPackage(udo_name.c_str());
    assert(udo_load == 1);

    /* intialize runtimes */
    intialize_model_runtime(models);

    /* allocate input buffer */
    std::vector<int> first_model_input_sizes = {
        (1 + 2 * DECODERS) * (4*4 + (MAX_SEQ_LEN * HIDDEN_SIZE) * DATASIZE), 
        MASK_SIZE, 
        POS_IDS_SIZE, 
        TOTAL_DECODER_WEIGHT_SIZE, 
        TOTAL_LM_WEIGHT_SIZE, 
        SIN_COS_TOTAL_SIZE, 
        SIN_COS_TOTAL_SIZE
    };
    allocate_model_input_buffers(models, first_model_input_sizes, debug);

    /* intialize input buffer */
    intialize_input_buffers(models, debug);

    /* allocate output buffer */
    std::vector<int> first_model_output_sizes = {
        (1 + 2 * DECODERS) * (4*4 + (MAX_SEQ_LEN * HIDDEN_SIZE) * DATASIZE)
    };
    allocate_model_output_buffers(models, first_model_output_sizes, debug);

    /* execution stage */
    #ifdef DEBUG
        std::cout << "execution stage\n";
    #endif

    /* tokenizer encode */
    std::string input_txt;
    // "What is your favorite color?. Mine is red."
    std::vector<uint32_t> token_collection = {
        2061, 318, 534, 4004, 3124, 30, 13, 11517, 318, 2266, 13
    };
    std::vector<uint32_t> tokens = token_collection;

    uint32_t tot_seq_len = tokens.size();
    uint32_t next_token;

    /* if kv_cache is supposed to be empty, dims should be [0,0,0,0] */
    if (kv_empty) {
        resetKV((datatype*)models[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
    }

    for (uint32_t iteration_num = 0; iteration_num < NUM_ITERS; iteration_num++) {

        #ifdef DEBUG
            printV("tokens", tokens);
        #endif

        /* embedding layer */
        writeEmbedding(
            "embed_tokens.dat", 
            tokens, 
            HIDDEN_SIZE, 
            (datatype*)models[0].applicationInputBuffers["hidden_states_and_kv:0"].data());
        #ifdef DEBUG
            std::cout << "preparing inputs\n";
        #endif


        /* generate proper mask and position_ids */
        prepareInputs(
            (float*)(models[0].applicationInputBuffers["attention_mask:0"].data()),
            (int*)(models[0].applicationInputBuffers["position_ids_1:0"].data()),
            tot_seq_len, iteration_num);
        #ifdef DEBUG
            std::cout << "executing model\n";
        #endif

        /* call model */
        models[0].snpe->execute(models[0].inputMap, models[0].outputMap);

        /* write kv cache from out to in */
        #ifdef DEBUG
            std::cout << "calling copyKV\n";
        #endif
        copyKV(
            (datatype*)models[0].applicationOutputBuffers["Output_1:0"].data(),
            (datatype*)models[0].applicationInputBuffers["hidden_states_and_kv:0"].data());

        /* grab next token */
        next_token = ((uint32_t*)models[0].applicationOutputBuffers["Output_1:0"].data())[0];
        #ifdef DEBUG
            std::cout << "next token grabbed: " << next_token << "\n";
        #endif

        /* insert token */
        token_collection.push_back(next_token);
        tokens = std::vector<uint32_t> {next_token};

        tot_seq_len++;
    }

    /* tokenizer decode */

    printV("tokens produced:", token_collection);
}



std::vector<std::vector<float>> read_embedding(
    std::string filename, const std::vector<int>& input_ids, 
    int rowSize, int colSize)
{

  FILE *fp = fopen(filename.c_str(), "rb");
  
  std::vector<std::vector<float>> output(input_ids.size()); // "float" subject to change
  std::vector<float> temp_arry(rowSize); // "float" subject to change
  
  for (int i = 0; i < input_ids.size(); i++) {
    if (std::abs(input_ids[i]) >= colSize) {
        std::cerr << "Error: attempting to read from column " << input_ids[i] << 
                    " when the max size is " << colSize << "\n";
        std::exit(3);
    }
    const size_t ret_code = fread(temp_arry.data(), 4, rowSize, fp); // read a row
    if (ret_code != rowSize) {
        std::cerr << "Error: number of elements read " << ret_code << " instead of " << rowSize <<
                    " for column "  << input_ids[i] << "\n";
        std::exit(4);
    }
    output[i] = temp_arry; // probably not the most efficient
  }
  fclose(fp);
  return output;
}
