#ifndef MAIN_MACROS_H_
#define MAIN_MACROS_H_

#include <iostream>

// #define DEBUG // enabled through makefiles
#define DEBUG_2 // enabled through makefiles

// float
#define BATCH_SIZE 1
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560
#define QUERY_STATES_BUFF_SIZE  (BATCH_SIZE*MAX_SEQ_LEN*HIDDEN_SIZE)

#define INTERMEDIATE_SIZE 10240
#define INTERMEDIATE_STATES_BUFF_SIZE (BATCH_SIZE*MAX_SEQ_LEN*INTERMEDIATE_SIZE)

// partial_rotary_factor: 0.4
// head_dim = hidden_size // num_attention_heads i.e. (2560 / 32) = 80
// sin_cos(param:dim) = head_dim * partial_rotary_factor = 80 * .4 = 32
#define SIN_COS_DIM 32
#define SIN_COS_MAX_SEQ_LEN 2048 // a temporary solution

// ENSURE TO CHANGE THIS FOR RUNNING ON PHONE
// #define DATASIZE 4

// #if DATASIZE == 1
//     #define QUANTIZE
//     #define QUANT_TYPE uint8_t
//     #define UNQUANT_TYPE float
// #elif DATASIZE == 4
//     #define QUANT_TYPE float
//     #define UNQUANT_TYPE float
// #endif

#define FP16 ushort

// overflow fixing
// #define FP16_POS_INF 65504.0f
// #define FP16_NEG_INF -65504.0f
// #define FP32_POS_INF 3.4e38
// #define FP32_NEG_INF -3.4e38

#define FP16_POS_INF 65504.0f
#define FP16_NEG_INF -65504.0f
#define FP32_POS_INF 3.4e30
#define FP32_NEG_INF -3.4e30


// #define LOWEST -65504 // fp16
#define LOWEST -3.4e30 // fp32

#define VOCAB_SIZE 51200

#define ATTENTION_HEADS 32
// #define HEAD_DIM (HIDDEN_SIZE / ATTENTION_HEADS)
#define HEAD_DIM 80

// #define DECODERS 2
#define DECODERS 32
// #define DECODERS 32



#define SIN_COS_BUFF_SIZE  (SIN_COS_DIM*SIN_COS_MAX_SEQ_LEN)
// #define ATTN_WEIGHTS_SIZE (MAX_SEQ_LEN*MAX_SEQ_LEN*32)
// #define LARGE_BUFFER_SIZE ATTN_WEIGHTS_SIZE
// #define DECODER_WEIGHT_SIZE 78671360
// #define LM_WEIGHTS_SIZE 131128320
// #define HIDDEN_STATES_SIZE  (4+((HIDDEN_SIZE*MAX_SEQ_LEN)/2))
// #define MASK_SIZE ((4+MAX_SEQ_LEN*MAX_SEQ_LEN)*4)
// #define POS_IDS_SIZE (4*(4+MAX_SEQ_LEN))
// #define TOTAL_DECODER_WEIGHT_SIZE (DECODERS * DECODER_WEIGHT_SIZE * DATASIZE)
// #define TOTAL_LM_WEIGHT_SIZE (4*65564160)
// #define SIN_COS_TOTAL_SIZE (4*32768)

// typedef uint16_t datatype;

#define N_PRINT 5 // legnth of elements to print

void stall() {
    size_t dum = 0;
        for (size_t i = 0; i < 1000000000; i++) {
        dum++;
    }
    std::cout << "dum: " << dum << "\n";
}

#ifdef DEBUG_2
    #define CLOCK_TYPE std::chrono::steady_clock
    #define CLOCK_INIT CLOCK_TYPE::time_point start; CLOCK_TYPE::time_point end; int64_t duration;
    #define CLOCK_START start = CLOCK_TYPE::now();
    #define CLOCK_END end = CLOCK_TYPE::now(); duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); std::cout << "duration: " << duration << "ms\n";
#endif

struct RuntimeParams {
    zdl::DlSystem::Runtime_t runtime_type;
    size_t datasize;
    bool isTFBuffer;
    std::string dataDir;
};

void printRuntime(zdl::DlSystem::Runtime_t runtime) {
    std::cout << "runtime: ";
    switch (runtime)
    {
    case zdl::DlSystem::Runtime_t::CPU:
        std::cout << "CPU32\n";
        break;
    case zdl::DlSystem::Runtime_t::GPU:
        std::cout << "GPU32\n";
        break;
    case zdl::DlSystem::Runtime_t::GPU_FLOAT16:
        std::cout << "GPU16\n";
        break;
    case zdl::DlSystem::Runtime_t::DSP:
        std::cout << "DSP08\n";
        break;
    default:
        break;
    }
}

#endif