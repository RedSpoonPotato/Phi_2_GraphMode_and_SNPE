#ifndef MAIN_MACROS_H_
#define MAIN_MACROS_H_

#include <iostream>

#define DEBUG

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
#define DATASIZE 4

#if DATASIZE == 1
    #define QUANTIZE
    #define QUANT_TYPE uint8_t
#elif DATASIZE == 4
    #define QUANT_TYPE float
#endif



#define VOCAB_SIZE 51200

#define DECODERS 3

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

#endif