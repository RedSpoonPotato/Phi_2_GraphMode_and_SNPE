#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

// #include "tensorflow/core/framework/tensor_shape.h"
// #include "tensorflow/core/framework/register_types.h"

// #include <iostream> // temporary for print statements
// #include <initializer_list>

// .Input("params: float32")

// change this to appropriate size
#define DECODER_OUTPUT_SIZE 10
#define MAX_SEQ_LEN 2048
#define HIDDEN_SIZE 2560

/*
must combine inputs:
hd_states:
hd_states_dims
kv
kv_dims
*/


using namespace tensorflow;

REGISTER_OP("Decode")
    /* Inputs */
    // .Input("hd_states: float32")
    .Input("attn_and_kv_in: float32")
    .Input("mask: float32")
    .Input("position_ids: float32")
    // .Input("kv: float32")
    /* Weights */
    .Input("decoder_weights: float32")
    .Input("lm_head_weights: float32")
    .Input("sin_cached: float32")
    .Input("cos_cached: float32")
    /* Init Params */
    /* Outputs */
    .Output("attn_and_kv_out: float32")
    // .Output("attn_output: float32")
    // .Output("attn_output_dims: float32")
    // .Output("kv_out: float32")
    // .Output("kv_out_dims: float32")
    
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0)); // could be causing a memory leak

      // MODDED****
      // c->set_output(1, c->input(1)); // makre sure input buffer size is big enough
      // c->set_output(2, c->input(3)); // kv
      // c->set_output(3, c->input(1));
      return OkStatus();
    });

class NameOfOP : public OpKernel {
 public:
  explicit NameOfOP(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    /* inputs */
    const Tensor& input_0 = context->input(0);
    auto input_0_flat = input_0.flat<float>();
    const Tensor& input_1 = context->input(1);
    auto input_1_flat = input_1.flat<float>();
    const Tensor& input_2 = context->input(2);
    auto input_2_flat = input_2.flat<float>();
    const Tensor& input_3 = context->input(3);
    auto input_3_flat = input_3.flat<float>();
    const Tensor& input_4 = context->input(4);
    auto input_4_flat = input_4.flat<float>();
    const Tensor& input_5 = context->input(5);
    auto input_5_flat = input_5.flat<float>();
    const Tensor& input_6 = context->input(6);
    auto input_6_flat = input_6.flat<float>();
    // const Tensor& input_7 = context->input(7);
    // auto input_7_flat = input_7.flat<float>();


    // modifiying index params below to all be 0 (rather than 0,1,2,3)

    // std::cout << "\ninput vector shape:" << input_tensor.shape() << "\n\n";
    // std::cout << "first element of tensor:" << context->input(1).flat<int>().data()[0] << "\n\n";

    // int first_elem_of_num = context->input(1).flat<unsigned int>().data()[0]; 

    // std::initializer_list<int64_t> out_shape = {1, 1, 1, DECODER_OUTPUT_SIZE};
    // tensorflow::TensorShape outShape;
    // TensorShape::BuildTensorShape(out_shape, &outShape);

    /* outputs */
    Tensor* output_0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_0.shape(), &output_0));
    auto output_0_flat = output_0->flat<float>();


    // modified
    // Tensor* output_1 = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(1, input_0.shape(), &output_1));
    // auto output_1_flat = output_1->flat<float>();


    // Tensor* output_1 = NULL;
    // // OP_REQUIRES_OK(context, context->allocate_output(1, input_1.shape(), &output_1));
    // context, context->allocate_output(1, input_1.shape(), &output_1);
    // auto output_1_flat = output_1->flat<float>();
    // Tensor* output_2 = NULL;
    // // OP_REQUIRES_OK(context, context->allocate_output(2, input_3.shape(), &output_2));
    // context, context->allocate_output(2, input_3.shape(), &output_2);
    // auto output_2_flat = output_2->flat<float>();
    // Tensor* output_3 = NULL;
    // // OP_REQUIRES_OK(context, context->allocate_output(3, input_1.shape(), &output_3));
    // context, context->allocate_output(3, input_1.shape(), &output_3);
    // auto output_3_flat = output_3->flat<float>();

  }
};

REGISTER_KERNEL_BUILDER(Name("Decode").Device(DEVICE_CPU), NameOfOP);