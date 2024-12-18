/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include <math.h>
#include <stdio.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
// #include "perf.h"
#include "cfu.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // perf_enable_counter(6);
  
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  // const int dilation_width_factor = params.dilation_width_factor;
  // const int dilation_height_factor = params.dilation_height_factor;
  // const int pad_width = params.padding_values.width;
  // const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  // Get padding values.
  int pad_top = params.padding_values.height;
  // const int pad_bottom = params.padding_values.height_offset;
  int pad_left = params.padding_values.width;
  // const int pad_right = params.padding_values.width_offset;

  // const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int k = filter_height * filter_width * filter_input_depth;
  const int num_patches = output_height * output_width;

  // Fixed-size arrays.
  constexpr int max_k = 1024;
  constexpr int max_patches = 1024;
  constexpr int max_output_depth = 1024;

  TFLITE_DCHECK_LE(k, max_k);
  TFLITE_DCHECK_LE(num_patches, max_patches);
  TFLITE_DCHECK_LE(output_depth, max_output_depth);

  int8_t kernel_matrix[((max_output_depth-1)/4)+1][max_k*4] __attribute__((aligned(4)));
  // int8_t im2col_matrix[max_k][max_patches];
  int8_t im2col_matrix[((max_patches-1)/4)+1][max_k*4] __attribute__((aligned(4))); // transpose

  int32_t output_matrix[64][1024] __attribute__((aligned(4))); // transpose

  // printf("k = %d\n", k);
  // printf("n = %d\n", num_patches);
  // printf("m = %d\n", output_depth);
  // printf("(m, k, n) = (%4d, %4d, %4d)\n", output_depth, k, num_patches);
  // printf("output offset = %ld)\n", output_offset);

  // Build the kernel matrix.
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int kernel_index = 0;
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          int8_t filter_val = filter_data[Offset(
              filter_shape, out_channel, filter_y, filter_x, in_channel)];
          kernel_matrix[out_channel/4][4*kernel_index+out_channel%4] = filter_val;
          kernel_index++;
        }
      }
    }
  }

  // printf("stride_height = %d)\n", stride_height);
  // printf("pad_top = %d)\n", pad_top);
  // printf("stride_width = %d)\n", stride_width);
  // printf("pad_left = %d)\n", pad_left);
  // printf("output_activation_max = %ld)\n", output_activation_max);
  // printf("output_activation_min = %ld)\n", output_activation_min);

  // Build the im2col matrix.
  int patch_index = 0;
  int im2col_index;
  // int flag = 0;
  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_top;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_left;
      im2col_index = 0;
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        const int in_y = in_y_origin + filter_y;
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          const int in_x = in_x_origin + filter_x;
          for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
            int8_t input_val;
            if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) 
              input_val = input_data[Offset(input_shape, 0, in_y, in_x, in_channel)];
            else 
              input_val = (int8_t)(-input_offset);  // Zero padding
            im2col_matrix[patch_index/4][4*im2col_index+patch_index%4] = input_val;
            im2col_index++;
          }
        }
      }
      ++patch_index;
    }
  }

  // Perform matrix multiplication.
  // kernel_matrix[out_channel][index]
  // im2col_matrix[patch_index][index]
  // int ADDR_BITS = 10;
  // int buffer_row = pow(2, ADDR_BITS);
  // int sqrt_of_Cbuf    = floor(sqrt(buffer_row/4));
  // int A_B_buf_max     = floor(buffer_row/k);
  // int A_matrix_need   = output_depth%4==0 ? output_depth/4 : output_depth/4+1;
  // int B_matrix_need   = num_patches%4==0 ? num_patches/4 : num_patches/4+1;
  // int tiling_num = sqrt_of_Cbuf < A_B_buf_max ? sqrt_of_Cbuf : A_B_buf_max;
  // tiling_num = tiling_num < A_matrix_need ? tiling_num : A_matrix_need;
  // tiling_num = tiling_num < B_matrix_need ? tiling_num : B_matrix_need;
  // int tiling_num = 4;
  // printf("buffer_row = %d\n", buffer_row);
  // printf("sqrt_of_Cbuf = %d\n", sqrt_of_Cbuf);
  // printf("A_B_buf_max = %d\n", A_B_buf_max);
  // printf("A_matrix_need = %d\n", A_matrix_need);
  // printf("B_matrix_need = %d\n", B_matrix_need);
  // printf("tiling_num = %d\n", tiling_num);
  int M_index_buf = -1;
  int first_round_flag;
  int cnt_256;

  for(int M_index = 0; M_index < output_depth; M_index = M_index+16) {
    for(int N_index = 0; N_index < num_patches; N_index = N_index+16) {
      first_round_flag = (M_index_buf != M_index) ? 1 : 0;
      uint32_t combined_2 = ((uint32_t)(first_round_flag & 0x01) << 10) | ((uint32_t)k & 0x3FF);
      cfu_op0(/* funct7= */ 1, /* in0= */ input_offset, /* in1= */ combined_2); // reset C_matrix_buffer

      if(N_index == 0 && M_index == 0)    cnt_256 = 256; // first time
      else                                cnt_256 = 0;

      for(int tiling_round=0; tiling_round<4; tiling_round++) {
        int out_channels_check = M_index + tiling_round*4;
        int patches_check = N_index + tiling_round*4;
        int out_channels_check_d_4 = out_channels_check/4;
        int patches_check_d_4 = patches_check/4;
        if(M_index_buf != M_index) { // M has been changed to the next round, you should input the new M
          // for(int k_index = 0; k_index < k; k_index = k_index+1) {
          //   uint32_t in_0 = 0;
          //   uint32_t in_1 = 0;

          //   int8_t* kernel_ptr    = &kernel_matrix[out_channels_check / 4][4 * k_index];
          //   int8_t* im2col_ptr    = &im2col_matrix[patches_check / 4][4 * k_index];

          //   in_0 = *((uint32_t*)kernel_ptr);
          //   in_1 = *((uint32_t*)im2col_ptr);

          //   cfu_op0(/* funct7= */ 4, /* in0= */ in_0, /* in1= */ in_1); // input
          // }
          for(int k_index = 0; k_index < k; k_index = k_index+1) {
            uint32_t in_0 = 0;
            uint32_t in_1 = 0;
            // out_channels_check = M_index + tiling_round*4;
            // patches_check = N_index + tiling_round*4;
            int8_t kernel_values[4];
            int8_t im2col_values[4];

            int8_t* kernel_ptr    = &kernel_matrix[out_channels_check_d_4][4 * k_index];
            // int8_t* kernel_ptr_p3 = &kernel_matrix[out_channels_check / 4][4 * k_index + 3];
            int8_t* im2col_ptr    = &im2col_matrix[patches_check_d_4][4 * k_index];
            // int8_t* im2col_ptr_p3 = &im2col_matrix[patches_check / 4][4 * k_index + 3];


            if(out_channels_check+3 < output_depth && patches_check+3 < num_patches) {
              in_0 = *((uint32_t*)kernel_ptr);
              in_1 = *((uint32_t*)im2col_ptr);
            }
            else {
              for (int j = 0; j < 4; ++j) {
                kernel_values[j] =
                    (out_channels_check+j >= output_depth)
                        ? (int8_t)(-input_offset)
                        : kernel_matrix[out_channels_check_d_4][4*k_index + j];
                im2col_values[j] =
                    (patches_check+j >= num_patches)
                        ? (int8_t)(-input_offset)
                        : im2col_matrix[patches_check_d_4][4*k_index + j];
              }
              // package data
              for (int j = 0; j < 4; ++j) {
                in_0 |= ((uint32_t)(uint8_t)kernel_values[j] << j * 8);
                in_1 |= ((uint32_t)(uint8_t)im2col_values[j] << j * 8);
              }
            }

            int32_t cfu_out = cfu_op0(/* funct7= */ 4, /* in0= */ in_0, /* in1= */ in_1); // input

            if(cnt_256 < 256) {
              int input_M, input_N;
              if(first_round_flag && M_index!=0) {
                  input_M = (M_index-16)+(cnt_256/4)%16;
                  input_N = (num_patches-16)+(cnt_256%4)+(cnt_256/64)*4;
              }
              else {
                  input_M = (M_index)+(cnt_256/4)%16;
                  input_N = (N_index-16)+(cnt_256%4)+(cnt_256/64)*4;
              }
              output_matrix[input_M][input_N] = cfu_out;

              // if (bias_data) {
              //   cfu_out += bias_data[input_M];
              // }
              // cfu_out = MultiplyByQuantizedMultiplier(cfu_out, output_multiplier[input_M], output_shift[input_M]);
              // cfu_out += output_offset;
              // cfu_out = std::max(cfu_out, output_activation_min);
              // cfu_out = std::min(cfu_out, output_activation_max);

              // int out_y = input_N / output_width;
              // int out_x = input_N % output_width;
              // output_data[Offset(output_shape, 0, out_y, out_x, input_M)] = static_cast<int8_t>(cfu_out);

            }
            cnt_256 = cnt_256==256 ? cnt_256 : cnt_256+1;

          }
        }
        else {
          // for(int k_index = 0; k_index < k; k_index = k_index+2) {
          //   uint32_t in_0 = 0;
          //   uint32_t in_1 = 0;

          //   int8_t* im2col_ptr_1    = &im2col_matrix[patches_check / 4][4 * k_index];
          //   int8_t* im2col_ptr_2    = &im2col_matrix[patches_check / 4][4 * (k_index+1)];

          //   in_0 = *((uint32_t*)im2col_ptr_1);
          //   in_1 = (k_index+1 >= k) ? 0 : *((uint32_t*)im2col_ptr_2);

          //   cfu_op0(/* funct7= */ 4, /* in0= */ in_0, /* in1= */ in_1); // input
          // }
          for(int k_index = 0; k_index < k; k_index = k_index+2) {
            uint32_t in_0 = 0;
            uint32_t in_1 = 0;
            // out_channels_check = M_index + tiling_round*4;
            // patches_check = N_index + tiling_round*4;
            int8_t im2col_values_1[4];
            int8_t im2col_values_2[4];

            int8_t* im2col_ptr_1    = &im2col_matrix[patches_check_d_4][4 * k_index];
            // int8_t* im2col_ptr_1_p3 = &im2col_matrix[patches_check / 4][4 * k_index + 3];
            int8_t* im2col_ptr_2    = &im2col_matrix[patches_check_d_4][4 * (k_index+1)];
            // int8_t* im2col_ptr_2_p3 = &im2col_matrix[patches_check / 4][4 * (k_index+1) + 3];


            if(!(k==27 && k_index==26) && out_channels_check+3 < output_depth && patches_check+3 < num_patches) {
              in_0 = *((uint32_t*)im2col_ptr_1);
              in_1 = *((uint32_t*)im2col_ptr_2);
            }
            else {
              for (int j = 0; j < 4; ++j) {
                im2col_values_1[j] =
                    (patches_check+j >= num_patches)
                        ? (int8_t)(-input_offset)
                        : im2col_matrix[patches_check/4][4*k_index + j];
                im2col_values_2[j] =
                    (patches_check+j >= num_patches) ? (int8_t)(-input_offset)
                  : (k_index+1 >= k) ? 0 
                  : im2col_matrix[patches_check/4][4*(k_index+1) + j];
              }
              // package data
              for (int j = 0; j < 4; ++j) {
                in_0 |= ((uint32_t)(uint8_t)im2col_values_1[j] << j * 8);
                in_1 |= ((uint32_t)(uint8_t)im2col_values_2[j] << j * 8);
              }
            }

            int32_t cfu_out = cfu_op0(/* funct7= */ 4, /* in0= */ in_0, /* in1= */ in_1); // input
            
            if(cnt_256 < 256) {
              int input_M, input_N;
              if(first_round_flag && M_index!=0) {
                  input_M = (M_index-16)+(cnt_256/4)%16;
                  input_N = (num_patches-16)+(cnt_256%4)+(cnt_256/64)*4;
              }
              else {
                  input_M = (M_index)+(cnt_256/4)%16;
                  input_N = (N_index-16)+(cnt_256%4)+(cnt_256/64)*4;
              }

              output_matrix[input_M][input_N] = cfu_out;

              // if (bias_data) {
              //   cfu_out += bias_data[input_M];
              // }
              // cfu_out = MultiplyByQuantizedMultiplier(cfu_out, output_multiplier[input_M], output_shift[input_M]);
              // cfu_out += output_offset;
              // cfu_out = std::max(cfu_out, output_activation_min);
              // cfu_out = std::min(cfu_out, output_activation_max);

              // int out_y = input_N / output_width;
              // int out_x = input_N % output_width;
              // output_data[Offset(output_shape, 0, out_y, out_x, input_M)] = static_cast<int8_t>(cfu_out);

            }
            cnt_256 = cnt_256==256 ? cnt_256 : cnt_256+1;
          }
        }
      }
      
      while(cnt_256 != 256) {
        int32_t cfu_out = cfu_op0(/* funct7= */ 6, /* in0= */ 0, /* in1= */ 0); // output 32-bit answer
        if(cnt_256 < 256) {
          int input_M, input_N;
          if(first_round_flag && M_index!=0) {
              input_M = (M_index-16)+(cnt_256/4)%16;
              input_N = (num_patches-16)+(cnt_256%4)+(cnt_256/64)*4;
          }
          else {
              input_M = (M_index)+(cnt_256/4)%16;
              input_N = (N_index-16)+(cnt_256%4)+(cnt_256/64)*4;
          }

          output_matrix[input_M][input_N] = cfu_out;

          // if (bias_data) {
          //   cfu_out += bias_data[input_M];
          // }
          // cfu_out = MultiplyByQuantizedMultiplier(cfu_out, output_multiplier[input_M], output_shift[input_M]);
          // cfu_out += output_offset;
          // cfu_out = std::max(cfu_out, output_activation_min);
          // cfu_out = std::min(cfu_out, output_activation_max);

          // int out_y = input_N / output_width;
          // int out_x = input_N % output_width;
          // output_data[Offset(output_shape, 0, out_y, out_x, input_M)] = static_cast<int8_t>(cfu_out);

        }
        cnt_256 = cnt_256 + 1;
      }

      int input_m = 16;
      int input_n = 16;
      int input_k = k;
      // printf("m = %d, n = %d, k = %d\n", input_m, input_n, k);
      input_m &= 0xFF; // Mask to keep only the lowest 8 bits
      input_n &= 0xFF; // 8
      input_k &= 0x3FF; // 10
      // Combine the numbers into a single 32-bit integer
      unsigned int combined = ((unsigned int)input_k << 22) |  // Shift input_m to bits 31-22
                              ((unsigned int)input_m << 14) |  // Shift input_n to bits 21-14
                              ((unsigned int)input_n << 6);   // Shift input_k to bits 13-6

      cfu_op0(/* funct7= */ 5, /* in0= */ combined, /* in1= */ 0); // begin calculation

      if(N_index+16 == num_patches && M_index+16 == output_depth) {
        cnt_256 = 257;
        for(int tiling_round=0; tiling_round<16; tiling_round=tiling_round+4) {
          for(int i=0; i<16; i=i+1) {
            if(M_index+i < output_depth) {
              for(int j=0; j<4; j=j+1) {

                int32_t cfu_out = cfu_op0(/* funct7= */ 6, /* in0= */ 0, /* in1= */ 0); // output 32-bit answer

                if(M_index+i>=output_depth || N_index+j+tiling_round>=num_patches) continue;
                if(N_index+j+tiling_round < num_patches) {
                  output_matrix[M_index+i][N_index+j+tiling_round] = cfu_out;
                  // if (bias_data) {
                  //   cfu_out += bias_data[M_index+i];
                  // }
                  // cfu_out = MultiplyByQuantizedMultiplier(cfu_out, output_multiplier[M_index+i], output_shift[M_index+i]);
                  // cfu_out += output_offset;
                  // cfu_out = std::max(cfu_out, output_activation_min);
                  // cfu_out = std::min(cfu_out, output_activation_max);

                  // int out_y = (N_index+j+tiling_round) / output_width;
                  // int out_x = (N_index+j+tiling_round) % output_width;
                  // output_data[Offset(output_shape, 0, out_y, out_x, M_index+i)] = static_cast<int8_t>(cfu_out);
                }
              }
            }
          }
        }
      }
      M_index_buf = M_index;
    }
  }
  int32_t cfu_out;
  for(int i=0; i<output_depth; i++) {
    for(int j=0; j<num_patches; j++) {
      cfu_out = output_matrix[i][j];
      if (bias_data) {
        cfu_out += bias_data[i];
      }
      cfu_out = MultiplyByQuantizedMultiplier(cfu_out, output_multiplier[i], output_shift[i]);
      cfu_out += output_offset;
      cfu_out = std::max(cfu_out, output_activation_min);
      cfu_out = std::min(cfu_out, output_activation_max);

      int out_y = j / output_width;
      int out_x = j % output_width;
      output_data[Offset(output_shape, 0, out_y, out_x, i)] = static_cast<int8_t>(cfu_out);
    }
  }

  // perf_disable_counter(6);
}

// inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier,
//                                            int32_t shift) {
//   if (shift < 0) {
//     cfu_op0(7, x, quantized_multiplier);
//     __asm volatile("NOP");
//     int val = cfu_op0(8, 0, 0);
//     __asm volatile("NOP");
//     return cfu_op0(9, val, -shift);
//   }
//   else [[unlikely]] {
//     cfu_op0(7, x << shift, quantized_multiplier);
//     __asm volatile("NOP");
//     return cfu_op0(8, 0, 0);
//   }
// }

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_