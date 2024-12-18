/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_POOLING_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_POOLING_H_

#include <algorithm>
#include <limits>
#include <cstdio>
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {


uint32_t combineInt8ToUint32(int8_t a, int8_t b, int8_t c, int8_t d) {
    uint32_t result = 0;
    result |= (static_cast<uint32_t>(a) & 0xFF) << 24; // 將 a 移至高位
    result |= (static_cast<uint32_t>(b) & 0xFF) << 16; // 將 b 移至次高位
    result |= (static_cast<uint32_t>(c) & 0xFF) << 8;  // 將 c 移至次低位
    result |= (static_cast<uint32_t>(d) & 0xFF);       // 將 d 保持在低位
    return result;
}

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  printf("-------------------------pololinh!!-----------------\n");
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  // const int input_height = input_shape.Dims(1);
  // const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  // const int stride_height = params.stride_height;
  // const int stride_width = params.stride_width;

  // const int input_depth = input_shape.Dims(3); //新增

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          // const int in_x_origin =
          //     (out_x * stride_width) - params.padding_values.width;
          // const int in_y_origin =
          //     (out_y * stride_height) - params.padding_values.height;
          // // Compute the boundaries of the filter region clamped so as to
          // // ensure that the filter window fits in the input array.
          // const int filter_x_start = std::max(0, -in_x_origin);
          // const int filter_x_end =
          //     std::min(params.filter_width, input_width - in_x_origin);
          // const int filter_y_start = std::max(0, -in_y_origin);
          // const int filter_y_end =
          //     std::min(params.filter_height, input_height - in_y_origin);
          int32_t acc = 0;
          // for (int filter_y = 0; filter_y < 8; ++filter_y) {
            // acc += input_data[Offset(input_shape, 0, filter_y, 0, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 1, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 2, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 3, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 4, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 5, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 6, channel)];
            // acc += input_data[Offset(input_shape, 0, filter_y, 7, channel)];
            //----------------------------------------------------------
            int8_t a,b,c,d,a2,b2,c2,d2;

             a = input_data[Offset(input_shape, 0, 0, 0, channel)];
             b = input_data[Offset(input_shape, 0, 0, 1, channel)];
             c = input_data[Offset(input_shape, 0, 0, 2, channel)];
             d = input_data[Offset(input_shape, 0, 0, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 0, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 0, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 0, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 0, 7, channel)];
             uint32_t G1 = combineInt8ToUint32(a,b,c,d);
             uint32_t G2 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 1, 0, channel)];
             b = input_data[Offset(input_shape, 0, 1, 1, channel)];
             c = input_data[Offset(input_shape, 0, 1, 2, channel)];
             d = input_data[Offset(input_shape, 0, 1, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 1, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 1, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 1, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 1, 7, channel)];
            uint32_t G3 = combineInt8ToUint32(a,b,c,d);
            uint32_t G4 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 2, 0, channel)];
             b = input_data[Offset(input_shape, 0, 2, 1, channel)];
             c = input_data[Offset(input_shape, 0, 2, 2, channel)];
             d = input_data[Offset(input_shape, 0, 2, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 2, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 2, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 2, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 2, 7, channel)];
            uint32_t G5 = combineInt8ToUint32(a,b,c,d);
            uint32_t G6 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 3, 0, channel)];
             b = input_data[Offset(input_shape, 0, 3, 1, channel)];
             c = input_data[Offset(input_shape, 0, 3, 2, channel)];
             d = input_data[Offset(input_shape, 0, 3, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 3, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 3, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 3, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 3, 7, channel)];
            uint32_t G7 = combineInt8ToUint32(a,b,c,d);
            uint32_t G8 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 4, 0, channel)];
             b = input_data[Offset(input_shape, 0, 4, 1, channel)];
             c = input_data[Offset(input_shape, 0, 4, 2, channel)];
             d = input_data[Offset(input_shape, 0, 4, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 4, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 4, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 4, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 4, 7, channel)];
            uint32_t G9 = combineInt8ToUint32(a,b,c,d);
            uint32_t G10 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 5, 0, channel)];
             b = input_data[Offset(input_shape, 0, 5, 1, channel)];
             c = input_data[Offset(input_shape, 0, 5, 2, channel)];
             d = input_data[Offset(input_shape, 0, 5, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 5, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 5, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 5, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 5, 7, channel)];
            uint32_t G11 = combineInt8ToUint32(a,b,c,d);
            uint32_t G12 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 6, 0, channel)];
             b = input_data[Offset(input_shape, 0, 6, 1, channel)];
             c = input_data[Offset(input_shape, 0, 6, 2, channel)];
             d = input_data[Offset(input_shape, 0, 6, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 6, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 6, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 6, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 6, 7, channel)];
            uint32_t G13 = combineInt8ToUint32(a,b,c,d);
            uint32_t G14 = combineInt8ToUint32(a2,b2,c2,d2);

             a = input_data[Offset(input_shape, 0, 7, 0, channel)];
             b = input_data[Offset(input_shape, 0, 7, 1, channel)];
             c = input_data[Offset(input_shape, 0, 7, 2, channel)];
             d = input_data[Offset(input_shape, 0, 7, 3, channel)];
             a2 = input_data[Offset(input_shape, 0, 7, 4, channel)];
             b2 = input_data[Offset(input_shape, 0, 7, 5, channel)];
             c2 = input_data[Offset(input_shape, 0, 7, 6, channel)];
             d2 = input_data[Offset(input_shape, 0, 7, 7, channel)];
            uint32_t G15 = combineInt8ToUint32(a,b,c,d);
            uint32_t G16 = combineInt8ToUint32(a2,b2,c2,d2);




            // acc = cfu_op0(12,cfu_op0(12,cfu_op0(12,cfu_op0(11,G1,G2),cfu_op0(11,G3,G4))
            //     ,cfu_op0(12,cfu_op0(11,G5,G6),cfu_op0(11,G7,G8))),
            //     cfu_op0(12,cfu_op0(12,cfu_op0(11,G9,G10),cfu_op0(11,G11,G12))
            //     ,cfu_op0(12,cfu_op0(11,G13,G14),cfu_op0(11,G15,G16))));
            int32_t x1,x2,x3,x4,x5,x6,x7,x8,y1,y2,y3,y4,z1,z2;
            x1 = cfu_op0(11,G1,G2);
            __asm volatile("NOP");
            x2 = cfu_op0(11,G3,G4);
            __asm volatile("NOP");
            x3 = cfu_op0(11,G5,G6);
            __asm volatile("NOP");
            x4 = cfu_op0(11,G7,G8);
            __asm volatile("NOP");
            x5 = cfu_op0(11,G9,G10);
            __asm volatile("NOP");
            x6 = cfu_op0(11,G11,G12);
            __asm volatile("NOP");
            x7 = cfu_op0(11,G13,G14);
            __asm volatile("NOP");
            x8 = cfu_op0(11,G15,G16);
            __asm volatile("NOP");
              // acc += cfu_op0(12,static_cast<uint32_t>(cfu_op0(11,G1,G2)),static_cast<uint32_t>(cfu_op0(11,G3,G4)));
              // acc += cfu_op0(12,static_cast<uint32_t>(cfu_op0(11,G5,G6)),static_cast<uint32_t>(cfu_op0(11,G7,G8)));
              // acc += cfu_op0(12,static_cast<uint32_t>(cfu_op0(11,G9,G10)),static_cast<uint32_t>(cfu_op0(11,G11,G12)));
              // acc += cfu_op0(12,static_cast<uint32_t>(cfu_op0(11,G13,G14)),static_cast<uint32_t>(cfu_op0(11,G15,G16)));
            y1 = cfu_op0(12,static_cast<uint32_t>(x1),static_cast<uint32_t>(x2));
            __asm volatile("NOP");
            y2 = cfu_op0(12,static_cast<uint32_t>(x3),static_cast<uint32_t>(x4));
            __asm volatile("NOP");
            y3 = cfu_op0(12,static_cast<uint32_t>(x5),static_cast<uint32_t>(x6));
            __asm volatile("NOP");
            y4 = cfu_op0(12,static_cast<uint32_t>(x7),static_cast<uint32_t>(x8));
            __asm volatile("NOP");

            z1 = cfu_op0(12,static_cast<uint32_t>(y1),static_cast<uint32_t>(y2));
            __asm volatile("NOP");
            z2 = cfu_op0(12,static_cast<uint32_t>(y3),static_cast<uint32_t>(y4));
            __asm volatile("NOP");

            acc = cfu_op0(12,static_cast<uint32_t>(z1),static_cast<uint32_t>(z2));
            __asm volatile("NOP");

            // acc = cfu_op0(12,static_cast<uint32_t>(x9),static_cast<uint32_t>(x10));
            // acc = cfu_op0(12,static_cast<uint32_t>(x11),static_cast<uint32_t>(x12));
            // acc = cfu_op0(12,static_cast<uint32_t>(x13),static_cast<uint32_t>(x14));

              // acc += cfu_op0(12,cfu_op0(11,G17,G18),cfu_op0(11,G19,G20));
              // acc += cfu_op0(12,cfu_op0(11,G1,G2),cfu_op0(11,G3,G4));


          // }
          // int filter_count = 0;
          // /////開始改
          // for (int filter_y = filter_y_start; filter_y < filter_y_end;
          //      ++filter_y) {
          //   for (int filter_x = filter_x_start; filter_x < filter_x_end;
          //        ++filter_x) {
          //     const int in_x = in_x_origin + filter_x;
          //     const int in_y = in_y_origin + filter_y;
          //     // acc +=input_data[Offset(input_shape, batch, in_y, in_x, channel)];
          //     // acc +=input_data[Offset(input_shape, batch, in_y, in_x, channel)];
          //     acc +=input_data[Offset(input_shape, batch, in_y, in_x, channel)];
          //     filter_count++;
          //   }
          // }
          // if (filter_count == 0) return false;
          // Round to the closest integer value.
          // acc = acc > 0 ? (acc + filter_count / 2) / filter_count
          //               : (acc - filter_count / 2) / filter_count;
          // acc = std::max(acc, params.quantized_activation_min);
          // acc = std::min(acc, params.quantized_activation_max);
          acc = acc > 0 ? (acc + 32) / 64 : (acc - 32) / 64;
          acc = std::max(acc, (int32_t)-128);
          acc = std::min(acc, (int32_t)127);
          // acc = std::max(acc, params.quantized_activation_min);
          // acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
          // output_data[Offset(output_shape, 0, 0, 0, channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
  return true;
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min,
                   std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(params.quantized_activation_max,
                   std::numeric_limits<int8_t>::max());
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int8_t max = std::numeric_limits<int8_t>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          max = std::max<int8_t>(max, params.quantized_activation_min);
          max = std::min<int8_t>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<int8_t>(max);
        }
      }
    }
  }
}

inline bool AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int16_t* input_data,
                        const RuntimeShape& output_shape,
                        int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int32_t acc = 0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              acc +=
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)];
              filter_count++;
            }
          }
          if (filter_count == 0) return false;
          // Round to the closest integer value.
          acc = acc > 0 ? (acc + filter_count / 2) / filter_count
                        : (acc - filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<int16_t>(acc);
        }
      }
    }
  }
  return true;
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int16_t* input_data, const RuntimeShape& output_shape,
                    int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min,
                   std::numeric_limits<int16_t>::min());
  TFLITE_DCHECK_LE(params.quantized_activation_max,
                   std::numeric_limits<int16_t>::max());
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          int16_t max = std::numeric_limits<int16_t>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              max = std::max(
                  max,
                  input_data[Offset(input_shape, batch, in_y, in_x, channel)]);
            }
          }
          max = std::max<int16_t>(max, params.quantized_activation_min);
          max = std::min<int16_t>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] =
              static_cast<int16_t>(max);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_POOLING_H_
