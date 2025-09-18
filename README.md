# AOCL-DLP (Deep Learning Primitives)

AOCL-DLP is a library designed to provide optimized deep learning primitives for AMD processors. It implements Low Precision GEMM (LPGEMM) for machine learning applications, supporting multiple data types as well as pre-operations and post-operations. The library is tailored to leverage the full potential of AMD hardware, ensuring efficient computation, scalability, and accelerated machine learning workloads .

## Features of AOCL-DLP

- **Highly Optimized GEMM Operations**: Implements high-performance matrix multiplication operations targeting AMD CPUs with specialized instruction sets (AVX2, AVX512, AVX512_VNNI, AVX512_BF16)
- **Multiple Data Type Support**: Works with various precision formats for efficient model training and inference
- **Pre-operations and Post-operations**: Includes comprehensive support for operations common in deep learning workloads
- **Batch GEMM Support**: Optimized for handling multiple GEMM operations simultaneously
- **Symmetric Quantization Support**: Provides specialized routines for symmetric quantization
- **Extensive Thread Support**: Optimized for parallel execution via OpenMP

## Supported Data Types

AOCL-DLP provides support for various data type combinations for GEMM operations:

| Input A    | Input B    | Output C   | Accumulator | Function Suffix     |
|------------|------------|------------|-------------|---------------------|
| float      | float      | float      | float       | f32f32f32of32       |
| bfloat16   | bfloat16   | float      | float       | bf16bf16f32of32     |
| bfloat16   | bfloat16   | bfloat16   | float       | bf16bf16f32obf16    |
| bfloat16   | int8_t     | float      | float       | bf16s4f32of32       |
| bfloat16   | int8_t     | bfloat16   | float       | bf16s4f32obf16      |
| uint8_t    | int8_t     | int32_t    | int32_t     | u8s8s32os32         |
| uint8_t    | int8_t     | int8_t     | int32_t     | u8s8s32os8          |
| uint8_t    | int8_t     | uint8_t    | int32_t     | u8s8s32ou8          |
| uint8_t    | int8_t     | float      | int32_t     | u8s8s32of32         |
| uint8_t    | int8_t     | bfloat16   | int32_t     | u8s8s32obf16        |
| int8_t     | int8_t     | int32_t    | int32_t     | s8s8s32os32         |
| int8_t     | int8_t     | int8_t     | int32_t     | s8s8s32os8          |
| int8_t     | int8_t     | uint8_t    | int32_t     | s8s8s32ou8          |
| int8_t     | int8_t     | float      | int32_t     | s8s8s32of32         |
| int8_t     | int8_t     | bfloat16   | int32_t     | s8s8s32obf16        |

Additionally, symmetric quantization variants are provided for:
- `s8s8s32of32_sym_quant`
- `s8s8s32obf16_sym_quant`

## Pre-Operations

AOCL-DLP supports the following pre-operations:

| Pre-Op Type | Description |
|-------------|-------------|
| Zero Point  | Input tensor zero point compensation for quantized operations |
| Scale Factor| Input tensor scaling for quantized operations |

These pre-operations support different storage types:
- `AOCL_GEMM_F32` (float)
- `AOCL_GEMM_BF16` (bfloat16)
- `AOCL_GEMM_INT8` (int8_t)
- `AOCL_GEMM_UINT8` (uint8_t)
- `AOCL_GEMM_INT32` (int32_t)

## Post-Operations

AOCL-DLP supports the following post-operations:

| Post-Op Type | Description |
|--------------|-------------|
| ELTWISE      | Element-wise operations including activation functions |
| BIAS         | Bias addition to result |
| SCALE        | Scaling operation |
| MATRIX_ADD   | Matrix addition with optional scaling |
| MATRIX_MUL   | Matrix multiplication with optional scaling |

### Eltwise Algorithm Types

The following eltwise algorithm types are supported:

| Eltwise Type | Description |
|--------------|-------------|
| RELU         | Rectified Linear Unit activation |
| PRELU        | Parametric Rectified Linear Unit activation |
| GELU_TANH    | Gaussian Error Linear Unit (tanh approximation) |
| GELU_ERF     | Gaussian Error Linear Unit (erf approximation) |
| CLIP         | Clipping values to a specified range |
| SWISH        | Swish activation function |
| TANH         | Hyperbolic tangent activation |
| SIGMOID      | Sigmoid activation function |

## Utility Functions

Standalone utility functions include:

| Utility Function | Description |
|-----------------|-------------|
| gelu_tanh_f32   | GELU activation with tanh approximation for float |
| gelu_erf_f32    | GELU activation with erf approximation for float |
| softmax_f32     | Softmax function for float |

## Eltwise Operations

The library provides specialized element-wise operations:

| Eltwise Operation | Description |
|------------------|-------------|
| bf16of32        | bfloat16 input to float output |
| bf16obf16       | bfloat16 input to bfloat16 output |
| f32of32         | float input to float output |
| f32obf16        | float input to bfloat16 output |
| f32os32         | float input to int32_t output |
| f32os8          | float input to int8_t output |

## Hardware Support

AOCL-DLP is optimized for AMD processors with the following instruction sets:
- AVX2/FMA3      (available on Zen1 and newer)
- AVX512         (available on Zen4 and newer)
- AVX512_VNNI    (available on Zen4 and newer, for int8 operations)
- AVX512_BF16    (available on Zen4 and newer, for bfloat16 operations)

It also runs on any x86_64 (AMD64) CPU that supports these instruction sets.

## Build

Refer to [BUILD.md](BUILD.md) for detailed build instructions, including support for both GNU Make and Ninja.

## Install

Refer to [INSTALL.md](INSTALL.md) for installation steps, including default and custom prefix install.

## License

AOCL-DLP is licensed under the terms and conditions as specified in the LICENSE file.
