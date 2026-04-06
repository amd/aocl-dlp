# AOCL-DLP (Deep Learning Primitives)

AOCL-DLP is a library designed to provide optimized deep learning primitives for AMD processors. It implements Low Precision GEMM and batch GEMM for deep learning applications, supporting multiple data types as well as pre-operations and post-operations. The library is tailored to leverage the full potential of AMD hardware, ensuring efficient computation, scalability, and accelerated deep learning workloads.

## Features of AOCL-DLP

- **Highly Optimized GEMM Operations**: Implements high-performance matrix multiplication operations targeting AMD CPUs with specialized instruction sets (AVX2, AVX512, AVX512_VNNI, AVX512_BF16, AVX512_FP16)
- **Multiple Data Type Support**: Works with various precision formats including FP32, FP16, BF16, INT8, INT4 for efficient model training and inference
- **Pre-operations and Post-operations**: Includes comprehensive support for operations common in deep learning workloads
- **Batch GEMM Support**: Optimized for handling multiple GEMM operations simultaneously
- **Symmetric Quantization Support**: Provides specialized routines for symmetric quantization
- **Extensive Thread Support**: Optimized for parallel execution via OpenMP


## Data Type Terminology

| Terminology | Description |
|-------------|-------------|
| u4/s4       | uint4_t or int4_t |
| u8/s8       | uint8_t or int8_t |
| u32/s32     | uint32_t or int32_t |
| f32         | float32 |
| f16         | float16 |
| bf16        | bfloat16 |

## Supported Data Types

AOCL-DLP provides support for various data type combinations for GEMM operations:

| Input A | Input B | Output C | Accumulator | Function Suffix |
|---------|---------|----------|-------------|-----------------|
| u8/s8 | s8 | s32/s8/u8/f32/bf16 | s32 | <u8\|s8>s8s32o<s32\|s8\|u8\|f32\|bf16> |
| bf16/f32 | s8 | s32/s8/u8/f32/bf16 | s32 | <bf16\|f32>s8s32o<s32\|s8\|u8\|f32\|bf16> |
| bf16 | s4/u4 | f32/bf16 | f32 | bf16<s4\|u4>f32o<f32\|bf16> |
| bf16 | bf16 | f32/bf16 | f32 | bf16bf16f32o<f32\|bf16> |
| f32 | f32 | f32 | f32 | f32f32f32of32 |
| f16 | f16 | f16 | f16 | f16f16f16of16 |
| u8 | s4 | s32 | s32 | u8s4s32os32 |

**Notes:**
- `u8s4s32os32` only has `reorder` and `get_reorder_buf_size` APIs (no gemm)
- `s8s8s32o<f32|bf16>` also has `_sym_quant` variants for symmetric quantization
- Mixed-precision reorder: `f32obf16` (converts f32 input to bf16 reordered output)

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

## Hardware Requirements

AOCL-DLP is optimized for AMD processors and requires specific minimum architecture support based on the functions being used:

### Minimum Architecture Requirements

| Function Type          | Minimum Required ISA      | Available On                                        |
|------------------------|---------------------------|-----------------------------------------------------|
| f32 (float)            | AVX2/FMA3                 | AMD Zen1 and newer, Intel Haswell and newer         |
| bf16 (bfloat16)        | AVX2/FMA3                 | AMD Zen1 and newer, Intel Haswell and newer         |
| ↳                      | AVX512_BF16 (optimal)     | AMD Zen4 and newer, Intel Cooper Lake and newer     |
| int8 (int8, uint8)     | AVX512_VNNI               | AMD Zen4 and newer, Intel Cascade Lake and newer    |

While optimized for AMD processors, the library is compatible with any x86_64 CPU that meets these minimum requirements. For best performance on AMD processors, it is recommended to use Zen4 or newer architectures which support all instruction sets.

## Build

Refer to [BUILD.md](BUILD.md) for detailed build instructions, including support for both GNU Make and Ninja.

## Install

Refer to [INSTALL.md](INSTALL.md) for installation steps, including default and custom prefix install.

## Wiki

Refer to [Wiki](https://github.com/amd/aocl-dlp/wiki) for more information.

## License

AOCL-DLP is licensed under the terms and conditions as specified in the LICENSE file.
