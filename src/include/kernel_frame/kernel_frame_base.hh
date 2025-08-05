/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#pragma once

#include <cstdint>
#include <vector>

#include "classic/dlp_base_types.h"
#include "utils/type_utils.hh"

namespace dlp::kernel_frame {

enum class kernelFrameError
{
    success,
    failure
};

enum class kernelRoutineType : uint8_t
{
    gemm = 0,
    reorder,
    max_kernel_routines
};

enum class kernelDatatype : uint8_t
{
    invalid = 0,
    u8s8s32os32,
    u8s8s32os8,
    bf16bf16f32obf16,
    bf16bf16f32of32,
    f32f32f32of32,
    max_kernel_datatypes
};

// Enum for individual datatypes (similar to AOCL_STORAGE_TYPE)
enum class DataType : uint8_t
{
    invalid = 0,
    s8,   // int8_t
    u8,   // uint8_t
    s16,  // int16_t
    u16,  // uint16_t
    bf16, // bfloat16
    s32,  // int32_t
    u32,  // uint32_t
    f32,  // float
    max_datatypes
};

enum class kernelOps : uint8_t
{
    invalid = 0,
    bias,
    geluTanh,
    geluErf,
    relu,
    reluScale,
    clip,
    downscale,
    matAdd,
    matMul,
    swish,
    tanh,
    sigmoid,
    max_kernel_ops
};

// Enum for matrix storage format
enum class storageFormat : uint8_t
{
    rowMajor = 0, // Row-major layout (C/C++ style)
    colMajor = 1  // Column-major layout (Fortran style)
};

struct kernelOpsMetaData
{
    kernelOps     type;
    DataType      scaleFactorDt; // Data type of the scale factor
    bool          scalarScaleFactorRequired;
    bool          vectorScaleFactorRequired;
    DataType      zeroPointDt; // Data type of the zero point
    bool          scalarZeroPointRequired;
    bool          vectorZeroPointRequired;
    DataType      paramStorageDt; // Data type of the parameter storage
    storageFormat cMatFormat; // Storage format of the C matrix (row-major or
                              // column-major)

    kernelOpsMetaData()
        : type(kernelOps::max_kernel_ops)
        , scaleFactorDt(DataType::max_datatypes)
        , scalarScaleFactorRequired(false)
        , vectorScaleFactorRequired(false)
        , zeroPointDt(DataType::max_datatypes)
        , scalarZeroPointRequired(false)
        , vectorZeroPointRequired(false)
        , paramStorageDt(DataType::max_datatypes)
        , cMatFormat(storageFormat::rowMajor)
    {
    }

    kernelOpsMetaData(kernelOps     type,
                      DataType      _scaleFactorDt,
                      bool          _scalarScaleFactorRequired,
                      bool          _vectorScaleFactorRequired,
                      DataType      _zeroPointDt,
                      bool          _scalarZeroPointRequired,
                      bool          _vectorZeroPointRequired,
                      DataType      _paramStorageDt,
                      storageFormat _CFormat)
        : type(type)
        , scaleFactorDt(_scaleFactorDt)
        , scalarScaleFactorRequired(_scalarScaleFactorRequired)
        , vectorScaleFactorRequired(_vectorScaleFactorRequired)
        , zeroPointDt(_zeroPointDt)
        , scalarZeroPointRequired(_scalarZeroPointRequired)
        , vectorZeroPointRequired(_vectorZeroPointRequired)
        , paramStorageDt(_paramStorageDt)
        , cMatFormat(_CFormat)
    {
    }

    kernelOpsMetaData(const kernelOpsMetaData& other)
        : type(other.type)
        , scaleFactorDt(other.scaleFactorDt)
        , scalarScaleFactorRequired(other.scalarScaleFactorRequired)
        , vectorScaleFactorRequired(other.vectorScaleFactorRequired)
        , zeroPointDt(other.zeroPointDt)
        , scalarZeroPointRequired(other.scalarZeroPointRequired)
        , vectorZeroPointRequired(other.vectorZeroPointRequired)
        , paramStorageDt(other.paramStorageDt)
        , cMatFormat(other.cMatFormat)
    {
    }

    kernelOpsMetaData(kernelOpsMetaData&& other)
        : type(other.type)
        , scaleFactorDt(other.scaleFactorDt)
        , scalarScaleFactorRequired(other.scalarScaleFactorRequired)
        , vectorScaleFactorRequired(other.vectorScaleFactorRequired)
        , zeroPointDt(other.zeroPointDt)
        , scalarZeroPointRequired(other.scalarZeroPointRequired)
        , vectorZeroPointRequired(other.vectorZeroPointRequired)
        , paramStorageDt(other.paramStorageDt)
        , cMatFormat(other.cMatFormat)
    {
    }

    kernelOpsMetaData& operator=(const kernelOpsMetaData& other)
    {
        if (this != &other) {
            this->type                      = other.type;
            this->scaleFactorDt             = other.scaleFactorDt;
            this->scalarScaleFactorRequired = other.scalarScaleFactorRequired;
            this->vectorScaleFactorRequired = other.vectorScaleFactorRequired;
            this->zeroPointDt               = other.zeroPointDt;
            this->scalarZeroPointRequired   = other.scalarZeroPointRequired;
            this->vectorZeroPointRequired   = other.vectorZeroPointRequired;
            this->paramStorageDt            = other.paramStorageDt;
            this->cMatFormat                = other.cMatFormat;
        }
        return *this;
    }

    kernelOpsMetaData& operator=(kernelOpsMetaData&& other)
    {
        if (this != &other) {
            *this = other;
        }
        return *this;
    }

    bool operator==(const kernelOpsMetaData& rhs) const
    {
        return (
            (this->type == rhs.type)
            && (this->scaleFactorDt == rhs.scaleFactorDt)
            && (this->scalarScaleFactorRequired
                == rhs.scalarScaleFactorRequired)
            && (this->vectorScaleFactorRequired
                == rhs.vectorScaleFactorRequired)
            && (this->zeroPointDt == rhs.zeroPointDt)
            && (this->scalarZeroPointRequired == rhs.scalarZeroPointRequired)
            && (this->vectorZeroPointRequired == rhs.vectorZeroPointRequired)
            && (this->paramStorageDt == rhs.paramStorageDt)
            && (this->cMatFormat == rhs.cMatFormat));
    }

    bool operator!=(const kernelOpsMetaData& rhs) const
    {
        return !(*this == rhs);
    }
};

struct kernelInfo
{
    md_t               mr;
    md_t               nr;
    md_t               k_unroll;
    bool               is_beta_zero;
    bool               is_alpha_one;
    kernelOpsMetaData* kOpsArr;
    std::size_t        kOpsArrSize;
    bool               anyKOpsOrder;

    kernelInfo(md_t                               mr,
               md_t                               nr,
               md_t                               k_unroll,
               bool                               is_beta_zero,
               bool                               is_alpha_one,
               std::unique_ptr<kernelOpsMetaData> kOpsArr,
               std::size_t                        kOpsArrSize,
               bool                               anyKOpsOrder)
        : mr(mr)
        , nr(nr)
        , k_unroll(k_unroll)
        , is_beta_zero(is_beta_zero)
        , is_alpha_one(is_alpha_one)
        , kOpsArr(((kOpsArr != nullptr) && (kOpsArrSize > 0))
                      ? kOpsArr.release()
                      : nullptr)
        , kOpsArrSize(((kOpsArr != nullptr) && (kOpsArrSize > 0)) ? kOpsArrSize
                                                                  : 0)
        , anyKOpsOrder(anyKOpsOrder)
    {
    }

    kernelInfo(const kernelInfo& other)
        : mr(other.mr)
        , nr(other.nr)
        , k_unroll(other.k_unroll)
        , is_beta_zero(other.is_beta_zero)
        , is_alpha_one(other.is_alpha_one)
        , kOpsArr(nullptr)
        , kOpsArrSize(((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0))
                          ? other.kOpsArrSize
                          : 0)
        , anyKOpsOrder(other.anyKOpsOrder)
    {
        if ((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0)) {
            this->kOpsArr =
                kernelInfo::allocateKernelOpsArray(other.kOpsArrSize);
            for (std::size_t i = 0; i < other.kOpsArrSize; i++) {
                this->kOpsArr[i] = other.kOpsArr[i];
            }
        }
    }

    // Unsafe constructor, used purely for internal and performance reasons.
    kernelInfo(kernelInfo* other)
        : mr(other->mr)
        , nr(other->nr)
        , k_unroll(other->k_unroll)
        , is_beta_zero(other->is_beta_zero)
        , is_alpha_one(other->is_alpha_one)
        , kOpsArr(((other->kOpsArr != nullptr) && (other->kOpsArrSize > 0))
                      ? other->kOpsArr
                      : nullptr)
        , kOpsArrSize(((other->kOpsArr != nullptr) && (other->kOpsArrSize > 0))
                          ? other->kOpsArrSize
                          : 0)
        , anyKOpsOrder(other->anyKOpsOrder)
    {
        if ((other->kOpsArr != nullptr) && (other->kOpsArrSize > 0)) {
            other->kOpsArr     = nullptr;
            other->kOpsArrSize = 0;
        }
    }

    kernelInfo(kernelInfo&& other)
        : mr(other.mr)
        , nr(other.nr)
        , k_unroll(other.k_unroll)
        , is_beta_zero(other.is_beta_zero)
        , is_alpha_one(other.is_alpha_one)
        , kOpsArr(((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0))
                      ? other.kOpsArr
                      : nullptr)
        , kOpsArrSize(((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0))
                          ? other.kOpsArrSize
                          : 0)
        , anyKOpsOrder(other.anyKOpsOrder)
    {
        if ((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0)) {
            other.kOpsArr     = nullptr;
            other.kOpsArrSize = 0;
        }
    }

    kernelInfo& operator=(const kernelInfo& other)
    {
        if (this != &other) {
            this->mr           = other.mr;
            this->nr           = other.nr;
            this->k_unroll     = other.k_unroll;
            this->is_beta_zero = other.is_beta_zero;
            this->is_alpha_one = other.is_alpha_one;
            if (this->kOpsArr != nullptr) {
                delete[] this->kOpsArr;
            }
            if ((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0)) {
                this->kOpsArr =
                    kernelInfo::allocateKernelOpsArray(other.kOpsArrSize);
                for (std::size_t i = 0; i < other.kOpsArrSize; i++) {
                    this->kOpsArr[i] = other.kOpsArr[i];
                }
                this->kOpsArrSize = other.kOpsArrSize;
            }
            this->anyKOpsOrder = other.anyKOpsOrder;
        }
        return *this;
    }

    kernelInfo& operator=(kernelInfo&& other)
    {
        if (this != &other) {
            this->mr           = other.mr;
            this->nr           = other.nr;
            this->k_unroll     = other.k_unroll;
            this->is_beta_zero = other.is_beta_zero;
            this->is_alpha_one = other.is_alpha_one;
            if (this->kOpsArr != nullptr) {
                delete[] this->kOpsArr;
            }
            if ((other.kOpsArr != nullptr) && (other.kOpsArrSize > 0)) {
                this->kOpsArr     = other.kOpsArr;
                this->kOpsArrSize = other.kOpsArrSize;
                other.kOpsArr     = nullptr;
                other.kOpsArrSize = 0;
            }
            this->anyKOpsOrder = other.anyKOpsOrder;
        }
        return *this;
    }

    ~kernelInfo() { kernelInfo::deallocateKernelOpsArray(this->kOpsArr); }

    bool operator==(const kernelInfo& rhs) const
    {
        bool isKOpsArrEqual = true;
        if ((this->kOpsArr != nullptr) && (rhs.kOpsArr != nullptr)) {
            for (std::size_t i = 0; i < this->kOpsArrSize; i++) {
                if (this->kOpsArr[i] != rhs.kOpsArr[i]) {
                    isKOpsArrEqual = false;
                    break;
                }
            }
        }
        return ((this->mr == rhs.mr) && (this->nr == rhs.nr)
                && (this->k_unroll == rhs.k_unroll)
                && (this->is_beta_zero == rhs.is_beta_zero)
                && (this->is_alpha_one == rhs.is_alpha_one)
                && (this->kOpsArrSize == rhs.kOpsArrSize) && isKOpsArrEqual
                && (this->anyKOpsOrder == rhs.anyKOpsOrder));
    }

    // TODO: Need to implement a subset function for kernelInfo

    static kernelOpsMetaData* allocateKernelOpsArray(std::size_t numKernelOps)
    {
        std::size_t        numElems = (numKernelOps > 0) ? numKernelOps : 1;
        kernelOpsMetaData* kOpsArr  = nullptr;
        kOpsArr                     = new kernelOpsMetaData[numElems];
        return kOpsArr;
    }

    static void deallocateKernelOpsArray(kernelOpsMetaData* kOpsArr)
    {
        if (kOpsArr != nullptr) {
            delete[] kOpsArr;
        }
    }
};

} // namespace dlp::kernel_frame
