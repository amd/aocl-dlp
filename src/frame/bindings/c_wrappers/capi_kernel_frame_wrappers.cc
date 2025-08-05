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

#include <iostream>
#include <optional>

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "jit/jit_kernel_adapter.hh"
#include "jit_register/jit_register.hh"
#include "kernel_register/kernel_register.hh"

using namespace dlp::kernel_frame;
using namespace dlp::jit;
using namespace dlp::kernels;

inline kernelDatatype
getKernelDatatype(kernel_datatype_t kDtype)
{
    if ((kDtype < DLP_KERNEL_U8S8S32OS32)
        || (kDtype >= DLP_KERNEL_DATATYPE_MAX)) {
        return kernelDatatype::invalid;
    }
    return static_cast<kernelDatatype>(kDtype);
}

DataType
getStorageDtFromAoclKernelDatatype(kernel_datatype_t kDtype)
{
    DataType retType = DataType::invalid;

    switch (kDtype) {
        case DLP_KERNEL_U8S8S32OS32:
            retType = DataType::s32;
            break;
        case DLP_KERNEL_U8S8S32OS8:
            retType = DataType::s32;
            break;
        case DLP_KERNEL_BF16BF16F32OBF16:
            retType = DataType::f32;
            break;
        case DLP_KERNEL_BF16BF16F32OF32:
            retType = DataType::f32;
            break;
        case DLP_KERNEL_F32F32F32OF32:
            retType = DataType::f32;
            break;
        default:
            retType = DataType::invalid;
            break;
    }

    return retType;
}

DataType
getStorageDtFromAoclStorageType(AOCL_STORAGE_TYPE st)
{
    DataType dt = DataType::invalid;
    switch (st) {
        case S8:
            dt = DataType::s8;
            break;
        case U8:
            dt = DataType::u8;
            break;
        case S16:
            dt = DataType::s16;
            break;
        case U16:
            dt = DataType::u16;
            break;
        case BF16:
            dt = DataType::bf16;
            break;
        case S32:
            dt = DataType::s32;
            break;
        case U32:
            dt = DataType::u32;
            break;
        case F32:
            dt = DataType::f32;
            break;
        default:
            dt = DataType::invalid;
            break;
    }

    return dt;
}

static inline void
setKernelOps(kernelOpsMetaData* metaData,
             lpgemm_post_op*    post_op,
             kernel_datatype_t  k_dtype)
{
    kernelOps kOpsType = kernelOps::invalid;
    switch (post_op->op_code) {
        case POST_OPS_BIAS: {
            metaData->type           = kernelOps::bias;
            metaData->paramStorageDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? storageFormat::colMajor
                                       : storageFormat::rowMajor;
            break;
        }
        case POST_OPS_RELU:
            metaData->type = kernelOps::relu;
            break;
        case POST_OPS_RELU_SCALE:
            metaData->type = kernelOps::reluScale;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_GELU_TANH:
            metaData->type = kernelOps::geluTanh;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_GELU_ERF:
            metaData->type = kernelOps::geluErf;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_CLIP:
            metaData->type = kernelOps::clip;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_SWISH:
            metaData->type = kernelOps::swish;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_TANH:
            metaData->type = kernelOps::tanh;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_SIGMOID:
            metaData->type = kernelOps::sigmoid;
            metaData->paramStorageDt =
                getStorageDtFromAoclKernelDatatype(k_dtype);
            break;
        case POST_OPS_DOWNSCALE: {
            metaData->type           = kernelOps::downscale;
            metaData->paramStorageDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? storageFormat::colMajor
                                          : storageFormat::rowMajor;
            metaData->scaleFactorDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->sf_stor_type));
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            metaData->zeroPointDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->zp_stor_type));
            metaData->scalarZeroPointRequired =
                (*(static_cast<md_t*>(post_op->op_args3)) == 1) ? true : false;
            metaData->vectorZeroPointRequired =
                (*(static_cast<md_t*>(post_op->op_args3)) > 1) ? true : false;
            break;
        }
        case POST_OPS_MATRIX_ADD: {
            metaData->type           = kernelOps::matAdd;
            metaData->paramStorageDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? storageFormat::colMajor
                                       : storageFormat::rowMajor;
            metaData->scaleFactorDt =
                DataType::f32; // TODO: Always F32 for mat add
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            break;
        }
        case POST_OPS_MATRIX_MUL: {
            metaData->type           = kernelOps::matMul;
            metaData->paramStorageDt = getStorageDtFromAoclStorageType(
                static_cast<AOCL_STORAGE_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? storageFormat::colMajor
                                       : storageFormat::rowMajor;
            metaData->scaleFactorDt =
                DataType::f32; // TODO: Always F32 for mat mul
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            break;
        }
        case POST_OPS_DISABLE:
            metaData->type = kernelOps::invalid;
            break;
        default:
            metaData->type = kernelOps::invalid;
            break;
    }
}

// TODO: This is a pseudo decision engine. Need to replace it with a real one.
// Post-ops are not supported in this pseudo decision engine.
static std::optional<kernelInfo>
getKernelInfoForJitIntelligence(kernel_datatype_t k_dtype,
                                md_t              m,
                                md_t              n,
                                md_t              k,
                                void*             alpha,
                                void*             beta,
                                lpgemm_post_op*   post_ops,
                                md_t              mr_hint,
                                md_t              nr_hint)
{
    md_t mr           = mr_hint;
    md_t nr           = nr_hint;
    md_t k_unroll     = 1;
    bool anyKOpsOrder = false;

    // TODO: Only supports non GEMV kernels for now.
    if ((m == 1) || (n == 1)) {
        return std::nullopt;
    }

    if (post_ops == nullptr) {
        kernelInfo kI{
            mr, nr, k_unroll, false, false, nullptr, 0, anyKOpsOrder
        };
        return std::make_optional(kI);
    } else {
        // Iterate over the post_ops list to get the number of post-ops.
        md_t            numPostOps    = 0;
        lpgemm_post_op* temp_post_ops = post_ops;
        while ((temp_post_ops != NULL)
               && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
            temp_post_ops = temp_post_ops->next;
            numPostOps++;
        }

        if (numPostOps == 0) {
            kernelInfo kI{ mr,    nr,      k_unroll, false,
                           false, nullptr, 0,        anyKOpsOrder };
            return std::make_optional(kI);
        } else {
            kernelInfo kI{ mr,    nr,      k_unroll, false,
                           false, nullptr, 0,        anyKOpsOrder };
            kI.kOpsArrSize = numPostOps;
            kI.kOpsArr     = kernelInfo::allocateKernelOpsArray(numPostOps);

            md_t ii       = 0;
            temp_post_ops = post_ops;
            while ((temp_post_ops != NULL)
                   && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
                setKernelOps(std::addressof(kI.kOpsArr[ii]), temp_post_ops,
                             k_dtype);
                temp_post_ops = temp_post_ops->next;
                ii++;
            }

            return std::make_optional(kI);
        }
        return std::nullopt;
    }
}

dlp_kernel_hndl_t
dlp_init_and_get_kernel_hndl(kernel_datatype_t k_dtype,
                             md_t              m,
                             md_t              n,
                             md_t              k,
                             void*             alpha,
                             void*             beta,
                             lpgemm_post_op*   post_ops,
                             md_t              mr_hint,
                             md_t              nr_hint)
{
    dlp_kernel_hndl_t kernel_hndl{ DLP_KERNEL_INVALID, 0, 0, nullptr };

    // TODO: Generate the jitIntelligence via pseudo DecisionEngine.
    auto optKI = getKernelInfoForJitIntelligence(k_dtype, m, n, k, alpha, beta,
                                                 post_ops, mr_hint, nr_hint);
    if (!optKI.has_value()) {
        return kernel_hndl;
    }

    kernelDatatype kDtype = getKernelDatatype(k_dtype);
    if (kDtype == kernelDatatype::invalid) {
        return kernel_hndl;
    }

    auto kernPtr =
        dlpKernelRegisterInstance().getGemmKernel(&optKI.value(), kDtype);

    if (!kernPtr) {
        auto jitGen =
            dlpJitGeneratorRegisterInstance().getGemmJitGenerator(kDtype);
        auto kB = std::make_unique<jitKernelAdapter>(optKI.value(),
                                                     std::move(jitGen), true);

        if (!kB->isJitGenerated()) {
            // TODO: Fallback to the default kernel.
            kernPtr = kernelBaseRef(nullptr);
        } else {
            auto retVal = dlpKernelRegisterInstance().registerGemmKernel(
                std::move(kB), "FP32JitKernel");
            if (retVal != kernelFrameError::success) {
                std::cout << "Jit kernel registration failed." << std::endl;
                std::cout << "Exiting..." << std::endl;
                std::abort();
            }
            kernPtr = dlpKernelRegisterInstance().getGemmKernel(&optKI.value(),
                                                                kDtype);
        }
    }

    kernel_hndl.kernel_base = static_cast<void*>(kernPtr.getPtr());
    kernel_hndl.mr          = optKI.value().mr;
    kernel_hndl.nr          = optKI.value().nr;
    kernel_hndl.kDtype      = k_dtype;
    return kernel_hndl;
}

void
dlp_execute_kernel(dlp_kernel_hndl_t   kernel_hndl,
                   md_t                m,
                   md_t                n,
                   md_t                k,
                   void*               A,
                   md_t                rs_a,
                   md_t                cs_a,
                   md_t                ps_a,
                   void*               B,
                   md_t                rs_b,
                   md_t                cs_b,
                   void*               C,
                   md_t                rs_c,
                   md_t                cs_c,
                   void*               alpha,
                   void*               beta,
                   lpgemm_post_op*     post_ops_list,
                   lpgemm_post_op_attr post_ops_attr)
{
    gemmParams kP{
        A,    B,    C,    m,    n,     k,    rs_a,          cs_a,         ps_a,
        rs_b, cs_b, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
    };
    kernelBase* kB = static_cast<kernelBase*>(kernel_hndl.kernel_base);
    kB->operator()(std::addressof(kP));
    return;
}
