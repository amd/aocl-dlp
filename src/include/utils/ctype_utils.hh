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

#include <type_traits>

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "kernel_frame/kernel_frame_base.hh"
#include "type_utils.hh"

namespace dlp::utils {

inline dlp::kernel_frame::kernelDatatype
getKernelDatatype(kernel_datatype_t kDtype)
{
    if ((kDtype < DLP_KERNEL_U8S8S32OS32)
        || (kDtype >= DLP_KERNEL_DATATYPE_MAX)) {
        return dlp::kernel_frame::kernelDatatype::invalid;
    }
    return static_cast<dlp::kernel_frame::kernelDatatype>(kDtype);
}

inline dlp::kernel_frame::DataType
getStorageDtFromDlpKernelDatatype(dlp::kernel_frame::kernelDatatype kDtype)
{
    dlp::kernel_frame::DataType retType = dlp::kernel_frame::DataType::invalid;

    switch (kDtype) {
        case dlp::kernel_frame::kernelDatatype::u8s8s32os32:
        case dlp::kernel_frame::kernelDatatype::u8s8s32os8:
        case dlp::kernel_frame::kernelDatatype::u8s8s32ou8:
            retType = dlp::kernel_frame::DataType::s32;
            break;
        case dlp::kernel_frame::kernelDatatype::u8s8s32obf16:
        case dlp::kernel_frame::kernelDatatype::u8s8s32of32:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        case dlp::kernel_frame::kernelDatatype::bf16bf16f32obf16:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        case dlp::kernel_frame::kernelDatatype::bf16bf16f32of32:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        case dlp::kernel_frame::kernelDatatype::f32f32f32of32:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        case dlp::kernel_frame::kernelDatatype::f16f16f16of16:
            retType = dlp::kernel_frame::DataType::f16;
            break;
        case dlp::kernel_frame::kernelDatatype::f32f16f32of32:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        case dlp::kernel_frame::kernelDatatype::s8s8s32os8:
        case dlp::kernel_frame::kernelDatatype::s8s8s32ou8:
        case dlp::kernel_frame::kernelDatatype::s8s8s32os32:
            retType = dlp::kernel_frame::DataType::s32;
            break;
        case dlp::kernel_frame::kernelDatatype::s8s8s32of16:
            retType = dlp::kernel_frame::DataType::f16;
            break;
        case dlp::kernel_frame::kernelDatatype::s8s8s32obf16:
        case dlp::kernel_frame::kernelDatatype::s8s8s32of32:
            retType = dlp::kernel_frame::DataType::f32;
            break;
        default:
            retType = dlp::kernel_frame::DataType::invalid;
            break;
    }

    return retType;
}

inline dlp::kernel_frame::DataType
getStorageDtFromAoclStorageType(DLP_TYPE st)
{
    dlp::kernel_frame::DataType dt = dlp::kernel_frame::DataType::invalid;
    switch (st) {
        case DLP_S4:
            dt = dlp::kernel_frame::DataType::s4;
            break;
        case DLP_U4:
            dt = dlp::kernel_frame::DataType::u4;
            break;
        case DLP_F4:
            dt = dlp::kernel_frame::DataType::f4;
            break;
        case DLP_S8:
            dt = dlp::kernel_frame::DataType::s8;
            break;
        case DLP_U8:
            dt = dlp::kernel_frame::DataType::u8;
            break;
        case DLP_S16:
            dt = dlp::kernel_frame::DataType::s16;
            break;
        case DLP_U16:
            dt = dlp::kernel_frame::DataType::u16;
            break;
        case DLP_F16:
            dt = dlp::kernel_frame::DataType::f16;
            break;
        case DLP_BF16:
            dt = dlp::kernel_frame::DataType::bf16;
            break;
        case DLP_S32:
            dt = dlp::kernel_frame::DataType::s32;
            break;
        case DLP_U32:
            dt = dlp::kernel_frame::DataType::u32;
            break;
        case DLP_F32:
            dt = dlp::kernel_frame::DataType::f32;
            break;
        default:
            dt = dlp::kernel_frame::DataType::invalid;
            break;
    }

    return dt;
}

} // namespace dlp::utils
