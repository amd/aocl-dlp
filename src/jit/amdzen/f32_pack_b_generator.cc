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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "f32_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

template<utils::kernelInstrType KType>
jitPackBF32<KType>::jitPackBF32()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numVecLoads(0)
    , useMask_(false)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBF32<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_         = params.NR;
    numVecLoads = NR_ / numElemsPerReg;
    useMask_    = params.useMask;

    RETURN_IF_ERROR(allocateReg());

    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    initializeStackFrame(stackFrame);
    initializeParameters();

    if (useMask_) {
        generateLtBlockLoop();
    } else {
        generateFullBlockLoop();
    }

    mov(regKr, NR_);
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regKr);
    mov(regKr, 1);
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regKr);

    vzeroupper();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBF32<KType>::allocateReg()
{
    if (NR_ <= 0 || (NR_ % numElemsPerReg) != 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    if (useMask_) {
        // lt-NR kernel: one mask per SIMD block so each load/store uses a
        // distinct register, eliminating false dependencies.
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // Needs numVecLoads opmask registers (k1..k7).
            if (numVecLoads > utils::NUM_USABLE_MASKS) {
                return dlp::jit::jitGeneratorError::badKernelInfo;
            }
        } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            // Needs numVecLoads data YMMs + numVecLoads mask YMMs <= numRegs.
            if (numVecLoads * 2 > numRegs) {
                return dlp::jit::jitGeneratorError::badKernelInfo;
            }
        }
    } else {
        if (numVecLoads > numRegs) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::initializeStackFrame(Xbyak::util::StackFrame& sf)
{
    pParams           = sf.p[0];
    regSrc            = sf.t[0];
    regDst            = sf.t[1];
    regK              = sf.t[2];
    regLdbBytes       = sf.t[3];
    regDstPanelStride = sf.t[4];
    regSrcEnd         = sf.t[5];
    regNPartial       = sf.t[6];
    regSrcBase        = sf.t[7];
    regDstBase        = sf.t[8];
    regSrcRow         = sf.t[9];
    regDstRow         = sf.t[10];
    regKr             = sf.t[11];
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::initializeParameters()
{
    mov(regSrc, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDst, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);

    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, rs_src)]);
    shl(regLdbBytes, 2);

    mov(regDstPanelStride, regK);
    imul(regDstPanelStride, regDstPanelStride,
         static_cast<int>(NR_ * sizeof(float)));

    mov(regSrcEnd,
        ptr[pParams
            + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
    shl(regSrcEnd, 2);
    add(regSrcEnd, regSrc);

    mov(regNPartial,
        ptr[pParams + offsetof(dlp::kernels::packBParams, n_partial)]);
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::generateFullBlockLoop()
{
    Xbyak::Label l_jc_done, l_jc_loop, l_kr_loop;

    mov(regSrcBase, regSrc);
    mov(regDstBase, regDst);

    cmp(regSrcBase, regSrcEnd);
    jge(l_jc_done, T_NEAR);

    L(l_jc_loop);

    mov(regSrcRow, regSrcBase);
    mov(regDstRow, regDstBase);
    xor_(regKr, regKr);

    L(l_kr_loop);

    loadAndStoreRow();

    add(regSrcRow, regLdbBytes);
    add(regDstRow, static_cast<int>(NR_ * sizeof(float)));
    inc(regKr);
    cmp(regKr, regK);
    jb(l_kr_loop, T_NEAR);

    add(regSrcBase, static_cast<int>(NR_ * sizeof(float)));
    add(regDstBase, regDstPanelStride);

    cmp(regSrcBase, regSrcEnd);
    jb(l_jc_loop, T_NEAR);

    L(l_jc_done);
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::loadAndStoreRow()
{
    for (int v = 0; v < numVecLoads; v++) {
        vmovups(RegType(v), ptr[regSrcRow + v * RegBytes]);
    }
    for (int v = 0; v < numVecLoads; v++) {
        vmovups(ptr[regDstRow + v * RegBytes], RegType(v));
    }
}

// ltNR kernel: per-SIMD-block masked loads and stores.
//
// Each SIMD block has its own precomputed mask (set in executeKernel):
//   - Full blocks get an all-ones mask (0xFFFF / all -1 lanes).
//   - The partial block gets a partial mask.
//   - Blocks beyond n_partial get a zero mask.
//
// This eliminates the runtime sub-loop and uses a distinct register per
// block, avoiding false dependencies on the data registers.
template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::generateLtBlockLoop()
{
    Xbyak::Label l_done, l_kr_loop;

    mov(regSrcRow, regSrc);
    mov(regDstRow, regDst);

    loadPerBlockMasks();

    xor_(regKr, regKr);
    test(regK, regK);
    jz(l_done, T_NEAR);

    L(l_kr_loop);

    maskedLoadAndStoreRow();

    add(regSrcRow, regLdbBytes);
    add(regDstRow, static_cast<int>(NR_ * sizeof(float)));
    inc(regKr);
    cmp(regKr, regK);
    jb(l_kr_loop, T_NEAR);

    L(l_done);
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::loadPerBlockMasks()
{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        for (int v = 0; v < numVecLoads; ++v) {
            kmovw(Xbyak::Opmask(v + 1),
                  ptr[pParams
                      + offsetof(dlp::kernels::packBParams, nFringeMaskPerBlock)
                      + v * sizeof(uint32_t)]);
        }
    } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        for (int v = 0; v < numVecLoads; ++v) {
            vmovdqu(
                Xbyak::Ymm(numVecLoads + v),
                ptr[pParams + offsetof(dlp::kernels::packBParams, nMaskPerBlock)
                    + v * static_cast<int>(numElemsPerReg * sizeof(int32_t))]);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitPackBF32<KType>::maskedLoadAndStoreRow()
{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        for (int v = 0; v < numVecLoads; ++v) {
            vmovups(Xbyak::Zmm(v) | Xbyak::Opmask(v + 1) | T_z,
                    ptr[regSrcRow + v * RegBytes]);
        }
        for (int v = 0; v < numVecLoads; ++v) {
            vmovups(ptr[regDstRow + v * RegBytes], Xbyak::Zmm(v));
        }
    } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        for (int v = 0; v < numVecLoads; ++v) {
            vpxor(Xbyak::Ymm(v), Xbyak::Ymm(v), Xbyak::Ymm(v));
        }
        for (int v = 0; v < numVecLoads; ++v) {
            vmaskmovps(Xbyak::Ymm(v), Xbyak::Ymm(numVecLoads + v),
                       ptr[regSrcRow + v * RegBytes]);
        }
        for (int v = 0; v < numVecLoads; ++v) {
            vmovups(ptr[regDstRow + v * RegBytes], Xbyak::Ymm(v));
        }
    }
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBF32<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::PackBcodeGenerator::jitPackBF32<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
