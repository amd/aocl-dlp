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

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_utilities.hh"
#include "traits.hh"

#include <cstdint>
#include <functional>
#include <memory>

namespace amdzen::x86gen {

/**
 * @brief Base class for post-op code generation on x86 (AVX2/AVX-512).
 *
 * Contains common parameters and helpers shared by all post-op strategies.
 * Post-op specific classes inherit and add their own members.
 */
template<utils::kernelInstrType KType>
class kernelOpsGeneratorX86
{
  protected:
    using Traits                       = traits::ArchitectureTraits<KType>;
    using RegType                      = typename Traits::RegType;
    using halfRegType                  = typename Traits::halfRegType;
    static constexpr int RegBytes      = Traits::regBytes;
    static constexpr int RegisterCount = Traits::numRegs;

    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    Xbyak::CodeGenerator* jit;

    const Xbyak::Reg64& regkernelOpsList; // rdx - Post-op list pointer
    const Xbyak::Reg64&
        regkernelOpsAttr;            // r8  - Address of kernelOpsAttr struct
    const Xbyak::Reg64& regTmp1;     // r9
    const Xbyak::Reg64& regTmp2;     // r10
    const Xbyak::Reg64& regTmp3;     // r11
    const Xbyak::Reg64& regTmp4;     // rax
    const Xbyak::Reg64& regTmp5;     // rbx
    const Xbyak::Reg64& regTmp6;     // rdi
    const Xbyak::Reg64& regTmp7;     // rsi
    const Xbyak::Reg64& regcsC;      // r12 - Column stride
    const Xbyak::Reg32& regTmp4Half; // eax
    const Xbyak::Reg32& regTmp5Half; // ebx

    VecPoolType*  vecPool  = nullptr;
    MaskPoolType* maskPool = nullptr;
    int maskOffset; // Offset from stackPtr to mask data (-1 if no mask)

    uint8_t fringeMaskIdx[dlp::kernels::maxNumMasks];
    int     numFringeMasks;

    Xbyak::Ymm ymmMask; // AVX2 mask register

    dlp::jit::jitAlgoType algoType;
    int                   MR;
    int                   NR;
    bool                  useMask;
    int                   numMaskRegs;
    int                   cRegStartIdx;
    int                   cRegCount;
    int                   numRegsPerRow;
    int                   numFullRegsPerRow;
    int                   numMaskRegsPerRow;

    // Table ownership: the root instance owns the table labels and flags.
    // Strategy copies (Bias, GeluTanh, etc.) are stack-scoped temporaries
    // created via the copy constructor; they get default-constructed (unused)
    // labels and redirect all table access through tableOwner-> to the root.
    // This ensures multiple strategies accumulate their table requirements
    // into a single set of flags, and embedKernelOpsAttributes() emits each
    // table at most once. Thread-safe by construction: each generateKernelOps
    // call is self-contained with its own root and stack-scoped strategies.
    kernelOpsGeneratorX86* tableOwner;

    enum TableFlags : uint8_t
    {
        TABLE_NONE = 0,
        TABLE_GELU = 1 << 0,
        TABLE_EXP  = 1 << 1,
        TABLE_ERF  = 1 << 2,
    };
    uint8_t      requiredTables = TABLE_NONE;
    bool         tablesEmbedded = false;
    Xbyak::Label geluTable;
    Xbyak::Label expTable;
    Xbyak::Label erfTable;
    Xbyak::Label tableEnd;

    // Table helpers for strategy classes (access root's labels/flags)
    void requestTable(uint8_t flag) { tableOwner->requiredTables |= flag; }

    Xbyak::Address geluAddr(int byteOffset)
    {
        return jit->ptr[jit->rip + tableOwner->geluTable + byteOffset];
    }
    static constexpr int geluMacrosOff = sizeof(gen::tables::gelu_consts);

    Xbyak::Address expAddr(int byteOffset)
    {
        return jit->ptr[jit->rip + tableOwner->expTable + byteOffset];
    }

    Xbyak::Address erfAddr(int byteOffset)
    {
        return jit->ptr[jit->rip + tableOwner->erfTable + byteOffset];
    }

    // Shared elementwise subroutines (used by multiple post-ops)
    void polyEval6(int const1, int const2, int r, int r2, int z, int q);
    void expF(int x,
              int const1,
              int const2,
              int r,
              int r2,
              int z,
              int dn,
              int q,
              int expCmpMaskIdx);
    void tanhF(int x_tanh,
               int x,
               int const1,
               int const2,
               int r,
               int r2,
               int z,
               int dn,
               int q,
               int expCmpMaskIdx);

    void embedKernelOpsAttributes();

    dlp::jit::jitGeneratorError setPostOpsContext(int  MR_val,
                                                  int  NR_val,
                                                  bool useMask_val,
                                                  int  numMaskRegs_val,
                                                  int  cRegStartIdx_val,
                                                  int  cRegCount_val);

    dlp::jit::jitGeneratorError initializeFringeMasks(
        int                                               numMasks,
        int                                               maskOffset,
        const Xbyak::Reg64&                               stackPtr,
        std::vector<utils::registerGuard<Xbyak::Opmask>>& guards,
        utils::registerGuard<RegType>&                    vecMaskGuard);

    RegType toRegType(int regIdx) const { return RegType(regIdx); }

    bool isGEMVN1() const { return algoType == dlp::jit::jitAlgoType::gemv_n1; }

    void loadVectorFloat32(const Xbyak::Address& addr,
                           const RegType&        dest,
                           bool                  useMaskOp,
                           const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    dlp::jit::jitGeneratorError loadVectorBF16andConvertToF32(
        const Xbyak::Address& addr,
        const RegType&        dest,
        bool                  useMaskOp,
        const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    dlp::jit::jitGeneratorError loadVectorFP16andConvertToF32(
        const Xbyak::Address& addr,
        const RegType&        dest,
        bool                  useMaskOp,
        const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    dlp::jit::jitGeneratorError loadVectorInt8andConvertToF32(
        const Xbyak::Address& addr,
        const RegType&        dest,
        bool                  useMaskOp,
        const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    dlp::jit::jitGeneratorError loadVectorUInt8andConvertToF32(
        const Xbyak::Address& addr,
        const RegType&        dest,
        bool                  useMaskOp,
        const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    void loadVectorInt32andConvertToF32(
        const Xbyak::Address& addr,
        const RegType&        dest,
        bool                  useMaskOp,
        const Xbyak::Opmask&  mask = Xbyak::Opmask(0));

    void broadcastScalarFloat32(const Xbyak::Address& addr,
                                const RegType&        dest);
    dlp::jit::jitGeneratorError broadcastScalarBF16andConvertToF32(
        const Xbyak::Address& addr, const RegType& dest);
    dlp::jit::jitGeneratorError broadcastScalarFP16andConvertToF32(
        const Xbyak::Address& addr, const RegType& dest);
    dlp::jit::jitGeneratorError broadcastScalarInt8andConvertToF32(
        const Xbyak::Address& addr, const RegType& dest);
    dlp::jit::jitGeneratorError broadcastScalarUInt8andConvertToF32(
        const Xbyak::Address& addr, const RegType& dest);
    void broadcastScalarInt32andConvertToF32(const Xbyak::Address& addr,
                                             const RegType&        dest);

    dlp::jit::jitGeneratorError loadVector(
        dlp::kernel_frame::DataType dtype,
        const Xbyak::Address&       addr,
        const RegType&              dest,
        bool                        useMaskOp,
        const Xbyak::Opmask&        mask = Xbyak::Opmask(0))
    {
        switch (dtype) {
            case dlp::kernel_frame::DataType::f32:
                loadVectorFloat32(addr, dest, useMaskOp, mask);
                break;
            case dlp::kernel_frame::DataType::bf16:
                return loadVectorBF16andConvertToF32(addr, dest, useMaskOp,
                                                     mask);
            case dlp::kernel_frame::DataType::f16:
                return loadVectorFP16andConvertToF32(addr, dest, useMaskOp,
                                                     mask);
            case dlp::kernel_frame::DataType::s8:
                return loadVectorInt8andConvertToF32(addr, dest, useMaskOp,
                                                     mask);
            case dlp::kernel_frame::DataType::u8:
                return loadVectorUInt8andConvertToF32(addr, dest, useMaskOp,
                                                      mask);
            case dlp::kernel_frame::DataType::s32:
                loadVectorInt32andConvertToF32(addr, dest, useMaskOp, mask);
                break;
            default:
                return dlp::jit::jitGeneratorError::notSupported;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    dlp::jit::jitGeneratorError broadcastScalar(
        dlp::kernel_frame::DataType dtype,
        const Xbyak::Address&       addr,
        const RegType&              dest)
    {
        switch (dtype) {
            case dlp::kernel_frame::DataType::f32:
                broadcastScalarFloat32(addr, dest);
                break;
            case dlp::kernel_frame::DataType::bf16:
                return broadcastScalarBF16andConvertToF32(addr, dest);
            case dlp::kernel_frame::DataType::f16:
                return broadcastScalarFP16andConvertToF32(addr, dest);
            case dlp::kernel_frame::DataType::s8:
                return broadcastScalarInt8andConvertToF32(addr, dest);
            case dlp::kernel_frame::DataType::u8:
                return broadcastScalarUInt8andConvertToF32(addr, dest);
            case dlp::kernel_frame::DataType::s32:
                broadcastScalarInt32andConvertToF32(addr, dest);
                break;
            default:
                return dlp::jit::jitGeneratorError::notSupported;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    static constexpr int getLoadBytes(dlp::kernel_frame::DataType dtype)
    {
        switch (dtype) {
            case dlp::kernel_frame::DataType::f32:
                return RegBytes;
            case dlp::kernel_frame::DataType::s32:
                return RegBytes;
            case dlp::kernel_frame::DataType::bf16:
                return RegBytes / 2;
            case dlp::kernel_frame::DataType::f16:
                return RegBytes / 2;
            case dlp::kernel_frame::DataType::s8:
                return RegBytes / 4;
            case dlp::kernel_frame::DataType::u8:
                return RegBytes / 4;
            default:
                return RegBytes;
        }
    }

    static constexpr int getElementSize(dlp::kernel_frame::DataType dtype)
    {
        switch (dtype) {
            case dlp::kernel_frame::DataType::f32:
                return sizeof(float);
            case dlp::kernel_frame::DataType::s32:
                return sizeof(int32_t);
            case dlp::kernel_frame::DataType::bf16:
                return sizeof(bfloat16);
            case dlp::kernel_frame::DataType::f16:
                return sizeof(dlp::float16);
            case dlp::kernel_frame::DataType::s8:
                return sizeof(int8_t);
            case dlp::kernel_frame::DataType::u8:
                return sizeof(uint8_t);
            default:
                return sizeof(float);
        }
    }

    kernelOpsGeneratorX86(kernelOpsGeneratorX86& base);

  public:
    explicit kernelOpsGeneratorX86(Xbyak::CodeGenerator* jit);

    /**
     * Generate post-operation code for the given kernel ops list.
     *
     * @param kernelOps       Post-ops metadata list (GELU, bias, scale, etc.).
     *                        Iterated in order; each op's code is emitted
     * inline.
     * @param postOpsArgWrapperPtrReg  GPR holding base pointer to the post-ops
     *                        argument wrapper (gemvM1Params / gemmParams,
     *                        selected by algoType).
     * @param algoType        Algorithm context (gemm, gemv_n1, gemv_m1).
     *                        Controls metadata pointer loading and loop
     * structure.
     * @param MR              Tile row count for the current micro-kernel.
     * @param NR              Tile column count for the current micro-kernel.
     * @param useMask         True if the fringe tile needs masked vector
     * stores.
     * @param numMaskRegs     Number of fringe mask registers the caller has
     *                        prepared (only relevant when useMask is true).
     * @param cRegStartIdx    First vector register index of C accumulators.
     * @param cRegCount       Number of C accumulator registers.
     * @param vecPool         Initialized vector register pool with accumulators
     *                        set via setAccumulators() and init() called.
     *                        Pool validity is enforced lazily via acquire().
     * @param maskPool        Initialized mask register pool with reserved bits
     *                        configured via init(). Null-safe at mask usage
     * sites.
     * @param maskOffset      Fringe mask source encoding:
     *                        >= 0  : byte offset from stackPtr into params,
     *                        == -1 : no mask,
     *                        <= -2 : encoded immediate (see
     * encodeMaskImmediate).
     */
    dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType,
        int                   MR,
        int                   NR,
        bool                  useMask,
        int                   numMaskRegs,
        int                   cRegStartIdx,
        int                   cRegCount,
        VecPoolType&          vecPool,
        MaskPoolType&         maskPool,
        int                   maskOffset);

    void advancePostOpsPtr();

    Xbyak::Opmask getFringeMask(int idx) const
    {
        return Xbyak::Opmask(fringeMaskIdx[idx]);
    }

    int getNumFringeMasks() const { return numFringeMasks; }
};

} // namespace amdzen::x86gen

extern template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
extern template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
extern template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
