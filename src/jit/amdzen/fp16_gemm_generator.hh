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
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"

namespace amdzen::gen {

/**
 * @brief JIT generator for FP16 GEMM kernels using Xbyak
 *
 * This class generates AVX-512-FP16 optimized kernels for fp16 GEMM
 * operations. Uses native FP16 accumulation with vfmadd231ph instruction.
 * Supports configurable MR/NR blocking with 32 FP16 elements per ZMM.
 */
template<utils::kernelInstrType KType>
class jitFP16_GEMM : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitFP16_GEMM(void* buffer, size_t bufferSize);
    ~jitFP16_GEMM()                         = default;
    jitFP16_GEMM(jitFP16_GEMM&)             = delete;
    jitFP16_GEMM& operator=(jitFP16_GEMM&)  = delete;
    jitFP16_GEMM(jitFP16_GEMM&&)            = delete;
    jitFP16_GEMM& operator=(jitFP16_GEMM&&) = delete;

    /**
     * @brief Generate the complete FP16 GEMM kernel
     */
    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    // =================================================================
    // TYPE DEFINITIONS AND ARCHITECTURE TRAITS
    // =================================================================
    using Traits    = amdzen::traits::ArchitectureTraits<KType>;
    using FP16Types = amdzen::traits::kernel_types<
        dlp::kernel_frame::kernelDatatype::f16f16f16of16>;
    using RegType = typename Traits::RegType;

    // =================================================================
    // FP16-SPECIFIC CONSTANTS
    // =================================================================
    static constexpr int FP16_ELEM_SIZE = FP16Types::elemSize;
    static constexpr int FP16_PER_ZMM   = FP16Types::elemsPerZmm;
    static constexpr int F32_ELEM_SIZE  = 4;

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int  numRegs  = Traits::numRegs;  // 32 for AVX-512
    int  RegSize  = Traits::regSize;  // 512 bits
    int  RegBytes = Traits::regBytes; // 64 bytes
    int  MR, NR;
    int  c_downscale;
    bool useMask =
        false; // Flag to indicate if masked instructions are generated
    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx;

    // =================================================================
    // GENERAL PURPOSE REGISTERS
    // =================================================================
    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr;
    Xbyak::Reg64 regRsA, regCsA, regRsB, regRsC;
    Xbyak::Reg64 regKIter, regMiter;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;

    // =================================================================
    // OPMASK REGISTERS
    // =================================================================
    // mask_regs[0]   - NR-fringe mask for FP16 B-loads / FP16 C-stores
    //                  (loaded from maskFP16; only used when useMask is true).
    // mask_regs[1-2] - F32-lane masks for the of32 rail's F32 in-place
    //                  beta-combine and store, plus the 32-col-tail
    //                  masking in the partial-spill path. Loaded from
    //                  maskF32[0..1].
    // Both pairs are added to the kernelOpsHandler mask-pool's preserve
    // list so the handler does not clobber them across CALLs/inline
    // emissions, eliminating the need for redundant kmovw re-loads after
    // the post-op chain returns.
    Xbyak::Opmask mask_regs[utils::NUM_USABLE_MASKS];

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================
    /**
     * @brief Allocate registers based on MR x NR tiling
     * For FP16: 32 elements per ZMM, so NR=128 uses 4 B registers
     */
    dlp::jit::jitGeneratorError allocateRegisters();

    /**
     * @brief Initialize stack frame registers
     */
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    /**
     * @brief Initialize parameter pointers from stack
     */
    void initializeParameters(bool mLoop = false);

    /**
     * @brief Initialize accumulator registers to zero
     */
    void initializeAccumulators(utils::generatorParams& params);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    /**
     * @brief Generate the M-loop that calls IR loop multiple times
     */
    dlp::jit::jitGeneratorError generateMLoop(utils::generatorParams& params);

    /**
     * @brief Generate the IR loop with K-loop, scaling and post-ops
     */
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // =================================================================
    // K-LOOP COMPUTATION
    // =================================================================
    /**
     * @brief Load B matrix values into registers using vmovdqu16
     */
    dlp::jit::jitGeneratorError loadBValues();

    /**
     * @brief Broadcast A element and FMA with B using vfmadd231ph
     */
    dlp::jit::jitGeneratorError broadcastAFMAwithB(bool isKRemainder);

    /**
     * @brief Unroll K-loop computation
     */
    dlp::jit::jitGeneratorError kUnroll(int unroll, bool isKRemainder);

    // =================================================================
    // SCALING OPERATIONS
    // =================================================================
    /**
     * @brief Apply alpha scaling using vmulph
     */
    dlp::jit::jitGeneratorError scaleAlpha();

    /**
     * @brief Apply beta scaling for existing C values.
     *
     * When @p betaIsOne is true, the broadcast + vmulph is skipped and only
     * the load-C + vaddph is emitted (correct because multiplying by 1 is
     * a no-op). Used when the Decision Engine has confirmed user beta
     * equals FP16_ONE.
     */
    dlp::jit::jitGeneratorError scaleBeta(bool betaIsOne = false);

    // =================================================================
    // POST-OPERATIONS AND RESULT STORAGE
    // =================================================================
    /**
     * @brief Generate post-operations sequence (orchestrator).
     *
     * Applies FP16 alpha, then dispatches to one or two chunk(s) of the
     * shared processChunk path. The of16 rail handles its own scaleBeta /
     * pure-FP16 store / intermediate-KC FP16 spill before the chunk path;
     * the of32 rail enters a chunk every KC and folds the F32 beta-combine
     * and F32 in-place store into the chunk itself.
     */
    dlp::jit::jitGeneratorError generatePostOps(utils::generatorParams& params);

    /**
     * @brief One unit of post-op work over a single FP16 sub-tile.
     *
     * Describes everything processChunk needs to know: where the FP16 data
     * lives (registers or a stack-spilled chunk), the F32-tile destination
     * geometry, the per-chunk mask seed for the of32 lane masks, and the
     * kernelOps emission strategy (inline vs CALL into a shared subroutine).
     */
    struct ChunkConfig
    {
        // Where the FP16 data the chunk widens lives.
        enum class Source
        {
            Registers, // FP16 accumulator at cRegIdx + row*bReg + fp16SrcOffset
            Stack      // FP16 spilled at [rsp + row*fp16RegsPerRow*RegBytes]
        };
        // Strategy for seeding maskF32[0..1] (the of32 lane masks). The
        // chunk re-loads mask_regs[1..2] from maskF32[] every entry; this
        // dial controls *what* gets stored at maskF32 first.
        enum class MaskSource
        {
            None,                // hasMaskedF32Pair is false; masks unused
            FromFp16Mask,        // derive maskF32[0..1] by splitting maskFP16
            ConstantAllOnes,     // both halves = 0xFFFF (full 32-col chunk)
            ConstantTailHalfOnly // low half = 0x0000 (NR=96 tail: 32 cols)
        };
        Source        source              = Source::Registers;
        int           fp16SrcOffset       = 0;
        int           fp16RegsPerRow      = 0;
        int           dstRegStart         = 0;
        int           scratchRegStart     = 0;
        int           numFullF32Pairs     = 0;
        bool          hasMaskedF32Pair    = false;
        int           numFullColsF32      = 0;
        int           colElemOffset       = 0;
        bool          maskedFp16Store     = false;
        bool          zeroUpperF32Pair    = false;
        MaskSource    maskSource          = MaskSource::None;
        Xbyak::Label* kernelOpsSubroutine = nullptr;
    };

    /**
     * @brief Run one post-op chunk: widen + beta-combine + kernelOps + store.
     *
     * The widen-step (FP16 → F32) reuses the existing helpers
     * (convertChunkFromRegsToF32 plus an inline stack-load path that
     * supports zero-padding the upper F32 pair for the NR=96 tail) and
     * lands the F32 tile at cfg.dstRegStart. The of32 beta-combine and F32
     * in-place store are no-ops on the of16 rail; conversely the FP16
     * narrow + store at the end is a no-op on of32. The kernelOps body
     * runs once per chunk and is gated on is_last_k at runtime so
     * intermediate-KC chunks (of32 only) skip it.
     */
    dlp::jit::jitGeneratorError processChunk(utils::generatorParams& params,
                                             const ChunkConfig&      cfg);

    /**
     * @brief Emit the kernelOpsHandler chain inline.
     *
     * Builds local register / mask pools, marks the FP16-fringe and F32
     * lane masks as preserved across the handler, and forwards to
     * kernelOpsHandler::generateKernelOps. Used directly on the
     * register-only path (single chunk, no benefit to a CALL); the
     * partial-spill path wraps this in emitKernelOpsSubroutine and CALLs
     * the resulting label from each chunk.
     */
    dlp::jit::jitGeneratorError emitKernelOpsBody(
        utils::generatorParams& params,
        int                     dstRegStart,
        int                     numCRegs,
        int                     numFullColsF32,
        bool                    hasMaskedF32Pair);

    /**
     * @brief Emit a CALL-able kernelOps subroutine at `label` (jmp-over).
     *
     * Wraps emitKernelOpsBody in a `jmp .end / L(label) / body / ret /
     * L(.end)` so both partial-spill chunks share a single emission of
     * the post-op chain.
     */
    dlp::jit::jitGeneratorError emitKernelOpsSubroutine(
        utils::generatorParams& params,
        Xbyak::Label&           label,
        int                     dstRegStart,
        int                     numCRegs,
        int                     numFullColsF32,
        bool                    hasMaskedF32Pair);

    /**
     * @brief Of32 rail: F32 beta-combine with user C from regCPtr.
     *
     * Loads the user's float C tile (regCPtr + colElemOffsetF32 in F32
     * elements, advancing by regRsC bytes per row) into ZMMs starting at
     * scratchRegStart and combines with the F32 accumulator tile in place:
     * vfmadd231ps for scalingType::generic, vaddps for scalingType::one,
     * skipped for scalingType::zero. mask_regs[1..2] (loaded from
     * maskF32[0..1]) gate the masked F32 pair when hasMaskedF32Pair is true.
     */
    dlp::jit::jitGeneratorError applyBetaCombineF32(
        utils::generatorParams& params,
        int                     numFullF32Pairs,
        bool                    hasMaskedF32Pair,
        int                     dstRegStart,
        int                     scratchRegStart,
        int                     colElemOffsetF32);

    /**
     * @brief Of32 rail: F32 in-place store of the F32 tile back to user C.
     *
     * vmovups from the F32 tile to regCPtr + colElemOffsetF32 (F32 elements),
     * advancing by regRsC bytes per row. The masked F32 pair (if any) is
     * gated by mask_regs[1..2] (already loaded from maskF32[0..1] and
     * preserved across the kernelOpsHandler call by the mask-pool's
     * preserve list).
     */
    dlp::jit::jitGeneratorError storeResultF32_inplace(int  numFullF32Pairs,
                                                       bool hasMaskedF32Pair,
                                                       int  dstRegStart,
                                                       int  colElemOffsetF32);

    /**
     * @brief Dispatch to appropriate store function
     */
    dlp::jit::jitGeneratorError storeResult();

    /**
     * @brief Store results as FP16 using vmovdqu16
     */
    dlp::jit::jitGeneratorError storeResultFP16();

    void spillChunkToStack(int srcRegOffset, int fp16RegsPerRow);
    void restoreStackAfterChunkSpill(int fp16RegsPerRow);
    void convertChunkFromRegsToF32(int srcRegOffset,
                                   int fp16RegsPerRow,
                                   int dstRegStart);
    /**
     * @brief Convert one row of an F32 chunk back to FP16 and store it.
     */
    void convertF32ChunkToFP16AndStoreRow(int  f32RegStart,
                                          int  rowIdx,
                                          int  colElemOffset,
                                          int  fp16RegsPerRow,
                                          bool maskedStore = false);

    /**
     * @brief Populate maskF32[0..1] from maskFP16 for F32 post-ops
     */
    void populateF32MasksFromFP16();

    /**
     * @brief Move C pointer for next M iteration
     */
    dlp::jit::jitGeneratorError moveCPtr();
};

// Type alias for AVX-512 FP16 GEMM generator
using jitGemmGenerator_fp16_avx512 =
    jitFP16_GEMM<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::gen
