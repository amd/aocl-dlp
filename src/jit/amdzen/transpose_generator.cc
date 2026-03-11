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
#include "transpose_generator.hh"

namespace amdzen::x86gen {

using namespace Xbyak;

/*
 * TransposeGenerator - JIT Code Generator for Matrix Transpose
 *
 * This class generates optimized x86 assembly code to transpose a matrix block
 * of dimensions MR x NR (rows x columns).
 *
 * ALGORITHM OVERVIEW:
 * -------------------
 * The transpose is performed by dividing the MR x NR block into smaller
 * sub-blocks of size numElemsPerReg x numElemsPerReg, where numElemsPerReg
 * is the number of elements that fit in one SIMD register:
 *   - AVX2 (YMM):   8 elements per register
 *   - AVX512 (YMM): 8 elements per register
 *   - AVX512 (ZMM): 16 elements per register
 *
 * TRANSPOSE STAGES:
 * -----------------
 * Each sub-block is transposed using a series of SIMD instructions:
 *
 * 1. unpack_ps (Stage 1): Interleave single-precision elements from pairs of
 *    registers, effectively transposing 2x2 sub-matrices.
 *    Input:  MR_local registers
 *    Output: numRegsAfterUnpackps registers (rounded up to multiple of 2)
 *
 * 2. unpack_pd (Stage 2): Interleave double-precision elements (pairs of
 * floats), transposing 4x4 sub-matrices. Input:  numRegsAfterUnpackps registers
 *    Output: numRegsAfterUnpackpd registers (rounded up to multiple of 4)
 *
 * 3. permute_r1 (Stage 3): Permute 128-bit lanes to transpose 8x8 sub-matrices.
 *    Only executed if MR_local > 4.
 *    Input:  numRegsAfterUnpackpd registers
 *    Output: numRegsAfterPermuteR1 registers (rounded up to multiple of 8)
 *
 * 4. permute_r2 (Stage 4): Final permutation for 16x16 sub-matrices (ZMM only).
 *    Only executed if MR_local > 8 (AVX512 ZMM only).
 *    Input:  numRegsAfterPermuteR1 registers
 *    Output: numRegsAfterPermuteR2 = NR_local registers
 *
 * REGISTER MANAGEMENT:
 * --------------------
 * The generator maintains a pool of scratch registers that can be borrowed
 * and returned as needed during transpose operations. When a register is no
 * longer needed, it's returned to the scratch pool for reuse.
 *
 * BLOCK ITERATION:
 * ----------------
 * The MR x NR matrix is processed in nested loops:
 *   - Outer loop: Iterate over MR dimension in blocks of numElemsPerReg
 *   - Inner loop: Iterate over NR dimension in blocks of numElemsPerReg
 * Each iteration transposes one numElemsPerReg x numElemsPerReg sub-block.
 */

// Utility function to round up x to the nearest multiple of y
static int
make_multiple_of(int x, int y)
{
    return ((x + y - 1) / y) * y;
}

/*
 * SCRATCH REGISTER MANAGEMENT
 * ----------------------------
 * These functions manage a pool of available scratch registers.
 * Registers are borrowed during transpose operations and returned when
 * no longer needed, allowing efficient reuse of the limited register set.
 */

// Get a scratch register from the pool
// Returns register index if available, -1 if pool is empty
template<utils::kernelInstrType KType>
int
TransposeGenerator<KType>::getScratchReg()
{
    // check if queue is empty
    if (!scratch_reg_queue.empty()) {
        int reg = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        numScratchRegs--;
        return reg;
    }
    // if queue is empty, return -1
    return -1;
}

// Return a scratch register to the pool for reuse
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::returnScratchReg(int reg)
{
    if (reg >= 0 && reg < numRegs) {
        scratch_reg_queue.push(reg);
        numScratchRegs++;
    }
}

// Remove a specific register from the scratch pool
// Used when storing and restoring data from stack
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::remove_from_scratch_reg_queue(int reg)
{
    // keep popping from the queue until the reg is found
    while (!scratch_reg_queue.empty()) {
        int r = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        if (r == reg) {
            numScratchRegs--;
            return;
        }
        scratch_reg_queue.push(r);
    }
}

// Swap the prev_output and curr_output arrays
// Used between transpose stages to make the output of one stage
// become the input to the next
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::swapOutputs()
{
    int* temp   = prev_output;
    prev_output = curr_output;
    curr_output = temp;
}

/*
 * TRANSPOSE PRIMITIVE OPERATIONS
 * -------------------------------
 * These functions implement the basic building blocks for matrix transpose
 * using SIMD unpack and permute instructions.
 */

// Unpack two registers (s1, s2) using vunpcklps/vunpckhps
// This interleaves single-precision values, transposing 2x2 sub-matrices
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpackPsPair(int s1, int s2, int i)
{
    int scratch = getScratchReg();
    jit_->vunpcklps(RegType(scratch), RegType(s1), RegType(s2));
    curr_output[i] = scratch;
    jit_->vunpckhps(RegType(s1), RegType(s1), RegType(s2));
    curr_output[i + 1] = s1;
    returnScratchReg(s2);
}

// Unpack a single register with itself (used when MR_local is odd)
// Similar to unpackPsPair but operates on a single register
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpackPs(int s1, int i)
{
    int scratch = getScratchReg();
    jit_->vunpcklps(RegType(scratch), RegType(s1), RegType(s1));
    curr_output[i] = scratch;
    jit_->vunpckhps(RegType(s1), RegType(s1), RegType(s1));
    curr_output[i + 1] = s1;
}

// Unpack two registers using vunpcklpd/vunpckhpd
// This interleaves double-precision values (pairs of floats), transposing 4x4
// sub-matrices
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpackPdPair(int s1,
                                        int s2,
                                        int op_idx1,
                                        int op_idx2)
{
    int scratch = getScratchReg();
    jit_->vunpcklpd(RegType(scratch), RegType(s1), RegType(s2));
    jit_->vunpckhpd(RegType(s1), RegType(s1), RegType(s2));
    curr_output[op_idx1] = scratch;
    curr_output[op_idx2] = s1;
    returnScratchReg(s2);
}

// Unpack a single register with itself (used when MR_local is odd)
// Similar to unpackPdPair but operates on a single register
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpackPd(int s1, int op_idx1, int op_idx2)
{
    int scratch = getScratchReg();
    jit_->vunpcklpd(RegType(scratch), RegType(s1), RegType(s1));
    jit_->vunpckhpd(RegType(s1), RegType(s1), RegType(s1));
    curr_output[op_idx1] = scratch;
    curr_output[op_idx2] = s1;
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permutePdR1Pair(
    int s1, int s2, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vperm2f128(RegType(scratch), RegType(s1), RegType(s2), 0x20);
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vperm2f128(RegType(s1), RegType(s1), RegType(s2), 0x31);
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
    returnScratchReg(s2);
}

template<>
void
TransposeGenerator<utils::kernelInstrType::avx512_ymm_32_reg>::permutePdR1Pair(
    int s1, int s2, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vshuff32x4(RegType(scratch), RegType(s1), RegType(s2), 0x00);
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vshuff32x4(RegType(s1), RegType(s1), RegType(s2), 0x03);
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
    returnScratchReg(s2);
}

template<>
void
TransposeGenerator<utils::kernelInstrType::avx512_zmm_32_reg>::permutePdR1Pair(
    int s1, int s2, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vmovups(RegType(scratch), RegType(s1));
    jit_->vpermt2pd(RegType(scratch), RegType(sel1), RegType(s2));
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vpermt2pd(RegType(s1), RegType(sel2), RegType(s2));
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
    returnScratchReg(s2);
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permutePdR1(
    int s1, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vperm2f128(RegType(scratch), RegType(s1), RegType(s1), 0x20);
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vperm2f128(RegType(s1), RegType(s1), RegType(s1), 0x31);
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
}

template<>
void
TransposeGenerator<utils::kernelInstrType::avx512_ymm_32_reg>::permutePdR1(
    int s1, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vshuff32x4(RegType(scratch), RegType(s1), RegType(s1), 0x00);
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vshuff32x4(RegType(s1), RegType(s1), RegType(s1), 0x3);
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
}

template<>
void
TransposeGenerator<utils::kernelInstrType::avx512_zmm_32_reg>::permutePdR1(
    int s1, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vmovups(RegType(scratch), RegType(s1));
    jit_->vpermt2pd(RegType(scratch), RegType(sel1), RegType(s1));
    curr_output[d1] = scratch;
    if (NR_local > numElemsPerReg / 2) {
        jit_->vpermt2pd(RegType(s1), RegType(sel2), RegType(s1));
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permuteR2Pair(
    int s1, int s2, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vmovups(Zmm(scratch), Zmm(s1));
    jit_->vpermt2pd(Zmm(scratch), Zmm(sel1), Zmm(s2));
    jit_->vpermt2pd(Zmm(s1), Zmm(sel2), Zmm(s2));
    curr_output[d1] = scratch;
    curr_output[d2] = s1;
    returnScratchReg(s2);
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permuteR2(int s1, int s2, int sel1, int d1)
{
    int scratch = getScratchReg();
    jit_->vmovups(Zmm(scratch), Zmm(s1));
    jit_->vpermt2pd(Zmm(scratch), Zmm(sel1), Zmm(s2));
    curr_output[d1] = scratch;
    returnScratchReg(s1);
    returnScratchReg(s2);
}

template<utils::kernelInstrType KType>
TransposeGenerator<KType>::TransposeGenerator(Xbyak::CodeGenerator* jit,
                                              int                   MR,
                                              int                   NR,
                                              bool                  useMask,
                                              bool                  mLoop,
                                              int                   numMaskRegs,
                                              int           cRegStartIdx,
                                              int           cRegCount,
                                              Xbyak::Reg64& regCPtr)
    : jit_(jit)
    // Note: we will be corrupting this register, but since this is the last
    // part of an IR loop, it is safe to do so.
    , regCPtr(regCPtr)
    , regCjr(jit->r8)         // Pointer to current column in output
    , regCsC(jit->r9)         // Column stride (in bytes)
    , regRsCBlock(jit->r10)   // Row stride for a block
    , regCsCBlock(jit->r11)   // Column stride for a block
    , regNleft(jit->r12)      // Number of columns remaining
    , regTmp1(jit->r13)       // Temporary register 1
    , regTmp2(jit->r14)       // Temporary register 2
    , regTmp3(jit->rax)       // Temporary register 3
    , regTmp4(jit->rbx)       // Temporary register 4
    , regTmpHalf(jit->eax)    // 32-bit temporary (for masks)
    , regNleftLocal(jit->ebx) // 32-bit local column count
    , MR(MR)
    , NR(NR)
    , useMask(useMask)
    , numMaskRegs(numMaskRegs)
    , cRegStartIdx(cRegStartIdx)
    , cRegCount(cRegCount)
{
    // Initialize scratch register pool with all registers before cRegStartIdx
    numScratchRegs = cRegStartIdx;
    for (iter_t i = 0; i < cRegStartIdx; i++) {
        scratch_reg_queue.push(i);
    }

    // maskRegIdx is hardcoded to 0 in f32_gemm_generator.cc.
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        // making sure that ymm0 is unused as it contains the mask array for n
        // dimension.
        if (useMask && mLoop) {
            int dummy = getScratchReg(); // Remove ymm0 from scratch pool
        }
    }
    MR_local              = 0;
    NR_local              = 0;
    betaRegIdx            = 0;
    numFullNRBlocks       = 0;
    numMaskNRBlocks       = 0;
    numNRBlocks           = 0;
    numFullMRBlocks       = 0;
    numMaskMRBlocks       = 0;
    numMRBlocks           = 0;
    numRegsAfterUnpackps  = 0;
    numRegsAfterUnpackpd  = 0;
    numRegsAfterPermuteR1 = 0;
    numRegsAfterPermuteR2 = 0;
}

/*
 * SET CONTEXT FOR TRANSPOSE
 * --------------------------
 * Calculate block counts and load runtime parameters (strides, masks, beta)
 * needed for the transpose operation.
 *
 * This must be called before generateTranspose() to set up the necessary
 * context for code generation.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::setContext(
    const Xbyak::Reg64& postOpsArgWrapperPtrReg, bool fuseBetaWithStore)
{
    // Calculate how many sub-blocks we need in each dimension
    numFullNRBlocks = NR / numElemsPerReg; // Full blocks in column dimension
    numMaskNRBlocks = useMask ? numMaskRegs : 0; // Partial/masked blocks
    numNRBlocks     = numFullNRBlocks + numMaskNRBlocks;

    numFullMRBlocks = MR / numElemsPerReg; // Full blocks in row dimension
    numMaskMRBlocks =
        MR % numElemsPerReg > 0 ? 1 : 0; // Partial block if MR not divisible
    numMRBlocks = numFullMRBlocks + numMaskMRBlocks;

    // load csC value multiply by sizeof(float)
    jit_->mov(regCsC, jit_->ptr[postOpsArgWrapperPtrReg
                                + offsetof(dlp::kernels::gemmParams, csC)]);
    jit_->lea(regCsC, jit_->ptr[regCsC * sizeof(float)]);

    // Calculate stride for a full sub-block: numElemsPerReg * csC *
    // sizeof(float) For 8 elements: multiply by 8
    jit_->mov(regCsCBlock, regCsC);
    jit_->lea(regCsCBlock, jit_->ptr[regCsCBlock * 8]);

    // For ZMM (16 elements), multiply by an additional 2
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit_->lea(regCsCBlock, jit_->ptr[regCsCBlock * 2]);
    }

    // Set up mask registers for AVX512
    if constexpr (KType != utils::kernelInstrType::avx2_ymm_16_reg) {
        // This logic is copied from the gemm generator to ensure consistency
        // between GEMM and transpose generator.
        // If this logic changes, we need to change the logic in transpose
        // generator as well.
        for (iter_t i = 0; i < numMaskRegs; i++) {
            fringeMask[i] = Opmask(i + 1);
        }

        mrMask = Opmask(numMaskRegs + 1);
    }

    // load n value
    jit_->mov(regNleft, jit_->ptr[postOpsArgWrapperPtrReg
                                  + offsetof(dlp::kernels::gemmParams, n)]);

    // If fusing beta scaling with store, load and broadcast beta value
    if (fuseBetaWithStore) {
        betaRegIdx = getScratchReg();
        jit_->mov(regTmp1,
                  jit_->ptr[postOpsArgWrapperPtrReg
                            + offsetof(dlp::kernels::gemmParams, beta)]);
        jit_->vbroadcastss(RegType(betaRegIdx), jit_->ptr[regTmp1]);
    }
}

// Initialize prev_output array with register indices for the current sub-block
// The registers are stored in row-major order in the register file, with
// stride of numNRBlocks between consecutive rows
// row_idx: which MR block we're processing (0 to numMRBlocks-1)
// col_idx: which NR block we're processing (0 to numNRBlocks-1)
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::setInitialIndices(int row_idx, int col_idx)
{
    int startIdx = cRegStartIdx + row_idx * numNRBlocks + col_idx;
    int i        = 0;
    // Set register indices for the actual data (MR_local rows)
    for (; i < MR_local; i++) {
        prev_output[i] = startIdx + i * numNRBlocks;
        curr_output[i] = -1;
    }
    // Mark remaining slots as unused (-1) for padding to numElemsPerReg
    for (; i < numElemsPerReg; i++) {
        prev_output[i] = -1;
        curr_output[i] = -1;
    }
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::generateNleftLocal(int j)
{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit_->kmovw(regTmpHalf, fringeMask[j]);
        jit_->popcnt(regNleftLocal, regTmpHalf);
    } else if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        jit_->kmovb(regTmpHalf, fringeMask[j]);
        jit_->popcnt(regNleftLocal, regTmpHalf);
    } else {
        jit_->mov(regNleftLocal, Xbyak::Reg32(regNleft.getIdx()));
    }
}

/*
 * REGISTER SPILLING TO STACK
 * ---------------------------
 * When there aren't enough scratch registers available for a sub-block
 * transpose, we temporarily spill some registers from the next block to
 * the stack, use them as scratch space, then restore them.
 */

// Push num_regs registers from the next block to the stack
// This makes them available for use as scratch registers
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::store_reg_in_stack(int num_regs,
                                              int nextBlockI,
                                              int nextBlockJ)
{
    int start_idx = cRegStartIdx + nextBlockI * numNRBlocks + nextBlockJ;
    // Allocate stack space
    jit_->sub(jit_->rsp, (num_regs * RegBytes));

    // Store registers to stack and add them to scratch pool
    for (iter_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(jit_->ptr[jit_->rsp + idx * RegBytes],
                      RegType(start_idx + idx * numNRBlocks));
        returnScratchReg(start_idx + idx * numNRBlocks);
    }
}

// Restore num_regs registers from the stack back to their original locations
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::get_reg_from_stack(int num_regs,
                                              int nextBlockI,
                                              int nextBlockJ)
{
    int start_idx = cRegStartIdx + nextBlockI * numNRBlocks + nextBlockJ;

    // Restore registers from stack and remove them from scratch pool
    for (iter_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(RegType(start_idx + idx * numNRBlocks),
                      jit_->ptr[jit_->rsp + idx * RegBytes]);
        remove_from_scratch_reg_queue(start_idx + idx * numNRBlocks);
    }

    // Free stack space
    jit_->add(jit_->rsp, (num_regs * RegBytes));
}

/*
 * MAIN TRANSPOSE GENERATION FUNCTION
 * -----------------------------------
 * Generates JIT code to transpose an MR x NR block by dividing it into
 * numElemsPerReg x numElemsPerReg sub-blocks and transposing each.
 *
 * The nested loop structure is:
 *   for i in [0, numMRBlocks):     // Iterate over row dimension
 *     for j in [0, numNRBlocks):   // Iterate over column dimension
 *       transpose sub-block[i][j]  // Size: numElemsPerReg x numElemsPerReg
 *
 * Each sub-block transpose goes through up to 4 stages (depending on size):
 *   Stage 1: unpack_ps  (2x2 transpose)
 *   Stage 2: unpack_pd  (4x4 transpose)
 *   Stage 3: permute_r1 (8x8 transpose) - if MR_local > 4
 *   Stage 4: permute_r2 (16x16 transpose) - if MR_local > 8 (ZMM only)
 */
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
TransposeGenerator<KType>::generateTranspose(
    const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
    dlp::jit::jitAlgoType algoType,
    bool                  fuseBetaWithStore)
{
    // Save registers that will be used as temporaries
    utils::registerGuard<Xbyak::Reg64> rG{ jit_ };
    rG.saveRegister(regCjr);
    rG.saveRegister(regRsCBlock);
    rG.saveRegister(regCsCBlock);
    rG.saveRegister(regCsC);
    rG.saveRegister(regNleft);
    rG.saveRegister(regTmp1);
    rG.saveRegister(regTmp2);
    rG.saveRegister(regTmp3);
    rG.saveRegister(regTmp4);

    // Set up context: load stride values, configure masks, etc.
    setContext(postOpsArgWrapperPtrReg, fuseBetaWithStore);

    int numRegsToStore = MR * numNRBlocks;

    // Outer loop: iterate over MR dimension in blocks of numElemsPerReg
    for (iter_t i = 0; i < numMRBlocks; i++) {
        // Copy the output matrix pointer for this row block
        jit_->mov(regCjr, regCPtr);

        // Determine actual size of this row block (may be partial in last
        // iteration)
        MR_local = (i == numFullMRBlocks) ? MR % numElemsPerReg
                                          : numElemsPerReg;

        // Inner loop: iterate over NR dimension in blocks of numElemsPerReg
        for (iter_t j = 0; j < numNRBlocks; j++) {

            // Set the dimensions of the current sub-block
            // Always setting NR_local to numElemsPerReg to avoid optimizations
            // based on NR dimension.
            // TODO: set it to variable when we pass n_left value at generation
            // time.
            NR_local = numElemsPerReg;

            // Determine if this is a partial column block (fringe case)
            // that requires variable/masked stores
            variableStores = (j >= numFullNRBlocks);
            if (variableStores) {
                generateNleftLocal(j - numFullNRBlocks);
            }

            // Calculate minimum scratch registers needed for this sub-block
            int minScratchReq =
                calculateScratchReq() + (fuseBetaWithStore ? 1 : 0);

            int numRegsToStoreLocal = MR_local;

            // Handle case where we don't have enough scratch registers:
            // Temporarily push some registers from the next block to stack
            int numRegsToPush = 0;
            int nextBlockI    = (i + 1) == numMRBlocks ? i : i + 1;
            int nextBlockJ    = (j + 1) == numNRBlocks ? j : j + 1;
            if (minScratchReq > numScratchRegs) {
                if (numRegsToStore > numRegsToStoreLocal) {
                    // Borrow registers from the next block by pushing to stack
                    numRegsToPush = minScratchReq - numScratchRegs;
                    store_reg_in_stack(numRegsToPush, nextBlockI, nextBlockJ);
                } else {
                    // Not enough registers and can't borrow from next block
                    // TODO: we can reduce the MR_local to a smaller value,
                    // and generate the transpose for the smaller block.
                    return dlp::jit::jitGeneratorError::notSupported;
                }
            }

            // Initialize register indices for the current sub-block
            setInitialIndices(i, j);

            // Generate the transpose code for this numElemsPerReg x
            // numElemsPerReg sub-block
            generateTransposeBlockMRxNR(fuseBetaWithStore);

            // Restore the registers we borrowed from the next block
            get_reg_from_stack(numRegsToPush, nextBlockI, nextBlockJ);

            // Move output pointer to next column block
            jit_->add(regCjr, regCsCBlock);
        }

        // Move output pointer to next row block (advance by MR_local rows)
        jit_->add(regCPtr, MR_local * sizeof(float));
    }

    embedSelectors();
    return dlp::jit::jitGeneratorError::success;
}

/*
 * Calculate minimum scratch registers needed for transposing current sub-block
 *
 * This function determines how many additional scratch registers are needed
 * beyond the registers holding the input data. The requirement depends on:
 *   - MR_local: the number of rows in the current sub-block
 *   - Which transpose stages need to be executed
 *
 * The calculation accounts for the "high water mark" of register usage
 * at each stage, considering that input registers are progressively freed
 * and can be reused as scratch space.
 */
template<utils::kernelInstrType KType>
int
TransposeGenerator<KType>::calculateScratchReq()
{
    int numStoreRegs = 0;
    // AVX2 needs 1 extra register for the mask
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        numStoreRegs = 1;
    }

    int is_odd = MR_local % 2;

    // Stage 1: unpack_ps (2x2 transpose)
    // Output registers needed (rounded up to multiple of 2)
    numRegsAfterUnpackps = make_multiple_of(MR_local, 2);

    // Scratch required: 1 for output of first instruction + 1 if MR_local is
    // odd
    int scratchReqForStage1 = 1 + is_odd;
    numStoreRegs += scratchReqForStage1;

    // Stage 2: unpack_pd (4x4 transpose)
    // Output registers needed (rounded up to multiple of 4)
    numRegsAfterUnpackpd = make_multiple_of(MR_local, 4);

    // Additional scratch needed beyond what's already allocated
    int scratchReqForStage2 = numRegsAfterUnpackpd - numRegsAfterUnpackps;
    numStoreRegs += scratchReqForStage2;

    if (MR_local <= 4) {
        return numStoreRegs;
    }

    // Stage 3: permute_r1 (8x8 transpose via 128-bit lane permutations)
    numRegsAfterPermuteR1 = NR_local > 8 ? make_multiple_of(MR_local, 8)
                                         : make_multiple_of(MR_local, 8) / 2;
    int scratchReqForStage3 =
        std::max(0, numRegsAfterPermuteR1 - numRegsAfterUnpackpd);

    // AVX512 ZMM needs 2 selector registers for permutation indices
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        scratchReqForStage3 += 2;
    }
    numStoreRegs += scratchReqForStage3;

    if (MR_local <= 8) {
        return numStoreRegs;
    }

    // Stage 4: permute_r2 (16x16 transpose, ZMM only)
    numRegsAfterPermuteR2 = NR_local;
    int scratchReqForStage4 =
        std::max(0, numRegsAfterPermuteR2 - numRegsAfterPermuteR1);

    numStoreRegs += scratchReqForStage4;

    return numStoreRegs;
}

/*
 * Generate transpose code for a single numElemsPerReg x numElemsPerReg
 * sub-block
 *
 * Executes up to 4 stages of SIMD transpose operations:
 *   Stage 1: unpack_ps  - 2x2 transpose (always executed)
 *   Stage 2: unpack_pd  - 4x4 transpose (always executed)
 *   Stage 3: permute_r1 - 8x8 transpose (if MR_local > 4)
 *   Stage 4: permute_r2 - 16x16 transpose (if MR_local > 8, ZMM only)
 *
 * After each stage, swapOutputs() is called to make the output of one stage
 * become the input to the next. Finally, the transposed data is stored to
 * the output matrix in column-major order.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::generateTransposeBlockMRxNR(bool fuseBetaWithStore)
{
    // generate the transpose for the block
    unpack_ps_MRxNR();

    swapOutputs();

    unpack_pd_MRxNR();

    int selector1 = -1, selector2 = -1;

    if (MR_local <= 4) {
        goto store;
    }

    // selectors are only needed for avx512 instructions.
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        selector1 = getScratchReg();
        selector2 = getScratchReg();
        jit_->vmovdqu64(Zmm(selector1), get_selector(selector1_off));
        jit_->vmovdqu64(Zmm(selector2), get_selector(selector2_off));
    }

    swapOutputs();
    permute_r1_MRxNR(selector1, selector2);

    if (MR_local <= 8) {
        // return the selectors if they are used.
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            returnScratchReg(selector1);
            returnScratchReg(selector2);
        }
        goto store;
    }

    // MR_local <=8 always when dealing with YMMs. so this code path
    // will only be taken for avx512 instructions. No need to check
    // explicitly for avx512_zmm_32_reg config.
    jit_->vmovdqu64(Zmm(selector1), get_selector(selector1_1_off));
    jit_->vmovdqu64(Zmm(selector2), get_selector(selector2_1_off));

    swapOutputs();
    permute_r2_MRxNR(selector1, selector2);

    returnScratchReg(selector1);
    returnScratchReg(selector2);

store:
    // Store the transposed data to memory in column-major order
    store_MRxNR(fuseBetaWithStore);
}

/*
 * STAGE 1: Transpose 2x2 sub-matrices using vunpcklps/vunpckhps
 *
 * This stage interleaves single-precision floats from pairs of registers.
 * This effectively transposes 2x2 blocks of the matrix.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpack_ps_MRxNR()
{
    int i           = 0;
    int full_panels = MR_local / 2 * 2;
    int half_panels = MR_local % 2;
    for (; i < full_panels; i += 2) {
        unpackPsPair(prev_output[i], prev_output[i + 1], i);
    }

    // Handle odd register if MR_local is odd
    if (half_panels) {
        unpackPs(prev_output[i], i);
    }
}

// Helper to calculate source and destination indices for unpack_pd stage
static void
calculate_unpack_pd_idx(
    int idx, int* src1, int* src2, int* dst1, int* dst2, int even_pairs)
{
    *src1 = idx;
    *src2 = idx + 1;
    *dst1 = (idx / 2);
    *dst2 = even_pairs + (idx / 2);
}

/*
 * STAGE 2: Transpose 4x4 sub-matrices using vunpcklpd/vunpckhpd
 *
 * This stage interleaves double-precision values (pairs of floats) from
 * pairs of registers. Operating on the output of Stage 1, this completes
 * the transpose of 4x4 blocks.
 *
 * The registers are processed in groups of 4, with special handling for
 * cases where numRegsAfterUnpackps is not a multiple of 4.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::unpack_pd_MRxNR()
{
    int src1, src2, dst1, dst2;
    int floor_4    = (numRegsAfterUnpackps / 4) * 4;
    int mod_4      = numRegsAfterUnpackps % 4;
    int ceil_4     = (numRegsAfterUnpackps + 4 - 1) / 4 * 4;
    int even_pairs = ceil_4 / 2;
    int i          = 0;

    // Process complete groups of 4 registers
    for (; i < floor_4; i += 4) {
        calculate_unpack_pd_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        unpackPdPair(prev_output[src1], prev_output[src1 + 2], dst1, dst1 + 1);
        unpackPdPair(prev_output[src2], prev_output[src2 + 2], dst2, dst2 + 1);
    }

    // Handle remaining registers if numRegsAfterUnpackps % 4 == 2
    calculate_unpack_pd_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
    if (mod_4 == 2) {
        unpackPd(prev_output[src1], dst1, dst1 + 1);
        unpackPd(prev_output[src2], dst2, dst2 + 1);
    }
}

// Helper to calculate indices for first half of registers in permute_r1 stage
static void
calculate_permute_r1_idx(
    int idx, int* src1, int* src2, int* dst1, int* dst2, int even_pairs)
{
    *src1 = idx;
    *src2 = idx + 1;
    *dst1 = (idx / 2);
    *dst2 = even_pairs + (idx / 2);
}

// Helper to calculate indices for second half of registers in permute_r1 stage
static void
calculate_permute_r1_idx2(int  idx,
                          int* s1,
                          int* s2,
                          int* d1,
                          int* d2,
                          int  even_pairs,
                          int  half_registers)
{
    *s1 = half_registers + idx;
    *s2 = *s1 + 1;
    *d1 = (even_pairs + idx) / 2;
    *d2 = even_pairs + *d1;
}

/*
 * STAGE 3: Transpose 8x8 sub-matrices by permuting 128-bit lanes
 *
 * This stage uses vperm2f128 (AVX2) or vshuff32x4/vpermt2pd (AVX512) to
 * rearrange 128-bit lanes within and between registers. This completes
 * the transpose of 8x8 blocks.
 *
 * The registers are split into two halves and processed separately,
 * with careful index calculation to place results in the correct order.
 *
 * For AVX512 ZMM, selector1 and selector2 contain indices for vpermt2pd.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permute_r1_MRxNR(int selector1, int selector2)
{
    int src1, src2, dst1, dst2;
    int half_registers = numRegsAfterUnpackpd / 2;
    int full_pairs = ((half_registers) / 4) * 4; // Groups of 4 in first half
    int half_pairs = (half_registers) % 4;       // Remainder
    int ceil_8     = (numRegsAfterUnpackpd + 8 - 1) / 8 * 8;
    int even_pairs = ceil_8 / 2;
    int i          = 0;

    // Process complete groups of 4 from both halves
    for (i = 0; i < full_pairs; i += 4) {
        // Operate on first half of the registers in prev_output
        calculate_permute_r1_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        permutePdR1Pair(prev_output[src1], prev_output[src1 + 2], selector1,
                        selector2, dst1, dst1 + 1);
        permutePdR1Pair(prev_output[src2], prev_output[src2 + 2], selector1,
                        selector2, dst2, dst2 + 1);

        // Operate on second half of the registers in prev_output
        calculate_permute_r1_idx2(i, &src1, &src2, &dst1, &dst2, even_pairs,
                                  half_registers);
        permutePdR1Pair(prev_output[src1], prev_output[src1 + 2], selector1,
                        selector2, dst1, dst1 + 1);
        permutePdR1Pair(prev_output[src2], prev_output[src2 + 2], selector1,
                        selector2, dst2, dst2 + 1);
    }

    // Handle remaining registers if half_registers % 4 != 0
    if (half_pairs) {
        // Operate on first half of the registers in prev_output
        calculate_permute_r1_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        permutePdR1(prev_output[src1], selector1, selector2, dst1, dst1 + 1);
        permutePdR1(prev_output[src2], selector1, selector2, dst2, dst2 + 1);

        // Operate on second half of the registers in prev_output
        calculate_permute_r1_idx2(i, &src1, &src2, &dst1, &dst2, even_pairs,
                                  half_registers);
        permutePdR1(prev_output[src1], selector1, selector2, dst1, dst1 + 1);
        permutePdR1(prev_output[src2], selector1, selector2, dst2, dst2 + 1);
    }
}

// Helper to calculate source register indices for permute_r2 stage
// The pattern maps logical column indices to physical register layout
static void
calculate_r2_idx(int idx, int* src1, int* src2)
{
    if (idx < 8) {
        *src1 = 2 * idx;
        *src2 = 2 * (idx + 1);
    } else {
        *src1 = 2 * (idx - 8) + 1;
        *src2 = 2 * (idx - 8) + 3;
    }
}

/*
 * STAGE 4: Transpose 16x16 sub-matrices (AVX512 ZMM only)
 *
 * This final stage uses vpermt2pd instructions to complete the transpose
 * of full 16x16 blocks. This is only needed when using ZMM registers
 * (512-bit) and MR_local > 8.
 *
 * The reg_indices array defines the mapping from the current register
 * layout to the final transposed column order.
 *
 * selector1 and selector2 contain permutation indices for vpermt2pd.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::permute_r2_MRxNR(int selector1, int selector2)
{
    // Mapping from register position to column index in transposed output
    int reg_indices[16] = {
        0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15
    };
    int src1, src2;

    // Process pairs of registers to generate output columns
    for (iter_t i = 0; i < numElemsPerReg; i += 2) {
        calculate_r2_idx(i, &src1, &src2);

        // Only generate code for columns that are within NR_local
        if (reg_indices[i] < NR_local && reg_indices[i + 1] < NR_local) {
            permuteR2Pair(prev_output[src1], prev_output[src2], selector1,
                          selector2, i, i + 1);
        } else if (reg_indices[i] < NR_local) {
            permuteR2(prev_output[src1], prev_output[src2], selector1, i);
        }
    }
}

/*
 * STORE OPERATIONS
 * ----------------
 * After transposing, the data is stored to memory in column-major order.
 * These functions handle both fixed and variable (masked) column stores,
 * with optional beta scaling (C = beta*C + A^T).
 */

// Generic helper template for variable column stores
// This abstracts the common pattern: loop through columns, conditionally
// check remaining count, and perform the store operation
template<utils::kernelInstrType KType>
template<typename StoreOp>
void
TransposeGenerator<KType>::storeColumns_variableCount(const int* reg_idx,
                                                      const int* half_idx,
                                                      int        numCols,
                                                      StoreOp    storeOp)
{
    if (!variableStores) {
        // Fixed number of columns - unroll completely
        for (iter_t i = 0; i < numCols; i++) {
            storeOp(i, reg_idx[i], half_idx[i]);
            jit_->add(regTmp1, regCsC);
        }
        return;
    }

    // Variable number of columns - unroll with conditional exits
    Xbyak::Label done;
    jit_->mov(regTmp2, 0); // Initialize counter

    for (iter_t i = 0; i < numCols; i++) {
        // Check if we've stored enough columns
        jit_->cmp(Xbyak::Reg32(regTmp2.getIdx()), regNleftLocal);
        jit_->jge(done, jit_->T_NEAR);

        // Perform the store operation
        storeOp(i, reg_idx[i], half_idx[i]);
        jit_->add(regTmp1, regCsC);

        // Increment counter
        jit_->inc(regTmp2);
    }

    jit_->L(done);
}

// Unified store functions - combine betaScale and betaZero into one

// avx512 specific function for store_4xNR
// reused for avx512_ymm_32_reg
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::store_4xNR(bool applyBetaScale)
{
    const int reg_idx[16] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
    const int quarter_idx[16] = {
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    };

    if (applyBetaScale) {
        auto betaScaleOp = [this](int col_idx, int reg_idx, int quarter_idx) {
            int cRegIdx = getScratchReg();
            jit_->vmovups(Xmm(cRegIdx) | mrMask, jit_->ptr[regTmp1]);

            int scratch = getScratchReg();
            jit_->vextractf32x4(Xmm(scratch), RegType(curr_output[reg_idx]),
                                quarter_idx);
            jit_->vfmadd231ps(Xmm(scratch), Xmm(betaRegIdx), Xmm(cRegIdx));
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Xmm(scratch));
            returnScratchReg(scratch);

            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_idx, quarter_idx, numElemsPerReg,
                                   betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int quarter_idx) {
            if (quarter_idx == 0) {
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                              Zmm(curr_output[reg_idx]));
            } else {
                int scratch = getScratchReg();
                jit_->vextractf32x4(Xmm(scratch), RegType(curr_output[reg_idx]),
                                    quarter_idx);
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Xmm(scratch));
                returnScratchReg(scratch);
            }
        };
        storeColumns_variableCount(reg_idx, quarter_idx, numElemsPerReg,
                                   betaZeroOp);
    }
}

// avx2 specific function for store_4xNR
template<>
void
TransposeGenerator<utils::kernelInstrType::avx2_ymm_16_reg>::store_4xNR(
    bool applyBetaScale)
{
    const int reg_idx[8]  = { 0, 1, 2, 3, 0, 1, 2, 3 };
    const int half_idx[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    if (applyBetaScale) {
        auto betaScaleOp = [this](int col_idx, int reg_idx, int quarter_idx) {
            int cRegIdx = getScratchReg();
            jit_->vmaskmovps(Xmm(cRegIdx), Xmm(maskRegIdx), jit_->ptr[regTmp1]);

            int scratch = getScratchReg();
            jit_->vextractf128(Xmm(scratch), RegType(curr_output[reg_idx]),
                               quarter_idx);
            jit_->vfmadd231ps(Xmm(scratch), Xmm(betaRegIdx), Xmm(cRegIdx));
            jit_->vmaskmovps(jit_->ptr[regTmp1], Xmm(maskRegIdx), Xmm(scratch));
            returnScratchReg(scratch);

            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_idx, half_idx, numElemsPerReg,
                                   betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int quarter_idx) {
            if (quarter_idx == 0) {
                jit_->vmaskmovps(jit_->ptr[regTmp1], Xmm(maskRegIdx),
                                 Xmm(curr_output[reg_idx]));
            } else {
                int scratch = getScratchReg();
                jit_->vextractf128(Xmm(scratch), RegType(curr_output[reg_idx]),
                                   quarter_idx);
                jit_->vmaskmovps(jit_->ptr[regTmp1], Xmm(maskRegIdx),
                                 Xmm(scratch));
                returnScratchReg(scratch);
            }
        };
        storeColumns_variableCount(reg_idx, half_idx, numElemsPerReg,
                                   betaZeroOp);
    }
}

#define MASK_MOV(dst, src)                                                     \
    if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {        \
        jit_->vmovups(dst | mrMask, src);                                      \
    } else {                                                                   \
        jit_->vmaskmovps(dst, Ymm(maskRegIdx), src);                           \
    }

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::store_8xNR(bool applyBetaScale)
{
    const int reg_indices[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };
    const int dummy_idx[8]   = { 0, 0, 0, 0, 0, 0, 0, 0 };

    if (applyBetaScale) {
        auto betaScaleOp = [this](int col_idx, int reg_idx, int unused) {
            int cRegIdx = getScratchReg();
            MASK_MOV(Ymm(cRegIdx), jit_->ptr[regTmp1])
            jit_->vfmadd231ps(Ymm(curr_output[reg_idx]), Ymm(betaRegIdx),
                              Ymm(cRegIdx));
            MASK_MOV(jit_->ptr[regTmp1], Ymm(curr_output[reg_idx]))
            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_indices, dummy_idx, numElemsPerReg,
                                   betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int unused) {
            MASK_MOV(jit_->ptr[regTmp1], Ymm(curr_output[reg_idx]))
        };
        storeColumns_variableCount(reg_indices, dummy_idx, numElemsPerReg,
                                   betaZeroOp);
    }
}

template<>
void
TransposeGenerator<utils::kernelInstrType::avx512_zmm_32_reg>::store_8xNR(
    bool applyBetaScale)
{
    const int reg_idx[16]  = { 0, 4, 2, 6, 0, 4, 2, 6, 1, 5, 3, 7, 1, 5, 3, 7 };
    const int half_idx[16] = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };

    if (applyBetaScale) {
        auto betaScaleOp = [this](int col_idx, int reg_idx, int half_idx) {
            int cRegIdx = getScratchReg();
            jit_->vmovups(Ymm(cRegIdx) | mrMask, jit_->ptr[regTmp1]);

            int scratch = getScratchReg();
            jit_->vextractf32x8(Ymm(scratch), Zmm(curr_output[reg_idx]),
                                half_idx);
            jit_->vfmadd231ps(Ymm(scratch), Ymm(betaRegIdx), Ymm(cRegIdx));
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Ymm(scratch));

            returnScratchReg(scratch);
            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_idx, half_idx, numElemsPerReg,
                                   betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int half_idx) {
            if (half_idx == 0) {
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                              Ymm(curr_output[reg_idx]));
            } else {
                int scratch = getScratchReg();
                jit_->vextractf32x8(Ymm(scratch), Zmm(curr_output[reg_idx]),
                                    half_idx);
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Ymm(scratch));
                returnScratchReg(scratch);
            }
        };
        storeColumns_variableCount(reg_idx, half_idx, numElemsPerReg,
                                   betaZeroOp);
    }
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::store_16xNR(bool applyBetaScale)
{
    const int reg_indices[16] = { 0, 4,  2,  6,  1, 5,  3,  7,
                                  8, 12, 10, 14, 9, 13, 11, 15 };
    const int dummy_idx[16]   = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    if (applyBetaScale) {
        auto betaScaleOp = [this](int col_idx, int reg_idx, int unused) {
            int cRegIdx = getScratchReg();
            jit_->vmovups(Zmm(cRegIdx) | mrMask, jit_->ptr[regTmp1]);
            jit_->vfmadd231ps(Zmm(curr_output[reg_idx]), Zmm(betaRegIdx),
                              Zmm(cRegIdx));
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                          Zmm(curr_output[reg_idx]));
            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_indices, dummy_idx, numElemsPerReg,
                                   betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int unused) {
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                          Zmm(curr_output[reg_idx]));
        };
        storeColumns_variableCount(reg_indices, dummy_idx, numElemsPerReg,
                                   betaZeroOp);
    }
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::createMaskFromConstant(int value)

{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit_->mov(regTmpHalf, 0xFFFF >> (numElemsPerReg - value));
        jit_->kmovw(mrMask, regTmpHalf);
    } else if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        jit_->mov(regTmpHalf, 0xFF >> (numElemsPerReg - value));
        jit_->kmovb(mrMask, regTmpHalf);
    } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        int scratch = getScratchReg();
        maskRegIdx  = getScratchReg();

        // Create all 1s in scratch
        jit_->vxorps(Ymm(scratch), Ymm(scratch), Ymm(scratch));
        jit_->vcmpeqps(Ymm(scratch), Ymm(scratch), Ymm(scratch));

        // Create all 0s in mask
        jit_->vxorps(Ymm(maskRegIdx), Ymm(maskRegIdx), Ymm(maskRegIdx));

        // Blend first MR_local lanes (handles all cases 0-8)
        if (value > 0 && value < 8) {
            uint8_t blend_imm = (1 << value) - 1;
            jit_->vblendps(Ymm(maskRegIdx), Ymm(maskRegIdx), Ymm(scratch),
                           blend_imm);
        } else if (value == 8) {
            // Just move all 1s to mask
            jit_->vmovaps(Ymm(maskRegIdx), Ymm(scratch));
        }

        returnScratchReg(scratch);
    } else {
        // saved for future use
    }
}

/*
 * Store the transposed sub-block to memory in column-major order
 *
 * The transposed data in curr_output[] is stored to memory, with each
 * register corresponding to one column of the output. Stores can be:
 *   - Full: All MR_local elements stored (for interior of matrix)
 *   - Masked: Partial stores (for edge cases where MR % numElemsPerReg != 0)
 *
 * The function branches at runtime based on whether beta is zero.
 */
template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::store_MRxNR(bool fuseBetaWithStore)
{
    jit_->inLocalLabel();

    // Set up pointer to start of current block in output matrix
    jit_->mov(regTmp1, regCjr);

    // Create mask for partial rows (when MR_local < numElemsPerReg)
    createMaskFromConstant(MR_local);

    if (fuseBetaWithStore) {
        // Runtime check: is beta == 0?
        int scratch = getScratchReg();
        jit_->vxorps(Xmm(scratch), Xmm(scratch), Xmm(scratch));
        jit_->vucomiss(Xmm(betaRegIdx), Xmm(scratch));
        returnScratchReg(scratch);
        jit_->je(".SBETAZERO", jit_->T_NEAR);

        // beta is non-zero, fuse with store
        if (MR_local <= 4) {
            store_4xNR(true); // Store 4 rows with beta scaling
        } else if (MR_local <= 8) {
            store_8xNR(true); // Store 8 rows with beta scaling
        } else {
            store_16xNR(true); // Store 16 rows with beta scaling (ZMM only)
        }

        jit_->jmp(".AfterStoreColMajorMR", jit_->T_NEAR);
    }

    jit_->L(".SBETAZERO");
    if (MR_local <= 4) {
        store_4xNR(false); // Store 4 rows without beta scaling
    } else if (MR_local <= 8) {
        store_8xNR(false); // Store 8 rows without beta scaling
    } else {
        store_16xNR(false); // Store 16 rows without beta scaling (ZMM only)
    }

    jit_->L(".AfterStoreColMajorMR");

    // Return all registers used for output to the scratch pool
    for (iter_t i = 0; i < numElemsPerReg; i++) {
        returnScratchReg(curr_output[i]);
    }

    // For AVX2, also return the mask register
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        returnScratchReg(maskRegIdx);
    }

    jit_->outLocalLabel();
}

template<utils::kernelInstrType KType>
void
TransposeGenerator<KType>::embedSelectors()
{
    jit_->jmp(".selectorsEnd", jit_->T_NEAR);
    {
        size_t remain = jit_->getSize() % 64;
        if (remain)
            jit_->nop(64 - remain);
    }
    jit_->L(selectors);
    jit_->db(reinterpret_cast<uint8_t*>(&selector1), sizeof(selector1));
    jit_->db(reinterpret_cast<uint8_t*>(&selector2), sizeof(selector2));
    jit_->db(reinterpret_cast<uint8_t*>(&selector1_1), sizeof(selector1_1));
    jit_->db(reinterpret_cast<uint8_t*>(&selector2_1), sizeof(selector2_1));
    jit_->L(".selectorsEnd");
}

} // namespace amdzen::x86gen

// Explicit template instantiations to resolve linker errors
template class amdzen::x86gen::TransposeGenerator<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
template class amdzen::x86gen::TransposeGenerator<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::x86gen::TransposeGenerator<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
