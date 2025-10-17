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

static int
make_multiple_of(int x, int y)
{
    return ((x + y - 1) / y) * y;
}

// Helper functions converted from macros
int
avx512TransposeGenerator::getScratchReg()
{
    // check if queue is empty
    if (!scratch_reg_queue.empty()) {
        int reg = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        return reg;
    }
    return -1;
}

void
avx512TransposeGenerator::returnScratchReg(int reg)
{
    if (reg >= 0 && reg < numRegs) {
        scratch_reg_queue.push(reg);
    }
}

void
avx512TransposeGenerator::swapOutputs()
{
    int* temp   = prev_output;
    prev_output = curr_output;
    curr_output = temp;
}

void
avx512TransposeGenerator::unpackPsPair(int s1, int s2, int i)
{
    int scratch = getScratchReg();
    jit_->vunpcklps(Zmm(scratch), Zmm(s1), Zmm(s2));
    curr_output[i] = scratch;
    jit_->vunpckhps(Zmm(s1), Zmm(s1), Zmm(s2));
    curr_output[i + 1] = s1;
    returnScratchReg(s2);
}

void
avx512TransposeGenerator::unpackPs(int s1, int i)
{
    int scratch = getScratchReg();
    jit_->vunpcklps(Zmm(scratch), Zmm(s1), Zmm(s1));
    curr_output[i] = scratch;
    jit_->vunpckhps(Zmm(s1), Zmm(s1), Zmm(s1));
    curr_output[i + 1] = s1;
}

void
avx512TransposeGenerator::unpackPdPair(int s1, int s2, int op_idx1, int op_idx2)
{
    int scratch = getScratchReg();
    jit_->vunpcklpd(Zmm(scratch), Zmm(s1), Zmm(s2));
    jit_->vunpckhpd(Zmm(s1), Zmm(s1), Zmm(s2));
    curr_output[op_idx1] = scratch;
    curr_output[op_idx2] = s1;
    returnScratchReg(s2);
}

void
avx512TransposeGenerator::unpackPd(int s1, int op_idx1, int op_idx2)
{
    int scratch = getScratchReg();
    jit_->vunpcklpd(Zmm(scratch), Zmm(s1), Zmm(s1));
    jit_->vunpckhpd(Zmm(s1), Zmm(s1), Zmm(s1));
    curr_output[op_idx1] = scratch;
    curr_output[op_idx2] = s1;
}

void
avx512TransposeGenerator::permutePdR1Pair(
    int s1, int s2, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vmovups(Zmm(scratch), Zmm(s1));
    jit_->vpermt2pd(Zmm(scratch), Zmm(sel1), Zmm(s2));
    curr_output[d1] = scratch;
    if (NR_local > 8) {
        jit_->vpermt2pd(Zmm(s1), Zmm(sel2), Zmm(s2));
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
    returnScratchReg(s2);
}

void
avx512TransposeGenerator::permutePdR1(
    int s1, int sel1, int sel2, int d1, int d2)
{
    int scratch = getScratchReg();
    jit_->vmovups(Zmm(scratch), Zmm(s1));
    jit_->vpermt2pd(Zmm(scratch), Zmm(sel1), Zmm(s1));
    curr_output[d1] = scratch;
    if (NR_local > 8) {
        jit_->vpermt2pd(Zmm(s1), Zmm(sel2), Zmm(s1));
        curr_output[d2] = s1;
    } else {
        returnScratchReg(s1);
    }
}

void
avx512TransposeGenerator::permuteR2Pair(
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

void
avx512TransposeGenerator::permuteR2(int s1, int s2, int sel1, int d1)
{
    int scratch = getScratchReg();
    jit_->vmovups(Zmm(scratch), Zmm(s1));
    jit_->vpermt2pd(Zmm(scratch), Zmm(sel1), Zmm(s2));
    curr_output[d1] = scratch;
    returnScratchReg(s1);
    returnScratchReg(s2);
}

avx512TransposeGenerator::avx512TransposeGenerator(Xbyak::CodeGenerator* jit,
                                                   int                   MR,
                                                   int                   NR,
                                                   bool          useMask,
                                                   int           numMaskRegs,
                                                   int           cRegStartIdx,
                                                   int           cRegCount,
                                                   Xbyak::Reg64& regCPtr)
    : jit_(jit)
    // we will be corrupting this register, but since this is the last part of
    // an IR loop, it is safe to do so.
    , regCPtr(regCPtr)
    , regCjr(jit->r8)
    , regCsC(jit->r9)
    , regRsCBlock(jit->r10)
    , regCsCBlock(jit->r11)
    , regNleft(jit->r12)
    , regTmp1(jit->r13)
    , regTmp2(jit->r14)
    , regTmp3(jit->rax)
    , regTmp4(jit->rbx)
    , regTmpHalf(jit->eax)
    , regNleftLocal(jit->ebx)
    , MR(MR)
    , NR(NR)
    , useMask(useMask)
    , numMaskRegs(numMaskRegs)
    , cRegStartIdx(cRegStartIdx)
    , cRegCount(cRegCount)
{
    numScratchRegs = cRegStartIdx;
    // move scratch registers to the queue
    for (int i = 0; i < cRegStartIdx; i++) {
        scratch_reg_queue.push(i);
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

void
avx512TransposeGenerator::setContext(
    const Xbyak::Reg64& postOpsArgWrapperPtrReg, bool fuseBetaWithStore)
{
    numFullNRBlocks = NR / numElemsPerReg;
    numMaskNRBlocks = useMask ? numMaskRegs : 0;
    numNRBlocks =
        numFullNRBlocks + numMaskNRBlocks; // set the context for the transpose

    numFullMRBlocks = MR / numElemsPerReg;
    numMaskMRBlocks = MR % numElemsPerReg > 0 ? 1 : 0;
    numMRBlocks =
        numFullMRBlocks + numMaskMRBlocks; // set the context for the transpose

    // load csC value multiply by sizeof(float)
    jit_->mov(regCsC, jit_->ptr[postOpsArgWrapperPtrReg
                                + offsetof(dlp::kernels::gemmParams, csC)]);
    jit_->lea(regCsC, jit_->ptr[regCsC * sizeof(float)]);

    // store 16*csC*sizeof(float) in regCsCBlock
    // TODO: Make it generic for any numElemsPerReg
    jit_->mov(regCsCBlock, regCsC);
    jit_->lea(regCsCBlock, jit_->ptr[regCsCBlock * 8]);
    jit_->lea(regCsCBlock, jit_->ptr[regCsCBlock * 2]);

    // load n value and calculate nleft
    // numElemsPerReg is always assumed to be a power of 2
    // so and with numElemsPerReg - 1 will give us n%numElemsPerReg
    jit_->mov(regNleft, jit_->ptr[postOpsArgWrapperPtrReg
                                  + offsetof(dlp::kernels::gemmParams, n)]);

    // this logic is copied from the gemm generator to ensure handshake
    // between GEMM and transpose generator.
    // If this logic changes, we need to change the logic in transpose generator
    // as well.
    for (int i = 0; i < numMaskRegs; i++) {
        fringeMask[i] = Opmask(i + 1);
    }

    mrMask = Opmask(numMaskRegs + 1);

    if (fuseBetaWithStore) {
        betaRegIdx = getScratchReg();
        // broadcast beta value
        jit_->mov(regTmp1,
                  jit_->ptr[postOpsArgWrapperPtrReg
                            + offsetof(dlp::kernels::gemmParams, beta)]);
        jit_->vbroadcastss(Zmm(betaRegIdx), jit_->ptr[regTmp1]);
    }
}

void
avx512TransposeGenerator::setInitialIndices(int row_idx, int col_idx)
{
    int startIdx = cRegStartIdx + row_idx * numNRBlocks + col_idx;
    int i        = 0;
    for (; i < MR_local; i++) {
        prev_output[i] = startIdx + i * numNRBlocks;
        curr_output[i] = -1;
    }
    for (; i < numElemsPerReg; i++) {
        prev_output[i] = -1;
        curr_output[i] = -1;
    }
}

void
avx512TransposeGenerator::generateNleftLocal(int j)
{
    jit_->kmovw(regTmpHalf, fringeMask[j]);
    jit_->popcnt(regNleftLocal, regTmpHalf);
}

dlp::jit::jitGeneratorError
avx512TransposeGenerator::generateTranspose(
    const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
    dlp::jit::jitAlgoType algoType,
    bool                  fuseBetaWithStore)
{

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

    setContext(postOpsArgWrapperPtrReg, fuseBetaWithStore);

    for (int i = 0; i < numMRBlocks; i++) {
        // copy the C pointer to cJR
        jit_->mov(regCjr, regCPtr);

        MR_local = (i == numFullMRBlocks) ? MR % numElemsPerReg
                                          : numElemsPerReg;

        for (int j = 0; j < numNRBlocks; j++) {

            // set the dimensions of the block
            // Always setting NR_local to numElemsPerReg to avoid optimizations
            // based on NR dimension.
            // TODO: set it to variable when we pass n_left value at generation
            // time.
            NR_local = numElemsPerReg;

            // hint that lt kernel is being generated and stores are variable
            variableStores = (j >= numFullNRBlocks);
            if (variableStores) {
                generateNleftLocal(j - numFullNRBlocks);
            }

            // Check if we have enough scratch registers for the current block
            int minScratchReq =
                calculateScratchReq() + (fuseBetaWithStore ? 1 : 0);
            if (minScratchReq > numScratchRegs) {
                return dlp::jit::jitGeneratorError::notSupported;
            }

            // fill the prev_output array with the values of the current block
            setInitialIndices(i, j);

            // generate the transpose for the block
            generateTransposeBlockMRxNR(fuseBetaWithStore);

            // move C pointer to next block along column dimension
            jit_->add(regCjr, regCsCBlock);
        }

        // move C pointer to next block along row dimension
        // TODO: MR_local can be anything, so we need to make it generic
        // jit_->lea(regCPtr, ptr[regCPtr + MR_local * sizeof(float)]);
        jit_->add(regCPtr, MR_local * sizeof(float));
    }

    embedSelectors();
    return dlp::jit::jitGeneratorError::success;
}

int
avx512TransposeGenerator::calculateScratchReq()
{
    int is_odd = MR_local % 2;
    /*
    int stage1 = 1 + is_odd;
    int stage2 = even%4>0 ? std::min(4,2*(even%4)):0;
    */
    numRegsAfterUnpackps = make_multiple_of(MR_local, 2);

    // 1 to store the output of first instruction before any registers
    // are released.
    int scratchReqForStage1 = 1 + is_odd;

    numRegsAfterUnpackpd = make_multiple_of(MR_local, 4);

    // 1 to use as src2 for cases where src2 is invalid.
    int scratchReqForStage2 = numRegsAfterUnpackpd - numRegsAfterUnpackps + 1;
    if (MR_local <= 4) {
        return scratchReqForStage1 + scratchReqForStage2;
    }

    // 2 for selectors
    numRegsAfterPermuteR1 = NR_local > 8 ? make_multiple_of(MR_local, 8)
                                         : make_multiple_of(MR_local, 8) / 2;
    int scratchReqForStage3 =
        std::max(0, numRegsAfterPermuteR1 - numRegsAfterUnpackpd) + 2;
    if (MR_local <= 8) {
        return scratchReqForStage1 + scratchReqForStage2 + scratchReqForStage3;
    }

    numRegsAfterPermuteR2 = NR_local;
    int scratchReqForStage4 =
        std::max(0, numRegsAfterPermuteR2 - numRegsAfterPermuteR1);

    return scratchReqForStage1 + scratchReqForStage2 + scratchReqForStage3
           + scratchReqForStage4;
}

void
avx512TransposeGenerator::generateTransposeBlockMRxNR(bool fuseBetaWithStore)
{
    // generate the transpose for the block
    unpack_ps_MRxNR();

    swapOutputs();

    unpack_pd_MRxNR();

    // before permuting, move the selectors to scratch registers
    int selector1 = getScratchReg();
    int selector2 = getScratchReg();

    if (MR_local <= 4) {
        returnScratchReg(selector1);
        returnScratchReg(selector2);
        goto store;
    }

    jit_->vmovdqu64(Zmm(selector1), get_selector(selector1_off));
    jit_->vmovdqu64(Zmm(selector2), get_selector(selector2_off));

    swapOutputs();
    permute_r1_MRxNR(selector1, selector2);

    if (MR_local <= 8) {
        returnScratchReg(selector1);
        returnScratchReg(selector2);
        goto store;
    }

    jit_->vmovdqu64(Zmm(selector1), get_selector(selector1_1_off));
    jit_->vmovdqu64(Zmm(selector2), get_selector(selector2_1_off));
    swapOutputs();
    permute_r2_MRxNR(selector1, selector2);
    returnScratchReg(selector1);
    returnScratchReg(selector2);
store:
    store_MRxNR(fuseBetaWithStore);
}

void
avx512TransposeGenerator::unpack_ps_MRxNR()
{
    int i           = 0;
    int full_panels = MR_local / 2 * 2;
    int half_panels = MR_local % 2;
    for (; i < full_panels; i += 2) {
        unpackPsPair(prev_output[i], prev_output[i + 1], i);
    }
    if (half_panels) {
        unpackPs(prev_output[i], i);
    }
}

static void
calculate_unpack_pd_idx(
    int idx, int* src1, int* src2, int* dst1, int* dst2, int even_pairs)
{
    *src1 = idx;
    *src2 = idx + 1;
    *dst1 = (idx / 2);
    *dst2 = even_pairs + (idx / 2);
}

void
avx512TransposeGenerator::unpack_pd_MRxNR()
{
    int src1, src2, dst1, dst2;
    int floor_4    = (numRegsAfterUnpackps / 4) * 4;
    int mod_4      = numRegsAfterUnpackps % 4;
    int ceil_4     = (numRegsAfterUnpackps + 4 - 1) / 4 * 4;
    int even_pairs = ceil_4 / 2;
    int i          = 0;
    for (; i < floor_4; i += 4) {
        calculate_unpack_pd_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        unpackPdPair(prev_output[src1], prev_output[src1 + 2], dst1, dst1 + 1);
        unpackPdPair(prev_output[src2], prev_output[src2 + 2], dst2, dst2 + 1);
    }
    calculate_unpack_pd_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
    if (mod_4 == 2) {
        unpackPd(prev_output[src1], dst1, dst1 + 1);
        unpackPd(prev_output[src2], dst2, dst2 + 1);
    }
}

void
calculate_permute_r1_idx(
    int idx, int* src1, int* src2, int* dst1, int* dst2, int even_pairs)
{
    *src1 = idx;
    *src2 = idx + 1;
    *dst1 = (idx / 2);
    *dst2 = even_pairs + (idx / 2);
}

void
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

void
avx512TransposeGenerator::permute_r1_MRxNR(int selector1, int selector2)
{
    int src1, src2, dst1, dst2;
    int half_registers = numRegsAfterUnpackpd / 2;
    int full_pairs     = ((half_registers) / 4) * 4;
    int half_pairs     = (half_registers) % 4;
    int ceil_8         = (numRegsAfterUnpackpd + 8 - 1) / 8 * 8;
    int even_pairs     = ceil_8 / 2;
    int i              = 0;
    for (i = 0; i < full_pairs; i += 4) {
        /* operate on first half of the registers in prev_output */
        calculate_permute_r1_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        permutePdR1Pair(prev_output[src1], prev_output[src1 + 2], selector1,
                        selector2, dst1, dst1 + 1);
        permutePdR1Pair(prev_output[src2], prev_output[src2 + 2], selector1,
                        selector2, dst2, dst2 + 1);
        /* operate on second half of the registers in prev_output */
        calculate_permute_r1_idx2(i, &src1, &src2, &dst1, &dst2, even_pairs,
                                  half_registers);
        permutePdR1Pair(prev_output[src1], prev_output[src1 + 2], selector1,
                        selector2, dst1, dst1 + 1);
        permutePdR1Pair(prev_output[src2], prev_output[src2 + 2], selector1,
                        selector2, dst2, dst2 + 1);
    }
    if (half_pairs) {
        /* operate on first half of the registers in prev_output */
        calculate_permute_r1_idx(i, &src1, &src2, &dst1, &dst2, even_pairs);
        permutePdR1(prev_output[src1], selector1, selector2, dst1, dst1 + 1);
        permutePdR1(prev_output[src2], selector1, selector2, dst2, dst2 + 1);
        /* operate on second half of the registers in prev_output */
        calculate_permute_r1_idx2(i, &src1, &src2, &dst1, &dst2, even_pairs,
                                  half_registers);
        permutePdR1(prev_output[src1], selector1, selector2, dst1, dst1 + 1);
        permutePdR1(prev_output[src2], selector1, selector2, dst2, dst2 + 1);
    }
}

void
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

void
avx512TransposeGenerator::permute_r2_MRxNR(int selector1, int selector2)
{
    int reg_indices[16] = {
        0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15
    };
    int src1, src2;
    for (int i = 0; i < numElemsPerReg; i += 2) {
        calculate_r2_idx(i, &src1, &src2);
        if (reg_indices[i] < NR_local && reg_indices[i + 1] < NR_local) {
            permuteR2Pair(prev_output[src1], prev_output[src2], selector1,
                          selector2, i, i + 1);
        } else if (reg_indices[i] < NR_local) {
            permuteR2(prev_output[src1], prev_output[src2], selector1, i);
        }
    }
}

// Generic helper template for variable column stores
// This abstracts the common pattern: loop through columns, check count, perform
// operation
template<typename StoreOp>
void
avx512TransposeGenerator::storeColumns_variableCount(const int* reg_idx,
                                                     const int* half_idx,
                                                     int        numCols,
                                                     StoreOp    storeOp)
{
    if (!variableStores) {
        // Fixed number of columns - unroll completely
        for (int i = 0; i < numCols; i++) {
            storeOp(i, reg_idx[i], half_idx[i]);
            jit_->add(regTmp1, regCsC);
        }
        return;
    }

    // Variable number of columns - unroll with conditional exits
    Xbyak::Label done;
    jit_->mov(regTmp2, 0); // Initialize counter

    for (int i = 0; i < numCols; i++) {
        // Check if we've stored enough columns
        jit_->cmp(regTmp2, regNleftLocal);
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
void
avx512TransposeGenerator::store_4xNR(bool applyBetaScale)
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
            jit_->vextractf32x4(Xmm(scratch), Zmm(curr_output[reg_idx]),
                                quarter_idx);
            jit_->vfmadd231ps(Xmm(scratch), Xmm(betaRegIdx), Xmm(cRegIdx));
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Xmm(scratch));
            returnScratchReg(scratch);

            returnScratchReg(cRegIdx);
        };
        storeColumns_variableCount(reg_idx, quarter_idx, 16, betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int quarter_idx) {
            if (quarter_idx == 0) {
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                              Zmm(curr_output[reg_idx]));
            } else {
                int scratch = getScratchReg();
                jit_->vextractf32x4(Xmm(scratch), Zmm(curr_output[reg_idx]),
                                    quarter_idx);
                jit_->vmovups(jit_->ptr[regTmp1] | mrMask, Xmm(scratch));
                returnScratchReg(scratch);
            }
        };
        storeColumns_variableCount(reg_idx, quarter_idx, 16, betaZeroOp);
    }
}

void
avx512TransposeGenerator::store_8xNR(bool applyBetaScale)
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
        storeColumns_variableCount(reg_idx, half_idx, 16, betaScaleOp);
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
        storeColumns_variableCount(reg_idx, half_idx, 16, betaZeroOp);
    }
}

void
avx512TransposeGenerator::store_16xNR(bool applyBetaScale)
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
        storeColumns_variableCount(reg_indices, dummy_idx, 16, betaScaleOp);
    } else {
        auto betaZeroOp = [this](int col_idx, int reg_idx, int unused) {
            jit_->vmovups(jit_->ptr[regTmp1] | mrMask,
                          Zmm(curr_output[reg_idx]));
        };
        storeColumns_variableCount(reg_indices, dummy_idx, 16, betaZeroOp);
    }
}

void
avx512TransposeGenerator::store_MRxNR(bool fuseBetaWithStore)
{
    jit_->inLocalLabel();
    jit_->mov(regTmpHalf, 0xFFFF >> (numElemsPerReg - MR_local));
    jit_->kmovw(mrMask, regTmpHalf);

    jit_->mov(regTmp1, regCjr);

    if (fuseBetaWithStore) {
        // add run-time check to check if beta is 0
        int scratch = getScratchReg();
        jit_->vxorps(Xmm(scratch), Xmm(scratch), Xmm(scratch));
        jit_->vucomiss(Xmm(betaRegIdx), Xmm(scratch));
        returnScratchReg(scratch);
        jit_->je(".SBETAZERO", jit_->T_NEAR);

        // beta is non-zero, fuse with store
        if (MR_local <= 4) {
            store_4xNR(true); // applyBetaScale = true
        } else if (MR_local <= 8) {
            store_8xNR(true); // applyBetaScale = true
        } else {
            store_16xNR(true); // applyBetaScale = true
        }

        jit_->jmp(".AfterStoreColMajorMR", jit_->T_NEAR);
    }

    jit_->L(".SBETAZERO");
    if (MR_local <= 4) {
        store_4xNR(false); // applyBetaScale = false
    } else if (MR_local <= 8) {
        store_8xNR(false); // applyBetaScale = false
    } else {
        store_16xNR(false); // applyBetaScale = false
    }

    jit_->L(".AfterStoreColMajorMR");
    jit_->outLocalLabel();

    // return all the accumulated registers
    for (int i = 0; i < numElemsPerReg; i++) {
        returnScratchReg(curr_output[i]);
    }
}

void
avx512TransposeGenerator::embedSelectors()
{
    jit_->jmp(".selectorsEnd", jit_->T_NEAR);
    jit_->align(64);
    jit_->L(selectors);
    jit_->db(reinterpret_cast<uint8_t*>(&selector1), sizeof(selector1));
    jit_->db(reinterpret_cast<uint8_t*>(&selector2), sizeof(selector2));
    jit_->db(reinterpret_cast<uint8_t*>(&selector1_1), sizeof(selector1_1));
    jit_->db(reinterpret_cast<uint8_t*>(&selector2_1), sizeof(selector2_1));
    jit_->L(".selectorsEnd");
}

} // namespace amdzen::x86gen
