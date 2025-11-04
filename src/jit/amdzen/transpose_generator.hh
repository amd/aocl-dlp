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

#include "jit_generator_utils.hh"
#include "traits.hh"
#include <queue>

namespace amdzen::x86gen {

template<utils::kernelInstrType KType>
class TransposeGenerator
{

    using Traits                    = amdzen::traits::ArchitectureTraits<KType>;
    using RegType                   = typename Traits::RegType;
    using halfRegType               = typename Traits::halfRegType;
    int              numRegs        = Traits::numRegs;
    int              RegBytes       = Traits::regBytes;
    static const int numElemsPerReg = Traits::regBytes / sizeof(float);

  public:
    TransposeGenerator(Xbyak::CodeGenerator* jit,
                       int                   MR,
                       int                   NR,
                       bool                  useMask,
                       bool                  mLoop,
                       int                   numMaskRegs,
                       int                   cRegStartIdx,
                       int                   cRegCount,
                       Xbyak::Reg64&         regCPtr);
    ~TransposeGenerator() = default;
    dlp::jit::jitGeneratorError generateTranspose(
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType          = dlp::jit::jitAlgoType::gemm,
        bool                  fuseBetaWithStore = false);

  private:
    void                  printOutput(int* output);
    Xbyak::CodeGenerator* jit_; // Back reference to access registers and state

    const Xbyak::Reg64 &regCPtr, regCjr, regCsC, regRsCBlock, regCsCBlock,
        regNleft, regTmp1, regTmp2, regTmp3, regTmp4;
    const Xbyak::Reg32 &regTmpHalf, regNleftLocal;

    Xbyak::Opmask fringeMask[dlp::kernels::maxNumMasks];
    Xbyak::Opmask mrMask;

    int numScratchRegs;

    int betaRegIdx;

    // these are the dimensions of the entire matrix that is being transposed
    int MR, NR, useMask;
    // These will be the dimensions of the block that is being transposed
    int MR_local, NR_local;
    int numMaskRegs;
    // this bool indicates that an lt kernel is being generated
    bool variableStores;
    // maskRegIdx is used to load mask for avx2 code.
    int cRegStartIdx, cRegCount, maskRegIdx;

    int numFullNRBlocks, numMaskNRBlocks, numNRBlocks;
    int numFullMRBlocks, numMaskMRBlocks, numMRBlocks;

    int numRegsAfterUnpackps;
    int numRegsAfterUnpackpd;
    int numRegsAfterPermuteR1;
    int numRegsAfterPermuteR2;

    // arrays to store indices of previous and current output
    int arr1[numElemsPerReg];
    int arr2[numElemsPerReg];

    // selector values for both permutes
    int64_t selector1[8]   = { 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB };
    int64_t selector2[8]   = { 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF };
    int64_t selector1_1[8] = { 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB };
    int64_t selector2_1[8] = { 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF };

    const md_t selector1_off   = 0;
    const md_t selector2_off   = selector1_off + sizeof(selector1);
    const md_t selector1_1_off = selector2_off + sizeof(selector2);
    const md_t selector2_1_off = selector1_1_off + sizeof(selector1_1);

    Xbyak::Address get_selector(md_t selector_off)
    {
        return jit_->ptr[jit_->rip + selectors + selector_off];
    }

    // pointers to previous and current output
    int* prev_output = arr1;
    int* curr_output = arr2;

    std::queue<int> scratch_reg_queue;
    Xbyak::Label    selectors;

    int popAndGetScratchReg()
    {
        int reg = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        return reg;
    }

    void generateNleftLocal(int j);
    void createMaskFromConstant(int value);

    int calculateScratchReq();

    void store_reg_in_stack(int num_regs, int nextBlockI, int nextBlockJ);
    void get_reg_from_stack(int num_regs, int nextBlockI, int nextBlockJ);

    void remove_from_scratch_reg_queue(int reg);

    void setInitialIndices(int row_idx, int col_idx);
    void generateTransposeBlockMRxNR(bool fuseBetaWithStore);
    void unpack_ps_MRxNR();
    void unpack_pd_MRxNR();
    void permute_r1_MRxNR(int selector1, int selector2);
    void permute_r2_MRxNR(int selector1, int selector2);
    // Unified store functions
    void store_4xNR(bool applyBetaScale);
    void store_8xNR(bool applyBetaScale);
    void store_16xNR(bool applyBetaScale);

    void store_MRxNR(bool fuseBetaWithStore);

    // Generic helper for storing with variable columns
    // StoreOp: lambda(col_idx, reg_idx, half_idx) that performs the actual
    // store
    template<typename StoreOp>
    void storeColumns_variableCount(const int* reg_idx,
                                    const int* half_idx,
                                    int        numCols,
                                    StoreOp    storeOp);
    void embedSelectors();
    void setContext(const Xbyak::Reg64& postOpsArgWrapperPtrReg,
                    bool                fuseBetaWithStore);

    // Helper functions (converted from macros)
    int  getScratchReg();
    void returnScratchReg(int reg);
    void swapOutputs();
    void unpackPsPair(int s1, int s2, int i);
    void unpackPs(int s1, int i);
    void unpackPdPair(int s1, int s2, int op_idx1, int op_idx2);
    void unpackPd(int s1, int op_idx1, int op_idx2);
    void permutePdR1Pair(int s1, int s2, int sel1, int sel2, int d1, int d2);
    void permutePdR1(int s1, int sel1, int sel2, int d1, int d2);
    void permuteR2Pair(int s1, int s2, int sel1, int sel2, int d1, int d2);
    void permuteR2(int s1, int s2, int sel1, int d1);
};

} // namespace amdzen::x86gen
