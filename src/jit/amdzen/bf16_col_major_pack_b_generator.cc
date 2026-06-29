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

#include "bf16_col_major_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

// ──────────────────────────────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
jitPackBBF16ColMajor<KType>::jitPackBBF16ColMajor()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numSubBlocks_(0)
    , useMask_(false)
    , nLoop_(true)
{
}

// ──────────────────────────────────────────────────────────────────────
// Register budget validation
//
// The transpose consumes 16 input ZMMs (the loaded columns) and 16 scratch
// ZMMs, independent of NR. AVX-512 provides exactly 32 ZMM, so the requirement
// is checked explicitly here.
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBBF16ColMajor<KType>::allocateReg()
{
    // BF16 pack-B is AVX-512 (ZMM) only and packs in 16-wide blocks.
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    if (NR_ <= 0 || (NR_ % BLOCK_N) != 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    // 16 input columns + 16 transpose scratch registers.
    if (numRegs < kNumVecRegs) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::initializeStackFrame(Xbyak::util::StackFrame& sf)
{
    pParams     = sf.p[0];
    regK        = sf.t[0];
    regKFull    = sf.t[1];
    regLdbBytes = sf.t[2];
    regSrcEnd   = sf.t[3];
    regSrcPanel = sf.t[4];
    regDstPanel = sf.t[5];
    regSrcSb    = sf.t[6];
    regDstSb    = sf.t[7];
    regKr       = sf.t[8];
    regSbCount  = sf.t[9];
    regColPtr   = sf.t[10];
    regTmp      = sf.t[11];
}

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::initializeParameters()
{
    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);

    // Column-to-column source stride (bytes): one n-step is cs_src elements.
    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, cs_src)]);
    shl(regLdbBytes, 1); // * sizeof(bf16)

    // Loop scaffolding is only needed by the main full-NR (looping) kernel.
    // The packed n-panel stride is recomputed inline in generateFullBlockLoop
    // (it is only used at panel-advance time), so no register is reserved for
    // it here.
    if (nLoop_) {
        // End of the full-NR source region (exclusive), in bytes:
        // src + n_full_pieces_limit columns * column-stride.
        mov(regSrcEnd,
            ptr[pParams
                + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
        imul(regSrcEnd, regLdbBytes);
        add(regSrcEnd, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    }

    // K rounded down to a whole number of 32-K tiles. The remainder (K % 32,
    // including any odd-K element) is handled by the masked K-fringe tile.
    mov(regKFull, regK);
    and_(regKFull, ~(numElemsPerReg - 1));
}

// ──────────────────────────────────────────────────────────────────────
// Top-level entry point
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBBF16ColMajor<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_           = params.NR;
    useMask_      = params.useMask;
    nLoop_        = params.nLoop;
    numSubBlocks_ = static_cast<int>(NR_ / BLOCK_N);

    RETURN_IF_ERROR(allocateReg());

    Xbyak::util::StackFrame stackFrame(
        this, 1, 12 | Xbyak::util::UseRBPAsFramePointer, 0);
    initializeStackFrame(stackFrame);
    initializeParameters();

    if (useMask_)
        generateLtBlockLoop();
    else
        generateFullBlockLoop();

    // Packed strides for this kernel width (the orchestrator overwrites these
    // with the full-NR values after the whole cascade; kept for parity with the
    // row-major generator and standalone use).
    mov(regTmp, static_cast<int>(NR_ * K_FACTOR));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regTmp);
    mov(regTmp, static_cast<int>(NR_ / K_FACTOR));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regTmp);

    vzeroupper();
    return dlp::jit::jitGeneratorError::success;
}

// ──────────────────────────────────────────────────────────────────────
// Full kernel: (optional N-panel loop) -> K-loop (outer) -> sub-block (inner)
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::generateFullBlockLoop()
{
    // Single-panel fringe kernel: pack one NR-wide panel from src/dst directly.
    if (!nLoop_) {
        mov(regSrcPanel,
            ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
        mov(regDstPanel,
            ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
        emitKLoop(/*partialNCols=*/false);
        return;
    }

    // Main kernel: iterate every full NR panel in [src, srcEnd).
    Xbyak::Label l_jc_done, l_jc_loop;

    mov(regSrcPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);

    cmp(regSrcPanel, regSrcEnd);
    jge(l_jc_done, T_NEAR);

    L(l_jc_loop);

    emitKLoop(/*partialNCols=*/false);

    // Advance to the next n-panel: NR columns of source.
    mov(regTmp, regLdbBytes);
    imul(regTmp, regTmp, static_cast<int>(NR_));
    add(regSrcPanel, regTmp);

    // Advance the packed base by one panel = KC_updated * NR * sizeof(bf16),
    // where KC_updated rounds K up to an even number of K-rows. This equals
    // (#K-pairs) * (NR * 2 bf16) — the same panel size as the row-major packer.
    // Recomputed here (cheaply) so no dedicated stride register is needed.
    mov(regTmp, regK);
    add(regTmp, 1);
    and_(regTmp, -2); // round up to even
    imul(regTmp, regTmp, static_cast<int>(NR_ * ELEM_BYTES));
    add(regDstPanel, regTmp);

    cmp(regSrcPanel, regSrcEnd);
    jb(l_jc_loop, T_NEAR);

    L(l_jc_done);
}

// ──────────────────────────────────────────────────────────────────────
// Lt kernel: a single partial sub-block of n_partial (< 16) columns.
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::generateLtBlockLoop()
{
    mov(regSrcPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    mov(regSbCount,
        ptr[pParams + offsetof(dlp::kernels::packBParams, n_partial)]);

    emitKLoop(/*partialNCols=*/true);
}

// ──────────────────────────────────────────────────────────────────────
// K-loop (outer): full 32-K tiles + the masked K-fringe tile. Each tile
// sweeps the panel's sub-blocks (inner), so the packed buffer is written in
// contiguous K-pair-major order — the same write stream as the intrinsic and
// F32 col-major packers.
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::emitKLoop(bool partialNCols)
{
    Xbyak::Label l_kfull_done, l_kfull_loop, l_kfringe_done;

    xor_(regKr, regKr);

    test(regKFull, regKFull);
    jz(l_kfull_done, T_NEAR);

    L(l_kfull_loop);

    emitTileSweep(/*kMasked=*/false, partialNCols);

    add(regKr, numElemsPerReg);
    cmp(regKr, regKFull);
    jb(l_kfull_loop, T_NEAR);

    L(l_kfull_done);

    // K-fringe: trailing K % 32 elements (covers odd K via the zero-mask).
    mov(regTmp, regK);
    and_(regTmp, numElemsPerReg - 1);
    test(regTmp, regTmp);
    jz(l_kfringe_done, T_NEAR);

    // Load the runtime epi16 K-fringe mask (up to 32 lanes -> kmovd).
    mov(regTmp.cvt32(),
        ptr[pParams + offsetof(dlp::kernels::packBParams, k_fringe_mask)]);
    kmovd(Xbyak::Opmask(MASK_KFRINGE), regTmp.cvt32());

    mov(regKr, regKFull);

    emitTileSweep(/*kMasked=*/true, partialNCols);

    L(l_kfringe_done);
}

// ──────────────────────────────────────────────────────────────────────
// One 32-K tile: set the per-tile bases from regKr, then sweep the sub-block(s)
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::emitTileSweep(bool kMasked, bool partialNCols)
{
    // Source base for column 0 of sub-block 0 at this K tile (regKr real-K).
    lea(regSrcSb, ptr[regSrcPanel + regKr * ELEM_BYTES]);

    // Packed base for K-pair (regKr/2) of sub-block 0: (regKr/2) * rs_dst_bytes
    // from the panel start. rs_dst_bytes = NR * 2 bf16.
    const int nrBytes = static_cast<int>(NR_ * K_FACTOR * ELEM_BYTES);
    mov(regDstSb, regKr);
    shr(regDstSb, 1);
    imul(regDstSb, regDstSb, nrBytes);
    add(regDstSb, regDstPanel);

    // Lt-block kernel: a single partial sub-block (regSbCount columns).
    if (partialNCols) {
        emitLoadCols(kMasked, /*partialNCols=*/true);
        transpose::emitTranspose16x16(*this);
        emitStoreRows(kMasked);
        return;
    }

    // Full / fringe-width kernel: sweep the NR/16 sub-blocks for this K tile.
    Xbyak::Label l_sb_loop;
    mov(regSbCount, numSubBlocks_);

    L(l_sb_loop);

    emitLoadCols(kMasked, /*partialNCols=*/false);
    transpose::emitTranspose16x16(*this);
    emitStoreRows(kMasked);

    // Advance to the next 16-column sub-block within this K tile.
    mov(regTmp, regLdbBytes);
    imul(regTmp, regTmp, BLOCK_N); // 16 columns of source
    add(regSrcSb, regTmp);
    add(regDstSb,
        static_cast<int>(BLOCK_N * K_FACTOR * ELEM_BYTES)); // 16 K-pairs packed

    dec(regSbCount);
    jnz(l_sb_loop, T_NEAR);
}

// ──────────────────────────────────────────────────────────────────────
// Load BLOCK_N columns into Zmm(0..15)
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::emitLoadCols(bool kMasked, bool partialNCols)
{
    using Xbyak::Zmm;

    // regSrcSb already points at column 0 of this sub-block at the current K
    // tile; regColPtr walks the columns from there.
    mov(regColPtr, regSrcSb);

    auto loadOne = [&](int i) {
        if (kMasked)
            vmovdqu16(Zmm(i) | Xbyak::Opmask(MASK_KFRINGE) | T_z,
                      ptr[regColPtr]);
        else
            vmovdqu64(Zmm(i), ptr[regColPtr]);
    };

    if (partialNCols) {
        // Zero every column first; only the live n_partial columns are loaded,
        // so the transposed rows are zero-padded beyond n_partial.
        for (int i = 0; i < BLOCK_N; ++i)
            vpxorq(Zmm(i), Zmm(i), Zmm(i));

        for (int i = 0; i < BLOCK_N; ++i) {
            Xbyak::Label l_skip;
            cmp(regSbCount, i + 1);
            jl(l_skip, T_NEAR);

            loadOne(i);

            L(l_skip);
            if (i < BLOCK_N - 1)
                add(regColPtr, regLdbBytes);
        }
    } else {
        for (int i = 0; i < BLOCK_N; ++i) {
            loadOne(i);
            if (i < BLOCK_N - 1)
                add(regColPtr, regLdbBytes);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Store the transposed K-pair rows
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBBF16ColMajor<KType>::emitStoreRows(bool kMasked)
{
    using Xbyak::Zmm;

    // rs_dst in bytes = NR * 2 bf16 (K-pair stride). regDstSb already points at
    // K-pair (regKr/2) of this sub-block, so transposed row r lands at
    // regDstSb + r * rs_dst_bytes.
    const int nrBytes = static_cast<int>(NR_ * K_FACTOR * ELEM_BYTES);

    if (!kMasked) {
        for (int r = 0; r < BLOCK_N; ++r) {
            const int srcReg = transpose::storeMap16[r];
            vmovdqu64(ptr[regDstSb + r * nrBytes], Zmm(srcReg));
        }
        return;
    }

    // K-fringe: store ceil(k_tail / 2) K-pair rows, k_tail = K % 32.
    mov(regTmp, regK);
    and_(regTmp, numElemsPerReg - 1);
    add(regTmp, 1);
    shr(regTmp, 1); // ceil(k_tail / 2)

    for (int r = 0; r < BLOCK_N; ++r) {
        Xbyak::Label l_skip;
        cmp(regTmp, r + 1);
        jb(l_skip, T_NEAR);

        const int srcReg = transpose::storeMap16[r];
        vmovdqu64(ptr[regDstSb + r * nrBytes], Zmm(srcReg));

        L(l_skip);
    }
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBBF16ColMajor<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
