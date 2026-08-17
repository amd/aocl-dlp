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

#include "aocl_dlp_config.h"

#include "s8_gemv_quant_generator.hh"

#include "classic/aocl_gemm_metadata.h" // DLP_F32 / DLP_BF16
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"

#include <memory>

namespace amdzen::gen {

using namespace Xbyak;

// ---------------------------------------------------------------------------
// s8s8s32 group-quantized GEMV, N == 1
// ---------------------------------------------------------------------------

// Byte offsets into gemvN1Params and its two nested attribute structs.
#define N1_OFF(field) offsetof(dlp::kernels::gemvN1Params, field)
#define N1_OPS_OFF(field)                                                      \
    (offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)                       \
     + offsetof(dlp_gemm_post_op_attr, field))
#define N1_GRP_OFF(field)                                                      \
    (offsetof(dlp::kernels::gemvN1Params, grpKernelOpsAttr)                    \
     + offsetof(dlp_gemm_grp_post_op_attr, field))

// Stack scratch used to bounce a strided C column through memory, so the
// gather/scatter is a plain scalar copy loop rather than a lane-insert chain.
static constexpr int N1_SCRATCH_BYTES = 64;

template<utils::kernelInstrType KType>
jitGEMVQuantN1<KType>::jitGEMVQuantN1(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantN1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr   = frame.p[0];
    regAptr    = frame.t[0];
    regTmpAptr = frame.t[1];
    regXptr    = frame.t[2];
    regYptr    = frame.t[3];
    regTmpYptr = frame.t[4];
    regRsA     = frame.t[5];
    regGIter   = frame.t[6];
    regRsC     = frame.t[7];
    regMIter   = frame.t[8];
    regKIter   = frame.t[9];
    regTmp1    = frame.t[10];
    regTmp2    = frame.t[11];
    regTmp3    = frame.t[12];
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::allocateRegisters()
{
    // Output bands after the horizontal reduction:
    // one register per vnniWidth rows.
    const int nBands = (MR + vnniWidth - 1) / vnniWidth;

    // group accumulator registers
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg;

    tmpReg     = 4; // reduceToXmm needs exactly four scratch registers
    tmpBaseIdx = 0;

    xReg     = 1;
    xBaseIdx = tmpBaseIdx + tmpReg;

    // Only the symmetric path biases A by +128, so only it owns this register.
    // The index is still computed either way; with a zero count the next bank
    // simply starts on top of it and reclaims the slot.
    vec128Reg = useVec128 ? 1 : 0;
    vec128Idx = xBaseIdx + xReg;

    // F32 accumulator registers
    fAccReg     = nBands;
    fAccBaseIdx = vec128Idx + vec128Reg;

    aSclReg     = nBands;
    aSclBaseIdx = fAccBaseIdx + fAccReg;

    bSclReg     = 1;
    bSclBaseIdx = aSclBaseIdx + aSclReg;

    yReg     = nBands;
    yBaseIdx = bSclBaseIdx + bSclReg;

    if ((yBaseIdx + yReg) > accumBaseIdx) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantN1<KType>::initializeParameters(
    utils::quantGemvN1GeneratorParams& params)
{
    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    MR               = params.base.MR;
    M_LEFT           = params.base.M_LEFT;
    yFormat          = params.base.yFormat;
    alphaScalingType = params.base.alphaScalingType;
    betaScalingType  = params.base.betaScalingType;
    c_downscale      = params.base.c_downscale;

    // Accumulation is int32, so a band is RegBytes/4 rows wide.
    vnniWidth = RegBytes / sizeof(int32_t);

    // Load strides from the stack
    mov(regRsA, ptr[stackPtr + N1_OFF(rsA)]);
    mov(regRsC, ptr[stackPtr + N1_OFF(rsC)]);

    // rs_c is in elements. The accumulator/output rail this kernel writes when
    // c_downscale == DLP_F32 is f32, so scale to bytes here; the DLP_BF16 rail
    // uses rs_c_downscale instead and scales it in updateCBufferPointers.
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    // Absolute output row origin, used to index the per-token A scales. The
    // m-loop advances this by MR per iteration.
    mov(regTmp2, ptr[stackPtr + N1_GRP_OFF(grp_post_op_i)]);
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantN1<KType>::regInit(int baseIdx, int count)
{
    // Zero out "count" number of registers starting from the baseIdx.
    for (iter_t i = 0; i < count; ++i) {
        vpxord(RegType(baseIdx + i), RegType(baseIdx + i),
               RegType(baseIdx + i));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::loadXValues(bool isFringe)
{
    if (isFringe) {
        // kmask_i8_avx512 (k1) is a byte-granular K-remainder mask, so it must
        // be consumed with a byte-granular load.
        vmovdqu8(RegType(xBaseIdx) | k1 | T_z, ptr[regXptr]);
    } else {
        vmovdqu32(RegType(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::loadAValues(int aRegIdx, bool isFringe)
{
    if (isFringe) {
        vmovdqu8(RegType(tmpBaseIdx + aRegIdx) | k1 | T_z,
                 ptr[regTmpAptr + regTmp1]);
        if (useVec128) {
            vpaddb(RegType(tmpBaseIdx + aRegIdx) | k1 | T_z,
                   RegType(tmpBaseIdx + aRegIdx), RegType(vec128Idx));
        }
    } else {
        vmovdqu32(RegType(tmpBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp1]);
        // tmp := tmp + 128
        if (useVec128) {
            vpaddb(RegType(tmpBaseIdx + aRegIdx), RegType(tmpBaseIdx + aRegIdx),
                   RegType(vec128Idx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::computeVNNI(int aRegIdx, int accumRegIdx)
{
    vpdpbusd(RegType(accumBaseIdx + accumRegIdx), RegType(tmpBaseIdx + aRegIdx),
             RegType(xBaseIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    int mIter = mSize / 4;
    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);

    for (iter_t i = 0; i < mIter; ++i) {
        for (iter_t j = 0; j < 4; ++j) {
            // Load this row's A values for the current K-sub-block (biasing
            // by +128 for VNNI's unsigned-A requirement if useVec128), then
            // vpdpbusd them against the shared X vector into that row's own
            // int32 accumulator (accumBaseIdx + row index).
            RETURN_IF_ERROR(loadAValues(j, isFringe));
            RETURN_IF_ERROR(computeVNNI(j, ((i * 4) + j)));

            add(regTmp1, regRsA);
        }
    }

    // Handle remaining mLeft rows.
    for (iter_t j = 0; j < mLeft; ++j) {
        RETURN_IF_ERROR(loadAValues(j, isFringe));
        RETURN_IF_ERROR(computeVNNI(j, ((mIter * 4) + j)));

        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

// Horizontally reduce up to 4 rows' 16-lane int32 accumulators
// (accumBaseIdx+startIdx..+blockSize-1) to one scalar per row, packed into
// the low `blockSize` dwords of Xmm(tmpIdx).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::reduceToXmm(int startIdx, int tmpIdx, int blockSize)
{
    // Only handles up to 4 rows per call hence reporting badKernelInfo for
    // blocksizes over 4.
    if (blockSize > 4) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Zero unused rows so a <4-row fringe doesn't add garbage below.
    for (iter_t i = blockSize; i < 4; ++i) {
        vxorps(RegType(tmpIdx + i), RegType(tmpIdx + i), RegType(tmpIdx + i));
    }

    // Fold each row's upper 256 bits into its lower half: 16 lanes -> 8.
    for (iter_t i = 0; i < blockSize; ++i) {
        vextracti32x8(Xbyak::Ymm(tmpIdx + i), RegType(startIdx + i), 1);
        vpaddd(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(startIdx + i));
    }

    // Pairwise horizontal-add across the 4 rows' 8-lane sums.
    vphaddd(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx + 1));
    vphaddd(Xbyak::Ymm(tmpIdx + 2), Xbyak::Ymm(tmpIdx + 2),
            Xbyak::Ymm(tmpIdx + 3));
    vphaddd(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx + 2));

    // Fold the last 256-bit halves: 4 row totals now sit in Xmm(tmpIdx).
    vextracti128(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx), 1);
    vpaddd(Xbyak::Xmm(tmpIdx), Xbyak::Xmm(tmpIdx + 1), Xbyak::Xmm(tmpIdx));

    return dlp::jit::jitGeneratorError::success;
}

// Horizontally reduces each row's register to a single scalar via reduceToXmm,
// then pack those 4 scalars into one lane-group so the result lands back in
// accumBaseIdx.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::reduceAccumulation(int mSize)
{
    for (iter_t i = 0; i < mSize; i += vnniWidth) {
        int blockSize = ((mSize - i) < vnniWidth) ? (mSize - i) : vnniWidth;

        for (iter_t j = 0; j < blockSize; j += 4) {
            int subBlockSize = ((blockSize - j) < 4) ? (blockSize - j) : 4;

            // Horizontally reduce rows [i+j, i+j+subBlockSize) from their
            // 16-lane partial sums down to one scalar each, landing in the
            // low subBlockSize dwords of tmpBaseIdx's xmm.
            RETURN_IF_ERROR(
                reduceToXmm((accumBaseIdx + i + j), tmpBaseIdx, subBlockSize));

            // Insert the resulting XMM into appropriate index in destination
            // ZMM
            vinserti32x4(RegType(accumBaseIdx + i / vnniWidth),
                         RegType(accumBaseIdx + i / vnniWidth),
                         Xbyak::Xmm(tmpBaseIdx), j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// dst = group_start + current relative group index.
template<utils::kernelInstrType KType>
void
jitGEMVQuantN1<KType>::absoluteGroupIndex(const Xbyak::Reg64& dst)
{
    mov(dst, ptr[stackPtr + N1_OFF(group_start)]);
    add(dst, regGIter);
}

// Subtract this group's B column sum from the reduced int32 row sums. VNNI
// consumes A as unsigned, so A was biased by +128 and the bias contribution
// (128 * sum(B) over the group) has to come back out. With n == 1 the sum is a
// single int32 per group, so one broadcast covers every row.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::conversionCompensationGroup(int mSize)
{
    // The bias and its compensation are a matched pair. If A was never biased
    // by +128 there is no 128*sum(B) term to remove, hence skipping.
    if (!useVec128) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int nBands = (mSize + vnniWidth - 1) / vnniWidth;

    // Load b_col_sum_vec address.
    mov(regTmp1, ptr[stackPtr + N1_OPS_OFF(b_col_sum_vec)]);
    absoluteGroupIndex(regTmp3); // gAbs = group_start + regGIter

    // Broadcast the single scalar b_col_sum_vec[gAbs] to every lane
    vpbroadcastd(RegType(xBaseIdx), ptr[regTmp1 + regTmp3 * sizeof(int32_t)]);

    // Undo the +128 bias so accumBaseIdx again holds the true (unbiased) int32
    // row sums for this group.
    for (iter_t b = 0; b < nBands; ++b) {
        vpsubd(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b),
               RegType(xBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Broadcast this group's B scale. With n == 1 there is a single output column,
// so the B scale is always a scalar; grp_post_op_j is 0 on this path.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::loadBScaleBroadcast()
{
    // Load address of b_scale_factor
    mov(regTmp1, ptr[stackPtr + N1_GRP_OFF(b_scale_factor)]);

    // Increment b_scale_factor address by group ldb if scales are per-group
    // (i.e., num_groups x 1)
    if (bPerGroupK) {
        absoluteGroupIndex(regTmp3);

        // b_scale_factor += grp_post_op_ldb * bScaleElemBytes() (4B for F32,
        // 2B for BF16)
        // regKIter is unused at this stage, hence temporarily loading it with
        // grp_post_op_ldb.
        mov(regKIter, ptr[stackPtr + N1_GRP_OFF(grp_post_op_ldb)]);
        imul(regTmp3, regKIter);
        lea(regTmp1, ptr[regTmp1 + regTmp3 * bScaleElemBytes()]);
    }

    // Broadcast B scale factor for supported types
    if (bScaleType == dlp::kernel_frame::DataType::f32) {
        vbroadcastss(RegType(bSclBaseIdx), ptr[regTmp1]);
    } else if (bScaleType == dlp::kernel_frame::DataType::bf16) {
        vpbroadcastw(RegType(bSclBaseIdx), ptr[regTmp1]);
        vpmovsxwd(RegType(bSclBaseIdx), Xbyak::Ymm(bSclBaseIdx));
        vpslld(RegType(bSclBaseIdx), RegType(bSclBaseIdx), 16);
    } else {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    return dlp::jit::jitGeneratorError::success;
}

// Build the per-row A scale vector for the current m tile.
//
// The address of row r's scale is
//   a_scale + ((grp_post_op_i + r) * lda + (aPerGroupK ? gAbs : 0)) * elemBytes
//
// When A collapses the K axis (PER_TOKEN) the frame sets lda to 1 and there is
// no group term, so the mSize scales are contiguous and group-invariant: one
// masked vector load, hoisted out of the group loop. That is also what makes
// PER_TOKEN correct for multi-group K, which the intrinsic kernel gets wrong.
// Otherwise lda is the group count and the rows are strided, so the values are
// bounced through the stack scratch a scalar at a time.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::loadAScaleVector(int mSize)
{
    const bool masked = ((mSize % vnniWidth) != 0);

    mov(regTmp1, ptr[stackPtr + N1_GRP_OFF(a_scale_factor)]);
    mov(regTmp3, ptr[stackPtr + N1_GRP_OFF(grp_post_op_lda)]);

    // Element offset of this tile's first row: row_base * lda (+ gAbs).
    mov(regKIter, regTmp2); // regTmp2 = grp_post_op_i
    imul(regKIter, regTmp3);
    if (aPerGroupK) {
        // gAbs = group_start + regGIter
        mov(regTmpYptr, ptr[stackPtr + N1_OFF(group_start)]);
        add(regTmpYptr, regGIter);
        add(regKIter, regTmpYptr);
    }
    lea(regTmp1, ptr[regTmp1 + regKIter * aScaleElemBytes()]);

    // A single row needs no gather. Going through the stack scratch put a
    // 4-byte store in front of a 64-byte load, which cannot store-forward, so
    // every group paid a forwarding stall.
    if (mSize == 1) {
        if (aScaleType == dlp::kernel_frame::DataType::f32) {
            vmovss(Xbyak::Xmm(aSclBaseIdx), ptr[regTmp1]);
        } else if (aScaleType == dlp::kernel_frame::DataType::bf16) {
            movzx(regKIter.cvt32(), word[regTmp1]);
            shl(regKIter.cvt32(), 16);
            vmovd(Xbyak::Xmm(aSclBaseIdx), regKIter.cvt32());
        } else {
            return dlp::jit::jitGeneratorError::notSupported;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    if (!aPerGroupK) {
        // lda == 1: contiguous run of mSize scales.
        if (aScaleType == dlp::kernel_frame::DataType::f32) {
            if (masked) {
                vmovups(RegType(aSclBaseIdx) | k2 | T_z, ptr[regTmp1]);
            } else {
                vmovups(RegType(aSclBaseIdx), ptr[regTmp1]);
            }
        } else if (aScaleType == dlp::kernel_frame::DataType::bf16) {
            if (masked) {
                vmovdqu16(Xbyak::Ymm(aSclBaseIdx) | k2 | T_z, ptr[regTmp1]);
            } else {
                vmovdqu16(Xbyak::Ymm(aSclBaseIdx), ptr[regTmp1]);
            }
            vpmovsxwd(RegType(aSclBaseIdx), Xbyak::Ymm(aSclBaseIdx));
            vpslld(RegType(aSclBaseIdx), RegType(aSclBaseIdx), 16);
        } else {
            return dlp::jit::jitGeneratorError::notSupported;
        }

        return dlp::jit::jitGeneratorError::success;
    }

    // Strided rows: scalar-copy into the stack scratch, widening bf16 to f32 on
    // the way in, then one vector load back.
    mov(regTmp3, ptr[stackPtr + N1_GRP_OFF(grp_post_op_lda)]);
    // regTmp3 now holds the per-row byte stride
    imul(regTmp3, regTmp3, aScaleElemBytes());

    for (iter_t r = 0; r < mSize; ++r) {
        if (aScaleType == dlp::kernel_frame::DataType::f32) {
            mov(regKIter.cvt32(), ptr[regTmp1]);
        } else if (aScaleType == dlp::kernel_frame::DataType::bf16) {
            movzx(regKIter.cvt32(), word[regTmp1]);
            shl(regKIter.cvt32(), 16);
        } else {
            return dlp::jit::jitGeneratorError::notSupported;
        }
        mov(ptr[rsp + r * static_cast<int>(sizeof(float))], regKIter.cvt32());
        add(regTmp1, regTmp3);
    }

    if (masked) {
        vmovups(RegType(aSclBaseIdx) | k2 | T_z, ptr[rsp]);
    } else {
        vmovups(RegType(aSclBaseIdx), ptr[rsp]);
    }

    return dlp::jit::jitGeneratorError::success;
}

// acc_f32 += cvt(int32 row sums) * b_scale[group] * a_scale[row, group]
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::dequantAccumToFAcc(int mSize)
{
    const int nBands = (mSize + vnniWidth - 1) / vnniWidth;

    for (iter_t b = 0; b < nBands; ++b) {
        // In-place int32 to f32 conversion
        vcvtdq2ps(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b));
        // accum[b] := accum[b] * b_sf
        vmulps(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b),
               RegType(bSclBaseIdx));
        // accum[b] := accum[b] * s_sf[r]
        vmulps(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b),
               RegType(aSclBaseIdx + b));
        // fAcc[r] := fAcc[r] + accum[b]
        vaddps(RegType(fAccBaseIdx + b), RegType(fAccBaseIdx + b),
               RegType(accumBaseIdx + b));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Walk the K groups for one m tile, leaving the F32 total in the accumulator
// registers.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::groupLoop(int mSize)
{
    // Live bands for this tile: ceil(mSize / vnniWidth), so an M fringe walks
    // fewer than MR/vnniWidth.
    const int nBands = (mSize + vnniWidth - 1) / vnniWidth;

    // Zero out `fAccReg` count of F32 accumulator registers starting from
    // fAccBaseIdx.
    regInit(fAccBaseIdx, fAccReg);

    if (alphaScalingType == dlp::kernel_frame::scalingType::zero) {
        // No A*x contribution
        for (iter_t b = 0; b < nBands; ++b) {
            vmovaps(RegType(accumBaseIdx + b), RegType(fAccBaseIdx + b));
        }
        return dlp::jit::jitGeneratorError::success;
    }

    // Scales that do not depend on the group index are loaded once per tile.
    // If bPerGroupK is true, scales are loaded post k-loop.
    if (!bPerGroupK) {
        RETURN_IF_ERROR(loadBScaleBroadcast());
    }
    if (!aPerGroupK) {
        RETURN_IF_ERROR(loadAScaleVector(mSize));
    }

    // The +128 bias VNNI needs for signed A is loop-invariant, so it is
    // materialised once per m tile and held live across every group and k
    // iteration.
    if (useVec128) {
        vxorps(RegType(vec128Idx), RegType(vec128Idx), RegType(vec128Idx));
        mov(regTmp3, 128);
        vpbroadcastb(RegType(vec128Idx), regTmp3.cvt8());
    }

    // A and x are walked contiguously across groups: group g starts exactly
    // where group g-1 ended, so the k-loop's own advance is all that is needed.
    xor_(regGIter, regGIter);
    mov(regTmpAptr, regAptr);
    mov(regXptr, ptr[stackPtr + N1_OFF(x)]);

    Xbyak::Label groupBegin, kLoop, kEnd, kFringeEnd;

    // Begin the group-loop over the k-loop.
    L(groupBegin);

    // Zero out `mSize` count of S32 accumulator registers starting from
    // accumBaseIdx.
    regInit(accumBaseIdx, mSize);

    // Full 64-element K blocks within this group.
    // regKIter is live now and should not be overriden.
    mov(regKIter, ptr[stackPtr + N1_OFF(k_iter)]);
    test(regKIter, regKIter);
    jz(kEnd, T_NEAR);

    // K block
    L(kLoop);
    RETURN_IF_ERROR(loadXValues(false));
    RETURN_IF_ERROR(processMRBlock(mSize, false));

    // Increment A and X for next iteration
    add(regTmpAptr, RegBytes);
    add(regXptr, RegBytes);

    sub(regKIter, 1);
    jnz(kLoop, T_NEAR);

    L(kEnd);

    // K-fringe block
    // Sub-64 K remainder of the group. Reachable whenever group_size is not a
    // multiple of 64 (group_size 16 and 32 take this path exclusively).
    mov(regKIter, ptr[stackPtr + N1_OFF(k_left)]);
    test(regKIter, regKIter);
    jz(kFringeEnd, T_NEAR);
    RETURN_IF_ERROR(loadXValues(true));
    RETURN_IF_ERROR(processMRBlock(mSize, true));
    // Advance by the remainder only, so the next group starts in the right
    // place.
    add(regTmpAptr, regKIter);
    add(regXptr, regKIter);

    L(kFringeEnd);

    RETURN_IF_ERROR(reduceAccumulation(mSize));
    RETURN_IF_ERROR(conversionCompensationGroup(mSize));

    if (bPerGroupK) {
        RETURN_IF_ERROR(loadBScaleBroadcast());
    }
    if (aPerGroupK) {
        RETURN_IF_ERROR(loadAScaleVector(mSize));
    }

    RETURN_IF_ERROR(dequantAccumToFAcc(mSize));

    add(regGIter, 1); // group += 1
    mov(regTmp1, ptr[stackPtr + N1_OFF(num_groups)]);
    cmp(regGIter, regTmp1);
    jl(groupBegin, T_NEAR);

    for (iter_t b = 0; b < nBands; ++b) {
        vmovaps(RegType(accumBaseIdx + b), RegType(fAccBaseIdx + b));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::scaleAccByAlphaF32(int mSize)
{
    if (alphaScalingType == dlp::kernel_frame::scalingType::one
        || alphaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int nBands = (mSize + vnniWidth - 1) / vnniWidth;

    mov(regTmp1, ptr[stackPtr + N1_OFF(alpha)]);
    vpbroadcastd(RegType(yBaseIdx), ptr[regTmp1]);
    vcvtdq2ps(RegType(yBaseIdx), RegType(yBaseIdx));

    for (iter_t b = 0; b < nBands; ++b) {
        vmulps(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b),
               RegType(yBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Retarget regTmpYptr at the downscaled output buffer and leave its byte row
// stride in regTmp1, which callers must treat as live for the whole walk.
template<utils::kernelInstrType KType>
void
jitGEMVQuantN1<KType>::updateCBufferPointers()
{
    mov(regTmpYptr, ptr[stackPtr + N1_OPS_OFF(buf_downscale)]);

    mov(regTmp1, ptr[stackPtr + N1_OPS_OFF(rs_c_downscale)]);
    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    // post_op_c_i is kept current in memory by the m-loop, so it is the row
    // origin of the tile being stored.
    mov(regKIter, ptr[stackPtr + N1_OPS_OFF(post_op_c_i)]);
    imul(regKIter, regTmp1);
    add(regTmpYptr, regKIter);
}

// acc += beta * C. C is f32 at y for a DLP_F32 output, or bf16 in
// buf_downscale for DLP_BF16.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::scaleYByBetaF32(int mSize)
{
    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int  nBands = (mSize + vnniWidth - 1) / vnniWidth;
    const bool masked = ((mSize % vnniWidth) != 0);
    const bool isUnitBeta =
        (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (!isUnitBeta) {
        mov(regTmp1, ptr[stackPtr + N1_OFF(beta)]);
        vpbroadcastd(RegType(bSclBaseIdx), ptr[regTmp1]);
        vcvtdq2ps(RegType(bSclBaseIdx), RegType(bSclBaseIdx));
    }

    // regTmpYptr = C base, regTmp1 = byte row stride.
    if (c_downscale == DLP_BF16) {
        updateCBufferPointers();
    } else {
        mov(regTmpYptr, regYptr);
        mov(regTmp1, regRsC);
    }

    if (yFormat != dlp::kernel_frame::storageFormat::colMajor) {
        // Strided C: scalar-copy into the scratch, widening bf16 as we go.
        for (iter_t r = 0; r < mSize; ++r) {
            if (c_downscale == DLP_BF16) {
                movzx(regKIter.cvt32(), word[regTmpYptr]);
                shl(regKIter.cvt32(), 16);
            } else {
                mov(regKIter.cvt32(), ptr[regTmpYptr]);
            }
            mov(ptr[rsp + r * static_cast<int>(sizeof(float))],
                regKIter.cvt32());
            add(regTmpYptr, regTmp1);
        }
        for (iter_t b = 0; b < nBands; ++b) {
            if (masked) {
                vmovups(RegType(yBaseIdx + b) | k2 | T_z,
                        ptr[rsp + b * RegBytes]);
            } else {
                vmovups(RegType(yBaseIdx + b), ptr[rsp + b * RegBytes]);
            }
        }
    } else if (c_downscale == DLP_BF16) {
        for (iter_t b = 0; b < nBands; ++b) {
            if (masked) {
                vmovdqu16(Xbyak::Ymm(yBaseIdx + b) | k2 | T_z,
                          ptr[regTmpYptr + b * (RegBytes / 2)]);
            } else {
                vmovdqu16(Xbyak::Ymm(yBaseIdx + b),
                          ptr[regTmpYptr + b * (RegBytes / 2)]);
            }
            vpmovsxwd(RegType(yBaseIdx + b), Xbyak::Ymm(yBaseIdx + b));
            vpslld(RegType(yBaseIdx + b), RegType(yBaseIdx + b), 16);
        }
    } else {
        for (iter_t b = 0; b < nBands; ++b) {
            if (masked) {
                vmovups(RegType(yBaseIdx + b) | k2 | T_z,
                        ptr[regTmpYptr + b * RegBytes]);
            } else {
                vmovups(RegType(yBaseIdx + b), ptr[regTmpYptr + b * RegBytes]);
            }
        }
    }

    for (iter_t b = 0; b < nBands; ++b) {
        if (isUnitBeta) {
            vaddps(RegType(accumBaseIdx + b), RegType(accumBaseIdx + b),
                   RegType(yBaseIdx + b));
        } else {
            vfmadd231ps(RegType(accumBaseIdx + b), RegType(bSclBaseIdx),
                        RegType(yBaseIdx + b));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::generatePostOps(utils::gemvN1GeneratorParams& params,
                                       int                           mSize)
{
    if (params.kernelOps.empty()) {
        return dlp::jit::jitGeneratorError::success;
    }

    Xbyak::Label skipPostOps;

    // Chunked-K frames would otherwise re-apply post-ops per chunk. The
    // sym-quant GEMV frame always marks a single chunk, but the gate keeps the
    // contract identical to every other kernel here.
    mov(regTmp1, ptr[stackPtr + N1_OPS_OFF(is_last_k)]);
    test(regTmp1, regTmp1);
    je(skipPostOps, T_NEAR);

    const int  numCRegs = (mSize + vnniWidth - 1) / vnniWidth;
    const bool useMask  = ((mSize % vnniWidth) != 0);

    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    VecPoolType vecPool;
    vecPool.setAccumulators(accumBaseIdx, numCRegs);
    RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

    // Two masks are preserved: k1 (K remainder) and k2 (M remainder).
    MaskPoolType maskPool;
    maskPool.addPreserve(utils::MASK_START_IDX, 2);
    RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                  Traits::reservedMaskBits));

    int maskOffset = static_cast<int>(N1_OFF(mmask_avx512));

    gen::kernelOpsHandler<KType> handler(this);
    RETURN_IF_ERROR(handler.generateKernelOps(
        params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1, mSize,
        /*NR=*/1, useMask, /*numMaskRegs=*/1, accumBaseIdx, numCRegs, vecPool,
        maskPool, maskOffset));

    L(skipPostOps);

    return dlp::jit::jitGeneratorError::success;
}

// Store the F32 accumulators, converting to bf16 with round-to-nearest-even
// when the output rail is DLP_BF16.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::storeYF32(int mSize)
{
    if ((c_downscale != DLP_F32) && (c_downscale != DLP_BF16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    const int  nBands = (mSize + vnniWidth - 1) / vnniWidth;
    const bool masked = ((mSize % vnniWidth) != 0);

    if (c_downscale == DLP_BF16) {
        updateCBufferPointers();
    } else {
        mov(regTmpYptr, regYptr);
        mov(regTmp1, regRsC);
    }

    if (c_downscale == DLP_BF16) {
        // bf16 = (v + 0x7FFF + ((v >> 16) & 1)) >> 16
        mov(regKIter, 0x00000001);
        vpbroadcastd(RegType(xBaseIdx), regKIter.cvt32());
        mov(regKIter, 0x00007FFF);
        vpbroadcastd(RegType(bSclBaseIdx), regKIter.cvt32());

        for (iter_t b = 0; b < nBands; ++b) {
            const int c = accumBaseIdx + b;
            vpsrld(RegType(yBaseIdx), RegType(c), 16);
            vpandd(RegType(yBaseIdx), RegType(yBaseIdx), RegType(xBaseIdx));
            vpaddd(RegType(c), RegType(c), RegType(bSclBaseIdx));
            vpaddd(RegType(c), RegType(c), RegType(yBaseIdx));
            vpsrld(RegType(c), RegType(c), 16);
            vpmovdw(Xbyak::Ymm(c), RegType(c));
        }
    }

    if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
        // Contiguous output column.
        for (iter_t b = 0; b < nBands; ++b) {
            const int c = accumBaseIdx + b;
            if (c_downscale == DLP_BF16) {
                if (masked) {
                    vmovdqu16(ptr[regTmpYptr + b * (RegBytes / 2)] | k2,
                              Xbyak::Ymm(c));
                } else {
                    vmovdqu16(ptr[regTmpYptr + b * (RegBytes / 2)],
                              Xbyak::Ymm(c));
                }
            } else {
                if (masked) {
                    vmovups(ptr[regTmpYptr + b * RegBytes] | k2, RegType(c));
                } else {
                    vmovups(ptr[regTmpYptr + b * RegBytes], RegType(c));
                }
            }
        }

        return dlp::jit::jitGeneratorError::success;
    }

    // Strided output column: spill the band and scatter it a scalar at a time.
    for (iter_t b = 0; b < nBands; ++b) {
        const int c = accumBaseIdx + b;
        if (c_downscale == DLP_BF16) {
            vmovdqu16(ptr[rsp], Xbyak::Ymm(c));
        } else {
            vmovups(ptr[rsp], RegType(c));
        }

        int rowsInBand = mSize - (b * vnniWidth);
        if (rowsInBand > vnniWidth) {
            rowsInBand = vnniWidth;
        }

        for (iter_t r = 0; r < rowsInBand; ++r) {
            if (c_downscale == DLP_BF16) {
                movzx(regKIter.cvt32(),
                      word[rsp + r * static_cast<int>(sizeof(uint16_t))]);
                mov(word[regTmpYptr], regKIter.cvt16());
            } else {
                mov(regKIter.cvt32(),
                    ptr[rsp + r * static_cast<int>(sizeof(float))]);
                mov(ptr[regTmpYptr], regKIter.cvt32());
            }
            add(regTmpYptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantN1<KType>::generateKernel(utils::quantGemvN1GeneratorParams& params)
{
    RETURN_IF_ERROR(
        utils::jitGeneratorUtils::checkValidGemvN1Params(params.base));

    // NOTE Asymmetric quantization is not supported for now; return.
    if ((params.aQuant.zeroPoint.storeDt
         != dlp::kernel_frame::DataType::invalid)
        || (params.bQuant.zeroPoint.storeDt
            != dlp::kernel_frame::DataType::invalid)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Set A and B scale datatypes
    aScaleType = params.aQuant.scale.storeDt;
    bScaleType = params.bQuant.scale.storeDt;

    // Scale-factor storage: Only f32 and bf16 are supported.
    if ((aScaleType != dlp::kernel_frame::DataType::f32
         && aScaleType != dlp::kernel_frame::DataType::bf16)
        || (bScaleType != dlp::kernel_frame::DataType::f32
            && bScaleType != dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // A mixed f32/bf16 pair is not supported.
    if ((aScaleType == dlp::kernel_frame::DataType::bf16)
        != (bScaleType == dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Only the F32 and BF16 downscale types are supported
    if ((params.base.c_downscale != DLP_F32)
        && (params.base.c_downscale != DLP_BF16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Set whether each scale array is tiled along K, which decides if the
    // group index offsets that operand's scale address at all.
    // A false perGroupK implies PER_TOKEN for A and PER_CHANNEL for B
    // and disables the group index offsets.
    aPerGroupK = params.aQuant.scale.perGroupK;
    bPerGroupK = params.bQuant.scale.perGroupK;

    // Reserve one register for converting int8 to uint8 for symmetric
    // quantization path.
    useVec128 = ((params.aQuant.zeroPoint.storeDt
                  == dlp::kernel_frame::DataType::invalid)
                 || (params.bQuant.zeroPoint.storeDt
                     == dlp::kernel_frame::DataType::invalid));

    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64
    // (no-op on Linux/SysV). See utils::winAbiVectorGuard.
    utils::winAbiVectorGuard winAbiGuard(this);

    initializeParameters(params);

    RETURN_IF_ERROR(allocateRegisters());

    // Allocate stack scratch bytes
    sub(rsp, N1_SCRATCH_BYTES);

    // Load A and Y pointers from stack
    mov(regAptr, ptr[stackPtr + N1_OFF(a)]);
    mov(regYptr, ptr[stackPtr + N1_OFF(y)]);

    // Load masks for fringe handling
    kmovq(k1, ptr[stackPtr + N1_OFF(kmask_i8_avx512)]);
    kmovd(k2, ptr[stackPtr + N1_OFF(mmask_avx512)]);

    Xbyak::Label mLoopBegin, mLoopEnd, fringeEnd;

    // M-Loop handling
    if (params.base.mloop) {
        // Load m-loop iterations
        mov(regMIter, ptr[stackPtr + N1_OFF(m_iter)]);
        test(regMIter, regMIter);
        jz(mLoopEnd, T_NEAR); // Jump to end if no iterations are left

        L(mLoopBegin);

        RETURN_IF_ERROR(groupLoop(MR));
        RETURN_IF_ERROR(scaleAccByAlphaF32(MR));
        RETURN_IF_ERROR(scaleYByBetaF32(MR));
        RETURN_IF_ERROR(generatePostOps(params.base, MR));
        RETURN_IF_ERROR(storeYF32(MR));

        // Advance to the next m tile.
        mov(regTmp1, MR);      // load MR
        imul(regTmp1, regRsA); // MR * rsA
        add(regAptr, regTmp1); // A += (MR * rsA)

        mov(regTmp1, MR);      // reload MR
        imul(regTmp1, regRsC); // MR * rsC
        add(regYptr, regTmp1); // C += (MR * rsC)

        lea(regTmp2, ptr[regTmp2 + MR]); // grp_post_op_i += MR

        // Keep post_op_c_i current, since the post-op emitter and the
        // downscale-buffer addressing both read it from memory.
        mov(regTmp1, ptr[stackPtr + N1_OPS_OFF(post_op_c_i)]);
        add(regTmp1, MR); // post_op_c_i += MR
        mov(ptr[stackPtr + N1_OPS_OFF(post_op_c_i)], regTmp1);

        sub(regMIter, 1); // m_iter -= 1
        jnz(mLoopBegin, T_NEAR);

        L(mLoopEnd);
    }

    // M-fringe loop handling
    if (params.base.mfringe && (M_LEFT > 0)) {
        RETURN_IF_ERROR(groupLoop(M_LEFT));
        RETURN_IF_ERROR(scaleAccByAlphaF32(M_LEFT));
        RETURN_IF_ERROR(scaleYByBetaF32(M_LEFT));
        RETURN_IF_ERROR(generatePostOps(params.base, M_LEFT));
        RETURN_IF_ERROR(storeYF32(M_LEFT));

        L(fringeEnd);
    }

    add(rsp, N1_SCRATCH_BYTES);
    vzeroupper();

    return dlp::jit::jitGeneratorError::success;
}

// ---------------------------------------------------------------------------
// s8s8s32 group-quantized GEMV, M == 1
// ---------------------------------------------------------------------------

#define M1_OFF(field) offsetof(dlp::kernels::gemvM1Params, field)
#define M1_OPS_OFF(field)                                                      \
    (offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)                       \
     + offsetof(dlp_gemm_post_op_attr, field))
#define M1_GRP_OFF(field)                                                      \
    (offsetof(dlp::kernels::gemvM1Params, grpKernelOpsAttr)                    \
     + offsetof(dlp_gemm_grp_post_op_attr, field))

// Stack scratch: [rsp+8] groups remaining in the current KC panel. The
// absolute K-group index used to live at [rsp+0], but three address
// computations per group read it, which put a store-to-load round trip in
// front of each of them; it is now register-resident (regGAbs) and offset 0 is
// simply left unused rather than shuffling gLeft down for 8 bytes of a scratch
// area that is three-quarters empty.
static constexpr int M1_SCRATCH_BYTES = 64;
static constexpr int M1_GLEFT_OFF     = 8;

template<utils::kernelInstrType KType>
jitGEMVQuantM1<KType>::jitGEMVQuantM1(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantM1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr    = frame.p[0];
    regBptr     = frame.t[0];
    regXptr     = frame.t[1];
    regYptr     = frame.t[2];
    regTmpYptr  = frame.t[3];
    regNIter    = frame.t[4];
    regKIter    = frame.t[5];
    regKSubIter = frame.t[6];
    regRsB      = frame.t[7];
    regGAbs     = frame.t[8];
    regTmp1     = frame.t[9];
    regTmp2     = frame.t[10];
    regIncN     = frame.t[11];
    regIncK     = frame.t[12];
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::allocateRegisters()
{
    const int nBands = NR / vnniWidth;

    // group accumulator registers
    accumReg     = nBands * K_SUB_ITER;
    accumBaseIdx = numRegs - accumReg;

    xReg     = K_SUB_ITER;
    xBaseIdx = accumBaseIdx - xReg;

    bReg     = nBands;
    bBaseIdx = xBaseIdx - bReg;

    // Only the symmetric path biases A by +128, so only it owns this register.
    // The index is still computed either way; with a zero count the next bank
    // simply starts on top of it and reclaims the slot.
    vec128Reg     = useVec128 ? 1 : 0;
    vec128BaseIdx = bBaseIdx - vec128Reg;

    // F32 total across groups and KC panels, plus the broadcast A scale and two
    // spare scratch registers, all below the +128 constant.
    fAccReg     = nBands;
    fAccBaseIdx = 0;
    aSclBaseIdx = fAccBaseIdx + fAccReg;
    aSclReg     = 1;
    scratchIdx  = aSclBaseIdx + aSclReg;
    tmpIdx      = scratchIdx + 1;

    if ((tmpIdx + 1) > vec128BaseIdx) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantM1<KType>::initializeParameters(
    utils::quantGemvM1GeneratorParams& params)
{
    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    NR               = params.base.NR;
    N_LEFT           = params.base.N_LEFT;
    N_LEFT_16        = params.base.N_LEFT_16;
    N_LEFT_LT16      = params.base.N_LEFT_LT16;
    KC               = params.base.KC;
    K_SUB_ITER       = params.base.K_SUB_ITER;
    yFormat          = params.base.yFormat;
    alphaScalingType = params.base.alphaScalingType;
    betaScalingType  = params.base.betaScalingType;
    c_downscale      = params.base.c_downscale;

    // Accumulation is int32, so a band is RegBytes/4 rows wide.
    vnniWidth = RegBytes / sizeof(int32_t);

    // Load strides from the stack
    mov(regRsB, ptr[stackPtr + M1_OFF(rsB)]);
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantM1<KType>::regInit(int baseIdx, int count)
{
    // Zero out "count" number of registers starting from the baseIdx.
    for (iter_t i = 0; i < count; ++i) {
        vpxord(RegType(baseIdx + i), RegType(baseIdx + i),
               RegType(baseIdx + i));
    }
}

// Output bands this variant touches. The fringe variants only walk the bands
// that N_LEFT actually reaches; the rest are left untouched and never stored.
template<utils::kernelInstrType KType>
int
jitGEMVQuantM1<KType>::activeBands(bool nMask) const
{
    if (!nMask) {
        return NR / vnniWidth;
    }
    return (N_LEFT / vnniWidth) + ((N_LEFT % vnniWidth) ? 1 : 0);
}

template<utils::kernelInstrType KType>
int
jitGEMVQuantM1<KType>::maskedBand(bool nMask) const
{
    if (!nMask || ((N_LEFT % vnniWidth) == 0)) {
        return -1;
    }
    return N_LEFT / vnniWidth;
}

// N-fringe variant of computeKxNR
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::computeKxNfringe()
{
    for (iter_t j = 0; j < K_SUB_ITER; j++) {
        vpbroadcastd(RegType(xBaseIdx + j), ptr[regXptr + j * 4]);
        if (useVec128) {
            vpaddb(RegType(xBaseIdx + j), RegType(xBaseIdx + j),
                   RegType(vec128BaseIdx));
        }
    }

    int n_iter = N_LEFT / vnniWidth;
    int n_left = N_LEFT % vnniWidth;

    for (iter_t j = 0; j < K_SUB_ITER; j++) {
        for (iter_t i = 0; i < n_iter; i++) {
            vmovdqu32(RegType(bBaseIdx + i),
                      ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                     RegType(xBaseIdx + j), RegType(bBaseIdx + i));
        }
        if (n_left) {
            // The <16 remainder band lives in its own reordered sub-panel,
            // tracked by regTmpYptr.
            vmovdqu32(RegType(bBaseIdx + n_iter) | k1 | T_z,
                      ptr[regTmpYptr + j * 64]);
            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * n_iter + j),
                     RegType(xBaseIdx + j), RegType(bBaseIdx + n_iter));
        }
        add(regTmp2, regRsB);
    }

    return dlp::jit::jitGeneratorError::success;
}

// One VNNI step per K sub-iteration j in [0, K_SUB_ITER): broadcast x's 4
// packed K-elements (biasing +128 if useVec128), then per band i,
//   accum[band=i, subiter=j][lane=n] += dot4(x_j, B[j][n])
// i.e. each of the 16 lanes is one output column n; x_j is identical across
// lanes since M == 1. accum is indexed by (i, j) rather than folded, so the
// K_SUB_ITER chains stay independent until accumulateKSubIters combines them.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::computeKxNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (nMask) {
        return computeKxNfringe();
    }

    // Broadcast x
    for (iter_t i = 0; i < K_SUB_ITER; ++i) {
        vpbroadcastd(RegType(xBaseIdx + i), ptr[regXptr + i * 4]);
        // x := x + 128
        if (useVec128) {
            vpaddb(RegType(xBaseIdx + i), RegType(xBaseIdx + i),
                   RegType(vec128BaseIdx));
        }
    }

    int nIter = NR / vnniWidth;
    for (iter_t j = 0; j < K_SUB_ITER; ++j) {
        for (iter_t i = 0; i < nIter; ++i) {
            // Load this row's B values for the current K-sub-block, then
            // vpdpbusd them against the shared X vector into that row's own
            // int32 accumulator (accumBaseIdx + row index).
            vmovdqu32(RegType(bBaseIdx + i),
                      ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                     RegType(xBaseIdx + j), RegType(bBaseIdx + i));
        }
        add(regTmp2, regRsB); // next K sub-iteration's packed B row
    }

    return dlp::jit::jitGeneratorError::success;
}

// N-fringe variant of compute1xNR
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::compute1xNfringe(bool isLastKGroup)
{
    if (isLastKGroup) {
        vmovdqu8(Xbyak::Xmm(xBaseIdx) | k2 | T_z, ptr[regXptr]);
        vpbroadcastd(RegType(xBaseIdx), Xbyak::Xmm(xBaseIdx));
    } else {
        vpbroadcastd(RegType(xBaseIdx), ptr[regXptr]);
    }
    if (useVec128) {
        vpaddb(RegType(xBaseIdx), RegType(xBaseIdx), RegType(vec128BaseIdx));
    }

    int n_iter = N_LEFT / vnniWidth;
    int n_left = N_LEFT % vnniWidth;

    for (iter_t i = 0; i < n_iter; i++) {
        vmovdqu32(RegType(bBaseIdx + i),
                  ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
        vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                 RegType(bBaseIdx + i));
    }

    if (n_left) {
        vmovdqu32(RegType(bBaseIdx + n_iter) | k1 | T_z, ptr[regTmpYptr]);
        vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * n_iter), RegType(xBaseIdx),
                 RegType(bBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Single-dword-of-K (4 K-elements, or fewer) VNNI step, used for the tail
// when a KC panel's K isn't a multiple of K_SUB_ITER * 4.
// isLastKGroup: only the very last K element of the very last group can leave
// fewer than 4 valid bytes at regXptr, so it alone needs the masked byte load
// (k2) instead of a plain dword broadcast, which would otherwise read past
// the end of x.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::compute1xNR(bool nMask, bool isLastKGroup)
{
    mov(regTmp2, regBptr); // B address for this K-block

    if (nMask) {
        return compute1xNfringe(isLastKGroup);
    }

    // x := broadcast(x[0..4) or x[0..k_left)) + 128
    if (isLastKGroup) {
        vmovdqu8(Xbyak::Xmm(xBaseIdx) | k2 | T_z, ptr[regXptr]);
        vpbroadcastd(RegType(xBaseIdx), Xbyak::Xmm(xBaseIdx));
    } else {
        vpbroadcastd(RegType(xBaseIdx), ptr[regXptr]);
    }
    if (useVec128) {
        vpaddb(RegType(xBaseIdx), RegType(xBaseIdx), RegType(vec128BaseIdx));
    }

    int nIter = NR / vnniWidth;
    for (iter_t i = 0; i < nIter; ++i) {
        vmovdqu32(RegType(bBaseIdx + i),
                  ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
        // accum[i][0] += dot4(x, B[i])
        vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                 RegType(bBaseIdx + i));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Fold the K_SUB_ITER partial accumulators of each band down to one.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::accumulateKSubIters(bool nMask)
{
    // Only the bands this variant actually touches. An N fringe leaves the
    // rest of the accumulator bank zeroed and never reads it -- every consumer
    // downstream is already bounded by activeBands -- so folding all NR/16
    // bands was pure waste on every fringe tile, every group.
    const int nIter = activeBands(nMask);
    for (iter_t i = 0; i < nIter; ++i) {
        for (iter_t j = 1; j < K_SUB_ITER; ++j) {
            vpaddd(RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i + j));
        }
        vmovdqu32(RegType(accumBaseIdx + i),
                  RegType(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

// One K group's worth of VNNI steps: group_size / 16 blocks of K_SUB_ITER
// dwords, then the leftover whole dwords, then a masked sub-4 tail. The tail
// only exists for a single full-K group whose K is not a multiple of 4.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::groupKLoop(bool nMask)
{
    Xbyak::Label mainBegin, mainEnd, subBegin, subEnd, remEnd;

    // Load k-group iterations
    mov(regKSubIter, ptr[stackPtr + M1_OFF(k_iter_sub_iter)]);
    test(regKSubIter, regKSubIter);
    jz(mainEnd, T_NEAR);

    L(mainBegin);
    RETURN_IF_ERROR(computeKxNR(nMask));
    lea(regXptr, ptr[regXptr + K_SUB_ITER * 4]);        // x += K_SUB_ITER*4
    lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);   // B += rsB*K_SUB_ITER
    lea(regTmpYptr, ptr[regTmpYptr + 64 * K_SUB_ITER]); // y += 64*K_SUB_ITER
    sub(regKSubIter, 1);                                // k_sub_iter -= 1
    jnz(mainBegin, T_NEAR);

    L(mainEnd);

    // Leftover K iterations
    mov(regKSubIter, ptr[stackPtr + M1_OFF(k_iter_sub_left)]);
    test(regKSubIter, regKSubIter);
    jz(subEnd, T_NEAR);

    L(subBegin);
    RETURN_IF_ERROR(compute1xNR(nMask, false));
    lea(regXptr, ptr[regXptr + 4]);        // x += 4
    lea(regBptr, ptr[regBptr + regRsB]);   // B += rsB
    lea(regTmpYptr, ptr[regTmpYptr + 64]); // y += 64
    sub(regKSubIter, 1);                   // k_sub_iter -= 1
    jnz(subBegin, T_NEAR);

    L(subEnd);

    // Sub-4 K tail, masked by kLeftmask (k2).
    mov(regKSubIter, ptr[stackPtr + M1_OFF(is_k_odd)]);
    test(regKSubIter, regKSubIter);
    jz(remEnd, T_NEAR);

    // isLastKGroup = true: since x may have fewer than 4 valid bytes left,
    // hence the masked load inside compute1xNR.
    RETURN_IF_ERROR(compute1xNR(nMask, true));

    lea(regXptr, ptr[regXptr + 4]);        // x += 4
    lea(regBptr, ptr[regBptr + regRsB]);   // B += rsB
    lea(regTmpYptr, ptr[regTmpYptr + 64]); // y += 64

    L(remEnd);

    return dlp::jit::jitGeneratorError::success;
}

// Subtract each band's B column sums for the current group.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::conversionCompensationGroup(bool nMask)
{
    // The bias and its compensation are a matched pair. If A was never biased
    // by +128 there is no 128*sum(B) term to remove, and subtracting one would
    // corrupt the accumulators.
    if (!useVec128) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int bands = activeBands(nMask);
    const int mBand = maskedBand(nMask);

    mov(regTmp1, ptr[stackPtr + M1_OPS_OFF(b_col_sum_vec)]);
    mov(regTmp2, ptr[stackPtr + M1_OPS_OFF(b_sum_offset)]);

    // + gAbs * grp_post_op_sum_ld
    mov(regKSubIter, regGAbs);
    imul(regKSubIter, ptr[stackPtr + M1_GRP_OFF(grp_post_op_sum_ld)]);
    add(regTmp2, regKSubIter);

    lea(regTmp1, ptr[regTmp1 + regTmp2 * sizeof(int32_t)]);

    for (iter_t i = 0; i < bands; ++i) {
        if (i == mBand) {
            vmovdqu32(RegType(bBaseIdx + i) | k1 | T_z,
                      ptr[regTmp1 + i * RegBytes]);
        } else {
            vmovdqu32(RegType(bBaseIdx + i), ptr[regTmp1 + i * RegBytes]);
        }
        vpsubd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(bBaseIdx + i));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Load this group's B scales, one vector per band. The partial band must be
// masked: b_scale_factor holds exactly n (or num_groups * n) elements, so an
// unmasked 16-wide load past the last column reads out of bounds.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::loadBScaleVectors(bool nMask)
{
    const int bands = activeBands(nMask);
    const int mBand = maskedBand(nMask);

    mov(regTmp1, ptr[stackPtr + M1_GRP_OFF(b_scale_factor)]);
    mov(regTmp2, ptr[stackPtr + M1_GRP_OFF(grp_post_op_j)]);

    if (bPerGroupK) {
        mov(regKSubIter, regGAbs);
        imul(regKSubIter, ptr[stackPtr + M1_GRP_OFF(grp_post_op_ldb)]);
        add(regTmp2, regKSubIter);
    }

    lea(regTmp1, ptr[regTmp1 + regTmp2 * bScaleElemBytes()]);

    for (iter_t i = 0; i < bands; ++i) {
        const bool masked = (i == mBand);
        if (bScaleType == dlp::kernel_frame::DataType::f32) {
            if (masked) {
                vmovups(RegType(bBaseIdx + i) | k1 | T_z,
                        ptr[regTmp1 + i * RegBytes]);
            } else {
                vmovups(RegType(bBaseIdx + i), ptr[regTmp1 + i * RegBytes]);
            }
        } else if (bScaleType == dlp::kernel_frame::DataType::bf16) {
            if (masked) {
                vmovdqu16(Xbyak::Ymm(bBaseIdx + i) | k1 | T_z,
                          ptr[regTmp1 + i * (RegBytes / 2)]);
            } else {
                vmovdqu16(Xbyak::Ymm(bBaseIdx + i),
                          ptr[regTmp1 + i * (RegBytes / 2)]);
            }
            vpmovsxwd(RegType(bBaseIdx + i), Xbyak::Ymm(bBaseIdx + i));
            vpslld(RegType(bBaseIdx + i), RegType(bBaseIdx + i), 16);
        } else {
            return dlp::jit::jitGeneratorError::notSupported;
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// With m == 1 there is a single output row, so the A scale is always a scalar.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::loadAScaleBroadcast()
{
    mov(regTmp1, ptr[stackPtr + M1_GRP_OFF(a_scale_factor)]);

    // row_base * lda, where row_base is grp_post_op_i (0 on this path).
    mov(regTmp2, ptr[stackPtr + M1_GRP_OFF(grp_post_op_i)]);
    imul(regTmp2, ptr[stackPtr + M1_GRP_OFF(grp_post_op_lda)]);

    if (aPerGroupK) {
        add(regTmp2, regGAbs);
    }

    lea(regTmp1, ptr[regTmp1 + regTmp2 * aScaleElemBytes()]);

    if (aScaleType == dlp::kernel_frame::DataType::f32) {
        vbroadcastss(RegType(aSclBaseIdx), ptr[regTmp1]);
    } else if (aScaleType == dlp::kernel_frame::DataType::bf16) {
        vpbroadcastw(RegType(aSclBaseIdx), ptr[regTmp1]);
        vpmovsxwd(RegType(aSclBaseIdx), Xbyak::Ymm(aSclBaseIdx));
        vpslld(RegType(aSclBaseIdx), RegType(aSclBaseIdx), 16);
    } else {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::dequantAccumToFAcc(bool nMask)
{
    const int bands = activeBands(nMask);

    RETURN_IF_ERROR(loadBScaleVectors(nMask));
    RETURN_IF_ERROR(loadAScaleBroadcast());

    for (iter_t i = 0; i < bands; ++i) {
        vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
        vmulps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(bBaseIdx + i));
        vmulps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(aSclBaseIdx));
        vaddps(RegType(fAccBaseIdx + i), RegType(fAccBaseIdx + i),
               RegType(accumBaseIdx + i));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Point regBptr (and regTmpYptr, the <16 sub-panel cursor) at the start of the
// current KC panel for this n tile. Mirrors the reordered-B addressing the
// non-quant M=1 kernel and the intrinsic kernel both use:
//   b + jc_cur_loop_rem * psB + n_sub_updated * kOffset + nOffset * psB
template<utils::kernelInstrType KType>
void
jitGEMVQuantM1<KType>::setupPanelBBase(bool nMask, bool isLastPanel)
{
    mov(regBptr, ptr[stackPtr + M1_OFF(b)]);

    // The panel stride lives in regTmp1 rather than a dedicated register: this
    // function is the only consumer, its caller overwrites regTmp1 immediately
    // afterwards, and freeing the dedicated register is what lets the absolute
    // group index be register-resident (regGAbs).
    if (isLastPanel) {
        mov(regTmp1, ptr[stackPtr + M1_OFF(psB)]);
    } else {
        mov(regTmp1, KC);
    }

    mov(regTmpYptr, ptr[stackPtr + M1_OFF(jc_cur_loop_rem)]);
    mov(regTmp2, ptr[stackPtr + M1_OFF(n_sub_updated)]);
    imul(regTmpYptr, regTmp1);
    imul(regTmp2, regIncK);

    lea(regBptr, ptr[regBptr + regTmpYptr]);
    lea(regBptr, ptr[regBptr + regTmp2]);

    mov(regTmp2, regIncN);
    imul(regTmp2, regTmp1);
    add(regBptr, regTmp2);

    // The <16 remainder band sits in its own sub-panel, after the full-16
    // columns of this tile.
    mov(regTmpYptr, regBptr);
    if (nMask && (N_LEFT > 16)) {
        mov(regTmp2, regRsB);
        shr(regTmp2, 2);
        imul(regTmp2, regTmp1);
        lea(regTmpYptr, ptr[regTmpYptr + regTmp2]);
    }
}

// Walk the K groups of one KC panel. Within a panel the k-loop's own pointer
// advance lands exactly on the next group's base -- a group spans
// group_size/4 packed rows, i.e. group_size * (rsB/4) bytes, which is precisely
// the reordered group stride -- so no per-group B recomputation is needed.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::groupLoop(bool nMask, bool isLastPanel)
{
    Xbyak::Label grpBegin, grpEnd;

    setupPanelBBase(nMask, isLastPanel);

    // Load group panel
    mov(regTmp1, ptr[stackPtr
                     + (isLastPanel ? M1_OFF(groups_last_panel)
                                    : M1_OFF(groups_per_panel))]);
    mov(ptr[rsp + M1_GLEFT_OFF], regTmp1);
    test(regTmp1, regTmp1);
    jz(grpEnd, T_NEAR);

    L(grpBegin);

    // Zero out `accumReg` count of S32 accumulator registers starting from
    // accumBaseIdx.
    regInit(accumBaseIdx, accumReg);

    RETURN_IF_ERROR(groupKLoop(nMask));
    RETURN_IF_ERROR(accumulateKSubIters(nMask));
    RETURN_IF_ERROR(conversionCompensationGroup(nMask));
    RETURN_IF_ERROR(dequantAccumToFAcc(nMask));

    // Advance the absolute group index and the per-panel countdown.
    add(regGAbs, 1);

    mov(regTmp1, ptr[rsp + M1_GLEFT_OFF]);
    sub(regTmp1, 1);
    mov(ptr[rsp + M1_GLEFT_OFF], regTmp1);
    jnz(grpBegin, T_NEAR);

    L(grpEnd);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::scaleAccByAlphaF32(bool nMask)
{
    if (alphaScalingType == dlp::kernel_frame::scalingType::one
        || alphaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int bands = activeBands(nMask);

    mov(regKSubIter, ptr[stackPtr + M1_OFF(alpha)]);
    vpbroadcastd(RegType(scratchIdx), ptr[regKSubIter]);
    vcvtdq2ps(RegType(scratchIdx), RegType(scratchIdx));

    for (iter_t i = 0; i < bands; ++i) {
        vmulps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(scratchIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVQuantM1<KType>::updateYBufferPointers()
{
    mov(regTmpYptr, ptr[stackPtr + M1_OPS_OFF(buf_downscale)]);

    mov(regTmp1, ptr[stackPtr + M1_OPS_OFF(post_op_c_j)]);
    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpYptr, regTmp1);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::scaleYByBetaF32(bool nMask)
{
    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    const int  bands = activeBands(nMask);
    const int  mBand = maskedBand(nMask);
    const bool isUnitBeta =
        (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (!isUnitBeta) {
        mov(regKSubIter, ptr[stackPtr + M1_OFF(beta)]);
        vpbroadcastd(RegType(scratchIdx), ptr[regKSubIter]);
        vcvtdq2ps(RegType(scratchIdx), RegType(scratchIdx));
    }

    if (c_downscale == DLP_BF16) {
        updateYBufferPointers();
    } else {
        mov(regTmpYptr, regYptr);
    }

    for (iter_t i = 0; i < bands; ++i) {
        const bool masked = (i == mBand);

        if (c_downscale == DLP_BF16) {
            if (masked) {
                vmovdqu16(Xbyak::Ymm(bBaseIdx + i) | k1 | T_z,
                          ptr[regTmpYptr + i * (RegBytes / 2)]);
            } else {
                vmovdqu16(Xbyak::Ymm(bBaseIdx + i),
                          ptr[regTmpYptr + i * (RegBytes / 2)]);
            }
            vpmovsxwd(RegType(bBaseIdx + i), Xbyak::Ymm(bBaseIdx + i));
            vpslld(RegType(bBaseIdx + i), RegType(bBaseIdx + i), 16);
        } else {
            if (masked) {
                vmovups(RegType(bBaseIdx + i) | k1 | T_z,
                        ptr[regTmpYptr + i * RegBytes]);
            } else {
                vmovups(RegType(bBaseIdx + i), ptr[regTmpYptr + i * RegBytes]);
            }
        }

        if (isUnitBeta) {
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   RegType(bBaseIdx + i));
        } else {
            vfmadd231ps(RegType(accumBaseIdx + i), RegType(scratchIdx),
                        RegType(bBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::generatePostOps(utils::gemvM1GeneratorParams& params,
                                       bool                          nMask)
{
    if (params.kernelOps.empty()) {
        return dlp::jit::jitGeneratorError::success;
    }

    Xbyak::Label skipPostOps;

    mov(regTmp1, ptr[stackPtr + M1_OPS_OFF(is_last_k)]);
    test(regTmp1, regTmp1);
    je(skipPostOps, T_NEAR);

    const int numCRegs = activeBands(nMask);

    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    VecPoolType vecPool;
    // The +128 constant has to survive: the n-loop reuses it for the next tile.
    // Without it there is nothing to preserve, and the index aliases a bank the
    // pool is free to allocate.
    if (useVec128) {
        vecPool.addPreserve(vec128BaseIdx);
    }
    vecPool.setAccumulators(accumBaseIdx, numCRegs);
    RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

    MaskPoolType maskPool;
    maskPool.addPreserve(utils::MASK_START_IDX, 2);
    RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                  Traits::reservedMaskBits));

    int maskOffset = static_cast<int>(M1_OFF(nmask_avx512));

    gen::kernelOpsHandler<KType> handler(this);
    RETURN_IF_ERROR(handler.generateKernelOps(
        params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1, /*MR=*/1,
        nMask ? N_LEFT : NR, nMask, /*numMaskRegs=*/1, accumBaseIdx, numCRegs,
        vecPool, maskPool, maskOffset));

    L(skipPostOps);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::storeYF32(bool nMask)
{
    if ((c_downscale != DLP_F32) && (c_downscale != DLP_BF16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    const int bands = activeBands(nMask);
    const int mBand = maskedBand(nMask);

    if (c_downscale == DLP_BF16) {
        updateYBufferPointers();

        // bf16 = (v + 0x7FFF + ((v >> 16) & 1)) >> 16
        mov(regKSubIter, 0x00000001);
        vpbroadcastd(RegType(scratchIdx), regKSubIter.cvt32());
        mov(regKSubIter, 0x00007FFF);
        vpbroadcastd(RegType(tmpIdx), regKSubIter.cvt32());

        for (iter_t i = 0; i < bands; ++i) {
            const int c = accumBaseIdx + i;
            vpsrld(RegType(aSclBaseIdx), RegType(c), 16);
            vpandd(RegType(aSclBaseIdx), RegType(aSclBaseIdx),
                   RegType(scratchIdx));
            vpaddd(RegType(c), RegType(c), RegType(tmpIdx));
            vpaddd(RegType(c), RegType(c), RegType(aSclBaseIdx));
            vpsrld(RegType(c), RegType(c), 16);
            vpmovdw(Xbyak::Ymm(c), RegType(c));

            if (i == mBand) {
                vmovdqu16(ptr[regTmpYptr + i * (RegBytes / 2)] | k1,
                          Xbyak::Ymm(c));
            } else {
                vmovdqu16(ptr[regTmpYptr + i * (RegBytes / 2)], Xbyak::Ymm(c));
            }
        }

        return dlp::jit::jitGeneratorError::success;
    }

    mov(regTmpYptr, regYptr);
    for (iter_t i = 0; i < bands; ++i) {
        if (i == mBand) {
            vmovups(ptr[regTmpYptr + i * RegBytes] | k1,
                    RegType(accumBaseIdx + i));
        } else {
            vmovups(ptr[regTmpYptr + i * RegBytes], RegType(accumBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// One n tile: KC-panel loop with the group loop nested inside, then the F32
// epilogue. nMask selects the N_LEFT fringe variant.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::emitNTile(utils::quantGemvM1GeneratorParams& params,
                                 bool                               nMask)
{
    const int bands    = activeBands(nMask);
    const int tileCols = nMask ? N_LEFT : NR;

    regInit(fAccBaseIdx, fAccReg);
    xor_(regIncK, regIncK);

    // The absolute group index restarts at group_start for every n tile. It
    // stays live in regGAbs across every KC panel below.
    mov(regGAbs, ptr[stackPtr + M1_OFF(group_start)]);

    // Skip A*B if alpha == 0.
    if (alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        Xbyak::Label panelBegin, panelEnd, lastPanelEnd;

        // Load x pointer
        mov(regXptr, ptr[stackPtr + M1_OFF(x)]);

        // Load k_iters
        mov(regKIter, ptr[stackPtr + M1_OFF(k_iter)]);
        test(regKIter, regKIter);
        jz(panelEnd, T_NEAR);

        L(panelBegin);
        RETURN_IF_ERROR(groupLoop(nMask, false));
        mov(regTmp2, KC);
        add(regIncK, regTmp2);
        sub(regKIter, 1);
        jnz(panelBegin, T_NEAR);

        L(panelEnd);

        // Trailing partial KC panel.
        mov(regKIter, ptr[stackPtr + M1_OFF(k_left)]);
        test(regKIter, regKIter);
        jz(lastPanelEnd, T_NEAR);
        RETURN_IF_ERROR(groupLoop(nMask, true));

        L(lastPanelEnd);
    }

    // Hand the F32 total to the epilogue in the accumulator registers.
    for (iter_t i = 0; i < bands; ++i) {
        vmovaps(RegType(accumBaseIdx + i), RegType(fAccBaseIdx + i));
    }

    RETURN_IF_ERROR(scaleAccByAlphaF32(nMask));
    RETURN_IF_ERROR(scaleYByBetaF32(nMask));
    RETURN_IF_ERROR(generatePostOps(params.base, nMask));
    RETURN_IF_ERROR(storeYF32(nMask));

    // The post-op emitter may have reused the +128 register under pressure.
    if (useVec128) {
        vxorps(RegType(vec128BaseIdx), RegType(vec128BaseIdx),
               RegType(vec128BaseIdx));
        mov(regTmp2, 128);
        vpbroadcastb(RegType(vec128BaseIdx), regTmp2.cvt8());
    }

    // Advance the per-tile column bases this kernel owns. b_sum_offset,
    // post_op_c_j and grp_post_op_j all index the absolute output column, so
    // they move with the n loop; the orchestrator restores them afterwards.
    mov(regTmp2, ptr[stackPtr + M1_OPS_OFF(b_sum_offset)]);
    add(regTmp2, tileCols);
    mov(ptr[stackPtr + M1_OPS_OFF(b_sum_offset)], regTmp2);

    mov(regTmp2, ptr[stackPtr + M1_OPS_OFF(post_op_c_j)]);
    add(regTmp2, tileCols);
    mov(ptr[stackPtr + M1_OPS_OFF(post_op_c_j)], regTmp2);

    mov(regTmp2, ptr[stackPtr + M1_GRP_OFF(grp_post_op_j)]);
    add(regTmp2, tileCols);
    mov(ptr[stackPtr + M1_GRP_OFF(grp_post_op_j)], regTmp2);

    mov(regTmp2, tileCols);
    add(regIncN, regTmp2);
    if (c_downscale == DLP_BF16) {
        // The f32 rail is unused for a bf16 output; y is not advanced.
    } else {
        lea(regYptr, ptr[regYptr + regTmp2 * sizeof(float)]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVQuantM1<KType>::generateKernel(utils::quantGemvM1GeneratorParams& params)
{
    RETURN_IF_ERROR(
        utils::jitGeneratorUtils::checkValidGemvM1Params(params.base));

    // NOTE Asymmetric quantization is not supported for now; return.
    if ((params.aQuant.zeroPoint.storeDt
         != dlp::kernel_frame::DataType::invalid)
        || (params.bQuant.zeroPoint.storeDt
            != dlp::kernel_frame::DataType::invalid)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Set A and B scale datatypes
    aScaleType = params.aQuant.scale.storeDt;
    bScaleType = params.bQuant.scale.storeDt;

    // Scale-factor storage: Only f32 and bf16 are supported.
    if ((aScaleType != dlp::kernel_frame::DataType::f32
         && aScaleType != dlp::kernel_frame::DataType::bf16)
        || (bScaleType != dlp::kernel_frame::DataType::f32
            && bScaleType != dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // A mixed f32/bf16 pair is not supported.
    if ((aScaleType == dlp::kernel_frame::DataType::bf16)
        != (bScaleType == dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Only the F32 and BF16 downscale types are supported
    if ((params.base.c_downscale != DLP_F32)
        && (params.base.c_downscale != DLP_BF16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Set whether each scale array is tiled along K, which decides if the
    // group index offsets that operand's scale address at all.
    // A false perGroupK implies PER_TOKEN for A and PER_CHANNEL for B
    // and disables the group index offsets.
    aPerGroupK = params.aQuant.scale.perGroupK;
    bPerGroupK = params.bQuant.scale.perGroupK;

    // Reserve one register for converting int8 to uint8 for symmetric
    // quantization path.
    useVec128 = ((params.aQuant.zeroPoint.storeDt
                  == dlp::kernel_frame::DataType::invalid)
                 || (params.bQuant.zeroPoint.storeDt
                     == dlp::kernel_frame::DataType::invalid));

    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64
    // (no-op on Linux/SysV). See utils::winAbiVectorGuard.
    utils::winAbiVectorGuard winAbiGuard(this);

    initializeParameters(params);

    RETURN_IF_ERROR(allocateRegisters());

    // Allocate stack scratch bytes
    sub(rsp, M1_SCRATCH_BYTES);

    // Broadcast 128 for sym_quant to convert signed int8 A to uint8.
    if (useVec128) {
        vxorps(RegType(vec128BaseIdx), RegType(vec128BaseIdx),
               RegType(vec128BaseIdx));
        mov(regTmp2, 128);
        vpbroadcastb(RegType(vec128BaseIdx), regTmp2.cvt8());
    }

    // Load Y pointer from stack
    mov(regYptr, ptr[stackPtr + M1_OFF(y)]);
    xor_(regIncN, regIncN);

    // Load mask for fringe handling
    kmovw(k2, ptr[stackPtr + M1_OFF(kLeftmask)]);

    Xbyak::Label nLoopBegin, nLoopEnd, nFringeEnd;

    // N-Loop handling
    if (params.base.nloop) {
        // Load n-loop iterations
        mov(regNIter, ptr[stackPtr + M1_OFF(n_iter)]);
        test(regNIter, regNIter);
        jz(nLoopEnd, T_NEAR); // Jump to end if no iterations are left

        L(nLoopBegin);
        RETURN_IF_ERROR(emitNTile(params, false));
        sub(regNIter, 1); // n_iter -= 1
        jnz(nLoopBegin, T_NEAR);

        L(nLoopEnd);
    }

    // N-fringe loop handling
    if (params.base.nfringe && (N_LEFT > 0)) {
        kmovw(k1, ptr[stackPtr + M1_OFF(nmask_avx512)]);

        mov(regNIter, ptr[stackPtr + M1_OFF(n_left)]);
        test(regNIter, regNIter);
        jz(nFringeEnd, T_NEAR);

        // The reordered fringe panel is only N_LEFT_16 columns wide, so the
        // packed row stride shrinks accordingly.
        if (N_LEFT < 32) {
            mov(regRsB, 64);
        } else if (N_LEFT < 48) {
            mov(regRsB, 128);
        } else if (N_LEFT < 64) {
            mov(regRsB, 192);
        }

        RETURN_IF_ERROR(emitNTile(params, true));

        L(nFringeEnd);
    }

    add(rsp, M1_SCRATCH_BYTES);
    vzeroupper();

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitGEMVQuantN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::gen::jitGEMVQuantM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
