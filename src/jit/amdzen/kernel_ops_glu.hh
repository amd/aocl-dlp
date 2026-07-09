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

#include "x86_kernel_ops_generator.hh"

namespace amdzen::x86gen {

// Forward declaration of the CRTP base defined in kernel_ops.hh. The GLU
// strategies only derive from it (a dependent base, checked at instantiation),
// so a declaration is sufficient here. Any activation they compose (e.g. Swish)
// is used only inside the out-of-line generateImpl bodies in the .tcc, where
// the full definitions are available -- hence this header can be included
// before the activation strategy classes.
template<typename Derived, utils::kernelInstrType KType>
class kernelopsBase;

// ─────────────────────────────────────────────────────────────────────────
// GLU fused post-op strategy classes
//
// Both GatedSwiglu and GatedSwigluAndMul are *terminal*, *shape-changing*
// post-ops: they consume a 2I-wide interleaved (gate, up) tile and produce I
// outputs (halved along N for row-major GEMM / GEMV M1, along M for
// column-major GEMM / GEMV N1). The strategy class does the GLU compute
// (de-interleave, swish on gate, multiply); the compacted tile is written to
// the caller-owned D buffer by the per-dtype generator's
// storeHalfWidthResult() / storeHalfWidthResultAlongM(), selected via the
// `storeHalfWidthResults` flag on generatorParams. Compute lives in the .tcc.
// ─────────────────────────────────────────────────────────────────────────

// GatedSwiglu: out[j] = swish(gate[j]) * up[j]
//   where gate[j] = c_acc[2j], up[j] = c_acc[2j+1] (column-interleaved)
template<utils::kernelInstrType KType>
class GatedSwiglu : public kernelopsBase<GatedSwiglu<KType>, KType>
{
    using opBase = kernelopsBase<GatedSwiglu<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

  public:
    explicit GatedSwiglu(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// GatedSwigluAndMul: the gpt-oss / OpenAI clamped GLU (vLLM
// `swigluoai_and_mul`):
//   out[j] = (clip(up[j], -limit, +limit) + 1)
//            * clip(gate[j], max=limit) * sigmoid(alpha * clip(gate[j],
//            max=limit))
//   gate[j] = c_acc[2j], up[j] = c_acc[2j+1] (column-interleaved). The OAI
//   constants (alpha=1.702, limit=7.0, +1 bias) are fixed by the gpt-oss spec
//   and baked into the kernel (gen::tables::glu_consts); the post-op's
//   alpha/beta args are ignored.
template<utils::kernelInstrType KType>
class GatedSwigluAndMul : public kernelopsBase<GatedSwigluAndMul<KType>, KType>
{
    using opBase = kernelopsBase<GatedSwigluAndMul<KType>, KType>;
    using typename opBase::RegType;
    using typename opBase::Traits;

  public:
    explicit GatedSwigluAndMul(kernelOpsGeneratorX86<KType>& base)
        : opBase(base)
    {
    }

    dlp::jit::jitGeneratorError generateImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);
};

// ─────────────────────────────────────────────────────────────────────────
// Both GatedSwiglu::generateImpl and GatedSwigluAndMul::generateImpl are
// defined out-of-line in x86_kernel_ops_generator.tcc (they compose Swish,
// whose definition is only complete in that translation unit).
// ─────────────────────────────────────────────────────────────────────────

} // namespace amdzen::x86gen
