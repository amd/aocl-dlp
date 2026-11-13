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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "aocl_dlp_config.h"
#include "jit/f32_tiny_gemm_kernel.h"
#include "xbyak/xbyak.h"

/*
 * Whole-problem f32 SGEMM leaves, emitted lazily on first use. Each kernel
 * takes the argument struct and walks m and n itself. The k loop is specialized
 * on the row and column block, so it is broadcast-A / load-B / FMA (RV) or
 * load-A / load-B / FMA (RD) with no calls. alpha must be nonzero.
 * beta == 0 does not read C. Post-ops are not emitted.
 *
 * RV is a 6 x 64 block on AVX-512 (four zmm along n) and 6 x 16 on AVX2,
 * with k unrolled by four, walked row strip first inside an L1 column block.
 * Only a partial last vector is masked.
 * AVX-512 RD is up to six rows, with its column block derived from the
 * registers left by that row count, and k in steps of 16. AVX2 uses the same
 * register-capacity rule with its smaller register file and k in steps of 8.
 */

namespace {

using namespace Xbyak;

constexpr int kFrame   = 520;
constexpr int kAlpha   = 160;
constexpr int kBeta    = 164;
constexpr int kKmask   = 168;
constexpr int kXmmSave = 320;

enum Slot
{
    S_A = 0,
    S_B,
    S_C,
    S_RSA,
    S_CSA,
    S_RSB,
    S_CSB,
    S_RSC,
    S_M,
    S_N,
    S_K,
    S_I,
    S_J,
    S_SCALE,
    S_LOADC,
    S_NC,
    S_JC,
    S_JE
};

/* Lanes 0..7 set, 8..15 clear. Indexed as tbl + 8 - nvalid. */
const int32_t kMaskTbl[16] = { -1, -1, -1, -1, -1, -1, -1, -1,
                               0,  0,  0,  0,  0,  0,  0,  0 };

class TinyGen : public CodeGenerator
{
  public:
    TinyGen()
        : CodeGenerator(256 * 1024, AutoGrow)
    {
    }

    void gen_rv_avx512();
    void gen_rd_avx512();
    void gen_rd_avx512_wide();
    void gen_rv_avx2();
    void gen_rd_avx2();

  private:
    void prologue();
    void epilogue();
    void load_args();

    void ld(const Reg64& r, int slot);
    void st(int slot, const Reg64& r);
    void st_imm(int slot, int64_t v);

    void emit_k1_from_rem(const Reg32& rem, int lanes);

    void gen_rv(bool avx512);
    void emit_rv_block(bool avx512, int mr, int nv, bool masked);
    void emit_rv_panels(bool avx512, int mr, const Label& after);

    void emit_rd512_block(int mr, int nr, bool stream_b);
    void emit_rd512_rows(int mr, bool stream_b, const Label& after);
    void emit_rd512_reduce_store(int mr, int nr);
    void gen_rd_avx512_impl(bool stream_b);

    void emit_rd2_block(int mr, int nr);
    void emit_rd2_rows(int mr, const Label& after);
    void emit_rd2_step(int mr, int nr, bool tail);
    void emit_rd2_reduce_store(int mr, int nr);
};

void
TinyGen::ld(const Reg64& r, int slot)
{
    mov(r, ptr[rsp + slot * 8]);
}

void
TinyGen::st(int slot, const Reg64& r)
{
    mov(ptr[rsp + slot * 8], r);
}

void
TinyGen::st_imm(int slot, int64_t v)
{
    mov(rax, v);
    st(slot, rax);
}

void
TinyGen::prologue()
{
#if DLP_OS_WINDOWS
    mov(rax, rcx);
#else
    mov(rax, rdi);
#endif
    /*
     * rbp is the frame pointer, as in the other JIT generators (see
     * UseRBPAsFramePointer), so perf and gdb can unwind through the kernel.
     * The body never uses rbp, so the saved value is only needed for the
     * caller's chain.
     */
    push(rbp);
    mov(rbp, rsp);
    push(rbx);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
#if DLP_OS_WINDOWS
    push(rdi);
    push(rsi);
#endif
    sub(rsp, kFrame);
#if DLP_OS_WINDOWS
    for (int i = 0; i < 10; ++i) {
        vmovdqu(ptr[rsp + kXmmSave + i * 16], Xmm(6 + i));
    }
#endif
    load_args();
}

void
TinyGen::epilogue()
{
#if DLP_OS_WINDOWS
    for (int i = 0; i < 10; ++i) {
        vmovdqu(Xmm(6 + i), ptr[rsp + kXmmSave + i * 16]);
    }
#endif
    vzeroupper();
    add(rsp, kFrame);
#if DLP_OS_WINDOWS
    pop(rsi);
    pop(rdi);
#endif
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbx);
    pop(rbp);
    ret();
}

void
TinyGen::load_args()
{
    /* rax is overwritten by the alpha/beta flag materialization below. */
    const Reg64 args = r11;
    const Reg64 tmp  = r10;
    mov(args, rax);

    auto take = [&](int slot, int off, bool to_bytes) {
        mov(tmp, ptr[args + off]);
        if (to_bytes) {
            shl(tmp, 2);
        }
        mov(ptr[rsp + slot * 8], tmp);
    };

    take(S_M, offsetof(dlp_sgemm_tiny_jit_args, m), false);
    take(S_N, offsetof(dlp_sgemm_tiny_jit_args, n), false);
    take(S_K, offsetof(dlp_sgemm_tiny_jit_args, k), false);
    take(S_A, offsetof(dlp_sgemm_tiny_jit_args, a), false);
    take(S_RSA, offsetof(dlp_sgemm_tiny_jit_args, rs_a), true);
    take(S_CSA, offsetof(dlp_sgemm_tiny_jit_args, cs_a), true);
    take(S_B, offsetof(dlp_sgemm_tiny_jit_args, b), false);
    take(S_RSB, offsetof(dlp_sgemm_tiny_jit_args, rs_b), true);
    take(S_CSB, offsetof(dlp_sgemm_tiny_jit_args, cs_b), true);
    take(S_C, offsetof(dlp_sgemm_tiny_jit_args, c), false);
    take(S_RSC, offsetof(dlp_sgemm_tiny_jit_args, rs_c), true);
    take(S_NC, offsetof(dlp_sgemm_tiny_jit_args, nc), false);

    movss(xmm0, ptr[args + offsetof(dlp_sgemm_tiny_jit_args, alpha)]);
    movss(ptr[rsp + kAlpha], xmm0);
    mov(eax, 0x3f800000);
    movd(xmm1, eax);
    ucomiss(xmm0, xmm1);
    /* Scale unless alpha is exactly 1.0. A NaN alpha still scales. */
    setp(al);
    setne(cl);
    or_(al, cl);
    movzx(rax, al);
    st(S_SCALE, rax);

    movss(xmm0, ptr[args + offsetof(dlp_sgemm_tiny_jit_args, beta)]);
    movss(ptr[rsp + kBeta], xmm0);
    xorps(xmm1, xmm1);
    ucomiss(xmm0, xmm1);
    /* Load C unless beta is exactly +0 or -0. A NaN beta still scales. */
    setp(al);
    setne(cl);
    or_(al, cl);
    movzx(rax, al);
    st(S_LOADC, rax);
}

/* mask = all-ones >> ((lanes - rem) & (lanes - 1)), rem in 0..lanes-1. */
void
TinyGen::emit_k1_from_rem(const Reg32& rem, int lanes)
{
    mov(ecx, lanes);
    sub(ecx, rem);
    and_(ecx, lanes - 1);
    mov(eax, lanes == 16 ? 0xFFFF : 0xFF);
    shr(eax, cl);
    kmovw(k1, eax);
}

/*
 * Independent accumulator sets for a small RV tile. Consecutive k steps use
 * different sets so the FMA chain is not latency bound. All sets together
 * must fit in the accumulator registers.
 */
int
rv_acc_sets(int tile, int capacity)
{
    int sets = 1;
    while ((sets < 4) && (tile * sets < 8) && (tile * sets * 2 <= capacity)) {
        sets *= 2;
    }
    return sets;
}

/*
 * One mr x (nv vectors) RV block. AVX-512 keeps accumulators in zmm0-23,
 * B in zmm24-27 and the A broadcast in zmm28. AVX2 uses ymm0-11, ymm12-13,
 * ymm14 and the tail mask in ymm15. `masked` loads and stores the last
 * vector under k1 (AVX-512) or ymm15 (AVX2).
 */
void
TinyGen::emit_rv_block(bool avx512, int mr, int nv, bool masked)
{
    const int vbytes   = avx512 ? 64 : 32;
    const int capacity = avx512 ? 24 : 12;
    const int b_reg    = capacity;
    const int a_reg    = capacity + (avx512 ? 4 : 2);
    const int tile     = mr * nv;
    const int sets     = rv_acc_sets(tile, capacity);

    auto vec = [avx512](int i) -> Xmm {
        return avx512 ? Xmm(Zmm(i)) : Xmm(Ymm(i));
    };
    auto load_last = [&](const Xmm& dst, const Address& src) {
        if (avx512) {
            vmovups(dst | k1 | T_z, src);
        } else {
            vmaskmovps(dst, Ymm(15), src);
        }
    };

    for (int i = 0; i < sets * tile; ++i) {
        vxorps(vec(i), vec(i), vec(i));
    }

    /* r12 = A + i * rs_a (rows 0-2), rdx = r12 + 3 * rs_a (rows 3-5). */
    ld(r11, S_RSA);
    ld(r12, S_I);
    imul(r12, r11);
    ld(rax, S_A);
    add(r12, rax);
    lea(rdx, ptr[r11 + r11 * 2]);
    add(rdx, r12);
    /* r13 = B + j. */
    ld(r13, S_J);
    shl(r13, 2);
    ld(rax, S_B);
    add(r13, rax);
    ld(r14, S_CSA);
    ld(r15, S_RSB);

    auto a_row = [&](int r) -> Address {
        const Reg64& base = (r < 3) ? r12 : rdx;
        const int    row  = r % 3;
        return (row == 0) ? ptr[base] : ptr[base + r11 * row];
    };
    auto step = [&](int set) {
        for (int v = 0; v < nv; ++v) {
            const Address src = ptr[r13 + v * vbytes];
            if (masked && (v == nv - 1)) {
                load_last(vec(b_reg + v), src);
            } else {
                vmovups(vec(b_reg + v), src);
            }
        }
        for (int r = 0; r < mr; ++r) {
            vbroadcastss(vec(a_reg), a_row(r));
            for (int v = 0; v < nv; ++v) {
                vfmadd231ps(vec(set * tile + r * nv + v), vec(a_reg),
                            vec(b_reg + v));
            }
        }
        add(r12, r14);
        if (mr > 3) {
            add(rdx, r14);
        }
        add(r13, r15);
    };

    /* k in steps of four, then the remainder. Both loops test at the bottom. */
    Label k_by4, k_rem, k_by1, k_done;
    ld(rbx, S_K);
    shr(rbx, 2);
    jz(k_rem, T_NEAR);
    L(k_by4);
    for (int u = 0; u < 4; ++u) {
        step(u % sets);
    }
    dec(rbx);
    jnz(k_by4, T_NEAR);
    L(k_rem);
    ld(rbx, S_K);
    and_(rbx, 3);
    jz(k_done, T_NEAR);
    L(k_by1);
    step(0);
    dec(rbx);
    jnz(k_by1, T_NEAR);
    L(k_done);

    for (int s = 1; s < sets; ++s) {
        for (int i = 0; i < tile; ++i) {
            vaddps(vec(i), vec(i), vec(s * tile + i));
        }
    }

    /* r8 = C + i * rs_c + j. */
    ld(r9, S_RSC);
    ld(r8, S_I);
    imul(r8, r9);
    ld(rax, S_J);
    lea(r8, ptr[r8 + rax * 4]);
    ld(rax, S_C);
    add(r8, rax);

    /* The B and A registers are free after the k loop. */
    const int alpha_reg = avx512 ? 29 : a_reg;
    const int beta_reg  = avx512 ? 30 : b_reg + 1;
    const int c_reg     = avx512 ? 31 : b_reg;

    Label no_scale, store;
    ld(rax, S_SCALE);
    test(rax, rax);
    jz(no_scale, T_NEAR);
    vbroadcastss(vec(alpha_reg), ptr[rsp + kAlpha]);
    for (int i = 0; i < tile; ++i) {
        vmulps(vec(i), vec(i), vec(alpha_reg));
    }
    L(no_scale);

    ld(rax, S_LOADC);
    test(rax, rax);
    jz(store, T_NEAR);
    vbroadcastss(vec(beta_reg), ptr[rsp + kBeta]);
    mov(r10, r8);
    for (int r = 0; r < mr; ++r) {
        for (int v = 0; v < nv; ++v) {
            const Address src = ptr[r10 + v * vbytes];
            if (masked && (v == nv - 1)) {
                load_last(vec(c_reg), src);
            } else {
                vmovups(vec(c_reg), src);
            }
            vfmadd231ps(vec(r * nv + v), vec(beta_reg), vec(c_reg));
        }
        add(r10, r9);
    }
    L(store);

    for (int r = 0; r < mr; ++r) {
        for (int v = 0; v < nv; ++v) {
            const Address dst = ptr[r8 + v * vbytes];
            if (masked && (v == nv - 1)) {
                if (avx512) {
                    vmovups(dst | k1, Zmm(r * nv + v));
                } else {
                    vmaskmovps(dst, Ymm(15), Ymm(r * nv + v));
                }
            } else {
                vmovups(dst, vec(r * nv + v));
            }
        }
        add(r8, r9);
    }
}

/*
 * The panels [S_JC, S_JE) of one mr-row strip. Full panels never use a mask;
 * only the last panel of the problem can be narrower.
 */
void
TinyGen::emit_rv_panels(bool avx512, int mr, const Label& after)
{
    const int lanes      = avx512 ? 16 : 8;
    const int lanes_log2 = avx512 ? 4 : 3;
    const int max_nv     = avx512 ? 4 : 2;
    const int panel      = lanes * max_nv;

    ld(rax, S_JC);
    st(S_J, rax);

    Label full, tail;
    L(full);
    ld(rdx, S_JE);
    ld(rax, S_J);
    sub(rdx, rax);
    cmp(rdx, panel);
    jl(tail, T_NEAR);
    emit_rv_block(avx512, mr, max_nv, false);
    ld(rax, S_J);
    add(rax, panel);
    st(S_J, rax);
    jmp(full, T_NEAR);

    /* rdx = nb = S_JE - S_J in 0..panel-1. */
    L(tail);
    test(rdx, rdx);
    jz(after, T_NEAR);
    mov(eax, edx);
    and_(eax, lanes - 1);
    if (avx512) {
        emit_k1_from_rem(eax, 16);
    } else {
        mov(r10, reinterpret_cast<uint64_t>(kMaskTbl));
        mov(ecx, 8);
        sub(ecx, eax);
        shl(ecx, 2);
        add(r10, rcx);
        vmovdqu(Ymm(15), ptr[r10]);
    }

    /* eax = 2 * vectors + (last vector partial). */
    mov(eax, edx);
    add(eax, lanes - 1);
    shr(eax, lanes_log2);
    add(eax, eax);
    xor_(ecx, ecx);
    test(edx, lanes - 1);
    setnz(cl);
    add(eax, ecx);
    for (int nv = 1; nv <= max_nv; ++nv) {
        for (int masked = 0; masked < 2; ++masked) {
            if ((nv == max_nv) && !masked) {
                continue;
            }
            Label next;
            cmp(eax, 2 * nv + masked);
            jne(next, T_NEAR);
            emit_rv_block(avx512, mr, nv, masked != 0);
            jmp(after, T_NEAR);
            L(next);
        }
    }
    jmp(after, T_NEAR);
}

/*
 * RV computes C in column blocks of nc (a multiple of the panel, chosen so
 * the block's B stays in L1). Inside a block it walks 6-row strips and then
 * panels of four zmm (64 columns) or two ymm (16 columns), so each strip
 * writes contiguous rows of C. A column-outer walk of a narrow panel would
 * touch one line per row at stride rs_c and thrash an L1 set when rs_c is a
 * multiple of 2 KiB.
 */
void
TinyGen::gen_rv(bool avx512)
{
    prologue();
    st_imm(S_JC, 0);

    Label jcloop, jcdone, jcnext;
    L(jcloop);
    ld(rax, S_JC);
    ld(rcx, S_N);
    cmp(rax, rcx);
    jge(jcdone, T_NEAR);
    /* S_JE = min(S_JC + nc, n). */
    ld(rdx, S_NC);
    add(rdx, rax);
    cmp(rdx, rcx);
    cmovg(rdx, rcx);
    st(S_JE, rdx);

    st_imm(S_I, 0);
    Label iloop;
    L(iloop);
    ld(rax, S_M);
    ld(rcx, S_I);
    sub(rax, rcx);
    jle(jcnext, T_NEAR);
    Label inext;
    for (int mr = 6; mr >= 1; --mr) {
        Label next;
        cmp(rax, mr);
        if (mr == 6) {
            jl(next, T_NEAR);
        } else {
            jne(next, T_NEAR);
        }
        emit_rv_panels(avx512, mr, inext);
        L(next);
    }
    L(inext);
    ld(rax, S_I);
    add(rax, 6);
    st(S_I, rax);
    jmp(iloop, T_NEAR);

    L(jcnext);
    ld(rax, S_JE);
    st(S_JC, rax);
    jmp(jcloop, T_NEAR);

    L(jcdone);
    epilogue();
}

void
TinyGen::gen_rv_avx512()
{
    gen_rv(true);
}

void
TinyGen::gen_rv_avx2()
{
    gen_rv(false);
}

void
TinyGen::emit_rd512_reduce_store(int mr, int nr)
{
    const int acc_stride = ((nr + 3) / 4) * 4;

    ld(rax, S_I);
    ld(r10, S_RSC);
    imul(rax, r10);
    ld(r10, S_J);
    shl(r10, 2);
    add(rax, r10);
    ld(r10, S_C);
    add(rax, r10);
    mov(r8, rax);
    ld(r9, S_RSC);

    ld(r12, S_SCALE);
    test(r12, r12);
    Label no_alpha;
    jz(no_alpha, T_NEAR);
    vbroadcastss(Xmm(28), ptr[rsp + kAlpha]);
    L(no_alpha);

    ld(r13, S_LOADC);
    test(r13, r13);
    Label no_beta;
    jz(no_beta, T_NEAR);
    vbroadcastss(Xmm(29), ptr[rsp + kBeta]);
    L(no_beta);

    for (int r = 0; r < mr; ++r) {
        /*
         * Four column sums per shuffle/add tree, on the full zmm accumulators
         * (at most zmm0-23) with zmm24-27 as temporaries. A fringe repeats
         * its last column; the masked store drops those lanes.
         */
        for (int c0 = 0; c0 < nr; c0 += 4) {
            const int cols = std::min(4, nr - c0);
            if (cols < 4) {
                mov(eax, (1 << cols) - 1);
                kmovw(k2, eax);
            }
            int src[4];
            for (int c = 0; c < 4; ++c) {
                src[c] = r * acc_stride + c0 + std::min(c, cols - 1);
            }
            vshufps(Zmm(24), Zmm(src[0]), Zmm(src[1]), 0x88);
            vshufps(Zmm(25), Zmm(src[0]), Zmm(src[1]), 0xDD);
            vaddps(Zmm(24), Zmm(24), Zmm(25));
            vshufps(Zmm(26), Zmm(src[2]), Zmm(src[3]), 0x88);
            vshufps(Zmm(27), Zmm(src[2]), Zmm(src[3]), 0xDD);
            vaddps(Zmm(26), Zmm(26), Zmm(27));
            vshufps(Zmm(25), Zmm(24), Zmm(26), 0x88);
            vshufps(Zmm(27), Zmm(24), Zmm(26), 0xDD);
            vaddps(Zmm(25), Zmm(25), Zmm(27));
            vextractf32x8(Ymm(24), Zmm(25), 1);
            vaddps(Ymm(24), Ymm(24), Ymm(25));
            vextractf32x4(Xmm(25), Ymm(24), 1);
            vaddps(Xmm(24), Xmm(25), Xmm(24));

            Label skip_scale;
            test(r12, r12);
            jz(skip_scale, T_NEAR);
            vmulps(Xmm(24), Xmm(24), Xmm(28));
            L(skip_scale);

            Label skip_beta;
            test(r13, r13);
            jz(skip_beta, T_NEAR);
            if (cols == 4) {
                vmovups(Xmm(27), ptr[r8 + c0 * sizeof(float)]);
            } else {
                vmovups(Xmm(27) | k2 | T_z, ptr[r8 + c0 * sizeof(float)]);
            }
            vfmadd231ps(Xmm(24), Xmm(29), Xmm(27));
            L(skip_beta);

            if (cols == 4) {
                vmovups(ptr[r8 + c0 * sizeof(float)], Xmm(24));
            } else {
                vmovups(ptr[r8 + c0 * sizeof(float)] | k2, Xmm(24));
            }
        }
        add(r8, r9);
    }
}

void
TinyGen::emit_rd512_block(int mr, int nr, bool stream_b)
{
    const int acc_stride = ((nr + 3) / 4) * 4;
    const int accum_regs = mr * acc_stride;
    const int a_reg      = stream_b ? accum_regs : accum_regs + nr;
    const int b_reg      = stream_b ? a_reg + mr : accum_regs;
    const int b_regs     = (b_reg + 1 < 32) ? 2 : 1;

    for (int r = 0; r < mr; ++r) {
        for (int c = 0; c < nr; ++c) {
            const int acc = r * acc_stride + c;
            vxorps(Zmm(acc), Zmm(acc), Zmm(acc));
        }
    }

    ld(rax, S_I);
    ld(r10, S_RSA);
    imul(rax, r10);
    ld(r10, S_A);
    add(rax, r10);
    mov(r12, rax);

    ld(rax, S_J);
    ld(r10, S_CSB);
    imul(rax, r10);
    ld(r10, S_B);
    add(rax, r10);
    mov(r13, rax);

    ld(r11, S_RSA);
    ld(r14, S_CSB);
    xor_(rbx, rbx);

    /* rcx = k rounded down to whole 16-lane steps. */
    Label full, full_done, tail_done;
    ld(rcx, S_K);
    and_(rcx, -16);
    jz(full_done, T_NEAR);
    L(full);

    if (stream_b) {
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r12);
        for (int r = 0; r < mr; ++r) {
            vmovups(Zmm(a_reg + r), ptr[r10]);
            add(r10, r11);
        }

        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r13);
        vmovups(Zmm(b_reg), ptr[r10]);
        add(r10, r14);
        for (int c = 0; c < nr; ++c) {
            const int current_b = b_reg + (c % b_regs);
            if (c + 1 < nr) {
                vmovups(Zmm(b_reg + ((c + 1) % b_regs)), ptr[r10]);
                add(r10, r14);
            }
            for (int r = 0; r < mr; ++r) {
                vfmadd231ps(Zmm(r * acc_stride + c), Zmm(a_reg + r),
                            Zmm(current_b));
            }
        }
    } else {
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r13);
        for (int c = 0; c < nr; ++c) {
            vmovups(Zmm(b_reg + c), ptr[r10]);
            add(r10, r14);
        }
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r12);
        for (int r = 0; r < mr; ++r) {
            vmovups(Zmm(a_reg), ptr[r10]);
            for (int c = 0; c < nr; ++c) {
                vfmadd231ps(Zmm(r * acc_stride + c), Zmm(a_reg),
                            Zmm(b_reg + c));
            }
            add(r10, r11);
        }
    }
    add(rbx, 16);
    cmp(rbx, rcx);
    jl(full, T_NEAR);
    L(full_done);

    ld(rax, S_K);
    cmp(rbx, rax);
    jge(tail_done, T_NEAR);

    if (stream_b) {
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r12);
        for (int r = 0; r < mr; ++r) {
            vmovups(Zmm(a_reg + r) | k1 | T_z, ptr[r10]);
            add(r10, r11);
        }

        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r13);
        vmovups(Zmm(b_reg) | k1 | T_z, ptr[r10]);
        add(r10, r14);
        for (int c = 0; c < nr; ++c) {
            const int current_b = b_reg + (c % b_regs);
            if (c + 1 < nr) {
                vmovups(Zmm(b_reg + ((c + 1) % b_regs)) | k1 | T_z, ptr[r10]);
                add(r10, r14);
            }
            for (int r = 0; r < mr; ++r) {
                vfmadd231ps(Zmm(r * acc_stride + c), Zmm(a_reg + r),
                            Zmm(current_b));
            }
        }
    } else {
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r13);
        for (int c = 0; c < nr; ++c) {
            vmovups(Zmm(b_reg + c) | k1 | T_z, ptr[r10]);
            add(r10, r14);
        }
        mov(r10, rbx);
        shl(r10, 2);
        add(r10, r12);
        for (int r = 0; r < mr; ++r) {
            vmovups(Zmm(a_reg) | k1 | T_z, ptr[r10]);
            for (int c = 0; c < nr; ++c) {
                vfmadd231ps(Zmm(r * acc_stride + c), Zmm(a_reg),
                            Zmm(b_reg + c));
            }
            add(r10, r11);
        }
    }
    L(tail_done);

    emit_rd512_reduce_store(mr, nr);
}

void
TinyGen::emit_rd512_rows(int mr, bool stream_b, const Label& after)
{
    const int accum_regs = stream_b ? 24 : 28;
    const int column_block =
        std::max(4, (accum_regs / (stream_b ? mr : mr + 1) / 4) * 4);

    st_imm(S_J, 0);
    Label loop, fringe;
    L(loop);
    ld(rax, S_J);
    add(rax, column_block);
    ld(rcx, S_N);
    cmp(rax, rcx);
    jg(fringe, T_NEAR);
    emit_rd512_block(mr, column_block, stream_b);
    ld(rax, S_J);
    add(rax, column_block);
    st(S_J, rax);
    jmp(loop, T_NEAR);

    L(fringe);
    if (stream_b && (column_block > 8)) {
        Label eight_loop, less_than_eight;
        L(eight_loop);
        ld(rax, S_J);
        add(rax, 8);
        ld(rcx, S_N);
        cmp(rax, rcx);
        jg(less_than_eight, T_NEAR);
        emit_rd512_block(mr, 8, stream_b);
        ld(rax, S_J);
        add(rax, 8);
        st(S_J, rax);
        jmp(eight_loop, T_NEAR);
        L(less_than_eight);
    }
    if (column_block > 4) {
        Label four_loop, less_than_four;
        L(four_loop);
        ld(rax, S_J);
        add(rax, 4);
        ld(rcx, S_N);
        cmp(rax, rcx);
        jg(less_than_four, T_NEAR);
        emit_rd512_block(mr, 4, stream_b);
        ld(rax, S_J);
        add(rax, 4);
        st(S_J, rax);
        jmp(four_loop, T_NEAR);
        L(less_than_four);
    }
    ld(rax, S_N);
    ld(rcx, S_J);
    sub(rax, rcx);
    for (int nr = 3; nr >= 1; --nr) {
        Label next;
        cmp(rax, nr);
        jne(next, T_NEAR);
        emit_rd512_block(mr, nr, stream_b);
        jmp(after, T_NEAR);
        L(next);
    }
    jmp(after, T_NEAR);
}

void
TinyGen::gen_rd_avx512()
{
    gen_rd_avx512_impl(false);
}

void
TinyGen::gen_rd_avx512_wide()
{
    gen_rd_avx512_impl(true);
}

void
TinyGen::gen_rd_avx512_impl(bool stream_b)
{
    prologue();

    /* k1 is preserved by every RD block, so compute its tail mask once. */
    ld(rax, S_K);
    and_(eax, 15);
    emit_k1_from_rem(eax, 16);

    st_imm(S_I, 0);
    Label loop, fringe, done;
    L(loop);
    ld(rax, S_I);
    add(rax, 6);
    ld(rcx, S_M);
    cmp(rax, rcx);
    jg(fringe, T_NEAR);
    {
        Label row_done;
        emit_rd512_rows(6, stream_b, row_done);
        L(row_done);
    }
    ld(rax, S_I);
    add(rax, 6);
    st(S_I, rax);
    jmp(loop, T_NEAR);

    L(fringe);
    ld(rax, S_M);
    ld(rcx, S_I);
    sub(rax, rcx);
    for (int mr = 5; mr >= 1; --mr) {
        Label next;
        cmp(rax, mr);
        jne(next, T_NEAR);
        /* rows() jumps to `after` between column blocks. Use a local. */
        Label row_done;
        emit_rd512_rows(mr, stream_b, row_done);
        L(row_done);
        jmp(done, T_NEAR);
        L(next);
    }
    L(done);
    epilogue();
}

void
TinyGen::emit_rd2_step(int mr, int nr, bool tail)
{
    const int acc_stride = ((nr + 3) / 4) * 4;

    /*
     * Every register holds an accumulator or an operand, so the k tail loads
     * the lane mask into the destination and masks the load with itself.
     */
    auto load = [&](const Ymm& dst, const Address& src) {
        if (tail) {
            vmovdqu(dst, ptr[rax]);
            vmaskmovps(dst, dst, src);
        } else {
            vmovups(dst, src);
        }
    };
    if (tail) {
        mov(rax, ptr[rsp + kKmask]);
    }

    mov(r10, rbx);
    shl(r10, 2);
    add(r10, r12);
    for (int r = 0; r < mr; ++r) {
        load(Ymm(12 + r), ptr[r10]);
        add(r10, r11);
    }

    mov(r10, rbx);
    shl(r10, 2);
    add(r10, r13);
    for (int c = 0; c < nr; ++c) {
        load(Ymm(15), ptr[r10]);
        for (int r = 0; r < mr; ++r) {
            vfmadd231ps(Ymm(r * acc_stride + c), Ymm(12 + r), Ymm(15));
        }
        add(r10, r14);
    }
}

void
TinyGen::emit_rd2_reduce_store(int mr, int nr)
{
    const int acc_stride = ((nr + 3) / 4) * 4;

    ld(rax, S_I);
    ld(r10, S_RSC);
    imul(rax, r10);
    ld(r10, S_J);
    shl(r10, 2);
    add(rax, r10);
    ld(r10, S_C);
    add(rax, r10);
    mov(r8, rax);
    ld(r9, S_RSC);

    ld(r12, S_SCALE);
    test(r12, r12);
    Label no_alpha;
    jz(no_alpha, T_NEAR);
    vbroadcastss(Xmm(13), ptr[rsp + kAlpha]);
    L(no_alpha);
    ld(r13, S_LOADC);
    test(r13, r13);
    Label no_beta;
    jz(no_beta, T_NEAR);
    vbroadcastss(Xmm(14), ptr[rsp + kBeta]);
    L(no_beta);

    for (int r = 0; r < mr; ++r) {
        for (int c0 = 0; c0 < nr; c0 += 4) {
            const int cols = std::min(4, nr - c0);
            const int base = r * acc_stride + c0;
            if (cols < 4) {
                mov(r10, reinterpret_cast<uint64_t>(kMaskTbl + 8 - cols));
                vmovdqu(Xmm(15), ptr[r10]);
                for (int c = cols; c < 4; ++c) {
                    vxorps(Ymm(base + c), Ymm(base + c), Ymm(base + c));
                }
            }
            /*
             * Reduce four columns in place with one shuffle/add tree. Other
             * accumulators stay live; ymm12 held A, which is dead here.
             */
            const Ymm t(12);
            vshufps(t, Ymm(base), Ymm(base + 1), 0x88);
            vshufps(Ymm(base + 1), Ymm(base), Ymm(base + 1), 0xDD);
            vaddps(Ymm(base), t, Ymm(base + 1));
            vshufps(t, Ymm(base + 2), Ymm(base + 3), 0x88);
            vshufps(Ymm(base + 3), Ymm(base + 2), Ymm(base + 3), 0xDD);
            vaddps(Ymm(base + 2), t, Ymm(base + 3));
            vshufps(t, Ymm(base), Ymm(base + 2), 0x88);
            vshufps(Ymm(base + 2), Ymm(base), Ymm(base + 2), 0xDD);
            vaddps(Ymm(base), t, Ymm(base + 2));
            vextractf128(Xmm(base + 1), Ymm(base), 1);
            vaddps(Xmm(base), Xmm(base + 1), Xmm(base));

            Label skip_scale;
            test(r12, r12);
            jz(skip_scale, T_NEAR);
            vmulps(Xmm(base), Xmm(base), Xmm(13));
            L(skip_scale);

            Label skip_beta;
            test(r13, r13);
            jz(skip_beta, T_NEAR);
            if (cols == 4) {
                vmovups(Xmm(12), ptr[r8 + c0 * sizeof(float)]);
            } else {
                vmaskmovps(Xmm(12), Xmm(15), ptr[r8 + c0 * sizeof(float)]);
            }
            vfmadd231ps(Xmm(base), Xmm(14), Xmm(12));
            L(skip_beta);

            if (cols == 4) {
                vmovups(ptr[r8 + c0 * sizeof(float)], Xmm(base));
            } else {
                vmaskmovps(ptr[r8 + c0 * sizeof(float)], Xmm(15), Xmm(base));
            }
        }
        add(r8, r9);
    }
}

void
TinyGen::emit_rd2_block(int mr, int nr)
{
    const int acc_stride = ((nr + 3) / 4) * 4;
    for (int r = 0; r < mr; ++r) {
        for (int c = 0; c < nr; ++c) {
            const int acc = r * acc_stride + c;
            vxorps(Ymm(acc), Ymm(acc), Ymm(acc));
        }
    }

    ld(rax, S_I);
    ld(r10, S_RSA);
    imul(rax, r10);
    ld(r10, S_A);
    add(rax, r10);
    mov(r12, rax);

    ld(rax, S_J);
    ld(r10, S_CSB);
    imul(rax, r10);
    ld(r10, S_B);
    add(rax, r10);
    mov(r13, rax);

    ld(r11, S_RSA);
    ld(r14, S_CSB);
    xor_(rbx, rbx);

    /* rcx = k rounded down to whole 8-lane steps. */
    Label full, full_done, tail_done;
    ld(rcx, S_K);
    and_(rcx, -8);
    jz(full_done, T_NEAR);
    L(full);
    emit_rd2_step(mr, nr, false);
    add(rbx, 8);
    cmp(rbx, rcx);
    jl(full, T_NEAR);
    L(full_done);

    ld(rax, S_K);
    cmp(rbx, rax);
    jge(tail_done, T_NEAR);
    emit_rd2_step(mr, nr, true);
    L(tail_done);

    emit_rd2_reduce_store(mr, nr);
}

void
TinyGen::emit_rd2_rows(int mr, const Label& after)
{
    // Accumulators, one A vector per row and one streamed B vector share the
    // sixteen-register file. Use every complete four-column store group that
    // fits; AVX2 therefore widens only where its register budget permits.
    constexpr int compute_regs = 16;
    int           column_block = (compute_regs - 1 - mr) / mr;
    column_block               = std::max(4, (column_block / 4) * 4);

    st_imm(S_J, 0);
    Label loop, fringe;
    L(loop);
    ld(rax, S_J);
    add(rax, column_block);
    ld(rcx, S_N);
    cmp(rax, rcx);
    jg(fringe, T_NEAR);
    emit_rd2_block(mr, column_block);
    ld(rax, S_J);
    add(rax, column_block);
    st(S_J, rax);
    jmp(loop, T_NEAR);

    L(fringe);
    if (column_block > 4) {
        Label four_loop, short_fringe;
        L(four_loop);
        ld(rax, S_J);
        add(rax, 4);
        ld(rcx, S_N);
        cmp(rax, rcx);
        jg(short_fringe, T_NEAR);
        emit_rd2_block(mr, 4);
        ld(rax, S_J);
        add(rax, 4);
        st(S_J, rax);
        jmp(four_loop, T_NEAR);
        L(short_fringe);
    }
    ld(rax, S_N);
    ld(rcx, S_J);
    sub(rax, rcx);
    for (int nr = 3; nr >= 1; --nr) {
        Label next;
        cmp(rax, nr);
        jne(next, T_NEAR);
        emit_rd2_block(mr, nr);
        jmp(after, T_NEAR);
        L(next);
    }
    jmp(after, T_NEAR);
}

void
TinyGen::gen_rd_avx2()
{
    prologue();

    /* Pointer to the 8-lane k tail mask. Unused when k is a multiple of 8. */
    mov(r10, reinterpret_cast<uint64_t>(kMaskTbl));
    ld(rax, S_K);
    and_(rax, 7);
    mov(ecx, 8);
    sub(rcx, rax);
    shl(rcx, 2);
    add(r10, rcx);
    mov(ptr[rsp + kKmask], r10);

    st_imm(S_I, 0);
    Label loop, fringe, done;
    L(loop);
    ld(rax, S_I);
    add(rax, 3);
    ld(rcx, S_M);
    cmp(rax, rcx);
    jg(fringe, T_NEAR);
    {
        Label row_done;
        emit_rd2_rows(3, row_done);
        L(row_done);
    }
    ld(rax, S_I);
    add(rax, 3);
    st(S_I, rax);
    jmp(loop, T_NEAR);

    L(fringe);
    ld(rax, S_M);
    ld(rcx, S_I);
    sub(rax, rcx);
    for (int mr = 2; mr >= 1; --mr) {
        Label next, row_done;
        cmp(rax, mr);
        jne(next, T_NEAR);
        emit_rd2_rows(mr, row_done);
        L(row_done);
        jmp(done, T_NEAR);
        L(next);
    }
    L(done);
    epilogue();
}

struct Held
{
    std::unique_ptr<TinyGen>           gen;
    std::atomic<dlp_sgemm_tiny_jit_fn> fn{ nullptr };
    std::once_flag                     once;
};

Held g_rv512{};
Held g_rd512{};
Held g_rd512_wide{};
Held g_rv2{};
Held g_rd2{};

dlp_sgemm_tiny_jit_fn
get_or_generate(void (TinyGen::*emit)(), Held& held)
{
    if (const auto fn = held.fn.load(std::memory_order_acquire)) {
        return fn;
    }
    /*
     * A throwing call_once callable leaves the flag unset and would retry
     * generation on every call. Catch inside so a failure is final and later
     * calls fall back to the framework at the cost of a load.
     */
    std::call_once(held.once, [&held, emit]() {
        try {
            auto generator = std::make_unique<TinyGen>();
            (generator.get()->*emit)();
            generator->ready();
            const auto fn =
                reinterpret_cast<dlp_sgemm_tiny_jit_fn>(const_cast<void*>(
                    static_cast<const void*>(generator->getCode())));
            held.gen = std::move(generator);
            held.fn.store(fn, std::memory_order_release);
        } catch (...) {
        }
    });
    return held.fn.load(std::memory_order_acquire);
}

} // namespace

extern "C"
{

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rv_avx512(void)
    {
        return get_or_generate(&TinyGen::gen_rv_avx512, g_rv512);
    }

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx512(void)
    {
        return get_or_generate(&TinyGen::gen_rd_avx512, g_rd512);
    }

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx512_wide(void)
    {
        return get_or_generate(&TinyGen::gen_rd_avx512_wide, g_rd512_wide);
    }

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rv_avx2(void)
    {
        return get_or_generate(&TinyGen::gen_rv_avx2, g_rv2);
    }

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx2(void)
    {
        return get_or_generate(&TinyGen::gen_rd_avx2, g_rd2);
    }

} // extern "C"
