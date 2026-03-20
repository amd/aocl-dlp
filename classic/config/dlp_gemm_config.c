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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#include <stdlib.h>

#include "classic/aocl_gemm_post_ops.h"
#include "classic/aocl_lib_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_blksz_map.h"
#include "dlp_gemm_func_map.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/bf16bf16f32/dlp_gemm_pack_bf16.h"
#include "kernels/dlp_gemm_eltwise_ops_kernels.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/dlp_gemm_utils_kernels.h"
#include "kernels/f32f32f32/dlp_gemm_pack_f32.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"
#include "kernels/s8s8s32/dlp_gemm_packa_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"
#include "kernels/s8s8s32/dlp_gemm_quanta_s8.h"
#include "kernels/u8s8s32/dlp_gemm_packa.h"
#include "kernels/u8s8s32/dlp_gemm_packb.h"
#include "logging/dlp_gemm_logger.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

static dlp_gemm_cntx_t global_cntx_t_list[AOCL_DLP_OPERATION_TYPE_LEN]
    __attribute__((aligned(64))); // Only one op type supported now.
static dlp_gemm_util_cntx_t
    global_util_cntx_t_list[AOCL_DLP_UTIL_OPERATION_TYPE_LEN]
    __attribute__((aligned(64))); // Only post-ops like utils.
static dlp_gemm_eltwise_ops_cntx_t
    global_eltwise_ops_cntx_t_list[AOCL_DLP_ELTWISE_OPS_OPERATION_TYPE_LEN]
    __attribute__((aligned(64))); // Post-ops only utils without gemm.

static dlp_arch_t       global_dlp_gemmenable_arch       = DLP_ARCH_ERROR;
static dlp_instr_pref_t global_dlp_gemmenable_instr_pref = DLP_INSTR_PREF_NONE;

#ifdef DLP_GEMM_BF16_JIT
// This bool indicates whether JIT kernel generation has been successful.
static bool jit_kernels_generated = FALSE;
bool
get_jit_kernels_generated()
{
    return jit_kernels_generated;
}
#endif

// This array is to store function pointers to jit generated kernels.
static void* global_jit_kernels[DLP_GEMM_BF16_MR]
                               [(DLP_GEMM_BF16_NR / NUM_F32_ELEMS_PER_ZMM) + 1]
    __attribute__((aligned(64)));

// Buffer size is chosen in order to accommodate the
// worst-case scenario for MR=6 and NR=64.
// The buffersize is chosen using bruteforce method.
#define DLP_JIT_REQ_OS_PAGE_SIZE 4096
#define JIT_KERNEL_SIZE          (14 * DLP_JIT_REQ_OS_PAGE_SIZE)

#ifdef DUMP_JIT_CODE
// Funtion to Dump JIT generated kernel
void
dump_jit_code(
    const void* code, int code_size, const char* code_name, int m, int n)
{
    if (code) {
        static int counter = 0;
#define MAX_FNAME_LEN 256
        char fname[MAX_FNAME_LEN + 1];
        // TODO (Roma): support prefix for code / linux perf dumps
        snprintf(fname, MAX_FNAME_LEN, "dnnl_dump_cpu_%s_%dx%d.%d.bin",
                 code_name, m, n, counter);
        counter++;
        FILE* fp = fopen(fname, "wb+");
        // Failure to dump code is not fatal
        if (fp) {
            int unused = fwrite(code, code_size, 1, fp);
            // UNUSED(unused);
            fclose(fp);
        }
    }
#undef MAX_FNAME_LEN
}
#endif

static dlp_pthread_once_t once_check_dlp_gemm_func_map_init =
    DLP_PTHREAD_ONCE_INIT;

static void
_dlp_gemm_init_enable_arch()
{
    dlp_arch_t arch_id    = dlp_get_arch();
    bool       enbl_instr = dlp_aocl_enable_instruction_query();

    if ((enbl_instr == TRUE)
        && ((arch_id == DLP_ARCH_ZEN3) || (arch_id == DLP_ARCH_ZEN2)
            || (arch_id == DLP_ARCH_ZEN))) {
        global_dlp_gemmenable_arch = DLP_ARCH_ZEN3;
    } else {
        global_dlp_gemmenable_arch = arch_id;
    }
}

static void
_dlp_gemm_init_enable_instr_pref()
{
    bool enbl_instr = dlp_aocl_enable_instruction_query();

    if (enbl_instr == TRUE) {
        global_dlp_gemmenable_instr_pref =
            dlp_env_get_kernel_instr_pref("AOCL_DLP_ENABLE_INSTRUCTIONS");
    }
}

dlp_arch_t
dlp_gemm_get_enabled_arch()
{
    return global_dlp_gemmenable_arch;
}

static void
_dlp_gemm_util_cntx_init_func_map()
{
#define UMACRO(ID, FUNC_PTR)                                                   \
    global_util_cntx_t_list[ID].kern_fun_ptr = FUNC_PTR;

    global_util_cntx_t_list[F32_GELU_TANH].kern_fun_ptr = NULL;
    global_util_cntx_t_list[F32_GELU_ERF].kern_fun_ptr  = NULL;
    global_util_cntx_t_list[F32_SOFTMAX].kern_fun_ptr   = NULL;

    // Kernel dispatch object factory.
    if (dlp_cpuid_is_avx512bf16_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_UTIL_KERN_FUNC_MAP_AVX512_VNNI_BF16
#endif
    } else if (dlp_cpuid_is_avx512vnni_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_UTIL_KERN_FUNC_MAP_AVX512_VNNI
#endif
    } else if (dlp_cpuid_is_avx2fma3_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN3
        DLP_GEMM_UTIL_KERN_FUNC_MAP_AVX2
#endif
    }

#undef UMACRO
}

static void
_dlp_gemm_eltwise_ops_cntx_init_func_map()
{
#define POMACRO(ID, FUNC_PTR)                                                  \
    global_eltwise_ops_cntx_t_list[ID].eltwise_ops_kern_fun_ptr = FUNC_PTR;

    global_eltwise_ops_cntx_t_list[BF16OF32].eltwise_ops_kern_fun_ptr = NULL;

    // Kernel dispatch object factory.
    if (dlp_cpuid_is_avx512bf16_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_ELTWISE_OPS_KERN_FUNC_MAP_AVX512_VNNI_BF16
#endif
    }

#undef POMACRO
}

// Using #define instead of function since the underlying macros the DLP_GEMM_*
// expands into is defined inside _dlp_gemm_cntx_init_func_map.
#define _DLP_GEMM_CNTX_UPD_FUNC_MAP_FOR_CONFIGURED_ARCH()                      \
    if (global_dlp_gemmenable_arch == DLP_ARCH_ZEN3) {                         \
        DLP_GEMM_KERN_FUNC_UPD_MAP_AVX512_VNNI_BF16_TO_AVX2;                   \
        DLP_GEMM_PACKA_FUNC_UPD_MAP_AVX512_VNNI_BF16_TO_AVX2;                  \
        DLP_GEMM_PACKB_FUNC_UPD_MAP_AVX512_VNNI_BF16_TO_AVX2;                  \
    } else if (((global_dlp_gemmenable_arch == DLP_ARCH_ZEN5)                  \
                || (global_dlp_gemmenable_arch == DLP_ARCH_ZEN4))              \
               && (global_dlp_gemmenable_instr_pref                            \
                   == DLP_INSTR_PREF_AVX512_YMM_FAVOUR)) {                     \
        DLP_GEMM_KERN_FUNC_UPD_MAP_AVX512_VNNI_TO_AVX512_256;                  \
        DLP_GEMM_PACKA_FUNC_UPD_MAP_AVX512_VNNI_TO_AVX512_256;                 \
        DLP_GEMM_PACKB_FUNC_UPD_MAP_AVX512_VNNI_TO_AVX512_256;                 \
    }

static void
_dlp_gemm_cntx_init_func_map()
{
#define KMACRO(ID, FUNC_PTR)  global_cntx_t_list[ID].kern_fun_ptr = FUNC_PTR;
#define PAMACRO(ID, FUNC_PTR) global_cntx_t_list[ID].packa_fun_ptr = FUNC_PTR;
#define PBMACRO(ID, FUNC_PTR) global_cntx_t_list[ID].packb_fun_ptr = FUNC_PTR;
#define PBMXPMACRO(ID, FUNC_PTR)                                               \
    global_cntx_t_list[ID].packb_mxp_fun_ptr = FUNC_PTR;
#define UBMACRO(ID, FUNC_PTR) global_cntx_t_list[ID].unpackb_fun_ptr = FUNC_PTR;
#define PBSMACRO(ID, FUNC_PTR)                                                 \
    global_cntx_t_list[ID].packsclb_fun_ptr = FUNC_PTR;
#define JITMACRO(ID, FUNC_PTR) global_cntx_t_list[ID].jit_kernel = FUNC_PTR;
    // TODO: Default initialize with reference kernels so that kernel pointer
    //  will be valid even in case none of the zen optimized kernels are
    //  available. This scenario could happen if the addon was built using
    //  a different arch config (eg: skx).

    global_cntx_t_list[U8S8S16OS16].kern_fun_ptr     = NULL;
    global_cntx_t_list[U8S8S32OS32].kern_fun_ptr     = NULL;
    global_cntx_t_list[F32F32F32OF32].kern_fun_ptr   = NULL;
    global_cntx_t_list[BF16BF16F32OF32].kern_fun_ptr = NULL;
    global_cntx_t_list[BF16S4F32OF32].kern_fun_ptr   = NULL;
    global_cntx_t_list[F32OBF16].kern_fun_ptr        = NULL;
    global_cntx_t_list[BF16U4F32OF32].kern_fun_ptr   = NULL;

    // Kernel dispatch object factory.
    if (dlp_cpuid_is_avx512bf16_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_KERN_FUNC_MAP_AVX512_VNNI_BF16
        DLP_GEMM_PACKA_FUNC_MAP_AVX512_VNNI_BF16
        DLP_GEMM_PACKB_FUNC_MAP_AVX512_VNNI_BF16
        DLP_GEMM_PACKBMXP_FUNC_MAP_AVX512_VNNI_BF16
        DLP_GEMM_UNPACKB_FUNC_MAP_AVX512_VNNI_BF16
        DLP_GEMM_PACKSCLB_FUNC_MAP_AVX512_VNNI_BF16

#ifdef DLP_GEMM_BF16_JIT
        dlp_gemm_jit_inputs_t inputs;
        inputs.alpha_scale = TRUE;
        inputs.beta_scale  = DLP_BETA_GEN;

        dlp_clsc_err_t err;

        md_t num_N_vars = (DLP_GEMM_BF16_NR / NUM_F32_ELEMS_PER_ZMM) + 1;

        jit_kernels_generated = TRUE;
        for (iter_t m = 0; m < DLP_GEMM_BF16_MR; m++) {
            for (iter_t n = 0; n < num_N_vars; n++) {
                inputs.MR            = (m == 0) ? DLP_GEMM_BF16_MR : m;
                inputs.NR            = n * 16;
                inputs.m_loop        = (m == 0) ? TRUE : FALSE;
                inputs.generate_mask = (n == 0) ? TRUE : FALSE;
                global_jit_kernels[m][n] =
                    dlp_malloc_page_aligned(JIT_KERNEL_SIZE, &err);
                if (global_jit_kernels[m][n] != NULL) {
                    get_jit_kernel(&inputs, global_jit_kernels[m][n],
                                   JIT_KERNEL_SIZE);
#ifdef DUMP_JIT_CODE
                    dump_jit_code(global_jit_kernels[m][n], JIT_KERNEL_SIZE,
                                  "dlp_gemm", inputs.MR, inputs.NR);
#endif
                } else {
                    jit_kernels_generated = FALSE;
                }
            }
        }

#endif

        // If arch is updated at runtime, it is expeceted to be honoured.
        _DLP_GEMM_CNTX_UPD_FUNC_MAP_FOR_CONFIGURED_ARCH()
#endif
    } else if (dlp_cpuid_is_avx512vnni_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_KERN_FUNC_MAP_AVX512_VNNI
        DLP_GEMM_PACKA_FUNC_MAP_AVX512_VNNI
        DLP_GEMM_PACKB_FUNC_MAP_AVX512_VNNI
        DLP_GEMM_PACKBMXP_FUNC_MAP_AVX512_VNNI

        _DLP_GEMM_CNTX_UPD_FUNC_MAP_FOR_CONFIGURED_ARCH()
#endif
    } else if (dlp_cpuid_is_avx2fma3_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN3
        DLP_GEMM_KERN_FUNC_MAP_AVX2
        DLP_GEMM_PACKA_FUNC_MAP_AVX2
        DLP_GEMM_PACKB_FUNC_MAP_AVX2
#endif
    }

    // FP16 is an independent ISA feature; gate its function pointers
    // with the FP16-specific check rather than the BF16 check.
    if (dlp_cpuid_is_avx512fp16_supported() == TRUE) {
#ifdef DLP_KERNELS_ZEN4
        DLP_GEMM_KERN_FUNC_MAP_AVX512_FP16
        DLP_GEMM_PACKA_FUNC_MAP_AVX512_FP16
        DLP_GEMM_PACKB_FUNC_MAP_AVX512_FP16
        DLP_GEMM_UNPACKB_FUNC_MAP_AVX512_FP16
#endif
    }

    // If built with a config not supporting zen3/zen4/amdzen, error out
    // since reference kernels are not available.
    if (global_cntx_t_list[F32F32F32OF32].kern_fun_ptr == NULL) {
        dlp_print_msg("AOCL_DLP_GEMM is not compiled using correct Zen config."
                      " Compile using zen3/zen4/amdzen config.",
                      __FILE__, __LINE__);
        abort();
    }

#undef PBMACRO
#undef PBMXPMACRO
#undef PAMACRO
#undef KMACRO
}

void
dlp_gemm_set_jit_kernel(void* kernel_fp, md_t m_index, md_t n_index)
{
    global_jit_kernels[m_index][n_index] = kernel_fp;
}

void*
dlp_gemm_get_jit_kernel(md_t m_index, md_t n_index)
{
    return global_jit_kernels[m_index][n_index];
}

DLP_INLINE void
dlp_gemm_set_block_sizes_global_cntx(AOCL_DLP_OPERATION_TYPE op_type,
                                     md_t                    MC,
                                     md_t                    NC,
                                     md_t                    KC,
                                     md_t                    MR,
                                     md_t                    NR)
{
    global_cntx_t_list[op_type].blksz.MC = MC;
    global_cntx_t_list[op_type].blksz.NC = NC;
    global_cntx_t_list[op_type].blksz.KC = KC;
    global_cntx_t_list[op_type].blksz.MR = MR;
    global_cntx_t_list[op_type].blksz.NR = NR;
}

DLP_INLINE void
dlp_gemm_set_pack_strides_global_cntx(AOCL_DLP_OPERATION_TYPE op_type,
                                      md_t                    packa_rs,
                                      md_t                    packa_cs,
                                      md_t                    packb_rs,
                                      md_t                    packb_cs)
{
    global_cntx_t_list[op_type].pack_s.packa_rs = packa_rs;
    global_cntx_t_list[op_type].pack_s.packa_cs = packa_cs;
    global_cntx_t_list[op_type].pack_s.packb_rs = packb_rs;
    global_cntx_t_list[op_type].pack_s.packb_cs = packb_cs;
}

static void
_dlp_gemm_cntx_init_blksz_map()
{
#define XMACRO(ID, MC, NC, KC, MR, NR, PACKA_RS, PACKA_CS, PACKB_RS, PACKB_CS) \
    dlp_gemm_set_block_sizes_global_cntx(ID, MC, NC, KC, MR, NR);              \
    dlp_gemm_set_pack_strides_global_cntx(ID, PACKA_RS, PACKA_CS, PACKB_RS,    \
                                          PACKB_CS);

    // Ideally the blocksize needs to be set based on arch id. However
    // since this code is also expected to work on other vendor machines,
    // the blocksize for a particular version of zen id is generalized
    // for all machines that support the ISA supported by that particular
    // zen id.
    if (dlp_cpuid_is_avx512vnni_supported() == TRUE) {
        DLP_GEMM_BLKSZ_MAP_ZEN4

        if (global_dlp_gemmenable_arch == DLP_ARCH_ZEN3) {
            DLP_GEMM_BLKSZ_UPD_MAP_ZEN4_TO_ZEN
        }
    } else if (dlp_cpuid_is_avx2fma3_supported() == TRUE) {
        DLP_GEMM_BLKSZ_MAP_ZEN
    } else {
        DLP_GEMM_BLKSZ_MAP_ZEN
    }

#undef XMACRO
}

DLP_INLINE void
dlp_gemm_set_sup_thres_global_cntx(AOCL_DLP_OPERATION_TYPE op_type,
                                   md_t                    MT,
                                   md_t                    NT,
                                   md_t                    KT)
{
    global_cntx_t_list[op_type].sup_thres.MT = MT;
    global_cntx_t_list[op_type].sup_thres.NT = NT;
    global_cntx_t_list[op_type].sup_thres.KT = KT;
}

static void
_dlp_gemm_cntx_init_sup_thres_map()
{
#define STMACRO(ID, MT, NT, KT)                                                \
    dlp_gemm_set_sup_thres_global_cntx(ID, MT, NT, KT);

    if (dlp_cpuid_is_avx512vnni_supported() == TRUE) {
        DLP_GEMM_SUP_THRES_MAP_ZEN4

        if (global_dlp_gemmenable_arch == DLP_ARCH_ZEN3) {
            DLP_GEMM_SUP_THRES_UPD_MAP_ZEN4_TO_ZEN
        }
    } else if (dlp_cpuid_is_avx2fma3_supported() == TRUE) {
        DLP_GEMM_SUP_THRES_MAP_ZEN
    } else {
        DLP_GEMM_SUP_THRES_MAP_ZEN
    }

#undef STMACRO
}

DLP_INLINE void
dlp_gemm_set_block_sizes_global_eltwise_ops_cntx(
    AOCL_DLP_ELTWISE_OPS_OPERATION_TYPE op_type,
    md_t                                MC,
    md_t                                NC,
    md_t                                KC,
    md_t                                MR,
    md_t                                NR)
{
    global_eltwise_ops_cntx_t_list[op_type].blksz.MC = MC;
    global_eltwise_ops_cntx_t_list[op_type].blksz.NC = NC;
    global_eltwise_ops_cntx_t_list[op_type].blksz.KC = KC;
    global_eltwise_ops_cntx_t_list[op_type].blksz.MR = MR;
    global_eltwise_ops_cntx_t_list[op_type].blksz.NR = NR;
}

static void
_dlp_gemm_eltwise_ops_cntx_init_blksz_map()
{
#define XMACRO(ID, MC, NC, KC, MR, NR)                                         \
    dlp_gemm_set_block_sizes_global_eltwise_ops_cntx(ID, MC, NC, KC, MR, NR);

    // Ideally the blocksize needs to be set based on arch id. However
    // since this code is also expected to work on other vendor machines,
    // the blocksize for a particular version of zen id is generalized
    // for all machines that support the ISA supported by that particular
    // zen id.
    if (dlp_cpuid_is_avx512bf16_supported() == TRUE) {
        DLP_GEMM_ELTWISE_OPS_BLKSZ_MAP_ZEN4
    } else {
        DLP_GEMM_ELTWISE_OPS_BLKSZ_MAP_ZEN
    }

#undef XMACRO
}

static void
dlp_gemm_cntx_init_map()
{
    _dlp_gemm_init_enable_arch();
    _dlp_gemm_init_enable_instr_pref();
    _dlp_gemm_cntx_init_func_map();
    _dlp_gemm_cntx_init_blksz_map();
    _dlp_gemm_cntx_init_sup_thres_map();
    _dlp_gemm_eltwise_ops_cntx_init_blksz_map();
    _dlp_gemm_eltwise_ops_cntx_init_func_map();
    _dlp_gemm_util_cntx_init_func_map();
}

// Set default block sizes for dlp_gemm.
// Detect thread topology for dlp_gemm.
void
dlp_init_global_cntx()
{
    dlp_pthread_once(&once_check_dlp_gemm_func_map_init,
                     dlp_gemm_cntx_init_map);

    dlp_gemm_init_thread_attrs();
}

dlp_gemm_cntx_t*
dlp_gemm_get_global_cntx_obj(AOCL_DLP_OPERATION_TYPE op)
{
    return &global_cntx_t_list[op];
}

dlp_gemm_util_cntx_t*
dlp_gemm_util_get_global_cntx_obj(AOCL_DLP_UTIL_OPERATION_TYPE op)
{
    return &global_util_cntx_t_list[op];
}

dlp_gemm_eltwise_ops_cntx_t*
dlp_gemm_eltwise_ops_get_global_cntx_obj(AOCL_DLP_ELTWISE_OPS_OPERATION_TYPE op)
{
    return &global_eltwise_ops_cntx_t_list[op];
}

md_t
dlp_gemm_get_block_size_MC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].blksz.MC;
}

md_t
dlp_gemm_get_block_size_NC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].blksz.NC;
}

md_t
dlp_gemm_get_block_size_KC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].blksz.KC;
}

md_t
dlp_gemm_get_block_size_NR_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].blksz.NR;
}

md_t
dlp_gemm_get_block_size_MR_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].blksz.MR;
}

md_t
dlp_gemm_get_sup_thres_MT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].sup_thres.MT;
}

md_t
dlp_gemm_get_sup_thres_NT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].sup_thres.NT;
}

md_t
dlp_gemm_get_sup_thres_KT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type)
{
    return global_cntx_t_list[op_type].sup_thres.KT;
}

void
dlp_gemm_get_packa_strides(dlp_gemm_cntx_t* lcntx, md_t* rs, md_t* cs)
{
    *rs = lcntx->pack_s.packa_rs;
    *cs = lcntx->pack_s.packa_cs;
}

void
dlp_gemm_get_packb_strides(dlp_gemm_cntx_t* lcntx, md_t* rs, md_t* cs)
{
    *rs = lcntx->pack_s.packb_rs;
    *cs = lcntx->pack_s.packb_cs;
}

void
dlp_gemm_mod_block_size_s16(
    md_t m, md_t n, md_t k, md_t* MC, md_t* NC, md_t* KC)
{
    const md_t range[4] = { 1024, 512, 256, 128 };

    if (n < *NC) {
        for (iter_t i = 0; i < 4; ++i) {
            if (n <= range[i]) {
                *NC = range[i];
            }
        }
    }

    if (k < *KC) {
        for (iter_t i = 0; i < 4; ++i) {
            if (k <= range[i]) {
                *KC = range[i];
            }
        }
    }
}
