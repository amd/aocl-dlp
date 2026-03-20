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

#ifndef DLP_GEMM_UTILS_H
#define DLP_GEMM_UTILS_H

#include <stdio.h>

#include "dlp_gemm_types.h"
#include "runtime/dlp_runtime.h"

DLP_INLINE void
dlp_param_map_char_to_lpmtag(char mtag, AOCL_DLP_MEMORY_TAG* lp_mtag)
{
    if (mtag == 'n' || mtag == 'N')
        *lp_mtag = UNPACKED;
    else if (mtag == 'p' || mtag == 'P')
        *lp_mtag = PACK;
    else if (mtag == 'r' || mtag == 'R')
        *lp_mtag = REORDERED;
    else {
        *lp_mtag = UNPACKED;
    }
}

DLP_INLINE void
dlp_param_map_char_to_lpmat_type(const char            mtag,
                                 AOCL_DLP_MATRIX_TYPE* lp_mat_type)
{
    if (mtag == 'a' || mtag == 'A')
        *lp_mat_type = A_MATRIX;
    else if (mtag == 'b' || mtag == 'B')
        *lp_mat_type = B_MATRIX;
    else if (mtag == 'w' || mtag == 'W')
        *lp_mat_type = AWQ_B_MATRIX;
    else {
        *lp_mat_type = B_MATRIX;
    }
}

DLP_INLINE md_t
make_multiple_of_n(md_t k, md_t n)
{
    if (n <= 0) {
        return 0;
    }

    return (((k + n - 1) / n) * n);
}

DLP_INLINE md_t
get_Bpanel_width_for_kmd_traversal(md_t jc, md_t n, md_t NC, md_t NR)
{
    md_t n_mod_NR      = n % NR;
    md_t n_sub_updated = NC;

    if ((n % NC) != 0) {
        // Only applicable to final NC part of jc loop where jc + remaining
        // elements is less than NC; or when n < NC in which case panel width
        // is atmost n.
        md_t n_last_loop = (n / NC) * NC;
        if (jc >= n_last_loop) {
            n_sub_updated = n - n_last_loop;
            if (n_mod_NR != 0) {
                n_sub_updated += (NR - n_mod_NR);
            }
        }
    }

    return n_sub_updated;
}

DLP_INLINE void
get_B_panel_reordered_start_offset_width(md_t  jc,
                                         md_t  n,
                                         md_t  NC,
                                         md_t  NR,
                                         md_t* panel_start,
                                         md_t* panel_offset,
                                         md_t* panel_width,
                                         md_t* panel_width_kmd_trav)
{
    // Since n dimension is split across threads in units of NR blocks,
    // it could happen that B matrix chunk for a thread may be part of
    // two separate NCxKC panels. In this case nc0 is updated such that
    // the jr loop only accesses the remaining portion of current NCxKC
    // panel, with the next jc iteration taking care of the other panel.
    // This ensures that jr loop does not cross panel boundaries.
    (*panel_start)  = (jc / NC) * NC;
    (*panel_offset) = jc - (*panel_start);

    // Check if jc + current_panel_width (nc0) crosses panel boundaries.
    if ((jc + (*panel_width)) > ((*panel_start) + NC)) {
        (*panel_width) = NC - (*panel_offset);
    }

    (*panel_width_kmd_trav) = get_Bpanel_width_for_kmd_traversal(jc, n, NC, NR);
}

DLP_INLINE void
adjust_B_panel_reordered_jc(md_t* jc, md_t panel_start)
{
    // Since n dimension is split across threads in units of NR blocks,
    // it could happen that B matrix chunk for a thread may be part of
    // two separate NCxKC panels. In this case jc is reset to immediate
    // previous panel offset so that in the next iteration, the
    // following panel belonging to the B chunk is accessed. This
    // ensures that jr loop does not cross panel boundaries.
    (*jc) = panel_start;
}

static inline bool
is_single_thread(dlp_rntm_t* rntm_g)
{
    bool is_st = FALSE;

    md_t n_threads = rntm_g->num_threads;
    md_t jc_ways   = rntm_g->jc_ways;
    md_t ic_ways   = rntm_g->ic_ways;

    if (n_threads == 1) {
        is_st = TRUE;
    } else if ((ic_ways > 0) && (jc_ways > 0) && ((ic_ways * jc_ways) == 1)) {
        is_st = TRUE;
    }

    return is_st;
}

DLP_INLINE void
dlp_param_map_netlib_to_dlp_trans(char trans, dlp_trans_t* dlp_trans)
{
    if (trans == 'n' || trans == 'N')
        *dlp_trans = DLP_NO_TRANSPOSE;
    else if (trans == 't' || trans == 'T')
        *dlp_trans = DLP_TRANSPOSE;
    else if (trans == 'c' || trans == 'C')
        *dlp_trans = DLP_CONJ_TRANSPOSE;
    else if (trans == 'p' || trans == 'P')
        *dlp_trans = DLP_PACKED;
    else {
        *dlp_trans = DLP_NO_TRANSPOSE;
    }
}

DLP_INLINE bool
dlp_is_notrans(dlp_trans_t trans)
{
    return (bool)(trans == DLP_NO_TRANSPOSE);
}

DLP_INLINE bool
dlp_is_trans(dlp_trans_t trans)
{
    return (bool)(trans == DLP_TRANSPOSE);
}

DLP_INLINE void
dlp_print_msg(char* str, char* file, uint64_t line)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "libdlp: %s (line %lu):\n", file, (long unsigned int)line);
    fprintf(stderr, "libdlp: %s\n", str);
    fflush(stderr);
}

#endif // DLP_GEMM_UTILS_H
