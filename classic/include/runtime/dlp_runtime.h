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

#ifndef DLP_RNTM_H
#define DLP_RNTM_H

#include "classic/dlp_base_types.h"

typedef struct __attribute__((aligned(64)))
{
    md_t num_threads;
    md_t ic_ways;
    md_t jc_ways;

    // enable/disable packing of left-hand matrix A.
    bool pack_a;

    // enable/disable packing of right-hand matrix B.
    bool pack_b;

    // denotes whether external libs (eg: OMP) controls threading.
    bool ext_mt_ctr_var;
} dlp_rntm_t;

#define DLP_CLASSIC_RNTM_INITIALIZER                                           \
    {                                                                          \
        .num_threads    = -1,                                                  \
        .ic_ways        = -1,                                                  \
        .jc_ways        = -1,                                                  \
        .pack_a         = FALSE,                                               \
        .pack_b         = FALSE,                                               \
        .ext_mt_ctr_var = TRUE,                                                \
    }

#define DLP_CLASSIC_RNTM_RESET(rntm)                                           \
    do {                                                                       \
        (rntm).num_threads    = -1;                                            \
        (rntm).ic_ways        = -1;                                            \
        (rntm).jc_ways        = -1;                                            \
        (rntm).pack_a         = FALSE;                                         \
        (rntm).pack_b         = FALSE;                                         \
        (rntm).ext_mt_ctr_var = TRUE;                                          \
    } while (0)

void
dlp_rntm_init_from_global(dlp_rntm_t* rntm);

#endif // DLP_RNTM_H
