/*
 * Portable SIMD reinterpret-cast macros.
 *
 * All definitions have moved to dlp_compat.h (Section 14).
 * This header enables the SIMD section and re-exports it so that
 * existing #include "classic/dlp_simd_casts.h" sites keep working.
 */

#ifndef DLP_SIMD_CASTS_H
#define DLP_SIMD_CASTS_H

#ifndef DLP_COMPAT_INCLUDE_SIMD
#define DLP_COMPAT_INCLUDE_SIMD
#endif

#include "classic/dlp_compat.h"

#endif /* DLP_SIMD_CASTS_H */
