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

#ifndef DLP_MACROS_H
#define DLP_MACROS_H

/**
 * @brief Force inline macro - portable across compilers
 *
 * Use this macro to force the compiler to inline a function.
 * This is useful for performance-critical code paths where
 * inlining is essential.
 *
 * Usage:
 *   DLP_ALWAYS_INLINE void myFunction() { ... }
 *
 * Note: This macro includes 'inline' so don't add it separately.
 */
#if defined(_MSC_VER)
#define DLP_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define DLP_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define DLP_ALWAYS_INLINE inline
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DLP_ATTRIBUTE_USED [[gnu::used]]
#else
#define DLP_ATTRIBUTE_USED
#endif

#ifdef __cplusplus

#define DLP_BEGIN_EXTERN_C                                                     \
    extern "C"                                                                 \
    {
#define DLP_END_EXTERN_C }

#else

#define DLP_BEGIN_EXTERN_C
#define DLP_END_EXTERN_C

#endif

#endif // DLP_MACROS_H
