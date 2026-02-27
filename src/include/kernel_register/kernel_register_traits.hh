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

#include <type_traits>

namespace dlp::kernel_frame::kernelRegisterTraits {

// Complex but necessary - provides excellent error diagnostics
// when dispatch table interfaces are implemented incorrectly
// Detects: T.insert<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE>(KEY_TYPE*,
// voidFunctorPtr)
template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_REPLACER,
         typename VALUE_TYPE,
         typename = void>
struct has_insert_method : std::false_type
{};

template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_REPLACER,
         typename VALUE_TYPE>
struct has_insert_method<
    T,
    HASH_KEY_GETTER,
    KEY_COMPARATOR,
    KEY_TYPE,
    VALUE_REPLACER,
    VALUE_TYPE,
    std::void_t<decltype(std::declval<T>()
                             .template insert<HASH_KEY_GETTER,
                                              KEY_COMPARATOR,
                                              KEY_TYPE,
                                              VALUE_REPLACER,
                                              VALUE_TYPE>(
                                 std::declval<KEY_TYPE*>(),
                                 std::declval<voidFunctorPtr>()))>>
    : std::true_type
{};

// Detects: T.query<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE>(KEY_TYPE*)
template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_TYPE,
         typename = void>
struct has_query_method : std::false_type
{};

template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_TYPE>
struct has_query_method<
    T,
    HASH_KEY_GETTER,
    KEY_COMPARATOR,
    KEY_TYPE,
    VALUE_TYPE,
    std::void_t<decltype(std::declval<T>()
                             .template query<HASH_KEY_GETTER,
                                             KEY_COMPARATOR,
                                             KEY_TYPE,
                                             VALUE_TYPE>(
                                 std::declval<KEY_TYPE*>()))>> : std::true_type
{};

// Detects: T.getValues<VALUE_TYPE>()
template<typename T, typename VALUE_TYPE, typename = void>
struct has_getValues_method : std::false_type
{};

template<typename T, typename VALUE_TYPE>
struct has_getValues_method<
    T,
    VALUE_TYPE,
    std::void_t<decltype(std::declval<T>().template getValues<VALUE_TYPE>())>>
    : std::true_type
{};

// Detects: T.getKeys<KEY_TYPE>()
template<typename T, typename KEY_TYPE, typename = void>
struct has_getKeys_method : std::false_type
{};

template<typename T, typename KEY_TYPE>
struct has_getKeys_method<
    T,
    KEY_TYPE,
    std::void_t<decltype(std::declval<T>().template getKeys<KEY_TYPE>())>>
    : std::true_type
{};

// Extract return type of insert() method
template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_REPLACER,
         typename VALUE_TYPE>
using insert_return_type =
    decltype(std::declval<T>()
                 .template insert<HASH_KEY_GETTER,
                                  KEY_COMPARATOR,
                                  KEY_TYPE,
                                  VALUE_REPLACER,
                                  VALUE_TYPE>(std::declval<KEY_TYPE*>(),
                                              std::declval<voidFunctorPtr>()));

// Extract return type of query() method
template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_TYPE>
using query_return_type =
    decltype(std::declval<T>()
                 .template query<HASH_KEY_GETTER,
                                 KEY_COMPARATOR,
                                 KEY_TYPE,
                                 VALUE_TYPE>(std::declval<KEY_TYPE*>()));

// Extract return type of getValues() method
template<typename T, typename VALUE_TYPE>
using getValues_return_type =
    decltype(std::declval<T>().template getValues<VALUE_TYPE>());

// Extract return type of getKeys() method
template<typename T, typename KEY_TYPE>
using getKeys_return_type =
    decltype(std::declval<T>().template getKeys<KEY_TYPE>());

// Main validation function that combines method existence checks with return
// type validation. Used in static_assert in kernelRegister. Returns true if
// all methods exist AND have correct return types, false otherwise.
template<typename T,
         typename HASH_KEY_GETTER,
         typename KEY_COMPARATOR,
         typename KEY_TYPE,
         typename VALUE_REPLACER,
         typename VALUE_TYPE>
constexpr bool
hasDispatchTableInterface_v()
{
    // Step 1: Check if all required methods exist
    constexpr bool has_insert =
        has_insert_method<T, HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                          VALUE_REPLACER, VALUE_TYPE>::value;
    constexpr bool has_query =
        has_query_method<T, HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                         VALUE_TYPE>::value;
    constexpr bool has_getValues = has_getValues_method<T, VALUE_TYPE>::value;
    constexpr bool has_getKeys   = has_getKeys_method<T, KEY_TYPE>::value;

    // Early return if any method is missing
    if constexpr (!has_insert || !has_query || !has_getValues || !has_getKeys) {
        return false;
    } else {
        // Step 2: Check return types (only if all methods exist)
        constexpr bool correct_insert_return = std::is_same_v<
            insert_return_type<T, HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                               VALUE_REPLACER, VALUE_TYPE>,
            VALUE_TYPE*>;
        constexpr bool correct_query_return =
            std::is_same_v<query_return_type<T, HASH_KEY_GETTER, KEY_COMPARATOR,
                                             KEY_TYPE, VALUE_TYPE>,
                           VALUE_TYPE*>;
        constexpr bool correct_getValues_return =
            std::is_same_v<getValues_return_type<T, VALUE_TYPE>,
                           std::vector<VALUE_TYPE*>>;
        constexpr bool correct_getKeys_return =
            std::is_same_v<getKeys_return_type<T, KEY_TYPE>,
                           std::vector<KEY_TYPE*>>;

        return correct_insert_return && correct_query_return
               && correct_getValues_return && correct_getKeys_return;
    }
}

} // namespace dlp::kernel_frame::kernelRegisterTraits
