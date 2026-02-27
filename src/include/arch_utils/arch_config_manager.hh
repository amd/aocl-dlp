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

#include <cstdint>

namespace dlp::arch_utils {

/**
 * @brief Architecture type enumeration for type-safe architecture handling
 */
enum class ArchitectureType : int16_t
{
    Error = 0,
    Generic,
    Zen5,
    Zen4,
    Zen3,
    Zen2,
    Zen,

    // These archs are used to represent non zen architectures. However kernel
    // dispatch is to be done based on the underlying ISA instead of arch.
    GenericAvx512Bf16,
    GenericAvx512Vnni,
    GenericAvx512,
    GenericAvx2
};

/**
 * @brief Architecture configuration manager for arch (and its capability
 * related) queries
 */
class archConfigManager
{
  public:
    /**
     * @brief Get the singleton instance of archConfigManager
     * @return Reference to the singleton instance (thread-safe initialization)
     */
    static archConfigManager& getInstance();

    /**
     * @brief Check if AVX2 and FMA3 instruction sets are supported
     * @return true if both AVX2 and FMA3 are supported, false otherwise
     */
    bool isAvx2Fma3SupportedByArch() const noexcept;

    /**
     * @brief Check if AVX512 instruction set is supported
     * @return true if AVX512 is supported, false otherwise
     */
    bool isAvx512SupportedByArch() const noexcept;

    /**
     * @brief Check if AVX512 VNNI (Vector Neural Network Instructions) is
     * supported
     * @return true if AVX512_VNNI is supported, false otherwise
     */
    bool isAvx512VnniSupportedByArch() const noexcept;

    /**
     * @brief Check if AVX512 BF16 (Brain Float16) instructions are supported
     * @return true if AVX512_BF16 is supported, false otherwise
     */
    bool isAvx512Bf16SupportedByArch() const noexcept;

    /**
     * @brief Check if AVX512 FP16 (IEEE Float16) instructions are supported
     * @return true if AVX512_FP16 is supported, false otherwise
     */
    bool isAvx512Fp16SupportedByArch() const noexcept;

    /**
     * @brief Get the floating-point/SIMD execution datapath width
     * @return Datapath width value (128, 256, or 512 bits typically)
     */
    std::uint32_t getFpDatapathWidthOfArch() const noexcept;

    /**
     * @brief Check if CPU is Zen5-similar architecture
     * @return true if CPU is Zen5 or Zen5-similar architecture, false otherwise
     */
    bool isZen5SimilarArch() const noexcept;

    /**
     * @brief Check if CPU is Zen4-similar architecture
     * @return true if CPU is Zen4 or Zen4-similar architecture, false otherwise
     */
    bool isZen4SimilarArch() const noexcept;

    /**
     * @brief Check if CPU is Zen-similar architecture (original Zen family)
     * @return true if CPU is Zen or Zen-similar architecture, false otherwise
     */
    bool isZenSimilarArch() const noexcept;

    /**
     * @brief Gets the current architecture type.
     *
     * This function returns the architecture type that the system is currently
     * configured to use. The architecture type determines which optimized code
     * paths and instruction sets will be utilized.
     *
     * @return ArchitectureType The current architecture type
     */
    ArchitectureType getArch() const noexcept;

    // Disable copy and move operations for singleton
    archConfigManager(const archConfigManager&)            = delete;
    archConfigManager& operator=(const archConfigManager&) = delete;
    archConfigManager(archConfigManager&&)                 = delete;
    archConfigManager& operator=(archConfigManager&&)      = delete;

  private:
    archConfigManager();
    ~archConfigManager() = default;

    void             setIsAvx2Fma3Supported();
    void             setIsAvx512Supported();
    void             setIsAvx512VnniSupported();
    void             setIsAvx512Bf16Supported();
    void             setIsAvx512Fp16Supported();
    void             setFpDatapathWidth();
    void             setIsZen5();
    void             setIsZen4();
    void             setIsZen();
    ArchitectureType queryUnderlyingArch(void);
    void             setArch(void);

    bool             isAvx2Fma3Supported;
    bool             isAvx512Supported;
    bool             isAvx512VnniSupported;
    bool             isAvx512Bf16Supported;
    bool             isAvx512Fp16Supported;
    uint32_t         fpDatapathWidth;
    bool             isZen5;
    bool             isZen4;
    bool             isZen;
    ArchitectureType thisArch;
};

} // namespace dlp::arch_utils
