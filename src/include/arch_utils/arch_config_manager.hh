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
    Zen6,
    Zen5,
    Zen4,
    Zen3,
    Zen2,
    Zen,

    // These archs are used to represent non zen architectures. However kernel
    // dispatch is to be done based on the underlying ISA instead of arch.
    // There is an assumption here that the ISA support is hierarchical, ex: if
    // a machine supports Avx512Bf16 then it also supports Avx512Vnni, Avx512
    // Avx2. This needs to be validated in future if more non-zen arch types
    // are added.
    GenericAvx512Fp16,
    GenericAvx512Bf16,
    GenericAvx512Vnni,
    GenericAvx512,
    GenericAvx2,
    MaxArchType
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
    static archConfigManager& getInstance()
    {
        // C++11 guarantees thread-safe static local initialization
        static archConfigManager instance;
        return instance;
    }

    /**
     * @brief Check if AVX2 and FMA3 instruction sets are supported
     * @return true if both AVX2 and FMA3 are supported, false otherwise
     */
    bool isAvx2Fma3SupportedByArch() const noexcept
    {
        return isAvx2Fma3Supported;
    }

    /**
     * @brief Check if AVX512 instruction set is supported
     * @return true if AVX512 is supported, false otherwise
     */
    bool isAvx512SupportedByArch() const noexcept { return isAvx512Supported; }

    /**
     * @brief Check if AVX512 VNNI (Vector Neural Network Instructions) is
     * supported
     * @return true if AVX512_VNNI is supported, false otherwise
     */
    bool isAvx512VnniSupportedByArch() const noexcept
    {
        return isAvx512VnniSupported;
    }

    /**
     * @brief Check if AVX512 BF16 (Brain Float16) instructions are supported
     * @return true if AVX512_BF16 is supported, false otherwise
     */
    bool isAvx512Bf16SupportedByArch() const noexcept
    {
        return isAvx512Bf16Supported;
    }

    /**
     * @brief Check if AVX512 FP16 (IEEE Float16) instructions are supported
     * @return true if AVX512_FP16 is supported, false otherwise
     */
    bool isAvx512Fp16SupportedByArch() const noexcept
    {
        return isAvx512Fp16Supported;
    }

    /**
     * @brief Get the floating-point/SIMD execution datapath width
     * @return Datapath width value (128, 256, or 512 bits typically)
     */
    std::uint32_t getFpDatapathWidthOfArch() const noexcept
    {
        return fpDatapathWidth;
    }

    /**
     * @brief Check if arch is Zen6-similar architecture
     * @return true if arch is Zen6 or Zen6-similar architecture, false
     * otherwise
     */
    bool isZen6SimilarArch() const noexcept { return isZen6; }

    /**
     * @brief Check if arch is Zen5-similar architecture
     * @return true if arch is Zen5 or Zen5-similar architecture, false
     * otherwise
     */
    bool isZen5SimilarArch() const noexcept { return isZen5; }

    /**
     * @brief Check if arch is Zen4-similar architecture
     * @return true if arch is Zen4 or Zen4-similar architecture, false
     * otherwise
     */
    bool isZen4SimilarArch() const noexcept { return isZen4; }

    /**
     * @brief Check if arch is Zen-similar architecture (original Zen family)
     * @return true if arch is Zen or Zen-similar architecture, false otherwise
     */
    bool isZenSimilarArch() const noexcept { return isZen; }

    /**
     * @brief Gets the underlying architecture type.
     *
     * This function returns the actual architecture type of the underlying
     * hardware.
     *
     * @return ArchitectureType The underlying architecture type
     */
    ArchitectureType getArch() const noexcept { return actualArch; }

    /**
     * @brief Check if AVX2 and FMA3 instruction sets are supported by
     * configured arch
     * @return true if both AVX2 and FMA3 are supported, false otherwise
     */
    bool isAvx2Fma3SupportedByConfiguredArch() const noexcept
    {
        return isAvx2Fma3Configured;
    }

    /**
     * @brief Check if AVX512 instruction set is supported by configured arch
     * @return true if AVX512 is supported, false otherwise
     */
    bool isAvx512SupportedByConfiguredArch() const noexcept
    {
        return isAvx512Configured;
    }

    /**
     * @brief Check if AVX512 VNNI (Vector Neural Network Instructions) is
     * supported by configured arch
     * @return true if AVX512_VNNI is supported, false otherwise
     */
    bool isAvx512VnniSupportedByConfiguredArch() const noexcept
    {
        return isAvx512VnniConfigured;
    }

    /**
     * @brief Check if AVX512 BF16 (Brain Float16) instructions are supported by
     * configured arch
     * @return true if AVX512_BF16 is supported, false otherwise
     */
    bool isAvx512Bf16SupportedByConfiguredArch() const noexcept
    {
        return isAvx512Bf16Configured;
    }

    /**
     * @brief Check if AVX512 FP16 (IEEE Float16) instructions are supported by
     * configured arch
     * @return true if AVX512_FP16 is supported, false otherwise
     */
    bool isAvx512Fp16SupportedByConfiguredArch() const noexcept
    {
        return isAvx512Fp16Configured;
    }

    /**
     * @brief Check if arch is configured as Zen6-similar architecture
     * @return true if arch is configured as Zen6 or Zen6-similar architecture,
     * false otherwise
     */
    bool isZen6SimilarConfiguredArch() const noexcept
    {
        return isZen6Configured;
    }

    /**
     * @brief Check if arch is configured as Zen5-similar architecture
     * @return true if arch is configured as Zen5 or Zen5-similar architecture,
     * false otherwise
     */
    bool isZen5SimilarConfiguredArch() const noexcept
    {
        return isZen5Configured;
    }

    /**
     * @brief Check if arch is configured as Zen4-similar architecture
     * @return true if arch is configured as Zen4 or Zen4-similar architecture,
     * false otherwise
     */
    bool isZen4SimilarConfiguredArch() const noexcept
    {
        return isZen4Configured;
    }

    /**
     * @brief Check if arch is configured as Zen-similar architecture (original
     * Zen family)
     * @return true if arch is configured as Zen or Zen-similar architecture,
     * false otherwise
     */
    bool isZenSimilarConfiguredArch() const noexcept { return isZenConfigured; }

    /**
     * @brief Gets the configured architecture type.
     *
     * This function returns the architecture type that the system is currently
     * configured to use. The architecture type determines which optimized code
     * paths and instruction sets will be utilized.
     *
     * @return ArchitectureType The configured architecture type
     */
    ArchitectureType getConfiguredArch() const noexcept { return thisArch; }

    // Disable copy and move operations for singleton
    archConfigManager(const archConfigManager&)            = delete;
    archConfigManager& operator=(const archConfigManager&) = delete;
    archConfigManager(archConfigManager&&)                 = delete;
    archConfigManager& operator=(archConfigManager&&)      = delete;

  private:
    archConfigManager();
    ~archConfigManager() = default;

    void setIsAvx2Fma3Supported();
    void setIsAvx512Supported();
    void setIsAvx512VnniSupported();
    void setIsAvx512Bf16Supported();
    void setIsAvx512Fp16Supported();
    void setFpDatapathWidth();
    void setIsZen6();
    void setIsZen5();
    void setIsZen4();
    void setIsZen();

    ArchitectureType queryUnderlyingArch(void);
    void             setArch(void);

    void setIsAvx2Fma3Configured();
    void setIsAvx512Configured();
    void setIsAvx512VnniConfigured();
    void setIsAvx512Bf16Configured();
    void setIsAvx512Fp16Configured();
    void setIsZen6Configured();
    void setIsZen5Configured();
    void setIsZen4Configured();
    void setIsZenConfigured();

    // Actual architectural feature support flags based on hardware detection.
    bool             isAvx2Fma3Supported;
    bool             isAvx512Supported;
    bool             isAvx512VnniSupported;
    bool             isAvx512Bf16Supported;
    bool             isAvx512Fp16Supported;
    uint32_t         fpDatapathWidth;
    bool             isZen6;
    bool             isZen5;
    bool             isZen4;
    bool             isZen;
    ArchitectureType actualArch;

    // Configured architecture based on environment variable or actual hardware
    // if env variable is not set or invalid.
    bool             isAvx2Fma3Configured;
    bool             isAvx512Configured;
    bool             isAvx512VnniConfigured;
    bool             isAvx512Bf16Configured;
    bool             isAvx512Fp16Configured;
    bool             isZen6Configured;
    bool             isZen5Configured;
    bool             isZen4Configured;
    bool             isZenConfigured;
    ArchitectureType thisArch;
};

} // namespace dlp::arch_utils
