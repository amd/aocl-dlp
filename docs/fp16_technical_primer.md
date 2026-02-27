# FP16 Technical Primer for AOCL-DLP

## 1. IEEE 754 Half-Precision Format

### Binary Representation

FP16 (IEEE 754 half-precision) is a 16-bit floating-point format with the following structure:

```
Bit Layout: [S][EEEEE][MMMMMMMMMM]
            |   |      |
            |   |      └─ 10-bit Mantissa (fraction)
            |   └──────── 5-bit Exponent
            └──────────── 1-bit Sign

Total: 1 + 5 + 10 = 16 bits
```

### Comparison with FP32 and BF16

| Format | Sign | Exponent | Mantissa | Total | Range (approx) | Precision |
|--------|------|----------|----------|-------|----------------|-----------|
| **FP32** | 1 | 8 bits | 23 bits | 32 | ±3.4×10³⁸ | ~7 decimal digits |
| **FP16** | 1 | 5 bits | 10 bits | 16 | ±6.55×10⁴ | ~3-4 decimal digits |
| **BF16** | 1 | 8 bits | 7 bits | 16 | ±3.4×10³⁸ | ~2-3 decimal digits |

**Key Insight**: FP16 has better precision than BF16 (more mantissa bits) but smaller dynamic range (fewer exponent bits). BF16 shares FP32's exponent, making truncation/conversion simpler.

### Bit Pattern Examples

```
Value: 1.0
Binary: 0 01111 0000000000
        │   │    └─ Mantissa = 0 (implicit 1.0)
        │   └────── Exponent = 15 (bias = 15, actual = 0)
        └────────── Sign = 0 (positive)

Value: -0.5
Binary: 1 01110 0000000000
        │   │    └─ Mantissa = 0
        │   └────── Exponent = 14 (actual = -1)
        └────────── Sign = 1 (negative)

Value: 65504 (max normal)
Binary: 0 11110 1111111111
        │   │    └─ Mantissa = all 1s
        │   └────── Exponent = 30 (actual = 15)
        └────────── Sign = 0
```

## 2. Representable Values

### Normal Numbers

- **Range**: ±6.10352×10⁻⁵ to ±65504
- **Minimum positive normal**: 2⁻¹⁴ ≈ 6.104×10⁻⁵
- **Maximum positive**: (2 - 2⁻¹⁰) × 2¹⁵ = 65504
- **Exponent bias**: 15 (so exponent field 00001 to 11110 represents -14 to +15)

### Subnormal (Denormal) Numbers

When exponent = 0 and mantissa ≠ 0, number is subnormal:
- **Range**: ±2⁻²⁴ to ±(1-2⁻¹⁰)×2⁻¹⁴
- **Smallest positive subnormal**: 2⁻²⁴ ≈ 5.96×10⁻⁸
- **Purpose**: Gradual underflow near zero (avoid discontinuity at zero)
- **Performance**: Often slower; consider Flush-To-Zero (FTZ) mode

### Special Values

| Exponent | Mantissa | Value |
|----------|----------|-------|
| 00000 | 00000... | **±0** (signed zero) |
| 00000 | ≠0 | **Subnormal** |
| 11111 | 00000... | **±∞** (infinity) |
| 11111 | ≠0, MSB=0 | **sNaN** (signaling NaN) |
| 11111 | ≠0, MSB=1 | **qNaN** (quiet NaN) |

### Precision Characteristics

- **Representable numbers**: Approximately 2¹⁶ = 65,536 distinct values (half positive, half negative)
- **Decimal precision**: ~3-4 significant decimal digits
- **Epsilon (machine precision)**: 2⁻¹⁰ ≈ 0.000977 (~0.1%)
- **Spacing near 1.0**: 2⁻¹⁰ ≈ 0.000977

## 3. FP16 vs BF16 Trade-offs

### Detailed Comparison

| Characteristic | FP16 | BF16 | Winner | Use Case |
|----------------|------|------|--------|----------|
| **Dynamic Range** | ±65504 | ±3.4×10³⁸ | BF16 | BF16 safer for training (no overflow) |
| **Mantissa Bits** | 10 | 7 | FP16 | FP16 better for precise values |
| **Precision (decimal)** | ~3-4 digits | ~2-3 digits | FP16 | FP16 for high-accuracy inference |
| **Conversion from FP32** | Complex | Truncate | BF16 | BF16 trivial to convert |
| **Overflow Risk** | High | Very low | BF16 | BF16 for gradients, updates |
| **Underflow Risk** | Moderate | Low | BF16 | FP16 needs careful scaling |
| **ML Training** | Less common | Very common | BF16 | BF16 industry standard |
| **ML Inference** | Common | Common | Tie | Both widely used |
| **Hardware Support** | Newer (2023+) | Widespread (2020+) | BF16 | BF16 more available today |

### When to Use FP16 vs BF16

**Use FP16 when:**
- Inference workload with controlled value ranges
- Need higher precision for small differences (e.g., attention scores)
- Memory bandwidth is the bottleneck (both are 16-bit, so equal here)
- Newer hardware with AVX512-FP16 available

**Use BF16 when:**
- Training neural networks (gradient updates)
- Large dynamic range needed (e.g., loss values)
- Converting directly from FP32 code
- Broader hardware compatibility needed

**Hybrid Approach:**
- Use BF16 for weights/gradients (training)
- Use FP16 for activations (inference)
- Per-layer decisions based on sensitivity analysis

## 4. FP16 ↔ FP32 Conversion

### Software Conversion Algorithm (FP16 → FP32)

```c
float fp16_to_fp32(uint16_t fp16) {
    uint32_t sign = (fp16 & 0x8000) << 16;  // Extract sign, move to bit 31
    uint32_t exp = (fp16 & 0x7C00) >> 10;   // Extract 5-bit exponent
    uint32_t mantissa = (fp16 & 0x03FF);    // Extract 10-bit mantissa

    uint32_t fp32_bits;

    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            fp32_bits = sign;
        } else {
            // Subnormal: normalize it
            exp = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x03FF;  // Remove leading 1
            fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        fp32_bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal number
        fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
    }

    return *(float*)&fp32_bits;
}
```

### Software Conversion Algorithm (FP32 → FP16)

```c
uint16_t fp32_to_fp16_rne(float value) {  // Round-to-nearest-even
    uint32_t fp32_bits = *(uint32_t*)&value;
    uint32_t sign = (fp32_bits & 0x80000000) >> 16;
    int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;  // Rebias
    uint32_t mantissa = (fp32_bits & 0x007FFFFF);

    if (exp <= 0) {
        if (exp < -10) {
            // Underflow to zero
            return (uint16_t)sign;
        }
        // Subnormal range: denormalize
        mantissa = (mantissa | 0x00800000) >> (1 - exp);
        // Round to nearest even
        uint16_t fp16_mantissa = (mantissa + 0x00001000 + ((mantissa >> 13) & 1)) >> 13;
        return (uint16_t)(sign | fp16_mantissa);
    } else if (exp >= 0x1F) {
        // Overflow to infinity
        return (uint16_t)(sign | 0x7C00);
    } else {
        // Normal number
        // Round to nearest even
        uint16_t fp16_mantissa = (mantissa + 0x00001000 + ((mantissa >> 13) & 1)) >> 13;
        if (fp16_mantissa == 0x0400) {
            // Rounding caused carry, increment exponent
            exp++;
            fp16_mantissa = 0;
        }
        return (uint16_t)(sign | (exp << 10) | fp16_mantissa);
    }
}
```

### Hardware Conversion (Preferred)

#### F16C Instructions (Available since Intel Ivy Bridge 2012, AMD Bulldozer 2011)

```asm
; FP16 → FP32 conversion (128-bit: 8×FP16 → 8×FP32, requires 2 instructions)
vcvtph2ps ymm0, xmm1    ; Convert 8×FP16 in XMM to 8×FP32 in YMM
                        ; Latency: ~4 cycles, Throughput: 1/cycle

; FP32 → FP16 conversion with rounding control
vcvtps2ph xmm0, ymm1, 0 ; Convert 8×FP32 in YMM to 8×FP16 in XMM
                        ; imm8=0: round to nearest even
                        ; Latency: ~4 cycles, Throughput: 1/cycle
```

**Rounding modes (imm8 in VCVTPS2PH):**
- `0`: Round to nearest even (default, IEEE 754 compliant)
- `1`: Round down (toward -∞)
- `2`: Round up (toward +∞)
- `3`: Round toward zero (truncate)
- `4`: Use MXCSR rounding control

#### AVX512-FP16 Native Instructions (Sapphire Rapids 2023+, Zen 5 2024+)

```asm
; FP16 → FP32 conversion (512-bit: 16×FP16 → 16×FP32)
vcvtph2ps zmm0, ymm1    ; Convert 16×FP16 to 16×FP32
                        ; Latency: ~4 cycles

; FP32 → FP16 conversion (512-bit: 16×FP32 → 16×FP16)
vcvtps2ph ymm0, zmm1, 0 ; Convert 16×FP32 to 16×FP16

; Native FP16 arithmetic (no conversion needed!)
vaddph zmm0, zmm1, zmm2 ; Add 32×FP16 values
vmulph zmm0, zmm1, zmm2 ; Multiply 32×FP16 values
vfmadd213ph zmm0, zmm1, zmm2  ; FMA in FP16

; FP16 dot product → FP32 accumulator (key for GEMM)
vdpphps zmm0, zmm1, zmm2  ; Dot product: 2×FP16 → FP32
                          ; Processes pairs of FP16, accumulates in FP32
                          ; Latency: ~6-7 cycles, Throughput: 2/cycle
```

### Performance Comparison

| Method | Throughput (values/cycle) | Latency (cycles) | Availability |
|--------|---------------------------|------------------|--------------|
| Software | ~0.1-0.2 (very slow) | 10-30 | Universal |
| F16C (128-bit) | 8 | 4 | Since 2011-2012 |
| F16C (256-bit) | 8 | 4 | AVX2+ |
| AVX512-FP16 | 32 (native ops) | 4-7 | 2023-2024+ |

**Recommendation**: Always use hardware instructions when available. F16C is ubiquitous; AVX512-FP16 is for newest CPUs only.

## 5. Numeric Considerations

### Overflow

**IEEE 754 Behavior**: When FP32 values exceed FP16 range (|value| > 65504), they overflow to ±infinity. This is the standard IEEE 754 behavior and matches hardware conversion instructions like `VCVTPS2PH`.

**AOCL-DLP Implementation**: Follows IEEE 754 - overflow to infinity (NOT saturation to max value). This design decision:
- Matches hardware instruction behavior
- Consistent with existing BF16 implementation
- Provides predictable, standard-compliant behavior
- Alerts user to range issues (infinity is detectable)

**The Problem**: FP16's maximum value (65504) is easily exceeded in deep learning:
- Weight * activation products
- Gradient accumulation over many samples
- Loss functions (cross-entropy can produce large logits)

**Example**:
```c
float16 a = 200.0f;   // OK
float16 b = 400.0f;   // OK
float16 c = a * b;    // 80000 → Overflow! → +Inf
```

**Mitigation Strategies**:
1. **Use F32 accumulators**: Always accumulate in FP32, downscale to FP16 only at output
2. **Input scaling**: Scale inputs to [-1, 1] or similar safe range
3. **Loss scaling**: Scale loss backward to prevent underflow, unscale gradients
4. **Per-tensor/per-channel scaling**: Track min/max per tensor, apply scale factors
5. **Monitoring**: Detect `Inf` in outputs, trigger re-scaling or fallback to FP32

### Underflow

**The Problem**: FP16's minimum normal value (6×10⁻⁵) can cause underflow:
- Small gradients vanish to zero (vanishing gradient problem)
- Attention scores with many tokens (softmax denominator grows)
- Batch normalization with small variances

**Example**:
```c
float16 grad = 0.00001f;  // 1e-5 → Becomes subnormal or flushes to zero
```

**Mitigation Strategies**:
1. **Gradient scaling**: Multiply loss by large factor (e.g., 1024), divide gradients afterward
2. **FTZ mode**: Flush-To-Zero for performance, but understand it zeros small values
3. **Mixed precision**: Use FP32 for parameters with small gradients
4. **Skip subnormals**: Check for denormal status, handle specially or clamp

### Subnormals (Denormals)

**Behavior**:
- Provide gradual underflow near zero (prevent sudden jump from tiny → 0)
- Much slower on many CPUs (10-100× slowdown)
- Can be disabled with Flush-To-Zero (FTZ) mode

**FTZ Mode**:
```c
// Enable FTZ in MXCSR (x86)
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
// Now subnormals → 0 (faster, but less accurate near zero)
```

**Recommendation**: Enable FTZ for performance unless you specifically need subnormal precision.

### Rounding Modes

**IEEE 754 Rounding Modes**:
1. **Round to nearest, ties to even** (default): Minimizes bias
2. **Round toward +∞** (ceiling): Always round up
3. **Round toward -∞** (floor): Always round down
4. **Round toward zero** (truncate): Like casting in C

**For AOCL-DLP**: Use round-to-nearest-even (mode 0) for correctness.

## 6. AVX512-FP16 Instructions

### Key Instructions for GEMM

```asm
; Data movement
vmovdqu16 zmm0, [mem]        ; Load 32×FP16 (64 bytes)
vmovdqu16 [mem], zmm0        ; Store 32×FP16

; Broadcast
vpbroadcastw zmm0, [mem]     ; Broadcast 1×FP16 to 32 lanes
vpbroadcastd zmm0, [mem]     ; Broadcast FP16 pair to 16 pairs

; Arithmetic (native FP16)
vaddph zmm0, zmm1, zmm2      ; Add 32×FP16
vsubph zmm0, zmm1, zmm2      ; Subtract 32×FP16
vmulph zmm0, zmm1, zmm2      ; Multiply 32×FP16
vfmadd231ph zmm0, zmm1, zmm2 ; FMA: zmm0 += zmm1 * zmm2 (in FP16)

; Dot product (CRITICAL for GEMM)
vdpphps zmm0, zmm1, zmm2     ; Dot product of FP16 pairs → FP32
; Input: zmm1 = [a0,a1, a2,a3, ..., a30,a31] (16 pairs)
;        zmm2 = [b0,b1, b2,b3, ..., b30,b31] (16 pairs)
; Output: zmm0 = [a0*b0+a1*b1, a2*b2+a3*b3, ..., a30*b30+a31*b31] (16 F32)
; Each pair of FP16 values multiplies and sums to one FP32 result

; Conversion
vcvtph2ps zmm0, ymm1         ; 16×FP16 → 16×FP32
vcvtps2ph ymm0, zmm1, imm8   ; 16×FP32 → 16×FP16 (with rounding)
```

### Performance Characteristics (Zen 5 / Sapphire Rapids)

| Instruction | Latency (cycles) | Throughput (ops/cycle) | Notes |
|-------------|------------------|------------------------|-------|
| `vmovdqu16` | 3-5 (L1 cache) | 2-3 | Load/store bandwidth limited |
| `vaddph` | 3-4 | 2 | 32 adds = 64 ops/cycle |
| `vmulph` | 3-4 | 2 | Same as add |
| `vfmadd231ph` | 4-5 | 2 | 64 ops/cycle (32 muls + 32 adds) |
| `vdpphps` | 6-7 | 2 | 32 ops/cycle (16 pairs × 2 ops) |
| `vcvtph2ps` | 4 | 1-2 | Conversion overhead |
| `vcvtps2ph` | 4 | 1-2 | Rounding overhead |

**Key Insight**: `vdpphps` is the hero instruction for GEMM. It computes 16 dot products (of FP16 pairs) per cycle, accumulating in F32 for accuracy.

### Register Capacity

```
ZMM register (512-bit):
- FP32: 16 values
- FP16: 32 values  ← 2× density!
- YMM (256-bit) for FP16 → FP32 conversion: 16 FP16 or 8 FP32

Total registers: 32 ZMM registers (zmm0-zmm31) on AVX512
```

## 7. Testing Considerations

### AOCL-DLP FP16 Test Suite

**Implementation Status**: ✅ Complete and validated

The FP16 support includes a comprehensive test suite (`tests/classic/test_fp16_conversions.cc`) with 32 tests covering:

```
Test Categories:
├── Basic conversions (zero, one, special values)
├── Overflow boundary (IEEE 754 compliance)
├── Underflow and denormals
├── Round-to-nearest-even rounding
├── Mantissa rounding with carry
├── Positive/negative symmetry
├── Consecutive value spacing (ULP)
├── Tie-breaking behavior
├── Edge cases (boundaries, max values)
└── Full range round-trip validation

Status: All 32 tests passing ✅
```

**Run Tests**:
```bash
cd build
make test_fp16_conversions
./tests/classic/test_fp16_conversions
```

### Tolerance Selection

**General Guidelines**:
```yaml
tolerances:
  # Standard operations (GEMM, bias, activations)
  absolute: 0.01      # 1% of value range
  relative: 0.001     # 0.1% relative error

  # Accumulation-heavy operations (large K in GEMM)
  absolute: 0.1       # 10% of range (rounding accumulates)
  relative: 0.005     # 0.5% relative

  # Near-overflow values (approaching ±65504)
  absolute: 1.0       # Coarse tolerance
  relative: 0.01      # 1% relative (larger errors acceptable)

  # Near-underflow values (approaching ±6e-5)
  absolute: 1e-4      # Absolute dominates at small scales
  relative: 0.1       # 10% relative (subnormals are noisy)
```

**Why tighter tolerances than BF16?**
- FP16 has 10-bit mantissa vs BF16's 7-bit
- ~8× more precision (2³ = 8)
- Can reliably expect ~0.1% error vs BF16's ~1%

### Test Data Generation

**Safe Value Ranges**:
```c
// Good: Safe range for GEMM inputs
float generate_safe_fp16() {
    // Range: [-100, 100] to avoid overflow in products
    return (rand() / (float)RAND_MAX) * 200.0f - 100.0f;
}

// Bad: Overflow-prone
float generate_unsafe_fp16() {
    return (rand() / (float)RAND_MAX) * 60000.0f;  // Too close to 65504!
}
```

**Strategies**:
1. **Normal distribution**: Mean=0, stddev=1 (most values in [-3, 3])
2. **Uniform distribution**: [-10, 10] for weights, [0, 10] for activations
3. **Scaled integers**: Use small integers (1, 2, 3) scaled by 0.1
4. **Power-of-two values**: Easy to convert exactly (1.0, 0.5, 0.25)

### Edge Case Testing

**Must-test values**:
```c
float16 test_values[] = {
    0.0f,          // Positive zero
    -0.0f,         // Negative zero (sign bit matters!)
    1.0f,          // Exactly representable
    -1.0f,
    0.5f, 0.25f,   // Powers of two (exact)
    6.104e-5f,     // Min positive normal
    65504.0f,      // Max positive
    6e-8f,         // Subnormal range
    INFINITY,      // +Inf
    -INFINITY,     // -Inf
    NAN,           // NaN (check propagation)
};
```

**Test patterns**:
- All zeros matrix
- Identity matrix
- Uniform values (all 1.0)
- Alternating signs
- Monotonic sequences (1, 2, 3, ...)
- Random with controlled range

### Denormal Handling

**Tests**:
```c
// Test 1: Underflow behavior
float16 tiny = 1e-8f;  // Forces subnormal
float16 result = tiny * tiny;  // Should it be 0 or subnormal?

// Test 2: FTZ mode
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
float16 x = 1e-6f;
float16 y = x * 0.01f;  // Should flush to zero with FTZ

// Test 3: Gradual underflow
for (float exp = -4; exp >= -8; exp -= 0.1) {
    float16 val = powf(10.0f, exp);
    // Verify smooth transition from normal → subnormal → zero
}
```

### NaN/Inf Propagation

**IEEE 754 rules**:
```c
// Arithmetic with Inf
Inf + x = Inf
Inf * x = Inf (if x != 0)
Inf / x = Inf (if x != Inf)
0 * Inf = NaN  ← Important!

// Arithmetic with NaN
NaN + x = NaN (NaN propagates)
NaN == NaN = false (!)

// Comparisons
Inf > any_finite = true
NaN > x = false (always)
```

**Test**:
```c
// Overflow produces Inf
float16 overflow = 65504.0f * 2.0f;  // = +Inf
assert(isinf(overflow));

// Invalid operations produce NaN
float16 invalid = 0.0f / 0.0f;  // = NaN
assert(isnan(invalid));

// NaN propagation in GEMM
// If any input element is NaN, entire output row/column becomes NaN
```

## 8. Implementation Patterns

### Typical GEMM Flow (FP16×FP16 → FP32)

```c
// Pseudocode for AOCL-DLP FP16 GEMM kernel

void fp16_gemm_kernel(float16* A, float16* B, float* C, int M, int N, int K) {
    // Step 1: Load FP16 data into registers
    for (int i = 0; i < M; i += MR) {      // MR = 6 (micro-kernel height)
        for (int j = 0; j < N; j += NR) {  // NR = 64 (16 F32 = 64 bytes)

            // Step 2: Initialize F32 accumulator registers (zero or beta*C)
            __m512 acc[MR];  // MR×16 F32 values
            for (int m = 0; m < MR; m++) {
                acc[m] = _mm512_setzero_ps();  // Or load C for beta != 0
            }

            // Step 3: Inner loop - compute dot products
            for (int k = 0; k < K; k += 2) {  // Process 2 FP16 at a time
                // Load B: 32×FP16 from B[k:k+2, j:j+32]
                __m512i b_fp16 = _mm512_loadu_epi16(&B[k*N + j]);

                // For each row of A
                for (int m = 0; m < MR; m++) {
                    // Broadcast A[i+m, k:k+2] (2×FP16 = 1 dword)
                    __m512i a_fp16 = _mm512_set1_epi32(*(int32_t*)&A[(i+m)*K + k]);

                    // Dot product: pairs of FP16 → F32
                    // vdpphps: a[0]*b[0] + a[1]*b[1] → F32, repeated 16 times
                    acc[m] = _mm512_dpwssd_ph(acc[m], a_fp16, b_fp16);
                    // (Note: Actual intrinsic may differ, this is illustrative)
                }
            }

            // Step 4: Post-operations in FP32 (bias, ReLU, etc.)
            for (int m = 0; m < MR; m++) {
                // Add bias (F32)
                if (bias != NULL) {
                    __m512 bias_vec = _mm512_loadu_ps(&bias[j]);
                    acc[m] = _mm512_add_ps(acc[m], bias_vec);
                }

                // ReLU (F32)
                acc[m] = _mm512_max_ps(acc[m], _mm512_setzero_ps());

                // Step 5: Store F32 result (no downscale)
                _mm512_storeu_ps(&C[(i+m)*N + j], acc[m]);
            }
        }
    }
}
```

### Typical GEMM Flow (FP16×FP16 → FP16 with downscale)

```c
// ... same as above until Step 5 ...

                // Step 5: Downscale F32 → FP16
                // Convert 16×F32 to 16×FP16
                __m256i c_fp16 = _mm512_cvtps_ph(acc[m], _MM_FROUND_TO_NEAREST_INT);

                // Store 16×FP16 (32 bytes)
                _mm256_storeu_epi16(&C_fp16[(i+m)*N + j], c_fp16);
```

### Conversion Best Practices

1. **Batch conversions**: Convert vectors, not scalars
   ```c
   // Good: Vectorized conversion
   __m512 fp32_vec = _mm512_cvtph_ps(fp16_vec);  // 16 at once

   // Bad: Scalar loop
   for (int i = 0; i < 16; i++) {
       fp32[i] = fp16_to_fp32(fp16[i]);  // Slow!
   }
   ```

2. **Minimize conversions**: Keep data in native format as long as possible
   ```c
   // Good: Operate in FP16, convert once at end
   fp16_add(); fp16_mul(); fp16_relu(); → convert_to_fp32();

   // Bad: Convert repeatedly
   convert_to_fp32(); fp32_add(); convert_to_fp16(); // Wasteful!
   ```

3. **Accumulate in higher precision**:
   ```c
   // Always accumulate in FP32, even if inputs are FP16
   __m512 sum_fp32 = _mm512_setzero_ps();  // F32 accumulator
   for (...) {
       __m256i x_fp16 = load_fp16();
       __m512 x_fp32 = _mm512_cvtph_ps(x_fp16);
       sum_fp32 = _mm512_add_ps(sum_fp32, x_fp32);  // Accumulate in F32
   }
   ```

4. **Set rounding mode appropriately**:
   ```c
   // For inference (speed): Use round-to-nearest-even (default)
   vcvtps2ph(ymm, zmm, 0);

   // For verification (accuracy): Also round-to-nearest-even
   // Only use directed rounding (1,2,3) if you understand the implications
   ```

## 9. Debugging Tips

### Watch for Silent Overflows

```c
// Overflow detection
void check_overflow_fp16(float16* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (isinf(data[i])) {
            fprintf(stderr, "Overflow at index %zu: value was ±Inf\n", i);
            // Options: clamp, scale inputs, use FP32
        }
    }
}

// Prevention: Clamp before storing
float16 safe_store_fp16(float fp32_value) {
    if (fp32_value > 65504.0f) return 65504.0f;      // Clamp to max
    if (fp32_value < -65504.0f) return -65504.0f;    // Clamp to min
    return (float16)fp32_value;  // Convert normally
}
```

### Check for Excessive Denormal Generation

```c
// Performance issue: Too many denormals slow down computation
size_t count_denormals_fp16(float16* data, size_t count) {
    size_t denormal_count = 0;
    for (size_t i = 0; i < count; i++) {
        uint16_t bits = *(uint16_t*)&data[i];
        uint16_t exp = (bits & 0x7C00) >> 10;
        uint16_t mantissa = bits & 0x03FF;
        if (exp == 0 && mantissa != 0) {
            denormal_count++;
        }
    }
    return denormal_count;
}

// If > 1% of values are denormal, consider:
// 1. Enable FTZ mode
// 2. Scale inputs to normal range
// 3. Use FP32 for that computation
```

### Validate Intermediate Results

```c
// Run in FP32 reference, compare intermediate stages
void debug_fp16_gemm() {
    // Compute in FP16
    fp16_gemm(A_fp16, B_fp16, C_fp16, ...);

    // Also compute in FP32
    fp32_gemm(A_fp32, B_fp32, C_fp32, ...);

    // Compare not just final output, but intermediate:
    // - After K-loop accumulation
    // - After bias addition
    // - After each activation

    // This pinpoints where divergence occurs
}
```

### Use FP32 Reference for Comparison

```c
// Always maintain a golden FP32 reference
float* compute_fp32_reference(float16* A, float16* B, int M, int N, int K) {
    float* A_f32 = convert_fp16_to_fp32(A, M*K);
    float* B_f32 = convert_fp16_to_fp32(B, K*N);
    float* C_f32 = malloc(M*N*sizeof(float));

    // Use naive FP32 GEMM (correctness over speed)
    naive_gemm_fp32(A_f32, B_f32, C_f32, M, N, K);

    return C_f32;  // Compare against FP16 result (converted to FP32)
}
```

### Common Pitfalls

1. **Sign zero confusion**: `-0.0 != 0.0` in bit pattern, but `-0.0 == 0.0` in comparison
   ```c
   float16 pos_zero = 0.0f;   // 0x0000
   float16 neg_zero = -0.0f;  // 0x8000
   assert(pos_zero == neg_zero);  // True (comparison ignores sign)
   assert(memcmp(&pos_zero, &neg_zero, 2) != 0);  // Different bits!
   ```

2. **NaN comparison pitfall**: `NaN != NaN` (always false)
   ```c
   float16 nan = NAN;
   assert(nan != nan);  // True! Use isnan() instead
   ```

3. **Truncation vs rounding**: FP32→FP16 conversion matters
   ```c
   float val = 1.0009765625f;  // Exactly between two FP16 values
   float16 truncated = (float16)val;  // C-cast might truncate
   float16 rounded = f32_to_fp16_rne(val);  // Round-to-nearest-even preferred
   ```

4. **Aliasing issues**: FP16 stored as `uint16_t`, not `float`
   ```c
   uint16_t fp16_bits = 0x3C00;  // Bit pattern for 1.0
   float16 val = *(float16*)&fp16_bits;  // Type-punning OK for FP16
   ```

## 10. References

### Standards and Specifications

- **IEEE 754-2008 Standard**: Defines FP16 (binary16) format officially
  - URL: https://standards.ieee.org/standard/754-2008.html
  - Covers all floating-point formats, rounding, exceptions

### Processor Documentation

- **Intel AVX512-FP16 ISA Extensions**:
  - Instruction Set Reference: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
  - Optimization Guide: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/
  - Sapphire Rapids (4th Gen Xeon) features AVX512-FP16

- **AMD Zen 5 Architecture**:
  - Software Optimization Guide: https://www.amd.com/en/support/tech-docs
  - EPYC 9005 Series and Ryzen 9000 series support AVX512-FP16

### Research Papers and Articles

- **"FP16 for Deep Learning"** - NVIDIA whitepaper on mixed precision training
- **"BFloat16: Hardware Numerics Definition"** - Intel/Google whitepaper comparing BF16 and FP16
- **"Mixed Precision Training"** (Micikevicius et al., ICLR 2018) - Seminal work on FP16 in ML

### Online Resources

- **Float Toy**: Interactive tool to visualize FP16/FP32/BF16 representations
  - URL: https://evanw.github.io/float-toy/
  - Great for understanding bit patterns

- **Compiler Explorer (godbolt.org)**: See generated assembly for FP16 intrinsics
  - Example: https://godbolt.org/ (search for vcvtph2ps, vdpphps)

### Tools and Libraries

- **AOCL (AMD Optimizing CPU Libraries)**: Vendor-optimized BLAS/LAPACK
  - URL: https://www.amd.com/en/developer/aocl.html

- **Intel MKL (Math Kernel Library)**: Intel's optimized math library
  - FP16 support in recent versions

- **SLEEF (SIMD Library for Evaluating Elementary Functions)**:
  - High-accuracy transcendental functions (sin, exp, etc.) for FP16
  - URL: https://sleef.org/

---

## 11. Implementation Status

### Completed Components ✅

1. **Type System**
   - `float16` typedef in `include/classic/aocl_fp16_type.h`
   - Integrated into operation type enums
   - CPU feature detection for AVX512-FP16

2. **Conversion Functions**
   - Software implementations: `tests/utils/conversion_utils.cc`
   - `fp16_to_f32()`: Lossless FP16 → FP32
   - `f32_to_fp16()`: Lossy FP32 → FP16 with round-to-nearest-even
   - `f32_to_fp16_vcvtps2ph()`: Hardware-accelerated path (when available)
   - **IEEE 754 compliant**: Overflow to infinity, proper denormal handling

3. **Test Infrastructure**
   - Matrix class FP16 support (`tests/framework/matrix.cc`)
   - Data generation with safe ranges
   - Tolerance-based comparison (abs=0.01, rel=0.001)
   - Pattern filling and value initialization
   - YAML configuration parsing for FP16 types

4. **Reference Implementations**
   - `aocl_gemm_fp16fp16f32of32_ref()`: FP16×FP16→FP32 GEMM
   - `aocl_gemm_fp16fp16f32of16_ref()`: FP16×FP16→FP16 GEMM
   - Naive triple-loop implementation for correctness
   - Integrated into UAL test dispatcher

5. **Comprehensive Testing**
   - **32 unit tests** for conversion functions (all passing)
   - Edge case coverage: overflow, underflow, denormals, rounding
   - Round-trip validation across full FP16 range
   - ULP spacing and monotonicity verification

6. **API Stubs**
   - `aocl_gemm_fp16fp16f32of32()` stub in `classic/aocl_gemm_fp16fp16f32of32.c`
   - `aocl_gemm_fp16fp16f32of16()` stub in `classic/aocl_gemm_fp16fp16f32of16.c`
   - Utility stubs in `classic/aocl_gemm_fp16_utils.c`

7. **Documentation**
   - Technical primer (this document)
   - Conversion reference with algorithms
   - Visual diagrams
   - Implementation status tracking

### Pending Components 🔄

**Note**: Build system integration (CMake) will be handled separately. Focus is on testing infrastructure correctness first.

Future work (not in current scope):
- Optimized FP16 kernels with AVX512-FP16 instructions
- JIT code generation for FP16 paths
- Decision engine backend for FP16
- Reordering/packing for FP16 matrices
- Post-operations support
- Batch GEMM variants

---

**Document Version**: 1.1
**Last Updated**: January 2025
**Target Audience**: AOCL-DLP developers implementing FP16 support
**Status**: Testing infrastructure complete and validated ✅

For questions or corrections, contact the AOCL-DLP team.
