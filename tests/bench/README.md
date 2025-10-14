# Benchmark Framework Tests

This directory contains unit tests for the benchmark framework utility functions.

## Test Coverage

The test suite `test_bench_utilities` validates the following benchmark framework components:

### 1. GemmBenchConfig Hash Function (`BenchConfigHashTest`)
- Hash determinism (same config produces same hash)
- Hash uniqueness for different dimensions
- Hash sensitivity to all config fields (data types, transpose, reordering)
- No collisions for common configurations

### 2. Benchmark Name Generation (`BenchmarkNameGenerationTest`)
- Basic name format validation
- Data type encoding in names
- Transpose flag variations (transA/transB)
- Reordering flag variations (mtagA/mtagB)
- Storage format encoding (row-major/column-major)
- Uniqueness for different configurations

### 3. Matrix Type Sizes (`MatrixTypeSizeTest`)
- Correct byte sizes for all supported matrix types:
  - 4-bit types (u4, s4): 1 byte
  - 8-bit types (u8, s8): 1 byte
  - 16-bit types (u16, s16, bf16): 2 bytes
  - 32-bit types (u32, s32, f32): 4 bytes

### 4. YAML Configuration Loading (`YamlConfigLoadingTest`)
- Valid YAML file parsing
- Configuration count verification
- Field value validation for loaded configs
- Automatic benchmark name generation
- Error handling for missing/invalid files

## Running Tests

```bash
# Build tests
cmake -B build -DTESTS=ON -DBENCHMARKS=ON
cmake --build build --target test_bench_utilities

# Run tests
./build/tests/bench/test_bench_utilities

# Or via CTest
cd build && ctest -R test_bench_utilities --output-on-failure
```

## Test Configuration

The tests use a minimal YAML configuration file located at:
- `tests/bench/configs/bench_test_minimal.yaml`

This file contains 2 test cases with known values for validation.

## Architecture

The test follows test-driven development principles by validating:
- Pure utility functions with clear inputs/outputs
- Edge cases and error handling
- Consistency of hash functions and name generation
- Correctness of type-to-size mappings

The test maximally reuses existing infrastructure:
- Uses Google Test framework (same as other tests)
- Reuses benchmark source files from `bench/src/`
- Shares framework/utils/adaptors sources with other tests
- Minimal CMake additions (only benchmark dependency)
