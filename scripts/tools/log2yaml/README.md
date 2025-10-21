# log2yaml - Test Log Parser for YAML Config Generation

A Rust utility that parses GoogleTest logs and generates YAML test configuration files for reproducing failed test cases.

## Overview

When tests fail, this tool extracts the failure details from the log and generates a YAML configuration file that can be used to reproduce only the failed test cases. This is particularly useful for:

- Debugging specific test failures
- Creating targeted regression test suites
- Analyzing failure patterns
- Re-running failed tests efficiently

## Building

From the project root:

```bash
cd scripts/tools/log2yaml
cargo build --release
```

The binary will be created at `target/release/log2yaml`

## Usage

### Basic Usage

Parse a test log and generate a YAML config:

```bash
./scripts/tools/log2yaml/target/release/log2yaml test.log
```

This creates `failed_tests_config.yaml` in the current directory.

### Specify Output File

```bash
./scripts/tools/log2yaml/target/release/log2yaml test.log -o my_failed_tests.yaml
```

### Verbose Mode

Show detailed parsing information:

```bash
./scripts/tools/log2yaml/target/release/log2yaml test.log -v
```

### Dry Run

Preview what would be generated without writing a file:

```bash
./scripts/tools/log2yaml/target/release/log2yaml test.log --dry-run
```

### Help

```bash
./scripts/tools/log2yaml/target/release/log2yaml --help
```

## Example Workflow

1. **Run tests and capture output:**
   ```bash
   cd build
   ctest > ../test.log 2>&1
   ```

2. **Generate config for failed tests:**
   ```bash
   cd ..
   ./scripts/tools/log2yaml/target/release/log2yaml test.log -o failed_tests.yaml
   ```

3. **Copy to test configs directory:**
   ```bash
   cp failed_tests.yaml tests/classic/configs/failed_tests_config.yaml
   ```

4. **Re-run only failed tests:**
   Use the generated config with your test framework to reproduce the failures.

## How It Works

### Parsing Strategy

The tool uses regex patterns to extract information from GoogleTest output:

1. **Identifies failed tests** by looking for `[  FAILED  ]` markers
2. **Extracts test details** from the structured output that precedes each failure:
   - Matrix dimensions (M, N, K)
   - Data types (A, B, C, accumulator)
   - Transposition flags
   - Leading dimensions (lda, ldb, ldc)
   - Alpha/Beta values
   - Reordering and packing flags

3. **Groups similar tests** by common parameters (except M, N, K)
4. **Generates optimized YAML** using ranges or lists as appropriate

### Smart Grouping

Tests with identical parameters (except M, N, K) are grouped into a single configuration entry. This creates minimal, efficient config files.

### Dimension Specification

The tool automatically chooses the most compact representation:

- **Single value**: `m: 143`
- **Short list**: `m: [143, 144, 145]`
- **Range**: `m: {lb: 10, ub: 20, step: 1}` (or `step: -1` for step=1)

## Output Format

The generated YAML follows the same structure as `tests/classic/configs/gemm_test_config.yaml`:

```yaml
# GEMM test configuration for failed tests
# Generated from: "test.log"

gemm_tests:
  - name: "failed_bf16bf16f32_group_0"
    a_type: ["bf16"]
    b_type: ["bf16"]
    c_type: ["f32"]
    acc_type: ["f32"]
    storage_format: ["row-major"]
    transA: [false]
    transB: [false]
    m: [143, 144, 145, 287, 288, 289]
    n: 1025
    k: [2048, 4095, 4096, 4097, 8191, 8192, 8193]
    alpha: [1.0]
    beta: [0.0]
    lda: 8193
    ldb: 8193
    ldc: 2048
    mtagA: ["none"]
    mtagB: ["none"]
    tolerances:
      float: 0.00001
      bfloat16: 0.01
      int8: 0
```

## Supported Test Formats

Currently supports GoogleTest output format with structured failure details including:

- Matrix dimensions
- Data types (f32, bf16, s8, u8, s32, etc.)
- Transposition flags
- Leading dimensions
- Alpha/Beta scalars
- Reordering/packing flags
- Post-ops (future enhancement)

## Limitations

- Only parses GoogleTest format logs
- Requires structured test details in the log (as printed by the test framework)
- Groups tests aggressively - review generated config if you need finer control

## Development

### Running Tests

```bash
cargo test
```

### Building for Development

```bash
cargo build
```

The debug binary will be at `target/debug/log2yaml`

### Code Structure

- `main.rs` - Complete implementation including:
  - CLI argument parsing (clap)
  - Log parsing with regex
  - Test grouping logic
  - YAML generation (serde_yaml)
  - Error handling (anyhow)

## Dependencies

- `regex` - Pattern matching for log parsing
- `serde` + `serde_yaml` - YAML serialization
- `clap` - Command-line argument parsing
- `anyhow` - Error handling

## License

Same as AOCL-DLP project.
