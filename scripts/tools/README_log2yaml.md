# log2yaml - Quick Reference Guide

Parse GoogleTest logs and generate YAML configs for failed tests.

## Quick Start

```bash
# From project root
./scripts/tools/log2yaml.sh test.log
```

Generates `failed_tests_config.yaml` with all failed test configurations.

## Common Commands

| Command | Description |
|---------|-------------|
| `./scripts/tools/log2yaml.sh test.log` | Generate failed_tests_config.yaml |
| `./scripts/tools/log2yaml.sh test.log -o custom.yaml` | Specify output file |
| `./scripts/tools/log2yaml.sh test.log -v` | Verbose mode (show all found failures) |
| `./scripts/tools/log2yaml.sh test.log --dry-run` | Preview without writing file |
| `./scripts/tools/log2yaml.sh --help` | Show all options |

## Example Output

For the test.log with 42 failures, the tool generates:

```yaml
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

This configuration will reproduce all 42 failed test combinations (6 M values × 7 K values = 42 tests).

## What Gets Parsed

From each failed test in the log, the tool extracts:

- **Dimensions**: M, N, K matrix sizes
- **Data types**: A, B, C matrix types and accumulator type
- **Operations**: Transposition flags (transA, transB)
- **Memory layout**: Leading dimensions (lda, ldb, ldc), storage format
- **Scalars**: Alpha and beta values
- **Optimizations**: Reordering and packing flags (mtagA, mtagB)

## Intelligent Grouping

Tests with identical parameters except M, N, K are automatically grouped into single configurations. This:

- Reduces config file size
- Makes patterns more visible
- Simplifies test reproduction

## Building from Source

If the tool isn't built yet:

```bash
cd scripts/tools/log2yaml
cargo build --release
cd ../../..
```

See full documentation in `scripts/tools/log2yaml/README.md`

## Integration with Test Framework

1. Run tests and save output:
   ```bash
   cd build
   ctest > ../test.log 2>&1
   ```

2. Generate failure config:
   ```bash
   cd ..
   ./scripts/tools/log2yaml.sh test.log -o tests/classic/configs/regression_tests.yaml
   ```

3. Run regression tests:
   ```bash
   # Use your test framework with the generated config
   ```

## Troubleshooting

**"No failed tests found"**
- Ensure the log contains GoogleTest formatted output
- Check that failures include detailed test information

**"Could not parse complete details"** (verbose mode)
- Some test details might be missing from the log
- Ensure tests print full parameter details on failure

**Tool not found**
- Run the build command above
- Or use direct path: `scripts/tools/log2yaml/target/release/log2yaml`

## See Also

- Full documentation: `scripts/tools/log2yaml/README.md`
- Example configs: `tests/classic/configs/gemm_test_config.yaml`
- Test framework: `tests/classic/test_yaml_gemm.cc`
