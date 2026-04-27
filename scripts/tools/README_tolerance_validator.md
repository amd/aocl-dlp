# Post-Op Tolerance Validation Tool

A Python tool that uses the **Error Propagation Model** to calculate and validate tolerance values for GEMM post-operation tests.

## Overview

This tool implements the error propagation mathematics in `scripts/tools/validators/validate_postop_tolerances.py` as a standalone Python script. It helps you:

1. **Calculate** correct tolerance values for any post-op chain
2. **Validate** existing YAML configurations against the model
3. **Update** YAML files with computed tolerances
4. **Detect issues** like missing tolerances, default parameters, and cartesian mode

## Features

- **Full parameter support**: Properly handles `scale_factor`, `alpha`, `min_val`, `max_val` for parameterized post-ops
- **Cartesian mode detection**: Adjusts tolerances by 1.5x when cartesian mode is enabled
- **Validation warnings**: Reports missing tolerances, default scale factors, and other potential issues
- **Error handling**: Raises clear errors for unsupported post-op types (no silent fallbacks)

## Prerequisites

```bash
pip install pyyaml
```

## Usage

### List All Post-Op Tolerances

Show tolerance values for all supported post-operations:

```bash
python ./scripts/tools/validators/validate_postop_tolerances.py --list-all
```

Output:
```
======================================================================
TOLERANCE VALUES FOR ALL POST-OPERATIONS
======================================================================

Post-Op         Tight (ULPs)    Loose (ULPs)    Rationale
----------------------------------------------------------------------
RELU            50.0            125.0           ReLU is exact...
GELU_ERF        104.2           375.0           GELU_ERF uses erf...
...
```

### Calculate Tolerance for Post-Op Chain

Calculate tolerances for specific post-ops with step-by-step error propagation:

```bash
# Single post-op
python ./scripts/tools/validators/validate_postop_tolerances.py --calculate GELU_ERF --dtype bf16 --k 32

# Chained post-ops
python ./scripts/tools/validators/validate_postop_tolerances.py --calculate GELU_ERF,SCALE --dtype bf16 --k 1 -v
```

Output shows:
- Error model parameters for each post-op
- Step-by-step error propagation
- Final ULP counts for tight and loose tolerances
- Suggested YAML configuration

### Validate YAML File

Check if tolerance values in a YAML configuration are correct:

```bash
python ./scripts/tools/validators/validate_postop_tolerances.py tests/classic/configs/gemm_test_config_postops_basic.yaml
```

The tool will:
- Parse each test case
- Extract post-op chain, data types, and k dimension
- Calculate expected tolerances
- Compare with existing values
- Report discrepancies

### Update YAML with Correct Tolerances

Automatically update YAML files with calculated tolerances:

```bash
python ./scripts/tools/validators/validate_postop_tolerances.py --update tests/classic/configs/gemm_test_config_postops_basic.yaml
```

### Verbose Mode

Show all test cases including correct ones:

```bash
python ./scripts/tools/validators/validate_postop_tolerances.py --verbose tests/classic/configs/*.yaml
```

### Save Report to File

```bash
python ./scripts/tools/validators/validate_postop_tolerances.py --output report.txt tests/classic/configs/*.yaml
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--list-all` | `-l` | List tolerance values for all post-ops |
| `--calculate` | `-c` | Calculate tolerance for comma-separated post-ops |
| `--dtype` | `-t` | Data type for calculation (f32, bf16, fp16). Default: f32 |
| `--k` | | K dimension for calculation. Default: 32 |
| `--base-multiplier` | `-b` | Base GEMM error in ULPs. Default: 50.0 |
| `--update` | `-u` | Update YAML files with calculated tolerances |
| `--verbose` | `-v` | Show detailed output |
| `--output` | `-o` | Write report to file |

## Error Propagation Model

The tool uses the following model for error propagation:

```
error_out = error_in × |f'(x)| + additive_error
```

Where:
- `error_in`: Error from previous operation (initially 50 ULPs from GEMM)
- `|f'(x)|`: Derivative (amplification factor) of the post-op function
- `additive_error`: New error introduced by the operation

### Supported Post-Operations

| Post-Op | Amplification | Additive ULPs | Notes |
|---------|---------------|---------------|-------|
| RELU | 1.0 | 0 | Exact operation |
| PRELU | max(alpha, 1.0) | 1 | Uses `alpha` parameter |
| SIGMOID | 0.25 | 4 | exp + division |
| TANH | 1.0 | 7 | Two exp + division |
| SWISH | 1.1 | 5 | sigmoid × x |
| GELU_TANH | 1.0 | 10 | tanh + cubic |
| GELU_ERF | 1.5 | 50 | erf (varies by impl) |
| CLIP | 1.0 | 0 | Uses `min`/`max` parameters |
| BIAS | 1.0 | 1 | One addition |
| SCALE | \|scale_factor\| | 1 | Uses `scale_factor` parameter |
| MATRIX_ADD | 1.0 | 2 | Two error sources |
| MATRIX_MUL | 1.0 | 2 | Two error sources |

### Parameterized Post-Ops

The tool extracts parameters from YAML for these post-ops:

- **SCALE**: Uses `scale_factor` or `scale` (default: 1.0). Amplification equals |scale_factor|.
- **PRELU**: Uses `alpha` (default: 0.01). Amplification is max(alpha, 1.0).
- **CLIP**: Uses `min`/`min_val` and `max`/`max_val` (defaults: 0.0 and 6.0).

### Cartesian Mode

When `cartesian: true` is set in the YAML (either at the post_operations level or per-operation), tolerances are multiplied by **1.5x** to account for additional error accumulation in cartesian mode.

### Error Handling

Unknown post-op types will raise an `UnsupportedPostOpError` with a helpful message listing all supported types. The tool **does not silently fall back** to any default.

## Example: GELU_ERF → SCALE Chain

```bash
$ python ./scripts/tools/validators/validate_postop_tolerances.py --calculate GELU_ERF,SCALE --dtype bf16 --k 32

Error Propagation (Step by Step):
------------------------------------------------------------
  Step 0 (GEMM): error = 50.0 ULPs
  Step 1 (GELU_ERF):
    input_value = 1.0000
    |f'(x)| = 1.0833
    additive = 50.0 ULPs
    error = 50.0 × 1.0833 + 50.0 = 104.2 ULPs
  Step 2 (SCALE):
    input_value = 0.8413
    |f'(x)| = 1.0000
    additive = 1.0 ULPs
    error = 104.2 × 1.0000 + 1.0 = 105.2 ULPs

YAML Configuration:
------------------------------------------------------------
  tolerances:
    relative: 377.5   # loose tolerance
    absolute: 105.2   # tight tolerance
```

## Warnings and Validation

The tool provides warnings for common issues:

- **Missing tolerances**: Warns if `relative` or `absolute` tolerance is not defined
- **Default scale factor**: Warns if SCALE operation uses the default `scale_factor=1.0`
- **Cartesian mode without tolerances**: Warns if cartesian mode is enabled but tolerances are missing
- **Unsupported post-ops**: Raises an error with clear message listing supported types


## Files

- `./scripts/tools/validators/validate_postop_tolerances.py` - Main tool
- `scripts/tools/README_tolerance_validator.md` - This documentation
