#!/usr/bin/env python3
"""
AOCL-DLP Post-Op Tolerance Validation Tool

This tool uses the error propagation model to calculate correct tolerance values
for GEMM post-operation tests and validates them against YAML configuration files.

Usage:
    # Validate tolerances in a YAML file
    python validate_postop_tolerances.py tests/classic/configs/gemm_test_config_postops_basic.yaml

    # Update YAML with correct tolerances
    python validate_postop_tolerances.py --update gemm_test_config_postops_basic.yaml

    # Calculate tolerances for specific post-ops
    python validate_postop_tolerances.py --calculate GELU_ERF,SCALE --dtype bf16 --k 32

    # Generate detailed report
    python validate_postop_tolerances.py --verbose --output report.txt *.yaml

"""

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

# Try to import yaml, provide helpful error if not available
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install it with: pip install pyyaml")
    sys.exit(1)


# =============================================================================
# DATA TYPE EPSILON VALUES
# =============================================================================

@dataclass
class DataTypeEpsilon:
    """Machine epsilon values for different data types"""
    f32: float = 1.1920929e-7   # 2^-23 (23 mantissa bits)
    bf16: float = 0.0078125      # 2^-7 (7 mantissa bits)
    fp16: float = 0.0009765625   # 2^-10 (10 mantissa bits)

    @classmethod
    def get(cls, dtype: str) -> float:
        """Get epsilon for a data type string"""
        dtype_lower = dtype.lower()
        if dtype_lower in ('f32', 'float', 'float32'):
            return cls.f32
        elif dtype_lower in ('bf16', 'bfloat16'):
            return cls.bf16
        elif dtype_lower in ('fp16', 'f16', 'float16', 'half'):
            return cls.fp16
        elif dtype_lower in ('s8', 's16', 's32', 'u8', 'u16', 'u32', 'int8', 'int16', 'int32'):
            return 0.0  # Integer types are exact
        elif dtype_lower in ('s4', 'u4', 'int4', 'uint4'):
            return 0.0  # 4-bit quantized types - use input type epsilon instead
        else:
            print(f"Warning: Unknown data type '{dtype}', using f32 epsilon")
            return cls.f32


# =============================================================================
# POST-OP ERROR MODEL
# =============================================================================

@dataclass
class PostOpConfig:
    """
    Configuration for a single post-operation extracted from YAML.

    Attributes:
        op_type: The post-operation type (e.g., "RELU", "SCALE", "GELU_ERF")
        scale_factor: Scale factor for SCALE operation (default: 1.0)
        alpha: Alpha parameter for PRELU (default: 0.01)
        min_val: Minimum value for CLIP (default: 0.0)
        max_val: Maximum value for CLIP (default: 6.0)
        is_cartesian: Whether cartesian mode is used (affects error calculation)
    """
    op_type: str
    scale_factor: float = 1.0
    alpha: float = 0.01
    min_val: float = 0.0
    max_val: float = 6.0
    is_cartesian: bool = False


@dataclass
class PostOpErrorModel:
    """
    Error propagation model for a single post-operation.

    Mirrors the C++ PostOpErrorModel struct from postop_tolerance.hh

    Attributes:
        amplification_factor: Typical |f'(x)| - how input error is scaled
        max_amplification: Maximum |f'(x)| over the domain (worst-case)
        additive_error_ulps: Error introduced by this operation (in ULPs)
        critical_threshold: Threshold for special handling near critical points
        derivative_fn: Function that computes |f'(x)| given input value
        transform_fn: Function that computes f(x) for tracking values through chain
        name: Post-op name for reporting
        rationale: Brief description of why these values were chosen
    """
    name: str
    amplification_factor: float
    max_amplification: float
    additive_error_ulps: float
    critical_threshold: float = 1e-6
    derivative_fn: Optional[Callable[[float], float]] = None
    transform_fn: Optional[Callable[[float], float]] = None
    rationale: str = ""


# =============================================================================
# PRE-DEFINED ERROR MODELS FOR ALL POST-OPS
# =============================================================================

def create_relu_model() -> PostOpErrorModel:
    """
    ReLU: y = max(0, x)

    Derivative: y' = 1 if x > 0, else 0
    - For positive inputs: passes error through unchanged
    - For negative inputs: error is "absorbed" (output is 0)
    - No computational error (exact operation)
    """
    return PostOpErrorModel(
        name="RELU",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=0.0,
        critical_threshold=1e-6,
        derivative_fn=lambda x: 1.0 if x > 0 else 0.0,
        transform_fn=lambda x: max(0.0, x),
        rationale="ReLU is exact; tolerance only needed at zero boundary"
    )


def create_prelu_model(alpha: float = 0.01) -> PostOpErrorModel:
    """
    PReLU: y = max(alpha*x, x) where alpha < 1

    Derivative: y' = 1 if x > 0, else alpha
    - Scales error by alpha for negative inputs
    """
    return PostOpErrorModel(
        name="PRELU",
        amplification_factor=max(alpha, 1.0),
        max_amplification=1.0,
        additive_error_ulps=1.0,
        critical_threshold=1e-6,
        derivative_fn=lambda x, a=alpha: 1.0 if x > 0 else a,
        transform_fn=lambda x, a=alpha: x if x > 0 else a * x,
        rationale="PReLU adds alpha multiplication error for negative inputs"
    )


def create_sigmoid_model() -> PostOpErrorModel:
    """
    SIGMOID: y = 1 / (1 + exp(-x))

    Derivative: y' = y * (1 - y) = sigmoid(x) * (1 - sigmoid(x))
    - Maximum at x=0 where y' = 0.25
    - Compresses errors for large |x|
    """
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x)) if x > -500 else 0.0

    return PostOpErrorModel(
        name="SIGMOID",
        amplification_factor=0.25,
        max_amplification=0.25,
        additive_error_ulps=4.0,
        critical_threshold=5.0,
        derivative_fn=lambda x: sigmoid(x) * (1.0 - sigmoid(x)),
        transform_fn=sigmoid,
        rationale="SIGMOID involves exp and division"
    )


def create_tanh_model() -> PostOpErrorModel:
    """
    TANH: y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Derivative: y' = 1 - tanh²(x) = sech²(x)
    - Maximum at x=0 where y' = 1.0
    - Compresses errors for large |x|
    """
    return PostOpErrorModel(
        name="TANH",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=7.0,
        critical_threshold=3.0,
        derivative_fn=lambda x: 1.0 - math.tanh(x) ** 2,
        transform_fn=math.tanh,
        rationale="TANH involves two exp, subtraction with potential cancellation"
    )


def create_swish_model() -> PostOpErrorModel:
    """
    SWISH: y = x * sigmoid(x) = x / (1 + exp(-x))

    Derivative: y' = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    - Has maximum around x ≈ 1.28 where y' ≈ 1.1
    """
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x)) if x > -500 else 0.0

    return PostOpErrorModel(
        name="SWISH",
        amplification_factor=1.1,
        max_amplification=1.1,
        additive_error_ulps=5.0,
        critical_threshold=5.0,
        derivative_fn=lambda x: sigmoid(x) * (1.0 + x * (1.0 - sigmoid(x))),
        transform_fn=lambda x: x / (1.0 + math.exp(-x)) if x > -500 else 0.0,
        rationale="SWISH involves exp, division, and multiplication"
    )


def create_gelu_tanh_model() -> PostOpErrorModel:
    """
    GELU_TANH: y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Complex derivative - approximated for efficiency
    Maximum derivative is approximately 1.0 at x ≈ 0
    """
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715

    def transform(x):
        x3 = x * x * x
        return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + coeff * x3)))

    def derivative(x):
        x3 = x * x * x
        inner = sqrt_2_over_pi * (x + coeff * x3)
        t = math.tanh(inner)
        sech2 = 1.0 - t * t
        inner_deriv = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
        return 0.5 * (1.0 + t) + 0.5 * x * sech2 * inner_deriv

    return PostOpErrorModel(
        name="GELU_TANH",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=10.0,
        critical_threshold=3.0,
        derivative_fn=derivative,
        transform_fn=transform,
        rationale="GELU_TANH involves tanh, cubic, and multiple multiplications"
    )


def create_gelu_erf_model() -> PostOpErrorModel:
    """
    GELU_ERF: y = 0.5 * x * (1 + erf(x / sqrt(2)))

    Note: erf implementations can vary significantly between libraries.
    When comparing DLP (optimized) vs REF implementations, larger errors are expected.
    The error is especially pronounced for bf16 inputs converted to f32 output.
    """
    inv_sqrt_2 = 0.7071067811865475

    def transform(x):
        return 0.5 * x * (1.0 + math.erf(x * inv_sqrt_2))

    def derivative(x):
        erf_val = math.erf(x * inv_sqrt_2)
        gaussian = math.exp(-0.5 * x * x) * inv_sqrt_2 * 1.1283791670955126  # 2/sqrt(π)
        return 0.5 * (1.0 + erf_val) + 0.5 * x * gaussian

    return PostOpErrorModel(
        name="GELU_ERF",
        amplification_factor=1.5,
        max_amplification=2.0,
        additive_error_ulps=50.0,  # erf implementations vary significantly
        critical_threshold=3.0,
        derivative_fn=derivative,
        transform_fn=transform,
        rationale="GELU_ERF uses erf which has polynomial approximation error"
    )


def create_clip_model(min_val: float = 0.0, max_val: float = 6.0) -> PostOpErrorModel:
    """
    CLIP: y = min(max(x, min_val), max_val)

    Derivative: y' = 1 if min_val < x < max_val, else 0
    """
    return PostOpErrorModel(
        name="CLIP",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=0.0,
        critical_threshold=1e-6,
        derivative_fn=lambda x, lo=min_val, hi=max_val: 1.0 if lo < x < hi else 0.0,
        transform_fn=lambda x, lo=min_val, hi=max_val: min(max(x, lo), hi),
        rationale="CLIP is exact; tolerance needed at min/max boundaries"
    )


def create_bias_model() -> PostOpErrorModel:
    """
    BIAS: y = x + bias

    Derivative: y' = 1
    Error passes through unchanged, but addition adds 1 ULP
    """
    return PostOpErrorModel(
        name="BIAS",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=1.0,
        critical_threshold=1e-10,
        derivative_fn=lambda x: 1.0,
        transform_fn=lambda x: x,  # Bias value unknown
        rationale="BIAS is simple addition but cancellation can occur"
    )


def create_scale_model(scale_factor: float = 1.0) -> PostOpErrorModel:
    """
    SCALE: y = x * scale_factor

    Derivative: y' = scale_factor
    Error is amplified by |scale_factor|
    """
    return PostOpErrorModel(
        name="SCALE",
        amplification_factor=abs(scale_factor),
        max_amplification=abs(scale_factor),
        additive_error_ulps=1.0,
        critical_threshold=1e-10,
        derivative_fn=lambda x, sf=scale_factor: abs(sf),
        transform_fn=lambda x, sf=scale_factor: x * sf,
        rationale="SCALE is multiplication; error scales with scale factor"
    )


def create_matrix_add_model() -> PostOpErrorModel:
    """
    MATRIX_ADD: y = x + matrix_element

    Like BIAS but accounting for error in the matrix being added
    """
    return PostOpErrorModel(
        name="MATRIX_ADD",
        amplification_factor=1.0,
        max_amplification=1.0,
        additive_error_ulps=2.0,
        critical_threshold=1e-10,
        derivative_fn=lambda x: 1.0,
        transform_fn=lambda x: x,
        rationale="MATRIX_ADD combines errors from two sources"
    )


def create_matrix_mul_model(typical_value: float = 1.0) -> PostOpErrorModel:
    """
    MATRIX_MUL: y = x * matrix_element

    For element-wise multiplication, relative errors add
    """
    return PostOpErrorModel(
        name="MATRIX_MUL",
        amplification_factor=abs(typical_value),
        max_amplification=abs(typical_value) * 2.0,
        additive_error_ulps=2.0,
        critical_threshold=1e-10,
        derivative_fn=lambda x, tv=typical_value: abs(tv),
        transform_fn=lambda x, tv=typical_value: x * tv,
        rationale="MATRIX_MUL compounds relative errors from two sources"
    )


# =============================================================================
# POST-OP MODEL REGISTRY
# =============================================================================

# Supported post-op types for validation
SUPPORTED_POSTOPS = {
    "RELU", "PRELU", "SIGMOID", "TANH", "SWISH",
    "GELU_TANH", "GELU_ERF", "CLIP",
    "BIAS", "SCALE", "MATRIX_ADD", "MATRIX_MUL"
}

# Map of post-op type to model factory function
POSTOP_MODEL_FACTORIES = {
    "RELU": create_relu_model,
    "PRELU": create_prelu_model,
    "SIGMOID": create_sigmoid_model,
    "TANH": create_tanh_model,
    "SWISH": create_swish_model,
    "GELU_TANH": create_gelu_tanh_model,
    "GELU_ERF": create_gelu_erf_model,
    "CLIP": create_clip_model,
    "BIAS": create_bias_model,
    "SCALE": create_scale_model,
    "MATRIX_ADD": create_matrix_add_model,
    "MATRIX_MUL": create_matrix_mul_model,
}


class UnsupportedPostOpError(ValueError):
    """Raised when an unsupported post-op type is encountered."""
    pass


def normalize_postop_type(postop_type: str) -> str:
    """
    Normalize a post-op type string.

    Args:
        postop_type: Post-operation type (e.g., "RELU", "Elementwise-RELU")

    Returns:
        Normalized post-op type string
    """
    normalized = postop_type.upper()
    normalized = normalized.replace("ELEMENTWISE-", "")
    normalized = normalized.replace("ELEMENTWISE_", "")
    normalized = normalized.replace("-", "_")
    return normalized


def get_error_model(postop_config: Union[str, PostOpConfig]) -> PostOpErrorModel:
    """
    Get error model for a post-op configuration.

    Args:
        postop_config: Either a post-operation type string (e.g., "RELU", "GELU_ERF")
                       or a PostOpConfig object with full configuration

    Returns:
        PostOpErrorModel for the specified operation

    Raises:
        UnsupportedPostOpError: If the post-op type is not supported
    """
    # Handle both string and PostOpConfig inputs
    if isinstance(postop_config, str):
        config = PostOpConfig(op_type=postop_config)
    else:
        config = postop_config

    # Normalize post-op type
    normalized = normalize_postop_type(config.op_type)

    if normalized not in POSTOP_MODEL_FACTORIES:
        raise UnsupportedPostOpError(
            f"Unknown post-op type '{config.op_type}'. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_POSTOPS))}. "
            "Please add support for this post-op type or check the spelling."
        )

    # Create model with appropriate parameters based on post-op type
    if normalized == "SCALE":
        # Use actual scale_factor from config
        return create_scale_model(scale_factor=config.scale_factor)
    elif normalized == "PRELU":
        # Use actual alpha from config
        return create_prelu_model(alpha=config.alpha)
    elif normalized == "CLIP":
        # Use actual min/max values from config
        return create_clip_model(min_val=config.min_val, max_val=config.max_val)
    else:
        return POSTOP_MODEL_FACTORIES[normalized]()


# =============================================================================
# ERROR PROPAGATION
# =============================================================================

@dataclass
class ErrorState:
    """Error propagation state through the chain"""
    cumulative_ulps: float
    value: float
    amplification: float


@dataclass
class ComputedTolerance:
    """Computed tolerance values"""
    tight_abs: float
    tight_rel: float
    loose_abs: float
    loose_rel: float

    def __str__(self):
        return f"tight_abs={self.tight_abs:.2f}, tight_rel={self.tight_rel:.2f}, loose_abs={self.loose_abs:.2f}, loose_rel={self.loose_rel:.2f}"


def propagate_error_through_chain(
    error_models: List[PostOpErrorModel],
    initial_error_ulps: float,
    initial_value: float = 1.0,
    use_value_dependent: bool = True
) -> ErrorState:
    """
    Propagate error through a chain of post-operations.

    This function efficiently computes the total error accumulation
    by propagating errors through each post-op in sequence.

    Args:
        error_models: List of error models for each post-op in the chain
        initial_error_ulps: Initial error (typically from GEMM = base_multiplier)
        initial_value: Typical initial value (for value-dependent calculations)
        use_value_dependent: If True, use derivative functions with actual values

    Returns:
        Final error state after propagation

    Algorithm:
        for each postop in chain:
            error_out = error_in × |f'(value)| + additive_error
            value = f(value)  # Update value for next stage
    """
    state = ErrorState(
        cumulative_ulps=initial_error_ulps,
        value=initial_value,
        amplification=1.0
    )

    for model in error_models:
        # Get amplification factor
        amp = model.amplification_factor
        if use_value_dependent and model.derivative_fn:
            try:
                amp = model.derivative_fn(state.value)
            except Exception:
                amp = model.amplification_factor

        # Propagate error: error_out = error_in * |f'(x)| + additive
        state.cumulative_ulps = state.cumulative_ulps * amp + model.additive_error_ulps
        state.amplification *= amp

        # Update value for next stage
        if model.transform_fn:
            try:
                state.value = model.transform_fn(state.value)
            except Exception:
                pass

    return state


def compute_worst_case_error(
    error_models: List[PostOpErrorModel],
    initial_error_ulps: float
) -> float:
    """
    Compute worst-case error for a chain (ignores value dependency).

    Uses maximum amplification factors for conservative estimate.
    """
    cumulative = initial_error_ulps

    for model in error_models:
        cumulative = cumulative * model.max_amplification + model.additive_error_ulps

    return cumulative


def compute_combined_tolerance(
    postop_sequence: List[Union[str, PostOpConfig]],
    epsilon: float,
    k: int,
    base_multiplier: float = 50.0,
    initial_value: float = 1.0,
    use_worst_case: bool = False,
    is_cartesian: bool = False
) -> ComputedTolerance:
    """
    Compute combined tolerance for a sequence of post-operations.

    EFFICIENT IMPLEMENTATION using error propagation model.
    Uses the chain rule to propagate errors: error_out = error_in × |f'(x)| + additive

    Args:
        postop_sequence: List of post-op type strings or PostOpConfig objects in execution order
        epsilon: Machine epsilon for the data type
        k: The k dimension of the GEMM operation
        base_multiplier: Base multiplier for initial GEMM error (default: 50.0)
        initial_value: Optional typical value for value-dependent calculation
        use_worst_case: If True, use max amplification factors (conservative)
        is_cartesian: If True, apply cartesian mode adjustments to tolerance

    Returns:
        ComputedTolerance with accumulated tolerances
    """
    # Base tolerance unit
    base_tolerance = epsilon * k

    if not postop_sequence:
        # No post-ops, return base GEMM tolerance
        return ComputedTolerance(
            tight_abs=base_tolerance * base_multiplier,
            tight_rel=base_tolerance * base_multiplier,
            loose_abs=base_tolerance * base_multiplier * 2.0,
            loose_rel=base_tolerance * base_multiplier * 2.0
        )

    # Build error models for the chain
    postop_error_models = [get_error_model(postop) for postop in postop_sequence]

    # Compute tight tolerance using value-dependent propagation
    tight_state = propagate_error_through_chain(
        postop_error_models,
        base_multiplier,
        initial_value,
        not use_worst_case
    )

    # Compute loose/worst-case tolerance
    if use_worst_case:
        loose_ulps = tight_state.cumulative_ulps
    else:
        loose_ulps = compute_worst_case_error(postop_error_models, base_multiplier)

    # Apply relaxation factor for loose tolerance
    LOOSE_RELAXATION = 2.5

    # Apply cartesian mode adjustment if enabled
    # In cartesian mode, errors can compound differently due to the
    # way operations are applied across matrix dimensions
    CARTESIAN_FACTOR = 1.5 if is_cartesian else 1.0

    return ComputedTolerance(
        tight_abs=tight_state.cumulative_ulps * base_tolerance * CARTESIAN_FACTOR,
        tight_rel=tight_state.cumulative_ulps * base_tolerance * CARTESIAN_FACTOR,
        loose_abs=loose_ulps * base_tolerance * LOOSE_RELAXATION * CARTESIAN_FACTOR,
        loose_rel=loose_ulps * base_tolerance * LOOSE_RELAXATION * CARTESIAN_FACTOR
    )


def get_tolerance_multipliers(
    postop_sequence: List[Union[str, PostOpConfig]],
    base_multiplier: float = 50.0,
    initial_value: float = 1.0,
    is_cartesian: bool = False
) -> Tuple[float, float]:
    """
    Get the ULP multipliers for tight and loose tolerances.

    This returns the multipliers that should be used in YAML files.

    Args:
        postop_sequence: List of post-op type strings or PostOpConfig objects
        base_multiplier: Base GEMM error in ULPs
        initial_value: Typical initial value for calculations
        is_cartesian: Whether cartesian mode is enabled

    Returns:
        (tight_multiplier, loose_multiplier) in ULPs
    """
    if not postop_sequence:
        return (base_multiplier, base_multiplier * 2.0)

    postop_error_models = [get_error_model(postop) for postop in postop_sequence]

    # Tight: value-dependent propagation
    tight_state = propagate_error_through_chain(
        postop_error_models, base_multiplier, initial_value, True
    )

    # Loose: worst-case propagation
    loose_ulps = compute_worst_case_error(postop_error_models, base_multiplier)

    LOOSE_RELAXATION = 2.5

    # Apply cartesian mode adjustment if enabled
    CARTESIAN_FACTOR = 1.5 if is_cartesian else 1.0

    return (tight_state.cumulative_ulps * CARTESIAN_FACTOR,
            loose_ulps * LOOSE_RELAXATION * CARTESIAN_FACTOR)


# =============================================================================
# YAML PARSING
# =============================================================================

def extract_postop_configs(test_config: dict) -> List[PostOpConfig]:
    """
    Extract post-op configurations from a test configuration.

    This extracts not just the post-op types, but also their parameters
    like scale_factor, alpha, min/max values, and cartesian mode.

    Args:
        test_config: Test configuration dictionary from YAML

    Returns:
        List of PostOpConfig objects with full configuration
    """
    postop_configs = []

    post_operations = test_config.get('post_operations', {})

    # Handle case where post_operations is None
    if post_operations is None:
        post_operations = {}

    operations = post_operations.get('operations', [])

    # Handle case where operations is None
    if operations is None:
        operations = []

    # Check for global cartesian setting
    global_cartesian = post_operations.get('cartesian', False)

    for op in operations:
        op_type = op.get('type', '')
        if op_type:
            # Normalize the type
            normalized = normalize_postop_type(op_type)

            # Extract parameters based on post-op type
            config = PostOpConfig(
                op_type=normalized,
                scale_factor=op.get('scale_factor', op.get('scale', 1.0)),
                alpha=op.get('alpha', 0.01),
                min_val=op.get('min', op.get('min_val', 0.0)),
                max_val=op.get('max', op.get('max_val', 6.0)),
                is_cartesian=op.get('cartesian', global_cartesian)
            )
            postop_configs.append(config)

    return postop_configs


def extract_postop_types(test_config: dict) -> List[str]:
    """
    Extract post-op types from a test configuration.

    This is a convenience function that returns just the type names.
    For full configuration including parameters, use extract_postop_configs().

    Args:
        test_config: Test configuration dictionary from YAML

    Returns:
        List of normalized post-op type strings
    """
    configs = extract_postop_configs(test_config)
    return [config.op_type for config in configs]


def is_cartesian_mode(test_config: dict) -> bool:
    """
    Check if cartesian mode is enabled for any post-op in the test.

    Args:
        test_config: Test configuration dictionary from YAML

    Returns:
        True if any post-op uses cartesian mode
    """
    configs = extract_postop_configs(test_config)
    return any(config.is_cartesian for config in configs)


def extract_data_types(test_config: dict) -> Tuple[str, str, str]:
    """Extract a_type, b_type, c_type from test configuration"""
    a_type = test_config.get('a_type', ['f32'])
    b_type = test_config.get('b_type', ['f32'])
    c_type = test_config.get('c_type', ['f32'])

    # Take first value if it's a list
    if isinstance(a_type, list):
        a_type = a_type[0] if a_type else 'f32'
    if isinstance(b_type, list):
        b_type = b_type[0] if b_type else 'f32'
    if isinstance(c_type, list):
        c_type = c_type[0] if c_type else 'f32'

    return a_type, b_type, c_type


def get_appropriate_epsilon(a_type: str, b_type: str, c_type: str) -> float:
    """
    Determine the appropriate epsilon based on data types.

    For GEMM, the accumulated error depends on the input types,
    but the tolerance is applied to the output type.

    For bf16 inputs → f32 output, we need bf16 epsilon because
    that's where the error accumulates.
    """
    # Get epsilon for input types (where error accumulates)
    eps_a = DataTypeEpsilon.get(a_type)
    eps_b = DataTypeEpsilon.get(b_type)

    # Use the larger epsilon (more error-prone type)
    return max(eps_a, eps_b)


def extract_k_values(test_config: dict) -> List[int]:
    """Extract k dimension values from test configuration"""
    k = test_config.get('k', [32])
    if isinstance(k, list):
        return k
    return [k]


def extract_existing_tolerances(test_config: dict) -> Tuple[Optional[float], Optional[float]]:
    """Extract existing tolerance values from test configuration"""
    tolerances = test_config.get('tolerances', {})

    # Handle case where tolerances is None (empty tolerances: section in YAML)
    if tolerances is None:
        tolerances = {}

    relative = tolerances.get('relative')
    absolute = tolerances.get('absolute')

    # Convert string values to float if needed (e.g., "1e-3" -> 0.001)
    if relative is not None and isinstance(relative, str):
        try:
            relative = float(relative)
        except ValueError:
            relative = None

    if absolute is not None and isinstance(absolute, str):
        try:
            absolute = float(absolute)
        except ValueError:
            absolute = None

    return relative, absolute


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a single test case"""
    test_name: str
    postops: List[str]
    a_type: str
    b_type: str
    c_type: str
    k_values: List[int]
    existing_relative: Optional[float]
    existing_absolute: Optional[float]
    calculated_tight: float
    calculated_loose: float
    relative_correct: bool
    absolute_correct: bool
    suggested_relative: float
    suggested_absolute: float
    is_cartesian: bool = False
    has_scale_factor: bool = False
    scale_factor_value: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def validate_test_case(test_config: dict, verbose: bool = False) -> ValidationResult:
    """Validate tolerance values for a single test case"""
    name = test_config.get('name', 'unnamed')
    warnings = []

    # Extract configuration with full parameters
    postop_configs = extract_postop_configs(test_config)
    postops = [config.op_type for config in postop_configs]
    a_type, b_type, c_type = extract_data_types(test_config)
    k_values = extract_k_values(test_config)
    existing_rel, existing_abs = extract_existing_tolerances(test_config)

    # Check for cartesian mode
    cartesian = is_cartesian_mode(test_config)

    # Check for scale factor
    has_scale = any(config.op_type == "SCALE" for config in postop_configs)
    scale_value = None
    if has_scale:
        for config in postop_configs:
            if config.op_type == "SCALE":
                scale_value = config.scale_factor
                # Warn if using default scale factor (might be unintended)
                if config.scale_factor == 1.0:
                    warnings.append(
                        f"SCALE operation uses default scale_factor=1.0. "
                        "Verify this is intentional."
                    )
                break

    # Get appropriate epsilon
    epsilon = get_appropriate_epsilon(a_type, b_type, c_type)

    # Use max k for tolerance calculation (worst case)
    k = max(k_values) if k_values else 32

    # Calculate tolerances using error propagation with full configs
    try:
        tight_mult, loose_mult = get_tolerance_multipliers(
            postop_configs,
            is_cartesian=cartesian
        )
    except UnsupportedPostOpError as e:
        warnings.append(str(e))
        # Use conservative defaults for unsupported post-ops
        tight_mult = 100.0
        loose_mult = 250.0

    # For YAML, we store multipliers, not actual tolerance values
    # The framework computes: tolerance = epsilon * k * multiplier
    #
    # Tolerance assignment:
    # - relative: Use LOOSE tolerance (allows proportional error for larger values)
    # - absolute: Use TIGHT tolerance (strict bound for near-zero values)

    # Suggested values (rounded for cleaner YAML)
    suggested_rel = round(loose_mult, 1)   # loose → relative (more forgiving)
    suggested_abs = round(tight_mult, 1)   # tight → absolute (stricter)

    # Check if existing values are reasonable
    # Allow 50% tolerance in the comparison
    rel_correct = True
    abs_correct = True

    if existing_rel is not None:
        rel_correct = (0.5 * suggested_rel <= existing_rel <= 2.0 * suggested_rel)
    else:
        # No tolerance defined - this should be flagged
        if postops:
            warnings.append(
                "No relative tolerance defined. This may cause test failures."
            )

    if existing_abs is not None:
        abs_correct = (0.5 * suggested_abs <= existing_abs <= 2.0 * suggested_abs)
    else:
        # No tolerance defined - this should be flagged
        if postops:
            warnings.append(
                "No absolute tolerance defined. This may cause test failures."
            )

    # Additional validation for cartesian mode
    if cartesian and (existing_rel is None or existing_abs is None):
        warnings.append(
            "Cartesian mode is enabled but tolerances are not defined. "
            "Cartesian mode typically requires adjusted tolerances."
        )

    return ValidationResult(
        test_name=name,
        postops=postops,
        a_type=a_type,
        b_type=b_type,
        c_type=c_type,
        k_values=k_values,
        existing_relative=existing_rel,
        existing_absolute=existing_abs,
        calculated_tight=tight_mult,
        calculated_loose=loose_mult,
        relative_correct=rel_correct,
        absolute_correct=abs_correct,
        suggested_relative=suggested_rel,
        suggested_absolute=suggested_abs,
        is_cartesian=cartesian,
        has_scale_factor=has_scale,
        scale_factor_value=scale_value,
        warnings=warnings
    )


def validate_yaml_file(yaml_path: Path, verbose: bool = False) -> List[ValidationResult]:
    """Validate all test cases in a YAML file"""
    results = []

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Support multiple YAML config keys: gemm_tests, batch_gemm_tests, yaml_test
    test_keys = ['gemm_tests', 'batch_gemm_tests', 'yaml_test']
    tests = []

    for key in test_keys:
        if key in config:
            key_tests = config.get(key, [])
            if key_tests:
                tests.extend(key_tests)

    for test in tests:
        result = validate_test_case(test, verbose)
        results.append(result)

    return results


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(results: List[ValidationResult], file_path: str, verbose: bool = False) -> str:
    """Generate validation report"""
    lines = []

    lines.append("=" * 70)
    lines.append(f"TOLERANCE VALIDATION REPORT")
    lines.append(f"File: {file_path}")
    lines.append("=" * 70)
    lines.append("")

    incorrect_count = 0
    correct_count = 0
    skipped_count = 0
    warning_count = 0

    for result in results:
        # Skip integer-only tests (no tolerance needed)
        if result.a_type in ('s8', 'u8', 's16', 'u16', 's32', 'u32'):
            if result.c_type in ('s8', 's16', 's32'):
                skipped_count += 1
                continue

        is_correct = result.relative_correct and result.absolute_correct
        has_warnings = len(result.warnings) > 0

        if has_warnings:
            warning_count += 1

        if is_correct and not has_warnings and not verbose:
            correct_count += 1
            continue

        if is_correct:
            correct_count += 1
            if has_warnings:
                status = "⚠ CORRECT (with warnings)"
            else:
                status = "✓ CORRECT"
        else:
            incorrect_count += 1
            status = "✗ NEEDS UPDATE"

        lines.append(f"Test: {result.test_name}")
        lines.append(f"  Post-ops: {' → '.join(result.postops) if result.postops else 'None'}")
        lines.append(f"  Data types: {result.a_type}×{result.b_type} → {result.c_type}, k={result.k_values}")

        # Show cartesian and scale factor info if present
        if result.is_cartesian:
            lines.append(f"  Cartesian mode: enabled (tolerance adjusted by 1.5x)")
        if result.has_scale_factor:
            lines.append(f"  Scale factor: {result.scale_factor_value}")

        lines.append(f"  Status: {status}")
        lines.append("")

        lines.append(f"  Current tolerances:")
        lines.append(f"    relative: {result.existing_relative}")
        lines.append(f"    absolute: {result.existing_absolute}")
        lines.append("")

        lines.append(f"  Calculated tolerances (Error Propagation Model):")
        lines.append(f"    tight (absolute): {result.calculated_tight:.1f} ULPs")
        lines.append(f"    loose (relative): {result.calculated_loose:.1f} ULPs")
        lines.append("")

        if not is_correct:
            lines.append(f"  Suggested values:")
            if not result.relative_correct:
                lines.append(f"    relative: {result.suggested_relative} (currently {result.existing_relative})")
            if not result.absolute_correct:
                lines.append(f"    absolute: {result.suggested_absolute} (currently {result.existing_absolute})")
            lines.append("")

        # Show warnings
        if has_warnings:
            lines.append(f"  ⚠ Warnings:")
            for warning in result.warnings:
                lines.append(f"    - {warning}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("")

    # Summary
    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total tests:    {len(results)}")
    lines.append(f"Correct:        {correct_count}")
    lines.append(f"Need update:    {incorrect_count}")
    lines.append(f"With warnings:  {warning_count}")
    lines.append(f"Skipped (int):  {skipped_count}")
    lines.append("")

    if incorrect_count > 0:
        lines.append("Run with --update to automatically fix tolerance values.")
    elif warning_count > 0:
        lines.append("Tolerance values are within range, but please review warnings.")
    else:
        lines.append("All tolerance values are within acceptable range.")

    return "\n".join(lines)


# =============================================================================
# CALCULATE MODE - Direct tolerance calculation
# =============================================================================

def calculate_tolerance_for_chain(
    postops: List[str],
    dtype: str,
    k: int,
    base_multiplier: float = 50.0,
    verbose: bool = False
) -> None:
    """Calculate and display tolerance for a post-op chain"""
    epsilon = DataTypeEpsilon.get(dtype)

    print("=" * 60)
    print("TOLERANCE CALCULATION")
    print("=" * 60)
    print(f"Post-op chain: {' → '.join(postops) if postops else 'None (base GEMM)'}")
    print(f"Data type: {dtype}")
    print(f"k dimension: {k}")
    print(f"Epsilon: {epsilon}")
    print(f"Base multiplier: {base_multiplier} ULPs")
    print("")

    # Get error models
    try:
        error_model_list = [get_error_model(postop) for postop in postops]
    except UnsupportedPostOpError as e:
        print(f"Error: {e}")
        return

    if verbose:
        print("Error Models:")
        print("-" * 60)
        for model in error_model_list:
            print(f"  {model.name}:")
            print(f"    Amplification factor: {model.amplification_factor}")
            print(f"    Max amplification: {model.max_amplification}")
            print(f"    Additive error: {model.additive_error_ulps} ULPs")
            print(f"    Rationale: {model.rationale}")
            print("")

    # Calculate step-by-step error propagation
    print("Error Propagation (Step by Step):")
    print("-" * 60)

    cumulative_ulps = base_multiplier
    value = 1.0

    print(f"  Step 0 (GEMM): error = {cumulative_ulps:.1f} ULPs")

    for i, model in enumerate(error_model_list):
        amp = model.amplification_factor
        if model.derivative_fn:
            try:
                amp = model.derivative_fn(value)
            except Exception:
                pass

        new_error = cumulative_ulps * amp + model.additive_error_ulps

        print(f"  Step {i+1} ({model.name}):")
        print(f"    input_value = {value:.4f}")
        print(f"    |f'(x)| = {amp:.4f}")
        print(f"    additive = {model.additive_error_ulps} ULPs")
        print(f"    error = {cumulative_ulps:.1f} × {amp:.4f} + {model.additive_error_ulps} = {new_error:.1f} ULPs")

        cumulative_ulps = new_error
        if model.transform_fn:
            try:
                value = model.transform_fn(value)
            except Exception:
                pass

    print("")
    print("Final Results:")
    print("-" * 60)

    tight_mult, loose_mult = get_tolerance_multipliers(postops, base_multiplier)

    print(f"  Tight tolerance (ULPs): {tight_mult:.1f}")
    print(f"  Loose tolerance (ULPs): {loose_mult:.1f}")
    print("")

    tight_actual = tight_mult * epsilon * k
    loose_actual = loose_mult * epsilon * k

    print(f"  For k={k}:")
    print(f"    Tight tolerance value: {tight_actual:.6e}")
    print(f"    Loose tolerance value: {loose_actual:.6e}")
    print("")

    print("YAML Configuration:")
    print("-" * 60)
    print("  tolerances:")
    print(f"    relative: {loose_mult:.1f}")
    print(f"    absolute: {tight_mult:.1f}")


# =============================================================================
# UPDATE YAML
# =============================================================================

def update_yaml_tolerances(yaml_path: Path, results: List[ValidationResult]) -> bool:
    """Update YAML file with calculated tolerances using YAML parsing"""
    import re

    # Read the file content for parsing
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Also read raw content for line-based replacement
    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    modified = False

    # Build a mapping of test name to suggested tolerances
    updates_needed = {}
    for result in results:
        # Skip integer-only tests
        if result.a_type in ('s8', 'u8', 's16', 'u16', 's32', 'u32'):
            if result.c_type in ('s8', 's16', 's32'):
                continue

        # Skip if already correct
        if result.relative_correct and result.absolute_correct:
            continue

        updates_needed[result.test_name] = {
            'suggested_relative': result.suggested_relative,
            'suggested_absolute': result.suggested_absolute,
            'existing_relative': result.existing_relative,
            'existing_absolute': result.existing_absolute,
            'relative_updated': False,
            'absolute_updated': False
        }

    if not updates_needed:
        return False

    # Process line by line to update tolerances
    current_test_name = None
    in_tolerances = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for test name
        if stripped.startswith('- name:'):
            # Extract test name
            match = re.search(r'- name:\s*"([^"]+)"', line)
            if match:
                current_test_name = match.group(1)

        # Check for tolerances section
        if stripped.startswith('tolerances:'):
            in_tolerances = True
        elif in_tolerances and current_test_name and current_test_name in updates_needed:
            updates = updates_needed[current_test_name]

            # Look for relative or absolute lines within tolerances block
            if stripped.startswith('relative:'):
                # Update relative value
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}relative: {updates['suggested_relative']}\n"
                updates['relative_updated'] = True
                modified = True
            elif stripped.startswith('absolute:'):
                # Update absolute value
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}absolute: {updates['suggested_absolute']}\n"
                updates['absolute_updated'] = True
                modified = True
            elif stripped.startswith('#'):
                # Skip comment lines
                pass
            elif stripped and not stripped.startswith('#'):
                # End of tolerances section (hit another key)
                # Check if we finished updating this test
                if updates['relative_updated'] or updates['absolute_updated']:
                    print(f"  Updated: {current_test_name}")
                    print(f"    relative: {updates['existing_relative']} → {updates['suggested_relative']}")
                    print(f"    absolute: {updates['existing_absolute']} → {updates['suggested_absolute']}")
                    del updates_needed[current_test_name]
                in_tolerances = False

        # Check for end of test block (next test starts)
        if stripped.startswith('- name:') and current_test_name and in_tolerances:
            # Check if we finished updating this test before moving to next
            if current_test_name in updates_needed:
                updates = updates_needed[current_test_name]
                if updates['relative_updated'] or updates['absolute_updated']:
                    print(f"  Updated: {current_test_name}")
                    print(f"    relative: {updates['existing_relative']} → {updates['suggested_relative']}")
                    print(f"    absolute: {updates['existing_absolute']} → {updates['suggested_absolute']}")
                    del updates_needed[current_test_name]
            in_tolerances = False

        i += 1

    if modified:
        with open(yaml_path, 'w') as f:
            f.writelines(lines)

    return modified


# =============================================================================
# PRINT ALL POST-OPS
# =============================================================================

def print_all_postop_tolerances():
    """Print tolerance values for all supported post-ops"""
    print("=" * 70)
    print("TOLERANCE VALUES FOR ALL POST-OPERATIONS")
    print("=" * 70)
    print("")
    print("These are the ULP multipliers for each post-op when used alone.")
    print("For chained post-ops, use the error propagation model.")
    print("")
    print(f"{'Post-Op':<15} {'Tight (ULPs)':<15} {'Loose (ULPs)':<15} {'Rationale'}")
    print("-" * 70)

    postops = [
        "RELU", "PRELU", "SIGMOID", "TANH", "SWISH",
        "GELU_TANH", "GELU_ERF", "CLIP",
        "BIAS", "SCALE", "MATRIX_ADD", "MATRIX_MUL"
    ]

    for postop in postops:
        tight, loose = get_tolerance_multipliers([postop])
        model = get_error_model(postop)
        rationale = model.rationale[:40] + "..." if len(model.rationale) > 40 else model.rationale
        print(f"{postop:<15} {tight:<15.1f} {loose:<15.1f} {rationale}")

    print("")
    print("Note: Base GEMM error is assumed to be 50 ULPs.")
    print("      Actual tolerance = epsilon × k × multiplier")


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AOCL-DLP Post-Op Tolerance Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate tolerances in a YAML file
  python validate_postop_tolerances.py gemm_test_config_postops_basic.yaml

  # Update YAML with correct tolerances
  python validate_postop_tolerances.py --update gemm_test_config_postops_basic.yaml

  # Calculate tolerances for specific post-ops
  python validate_postop_tolerances.py --calculate GELU_ERF,SCALE --dtype bf16 --k 32

  # Show all post-op tolerance values
  python validate_postop_tolerances.py --list-all

  # Generate verbose report
  python validate_postop_tolerances.py --verbose gemm_test_config*.yaml
"""
    )

    parser.add_argument(
        'files',
        nargs='*',
        help='YAML configuration files to validate'
    )

    parser.add_argument(
        '--update', '-u',
        action='store_true',
        help='Update YAML files with calculated tolerances'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including correct tolerances'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Write report to file'
    )

    parser.add_argument(
        '--calculate', '-c',
        type=str,
        help='Calculate tolerance for comma-separated post-ops (e.g., GELU_ERF,SCALE)'
    )

    parser.add_argument(
        '--dtype', '-t',
        type=str,
        default='f32',
        help='Data type for calculation (f32, bf16, fp16). Default: f32'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=32,
        help='K dimension for calculation. Default: 32'
    )

    parser.add_argument(
        '--base-multiplier', '-b',
        type=float,
        default=50.0,
        help='Base GEMM error in ULPs. Default: 50.0'
    )

    parser.add_argument(
        '--list-all', '-l',
        action='store_true',
        help='List tolerance values for all supported post-ops'
    )

    args = parser.parse_args()

    # Handle --list-all
    if args.list_all:
        print_all_postop_tolerances()
        return 0

    # Handle --calculate
    if args.calculate:
        postops = [p.strip() for p in args.calculate.split(',')]
        calculate_tolerance_for_chain(
            postops,
            args.dtype,
            args.k,
            args.base_multiplier,
            args.verbose
        )
        return 0

    # Validate YAML files
    if not args.files:
        parser.print_help()
        return 1

    all_results = []

    for file_pattern in args.files:
        path = Path(file_pattern)

        if not path.exists():
            print(f"Error: File not found: {path}")
            continue

        print(f"\nProcessing: {path}")

        try:
            results = validate_yaml_file(path, args.verbose)
            all_results.extend(results)

            # Generate report
            report = generate_report(results, str(path), args.verbose)

            if args.output:
                with open(args.output, 'a') as f:
                    f.write(report)
                    f.write("\n\n")
            else:
                print(report)

            # Update if requested
            if args.update:
                print(f"\nUpdating tolerances in {path}...")
                if update_yaml_tolerances(path, results):
                    print("  Done!")
                else:
                    print("  No changes needed.")

        except Exception as e:
            print(f"Error processing {path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
