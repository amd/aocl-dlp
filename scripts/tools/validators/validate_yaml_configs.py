#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
"""Validate AOCL-DLP YAML test configs against a whitelist.

The whitelist below IS the contract. When tests/framework/utils/yaml_parser.cc
or tests/framework/utils/parser.cc adds/renames a field, value, or operation
type, update the corresponding set in this file.

Usage:
    validate_yaml_configs.py PATH [PATH ...]
    validate_yaml_configs.py --strict PATH

Exit code: 0 on success, 1 if any failure (or warning under --strict).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "error: PyYAML is required (pip install pyyaml or system package)\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# This tool validates YAML configs against the AOCL-DLP YAML parser at:
#   tests/framework/utils/yaml_parser.cc
#   tests/framework/utils/parser.cc  (operation type handling)
# When updating the whitelists below, ensure they match what the parser
# accepts. If you add a value here that the parser doesn't yet handle, the
# tool will silently bless YAMLs that the parser would silently mishandle.
# ---------------------------------------------------------------------------

VALID_TOP_LEVEL_NODES = {
    "gemm_tests",
    "batch_gemm_tests",
}
# NOTE: "gemv_tests" was listed here but no C++ code ever constructs
# YamlParser(..., "gemv_tests"). Removed to avoid false PASSes.

VALID_FIELDS = {
    "name",
    "product_type",
    "a_type",
    "b_type",
    "c_type",
    "acc_type",
    "storage_format",
    "m", "n", "k",
    "lda", "ldb", "ldc",
    "alpha", "beta",
    "transA", "transB",
    "mtagA", "mtagB",
    "group_size",
    "tolerances",
    "post_operations",
    "pre_operations",
    "fill_value",
    "fill_pattern",
}

VALID_MTAG_VALUES = {"none", "reorder", "pack"}
# NOTE: "strided3d" is on a separate branch (bmm-3d) and not yet in amd-main.
# Re-add here once stringToMatrixTag() in yaml_parser.cc handles it.

VALID_OPERATION_TYPES = {
    "Elementwise-RELU",
    "Elementwise-PRELU",
    "Elementwise-GELU-TANH",
    "Elementwise-GELU-ERF",
    "Elementwise-SWISH",
    "Elementwise-MISH",
    "Elementwise-TANH",
    "Elementwise-SIGMOID",
    "Elementwise-CLIP",
    # NOTE: "Elementwise-LINEAR" has no handler in parser.cc; removed to avoid
    # blessing configs that would fail at runtime.
    "Bias",
    "Scale",
    "Matrix-Add",
    "Matrix-Mul",
    "A_Quant",
    "GroupScale",
    "WOQ",
}

VALID_PRODUCT_TYPES = {"cartesian", "simple"}
# Parser only matches "simple" / "cartesian" exactly (yaml_parser.cc ~L413);
# anything else silently falls through to cartesian.

VALID_MATRIX_TYPES = {
    "f32", "bf16", "fp16",
    "s8", "s16", "s32",
    "u8", "u16", "u32",
    "s4", "u4",
}

VALID_STORAGE_FORMATS = {
    "row_major", "row-major", "ROW_MAJOR",
    "column_major", "column-major", "COLUMN_MAJOR",
}

# Removed/renamed names — used for "did you mean" hints.
RENAMED_FIELDS = {
    "reorderA": "mtagA",
    "reorderB": "mtagB",
    "reorder_a": "mtagA",
    "reorder_b": "mtagB",
}

RENAMED_OPERATIONS = {
    "SymQuant": "GroupScale",
}


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def suggest(bad: str, candidates: Iterable[str], renamed: dict | None = None,
            max_dist: int = 3) -> str | None:
    """Return a "did you mean" string or None."""
    if renamed and bad in renamed:
        return renamed[bad]
    best, best_d = None, max_dist + 1
    bad_lc = bad.lower()
    for c in candidates:
        d = _levenshtein(bad_lc, c.lower())
        if d < best_d:
            best, best_d = c, d
    return best if best_d <= max_dist else None


def _hint(suggestion: str | None) -> str:
    return f" (did you mean '{suggestion}'?)" if suggestion else ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class Issue:
    __slots__ = ("level", "test_set", "msg")

    def __init__(self, level: str, test_set: str, msg: str):
        self.level = level   # "error" or "warning"
        self.test_set = test_set
        self.msg = msg

    def format(self) -> str:
        return f'  - test set "{self.test_set}": {self.msg}'


def _as_str_list(node) -> list[str]:
    """Coerce a YAML scalar / sequence into a list of strings (skipping
    non-strings — those are caught elsewhere as type errors)."""
    if node is None:
        return []
    if isinstance(node, list):
        return [str(x) for x in node if isinstance(x, (str, int, float, bool))]
    if isinstance(node, (str, int, float, bool)):
        return [str(node)]
    return []


def _validate_enum_field(values: list[str], valid: set[str], renamed: dict,
                         field_label: str, test_set: str,
                         issues: list[Issue]) -> None:
    for v in values:
        if v not in valid:
            sug = suggest(v, valid, renamed)
            valid_list = ", ".join(sorted(valid))
            issues.append(Issue(
                "error", test_set,
                f"invalid {field_label} value '{v}'{_hint(sug)} "
                f"(valid: {valid_list})",
            ))


def _validate_postops_block(block, kind: str, test_set: str,
                            issues: list[Issue]) -> None:
    """Validate a post_operations or pre_operations block."""
    if block is None:
        return
    if not isinstance(block, dict):
        issues.append(Issue(
            "error", test_set,
            f"{kind} must be a mapping with 'operations' (got "
            f"{type(block).__name__})",
        ))
        return
    ops = block.get("operations")
    if ops is None:
        # Empty block — allow (parser tolerates absence).
        return
    if not isinstance(ops, list):
        issues.append(Issue(
            "error", test_set,
            f"{kind}.operations must be a list (got {type(ops).__name__})",
        ))
        return
    for idx, op in enumerate(ops):
        if not isinstance(op, dict):
            issues.append(Issue(
                "error", test_set,
                f"{kind}[{idx}] must be a mapping with 'type' field",
            ))
            continue
        op_type = op.get("type")
        if op_type is None:
            issues.append(Issue(
                "error", test_set,
                f"{kind}[{idx}] missing required 'type' field",
            ))
            continue
        if not isinstance(op_type, str):
            issues.append(Issue(
                "error", test_set,
                f"{kind}[{idx}].type must be a string "
                f"(got {type(op_type).__name__})",
            ))
            continue
        if op_type not in VALID_OPERATION_TYPES:
            sug = suggest(op_type, VALID_OPERATION_TYPES, RENAMED_OPERATIONS)
            issues.append(Issue(
                "error", test_set,
                f"unknown operation type '{op_type}'{_hint(sug)}",
            ))


def validate_test_set(node, top_level: str,
                      issues: list[Issue]) -> None:
    """Validate a single test-set mapping. Mutates ``issues``."""
    name = "<unnamed>"
    if isinstance(node, dict) and isinstance(node.get("name"), str):
        name = node["name"]

    if not isinstance(node, dict):
        issues.append(Issue(
            "error", name,
            f"test set must be a mapping (got {type(node).__name__})",
        ))
        return

    # 2. Field whitelist
    for field in node.keys():
        if not isinstance(field, str):
            issues.append(Issue(
                "error", name,
                f"non-string field key '{field!r}'",
            ))
            continue
        if field not in VALID_FIELDS:
            sug = suggest(field, VALID_FIELDS, RENAMED_FIELDS)
            issues.append(Issue(
                "error", name,
                f"unknown field '{field}'{_hint(sug)}",
            ))

    # 3. mtagA / mtagB
    for tag_field in ("mtagA", "mtagB"):
        if tag_field in node:
            _validate_enum_field(
                _as_str_list(node[tag_field]), VALID_MTAG_VALUES, {},
                tag_field, name, issues,
            )

    # 5. product_type
    if "product_type" in node:
        _validate_enum_field(
            _as_str_list(node["product_type"]), VALID_PRODUCT_TYPES, {},
            "product_type", name, issues,
        )

    # 6. matrix types
    for mt_field in ("a_type", "b_type", "c_type", "acc_type"):
        if mt_field in node:
            _validate_enum_field(
                _as_str_list(node[mt_field]), VALID_MATRIX_TYPES, {},
                mt_field, name, issues,
            )

    # 7. storage_format
    if "storage_format" in node:
        _validate_enum_field(
            _as_str_list(node["storage_format"]), VALID_STORAGE_FORMATS, {},
            "storage_format", name, issues,
        )

    # 4. operation types
    _validate_postops_block(node.get("post_operations"), "post_operations",
                            name, issues)
    _validate_postops_block(node.get("pre_operations"), "pre_operations",
                            name, issues)

    # 8. multi-group + cartesian conflict (batch_gemm_tests only)
    # The C++ parser only enforces this for batch_gemm_tests; don't over-reject.
    if top_level == "batch_gemm_tests":
        gs = node.get("group_size")
        if isinstance(gs, list) and len(gs) > 1:
            pt_values = _as_str_list(node.get("product_type"))
            # Default product_type is cartesian; only "simple" silences the
            # parser's runtime check.
            if not pt_values:
                issues.append(Issue(
                    "error", name,
                    "multi-group group_size (list length > 1) requires "
                    "product_type: 'simple' (default 'cartesian' will fail at "
                    "parse time)",
                ))
            else:
                for pt in pt_values:
                    if pt != "simple":
                        issues.append(Issue(
                            "error", name,
                            f"multi-group group_size with product_type '{pt}' "
                            "is invalid; use product_type: 'simple'",
                        ))
                        break


def validate_file(path: Path) -> tuple[bool, int, list[Issue], list[str]]:
    """Returns (ok, num_test_sets_validated, issues, notes)."""
    issues: list[Issue] = []
    notes: list[str] = []

    try:
        with path.open("r") as f:
            doc = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        issues.append(Issue("error", "<file>", f"YAML parse error: {exc}"))
        return False, 0, issues, notes
    except OSError as exc:
        issues.append(Issue("error", "<file>", f"cannot read: {exc}"))
        return False, 0, issues, notes

    if doc is None:
        notes.append("empty document")
        return True, 0, issues, notes

    if not isinstance(doc, dict):
        issues.append(Issue(
            "error", "<file>",
            f"top-level must be a mapping (got {type(doc).__name__})",
        ))
        return False, 0, issues, notes

    # 1. Top-level node names
    recognized = [k for k in doc.keys() if k in VALID_TOP_LEVEL_NODES]
    unrecognized = [k for k in doc.keys() if k not in VALID_TOP_LEVEL_NODES]

    if not recognized:
        notes.append(
            "no recognized test nodes "
            f"({', '.join(sorted(VALID_TOP_LEVEL_NODES))}); skipped"
        )
        return True, 0, issues, notes

    for k in unrecognized:
        sug = suggest(str(k), VALID_TOP_LEVEL_NODES)
        issues.append(Issue(
            "warning", "<file>",
            f"unknown top-level node '{k}'{_hint(sug)} (valid: "
            f"{', '.join(sorted(VALID_TOP_LEVEL_NODES))})",
        ))

    n_sets = 0
    for top_key in recognized:
        suite = doc[top_key]
        if suite is None:
            continue
        if not isinstance(suite, list):
            issues.append(Issue(
                "error", f"<{top_key}>",
                f"must be a list of test sets (got {type(suite).__name__})",
            ))
            continue
        for ts in suite:
            n_sets += 1
            validate_test_set(ts, top_key, issues)

    has_error = any(i.level == "error" for i in issues)
    return not has_error, n_sets, issues, notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_yaml_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.yaml")))
            out.extend(sorted(path.rglob("*.yml")))
        elif path.is_file():
            out.append(path)
        else:
            sys.stderr.write(f"warning: {p} not found\n")
    # de-dupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate AOCL-DLP YAML test configs.",
    )
    parser.add_argument("paths", nargs="+",
                        help="YAML file(s) or directory(ies) to validate")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as errors")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only report failures")
    args = parser.parse_args(argv)

    files = collect_yaml_paths(args.paths)
    if not files:
        sys.stderr.write("error: no YAML files found\n")
        return 2

    n_pass = n_fail = n_skip = 0
    for path in files:
        ok, n_sets, issues, notes = validate_file(path)

        if args.strict and any(i.level == "warning" for i in issues):
            ok = False

        skipped = (n_sets == 0 and not any(i.level == "error" for i in issues))

        if skipped:
            n_skip += 1
            if not args.quiet:
                note_str = f" ({'; '.join(notes)})" if notes else ""
                print(f"SKIP: {path}{note_str}")
            continue

        if ok:
            n_pass += 1
            if not args.quiet:
                print(f"PASS: {path} ({n_sets} test sets validated)")
                for i in issues:  # warnings only
                    print(f"  ! {i.test_set}: {i.msg}")
        else:
            n_fail += 1
            print(f"FAIL: {path}")
            for i in issues:
                marker = "-" if i.level == "error" else "!"
                print(f"  {marker} test set \"{i.test_set}\": {i.msg}")

    print()
    print(f"Summary: {n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"(of {len(files)} files)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
