#!/bin/bash
# Generate baseline of functions that should remain inlined in dlp_init_and_get_kernel_hndl
# This script analyzes the current build to identify which functions are successfully inlined
# and creates a baseline file for future verification

set -e

LIBRARY="${1:-build/libaocl-dlp.a}"
OUTPUT_FILE="${2:-build/constraint_verification.log}"
INPUT_BASELINE="${3:-}"  # Optional: baseline file for comparison
FAILED=0

if [ ! -f "$LIBRARY" ]; then
    echo "Error: $LIBRARY not found. Build the library first."
    echo "Usage: $0 [library_path] [output_baseline] [input_baseline]"
    echo "  library_path: Path to library (default: build/libaocl-dlp.a)"
    echo "  output_baseline: Path to output baseline file (default: scripts/inlining_baseline.txt)"
    echo "  input_baseline: Optional baseline for comparison (enables constraint checks)"
    exit 1
fi

echo "=== Verifying Inlining Constraints for dlp_init_and_get_kernel_hndl ==="
echo "Library: $LIBRARY"
if [ -n "$INPUT_BASELINE" ]; then
    echo "Baseline: $INPUT_BASELINE (constraints enabled)"
else
    echo "Baseline: None (constraints disabled, will generate new baseline)"
fi
echo ""

# Read existing baseline for comparison FIRST (before truncating anything)
PREV_VTABLE_REFS=""
PREV_INDIRECT_CALLS=""
PREV_STACK_SIZE=""
PREV_CALL_COUNT=""

if [ -n "$INPUT_BASELINE" ] && [ -f "$INPUT_BASELINE" ]; then
    echo "[INFO] Reading baseline for comparison..."
    while IFS= read -r line; do
        if [[ "$line" =~ "vtable refs: "([0-9]+)", indirect calls: "([0-9]+) ]]; then
            PREV_VTABLE_REFS="${BASH_REMATCH[1]}"
            PREV_INDIRECT_CALLS="${BASH_REMATCH[2]}"
        elif [[ "$line" =~ "Stack Frame Size: "([0-9]+) ]]; then
            PREV_STACK_SIZE="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ "Call Instructions: "([0-9]+) ]]; then
            PREV_CALL_COUNT="${BASH_REMATCH[1]}"
        fi
    done < "$INPUT_BASELINE"
    echo "    Previous baseline loaded:"
    [ -n "$PREV_STACK_SIZE" ] && echo "      Stack frame: $PREV_STACK_SIZE bytes"
    [ -n "$PREV_CALL_COUNT" ] && echo "      Call count: $PREV_CALL_COUNT"
    [ -n "$PREV_INDIRECT_CALLS" ] && echo "      Indirect calls: $PREV_INDIRECT_CALLS"
    echo ""
fi

# Extract all symbols from the library first (needed for auto-discovery)
echo "[1/4] Extracting symbols from library..."
nm -C "$LIBRARY" > /tmp/library_symbols.txt
TOTAL_SYMBOLS=$(wc -l < /tmp/library_symbols.txt)
echo "    Found $TOTAL_SYMBOLS total symbols"

# List of critical functions that should be inlined in dlp_init_and_get_kernel_hndl call tree
CRITICAL_FUNCTIONS=(
    # Direct calls from dlp_init_and_get_kernel_hndl
    "dlp::utils::getKernelDatatype"
    "dlp_get_gemm_kernelInfo_by_dtype"
    "get_kernel_family_name"

    # Decision engine singleton and methods
    "dlp::de::decisionEngineInstance"
    "dlp::de::decisionEngine::getGemmKernelInfoForInputFastPath"

    # Kernel register singleton and methods
    "dlp::kernel_frame::dlpKernelRegisterInstance"
    "dlp::kernel_frame::dlpKernelRegister::getGemmKernel"
    "dlp::kernel_frame::dlpKernelRegister::registerGemmKernel"
    "dlp::kernel_frame::dlpKernelRegister::registerEmptyGemmKernel"

    # JIT generator register singleton and methods
    "dlp::jit::dlpJitGeneratorRegisterInstance"
    "dlp::jit::jitGeneratorRegister::getGemmJitGenerator"
)

# Auto-discover all iDEBackend-derived classes and their fast-path methods
echo ""
echo "[2/4] Auto-discovering iDEBackend-derived classes..."

# Find ALL implementations of getGemmKernelInfoForInputFastPath in ANY namespace
# Pattern: "::getGemmKernelInfoForInputFastPath" matches ANY class implementing this method
# Examples matched:
#   - dlp::de::gemmF32DEBackend::getGemmKernelInfoForInputFastPath(...)
#   - custom::int8BackDe::getGemmKernelInfoForInputFastPath(...)
#   - any::namespace::anyClass::getGemmKernelInfoForInputFastPath(...)
#
# Then filter OUT base classes and wrappers:
#   - grep -v "iDEBackend::..." = Exclude abstract base class
#   - grep -v "decisionEngine::..." = Exclude wrapper class
#
# Then clean up symbol table format:
#   - sed 's/^[0-9a-f]* [A-Z] //' = Remove "address type " prefix
#     Example: "0000000000000000 W func(...)" → "func(...)"
BACKEND_GEMM_FUNCS=$(grep -E "::getGemmKernelInfoForInputFastPath" /tmp/library_symbols.txt | \
    grep -v "iDEBackend::getGemmKernelInfoForInputFastPath" | \
    grep -v "decisionEngine::getGemmKernelInfoForInputFastPath" | \
    sed 's/^[0-9a-f]* [A-Z] //' | sort -u || true)

# Same pattern for GEMV functions
BACKEND_GEMV_FUNCS=$(grep -E "::getGemvKernelInfoForInputFastPath" /tmp/library_symbols.txt | \
    grep -v "iDEBackend::getGemvKernelInfoForInputFastPath" | \
    sed 's/^[0-9a-f]* [A-Z] //' | sort -u || true)

BACKEND_COUNT=0

# Add discovered backend GEMM functions
# Note: Here-string (<<<) feeds each line of BACKEND_GEMM_FUNCS to the loop
while IFS= read -r func; do
    if [ -n "$func" ]; then
        # Extract just the class name for reporting (namespace-agnostic)
        # Pattern: '(?:^|::)\K[^:]+(?=::getGemmKernelInfoForInputFastPath)'
        #   (?:^|::) = Non-capturing group: start of string OR "::"
        #   \K = Lookbehind reset: exclude left side from match
        #   [^:]+ = THE CAPTURED PART: one or more non-colon chars (the class name)
        #   (?=::getGemm...) = Lookahead: must be followed by this, but exclude from match
        #
        # Example: "dlp::de::gemmF32DEBackend::getGemmKernelInfoForInputFastPath(...)"
        #   Result: "gemmF32DEBackend" (just the class name)
        CLASS_NAME=$(echo "$func" | grep -oP '(?:^|::)\K[^:]+(?=::getGemmKernelInfoForInputFastPath)' || echo "Unknown")
        echo "    Found backend: $CLASS_NAME (GEMM fast path)"
        CRITICAL_FUNCTIONS+=("$func")  # Add FULL signature (with namespace and params)
        BACKEND_COUNT=$((BACKEND_COUNT + 1))
    fi
done <<< "$BACKEND_GEMM_FUNCS"  # Here-string: feed each line to 'func' variable

# Add discovered backend GEMV functions
while IFS= read -r func; do
    if [ -n "$func" ]; then
        # Extract class name (same pattern as above, but for getGemvKernelInfoForInputFastPath)
        CLASS_NAME=$(echo "$func" | grep -oP '(?:^|::)\K[^:]+(?=::getGemvKernelInfoForInputFastPath)' || echo "Unknown")
        echo "    Found backend: $CLASS_NAME (GEMV fast path)"
        CRITICAL_FUNCTIONS+=("$func")
        BACKEND_COUNT=$((BACKEND_COUNT + 1))
    fi
done <<< "$BACKEND_GEMV_FUNCS"

if [ $BACKEND_COUNT -eq 0 ]; then
    echo "    ⚠️  WARNING: No iDEBackend-derived classes found!"
    echo "       This may indicate a build issue or the pattern has changed"
else
    echo "    ✓ Discovered $BACKEND_COUNT backend fast-path methods"
fi

# Check which critical functions are actually CALLED (not inlined) vs just having symbols
echo ""
echo "[3/4] Analyzing call sites in dlp_init_and_get_kernel_hndl..."
echo "    Checking $(( ${#CRITICAL_FUNCTIONS[@]} )) functions ($(( ${#CRITICAL_FUNCTIONS[@]} - BACKEND_COUNT )) static + $BACKEND_COUNT auto-discovered backends)..."
INLINED_COUNT=0
NOT_INLINED_COUNT=0

# Initialize OUTPUT file (safe to truncate now since we've already read INPUT)
> "$OUTPUT_FILE"
echo "# Inlining Baseline for dlp_init_and_get_kernel_hndl Call Tree" >> "$OUTPUT_FILE"
echo "# Generated: $(date)" >> "$OUTPUT_FILE"
echo "# Library: $LIBRARY" >> "$OUTPUT_FILE"
echo "#" >> "$OUTPUT_FILE"
echo "# Functions below are NOT called in dlp_init_and_get_kernel_hndl (successfully inlined)" >> "$OUTPUT_FILE"
echo "# Note: Symbols may exist for vtable entries, but what matters is they're not CALLED" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Extract ONLY the disassembly of dlp_init_and_get_kernel_hndl function
# Command breakdown:
#   objdump -dC = Disassemble (-d) with C++ name demangling (-C)
#   sed -n = Quiet mode (don't auto-print)
#   '/<dlp_init_and_get_kernel_hndl>/,/^[0-9a-f]\+ <.*>:/p' = Range extraction:
#     - Start: Line containing "<dlp_init_and_get_kernel_hndl>"
#     - End: Next line matching "^[0-9a-f]\+ <.*>:" (next function header)
#     - p = Print the range
#
# Example output structure:
#   0000000000000000 <dlp_init_and_get_kernel_hndl>:  ← START
#      0:	push   %rbp
#      4:	call   9 <some_function>
#    ...
#   de88:	ret
#                                                     ← END (before next function)
#   0000000000000000 <next_function>:                ← This line triggers end pattern
#
# Result: Only dlp_init_and_get_kernel_hndl assembly in /tmp/dlp_init_disasm.txt
objdump -dC "$LIBRARY" | sed -n '/<dlp_init_and_get_kernel_hndl>/,/^[0-9a-f]\+ <.*>:/p' > /tmp/dlp_init_disasm.txt

for func in "${CRITICAL_FUNCTIONS[@]}"; do
    # Check if function is actually CALLED in the hot path
    if grep -q "$func" /tmp/dlp_init_disasm.txt; then
        echo "    ❌ CALLED (NOT INLINED): $func"
        NOT_INLINED_COUNT=$((NOT_INLINED_COUNT + 1))
        FAILED=1  # Mark as failed - critical function not inlined
    else
        echo "    ✓ NOT CALLED (INLINED): $func"
        echo "$func" >> "$OUTPUT_FILE"
        INLINED_COUNT=$((INLINED_COUNT + 1))
    fi
done

# Check for virtual dispatch patterns
echo ""
echo "[4/4] Checking for virtual dispatch patterns (indirect calls)..."
VTABLE_REFS=0
INDIRECT_CALLS=0

if objdump -d "$LIBRARY" | grep -q "<dlp_init_and_get_kernel_hndl>"; then
    # Extract the function disassembly
    DISASM=$(objdump -d "$LIBRARY" | sed -n '/<dlp_init_and_get_kernel_hndl>/,/<.*>:/p')

    # Check for vtable references (typically seen as _ZTV patterns or offset loads before calls)
    VTABLE_COUNT=$(echo "$DISASM" | grep "_ZTV" 2>/dev/null | wc -l)
    VTABLE_REFS=${VTABLE_COUNT:-0}

    # Check for indirect calls (call *%rax, call *%rcx, etc.) which indicate virtual dispatch
    INDIRECT_COUNT=$(echo "$DISASM" | grep -E "call[q]?\s+\*%" 2>/dev/null | wc -l)
    INDIRECT_CALLS=${INDIRECT_COUNT:-0}

    echo "    Vtable references: $VTABLE_REFS"
    echo "    Indirect calls: $INDIRECT_CALLS"

    # Compare with previous baseline if available
    if [ -n "$PREV_VTABLE_REFS" ] && [ -n "$PREV_INDIRECT_CALLS" ]; then
        VTABLE_DIFF=$((VTABLE_REFS - PREV_VTABLE_REFS))
        INDIRECT_DIFF=$((INDIRECT_CALLS - PREV_INDIRECT_CALLS))

        if [ $VTABLE_DIFF -gt 0 ] || [ $INDIRECT_DIFF -gt 0 ]; then
            echo "    ❌ FAIL: Virtual dispatch INCREASED!"
            echo "           Vtable refs increased by: $VTABLE_DIFF (was: $PREV_VTABLE_REFS)"
            echo "           Indirect calls increased by: $INDIRECT_DIFF (was: $PREV_INDIRECT_CALLS)"
            FAILED=1
        elif [ "$VTABLE_REFS" -gt 0 ] || [ "$INDIRECT_CALLS" -gt 0 ]; then
            echo "    ✓ Virtual dispatch stable (baseline: vtable=$PREV_VTABLE_REFS, indirect=$PREV_INDIRECT_CALLS)"
        else
            echo "    ✓ No virtual dispatch"
        fi
    else
        # No baseline to compare - just report (don't fail on first run)
        if [ "$VTABLE_REFS" -gt 0 ] || [ "$INDIRECT_CALLS" -gt 0 ]; then
            echo "    ⚠️  Virtual dispatch present (will be tracked in baseline)"
        else
            echo "    ✓ No virtual dispatch detected"
        fi
    fi

    echo "" >> "$OUTPUT_FILE"
    echo "# Virtual Dispatch: vtable refs: $VTABLE_REFS, indirect calls: $INDIRECT_CALLS" >> "$OUTPUT_FILE"
fi

# Analyze dlp_init_and_get_kernel_hndl disassembly for additional insights
echo ""
echo "[5/5] Analyzing dlp_init_and_get_kernel_hndl assembly..."
if objdump -d "$LIBRARY" | grep -q "<dlp_init_and_get_kernel_hndl>"; then
    # Extract stack frame size
    FRAME_SIZE=$(objdump -d "$LIBRARY" | \
        grep -A5 "<dlp_init_and_get_kernel_hndl>" | \
        grep "sub.*%rsp" | head -1 | \
        grep -oP '0x[0-9a-f]+' || echo "0x0")

    if [ "$FRAME_SIZE" != "0x0" ]; then
        SIZE_DEC=$((FRAME_SIZE))
        echo "    Stack frame size: $SIZE_DEC bytes (0x$(printf '%x' $SIZE_DEC))"

        # Compare with previous baseline if available
        if [ -n "$PREV_STACK_SIZE" ]; then
            SIZE_DIFF=$((SIZE_DEC - PREV_STACK_SIZE))
            if [ $SIZE_DIFF -gt 50 ]; then
                echo "    ❌ FAIL: Stack frame increased by $SIZE_DIFF bytes (was: $PREV_STACK_SIZE bytes)"
                echo "           Tolerance exceeded (±50 bytes)"
                FAILED=1
            elif [ $SIZE_DIFF -gt 0 ]; then
                echo "    ⚠️  Stack frame increased by $SIZE_DIFF bytes (was: $PREV_STACK_SIZE bytes, within tolerance)"
            elif [ $SIZE_DIFF -lt 0 ]; then
                SIZE_DIFF=$((SIZE_DIFF * -1))
                echo "    ✓ Stack frame decreased by $SIZE_DIFF bytes (was: $PREV_STACK_SIZE bytes)"
            else
                echo "    ✓ Stack frame unchanged ($PREV_STACK_SIZE bytes)"
            fi
        fi

        echo "" >> "$OUTPUT_FILE"
        echo "# Stack Frame Size: $SIZE_DEC bytes" >> "$OUTPUT_FILE"
    fi

    # Check for forced stack alignment
    if objdump -d "$LIBRARY" | grep -A20 "<dlp_init_and_get_kernel_hndl>" | \
       grep -q "and.*0xffffffffffffffc0,%rsp"; then
        echo "    ⚠️  64-byte forced stack alignment present"
        echo "# WARNING: Forced 64-byte stack alignment present" >> "$OUTPUT_FILE"
    else
        echo "    ✓ No forced stack alignment"
    fi

    # Count call instructions in the function
    CALL_COUNT=$(objdump -d "$LIBRARY" | \
        sed -n '/<dlp_init_and_get_kernel_hndl>/,/<.*>:/p' | \
        grep -c "call" || echo "0")
    echo "    Call instructions: $CALL_COUNT"

    # Compare with previous baseline if available
    if [ -n "$PREV_CALL_COUNT" ]; then
        CALL_DIFF=$((CALL_COUNT - PREV_CALL_COUNT))
        if [ $CALL_DIFF -gt 2 ]; then
            echo "    ❌ FAIL: Call count increased by $CALL_DIFF (was: $PREV_CALL_COUNT)"
            echo "           Tolerance exceeded (±2 calls)"
            FAILED=1
        elif [ $CALL_DIFF -gt 0 ]; then
            echo "    ⚠️  Call count increased by $CALL_DIFF (was: $PREV_CALL_COUNT, within tolerance)"
        elif [ $CALL_DIFF -lt 0 ]; then
            CALL_DIFF=$((CALL_DIFF * -1))
            echo "    ✓ Call count decreased by $CALL_DIFF (was: $PREV_CALL_COUNT)"
        else
            echo "    ✓ Call count unchanged ($PREV_CALL_COUNT)"
        fi
    fi

    echo "# Call Instructions: $CALL_COUNT" >> "$OUTPUT_FILE"
else
    echo "    ⚠️  WARNING: Could not find dlp_init_and_get_kernel_hndl in disassembly"
fi

# Cleanup
rm -f /tmp/library_symbols.txt /tmp/dlp_init_disasm.txt

# Summary and exit code handling
echo ""
echo "=========================================="
echo "Baseline Generation Summary"
echo "=========================================="
echo "Inlined functions:     $INLINED_COUNT"
echo "Not inlined functions: $NOT_INLINED_COUNT"
echo ""
echo "Baseline saved to: $OUTPUT_FILE"
echo ""

if [ $FAILED -eq 1 ]; then
    echo "❌ PERFORMANCE REGRESSION DETECTED"
    echo "=========================================="
    echo ""

    if [ $NOT_INLINED_COUNT -gt 0 ]; then
        echo "Issue 1: Critical functions are being CALLED (NOT inlined)"
        echo "  - $NOT_INLINED_COUNT functions marked '❌ CALLED' above"
        echo "  - Violates [[gnu::always_inline]] expectations"
        echo "  - Could cause 4-8 GFLOPS regression"
        echo ""
    fi

    echo "Action required:"
    echo "1. Review changes to capi_kernel_frame_wrappers.cc and related files"
    echo "2. Ensure [[gnu::always_inline]] is present on all critical functions"
    echo "3. Verify template parameter enables static dispatch (not virtual)"
    echo "4. Check that stack frame and call count increases are justified"
    echo "5. See CONFIRMED_FIX.md and PERFORMANCE_SAFEGUARDS.md for details"
    echo ""
    echo "All issues marked with '❌ FAIL' above must be addressed."
    echo ""
    exit 1
fi

echo "✓ All inlining checks passed"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - All $INLINED_COUNT critical functions successfully inlined"
echo "  - Auto-discovered $BACKEND_COUNT backend methods"
echo "  - Template dispatch working correctly"
echo "  - Baseline updated successfully"
echo ""
exit 0
