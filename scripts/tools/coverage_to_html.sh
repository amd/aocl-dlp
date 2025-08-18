#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (
# INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Code Coverage HTML Report Generator for AOCL-DLP
# This script generates comprehensive HTML coverage reports using gcov/lcov
# while leveraging the existing project infrastructure and exclusion patterns.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
COVERAGE_DIR="${PROJECT_ROOT}/coverage_reports"

# Default values
OPEN_BROWSER=true
CLEAN_PREVIOUS=true
VERBOSE=false
SERVE_HTTP=false
HTTP_PORT=8080

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate HTML coverage reports for AOCL-DLP using gcov/lcov.

OPTIONS:
    -h, --help              Show this help message
    -b, --build-dir DIR     Specify build directory (default: ${BUILD_DIR})
    -o, --output-dir DIR    Specify coverage output directory (default: ${COVERAGE_DIR})
    -n, --no-browser        Don't open browser automatically
    -k, --keep-previous     Keep previous coverage reports
    -v, --verbose           Enable verbose output
    -s, --serve             Start HTTP server to serve reports (great for SSH environments)
    -p, --port PORT         HTTP server port (default: ${HTTP_PORT}, auto-detects if busy)
    --clean-only            Only clean previous reports and exit

EXAMPLES:
    $0                      Generate coverage report with defaults
    $0 -v -n               Generate with verbose output, don't open browser
    $0 -s                  Generate and serve via HTTP server
    $0 -s -p 9000          Generate and serve on port 9000
    $0 -b ./my_build       Use custom build directory
    $0 --clean-only        Clean previous reports only

SSH/REMOTE USAGE:
    $0 -s                  Start HTTP server, then access via: http://your-server:8080
    $0 -s -v               Verbose output with HTTP server information

PREREQUISITES:
    1. Build project with coverage: cmake -DCMAKE_BUILD_TYPE=Coverage
    2. Run tests to generate coverage data
    3. Ensure lcov and genhtml are installed

EOF
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    local missing_deps=()

    if ! command -v lcov >/dev/null 2>&1; then
        missing_deps+=("lcov")
    fi

    if ! command -v genhtml >/dev/null 2>&1; then
        missing_deps+=("genhtml")
    fi

    if ! command -v gcov >/dev/null 2>&1; then
        missing_deps+=("gcov")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_info "To install on Ubuntu/Debian: sudo apt install lcov"
        print_info "To install on RHEL/CentOS: sudo yum install lcov"
        exit 1
    fi

    print_verbose "All dependencies found: lcov, genhtml, gcov"
}

# Function to validate directories
validate_directories() {
    print_info "Validating directories..."

    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project root not found: $PROJECT_ROOT"
        exit 1
    fi

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found: $BUILD_DIR"
        print_info "Please build the project first with: cmake -B build -DCMAKE_BUILD_TYPE=Coverage"
        exit 1
    fi

    # Check for coverage files
    if ! find "$BUILD_DIR" -name "*.gcno" -o -name "*.gcda" | grep -q .; then
        print_error "No coverage data found in build directory"
        print_info "Please ensure you:"
        print_info "  1. Built with coverage: cmake -DCMAKE_BUILD_TYPE=Coverage"
        print_info "  2. Ran tests to generate .gcda files"
        exit 1
    fi

    print_verbose "Found coverage data files in build directory"
}

# Function to clean previous reports
clean_previous_reports() {
    if [ "$CLEAN_PREVIOUS" = true ] && [ -d "$COVERAGE_DIR" ]; then
        print_info "Cleaning previous coverage reports..."
        rm -rf "$COVERAGE_DIR"
        print_verbose "Removed: $COVERAGE_DIR"
    fi
}

# Function to create coverage directory
create_coverage_directory() {
    print_info "Creating coverage output directory..."
    mkdir -p "$COVERAGE_DIR"
    print_verbose "Created: $COVERAGE_DIR"
}

# Function to generate exclusion patterns for lcov
generate_exclusion_patterns() {
    print_info "Generating exclusion patterns..."

    # Base exclusion patterns (extending the project's common/config.py patterns)
    local exclusions=(
        # Build and installation directories
        "*/build/*"
        "*/install/*"
        "*/exp/*"
        "*/.git/*"

        # Test and benchmark code (we want source coverage, not test coverage)
        "*/tests/*"
        "*/test/*"
        "*/bench/*"
        "*/benchmark/*"
        "*/*test*"
        "*/*bench*"

        # Documentation and tooling
        "*/docs/*"
        "*/scripts/*"
        "*/tools/*"

        # Examples (optional - can be included if desired)
        "*/examples/*"
        "*/example/*"

        # External dependencies and generated code
        "*/xbyak/*"
        "*/JIT/xbyak/*"
        "*/JIT/*"
        "*/external/*"
        "*/third_party/*"
        "*/3rdparty/*"

        # System headers and standard library
        "/usr/include/*"
        "/usr/local/include/*"
        "/opt/*/include/*"
        "*/c++/*"
        "*/gcc/*"
        "*/clang/*"

        # CMake generated files
        "*/CMakeFiles/*"
        "*/.cmake/*"

        # Temporary and cache files
        "*/tmp/*"
        "*/temp/*"
        "*/.cache/*"

        # Common test file patterns
        "*_test.c"
        "*_test.cc"
        "*_test.cpp"
        "*_test.h"
        "*_test.hh"
        "*_test.hpp"
        "test_*.c"
        "test_*.cc"
        "test_*.cpp"
        "test_*.h"
        "test_*.hh"
        "test_*.hpp"
    )

    # Convert to lcov remove pattern format
    EXCLUSION_ARGS=""
    for pattern in "${exclusions[@]}"; do
        EXCLUSION_ARGS="$EXCLUSION_ARGS --remove /tmp/coverage_raw.info '$pattern'"
    done

    print_verbose "Generated ${#exclusions[@]} exclusion patterns"
}

# Function to collect coverage data
collect_coverage_data() {
    print_info "Collecting coverage data with lcov..."

    cd "$PROJECT_ROOT"

    # Initialize coverage data collection
    print_verbose "Initializing lcov data collection"
    lcov --capture --directory "$BUILD_DIR" \
         --output-file /tmp/coverage_raw.info \
         --base-directory "$PROJECT_ROOT" \
         --no-external \
         --quiet

    if [ ! -f /tmp/coverage_raw.info ]; then
        print_error "Failed to collect coverage data"
        exit 1
    fi

    print_verbose "Raw coverage data collected: $(wc -l < /tmp/coverage_raw.info) lines"
}

# Function to filter coverage data
filter_coverage_data() {
    print_info "Filtering coverage data..."

    # Apply exclusion patterns
    print_verbose "Applying exclusion patterns"
    eval "lcov $EXCLUSION_ARGS --output-file /tmp/coverage_filtered.info --quiet"

    if [ ! -f /tmp/coverage_filtered.info ]; then
        print_error "Failed to filter coverage data"
        exit 1
    fi

    # Get coverage statistics
    local raw_lines=$(lcov --summary /tmp/coverage_raw.info 2>/dev/null | grep -E "lines\.\.\.*:" | awk '{print $2}' || echo "unknown")
    local filtered_lines=$(lcov --summary /tmp/coverage_filtered.info 2>/dev/null | grep -E "lines\.\.\.*:" | awk '{print $2}' || echo "unknown")

    print_verbose "Filtered coverage data: $filtered_lines lines (was $raw_lines)"

    # Copy filtered data to final location
    cp /tmp/coverage_filtered.info "$COVERAGE_DIR/coverage.info"
    print_verbose "Coverage data saved to: $COVERAGE_DIR/coverage.info"
}

# Function to generate HTML report
generate_html_report() {
    print_info "Generating HTML coverage report..."

    local html_dir="$COVERAGE_DIR/html"

    # Check genhtml version and use compatible options
    if genhtml --help 2>/dev/null | grep -q -- "--demangle-cpp"; then
        # Newer version with more options
        genhtml "$COVERAGE_DIR/coverage.info" \
                --output-directory "$html_dir" \
                --title "AOCL-DLP Code Coverage Report" \
                --show-details \
                --highlight \
                --legend \
                --branch-coverage \
                --function-coverage \
                --demangle-cpp \
                --sort \
                --num-spaces 4 \
                --quiet
    else
        # Older version with basic options
        genhtml "$COVERAGE_DIR/coverage.info" \
                --output-directory "$html_dir" \
                --title "AOCL-DLP Code Coverage Report" \
                --show-details \
                --branch-coverage \
                --function-coverage \
                --quiet
    fi

    if [ ! -f "$html_dir/index.html" ]; then
        print_error "Failed to generate HTML report"
        exit 1
    fi

    print_verbose "HTML report generated in: $html_dir"
}

# Function to generate coverage summary
generate_coverage_summary() {
    print_info "Generating coverage summary..."

    local summary_file="$COVERAGE_DIR/summary.txt"

    {
        echo "AOCL-DLP Code Coverage Summary"
        echo "============================="
        echo "Generated on: $(date)"
        echo "Build directory: $BUILD_DIR"
        echo "Coverage data: $COVERAGE_DIR/coverage.info"
        echo "HTML report: $COVERAGE_DIR/html/index.html"
        echo ""
        echo "Coverage Statistics:"
        echo "-------------------"
        lcov --summary "$COVERAGE_DIR/coverage.info" 2>/dev/null || echo "Failed to generate summary"
        echo ""
        echo "Top-level directories covered:"
        echo "-----------------------------"
        lcov --list "$COVERAGE_DIR/coverage.info" 2>/dev/null | head -20 || echo "Failed to list files"
    } > "$summary_file"

    print_verbose "Summary saved to: $summary_file"

    # Display key metrics
    if command -v lcov >/dev/null 2>&1; then
        local line_coverage=$(lcov --summary "$COVERAGE_DIR/coverage.info" 2>/dev/null | grep -E "lines\.\.\.*:" | awk '{print $2}' || echo "unknown")
        local func_coverage=$(lcov --summary "$COVERAGE_DIR/coverage.info" 2>/dev/null | grep -E "functions\.\.\.*:" | awk '{print $2}' || echo "unknown")
        local branch_coverage=$(lcov --summary "$COVERAGE_DIR/coverage.info" 2>/dev/null | grep -E "branches\.\.\.*:" | awk '{print $2}' || echo "unknown")

        echo ""
        print_success "Coverage Summary:"
        echo "  Line Coverage:     $line_coverage"
        echo "  Function Coverage: $func_coverage"
        echo "  Branch Coverage:   $branch_coverage"
    fi
}

# Function to open browser
open_browser() {
    if [ "$OPEN_BROWSER" = true ]; then
        local html_file="$COVERAGE_DIR/html/index.html"
        print_info "Opening coverage report in browser..."

        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$html_file" >/dev/null 2>&1 &
        elif command -v open >/dev/null 2>&1; then
            open "$html_file" >/dev/null 2>&1 &
        else
            print_warning "Could not open browser automatically"
            print_info "Please open: $html_file"
        fi
    fi
}

# Function to check if a port is available
is_port_available() {
    local port=$1
    if command -v netstat >/dev/null 2>&1; then
        ! netstat -tuln 2>/dev/null | grep -E -q ":${port}\b"
    elif command -v ss >/dev/null 2>&1; then
        ! ss -tuln 2>/dev/null | grep -E -q ":${port}\b"
    else
        # Fallback: try to bind to the port
        if command -v python3 >/dev/null 2>&1; then
            python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', $port))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null
        else
            return 0  # Assume available if we can't check
        fi
    fi
}

# Function to find an available port
find_available_port() {
    local start_port=$1
    local max_attempts=50
    local current_port=$start_port

    for ((i=0; i<max_attempts; i++)); do
        if is_port_available $current_port; then
            echo $current_port
            return 0
        fi
        current_port=$((current_port + 1))
    done

    print_error "Could not find an available port after $max_attempts attempts"
    return 1
}

# Function to ask user about HTTP server
ask_serve_http() {
    if [ "$SERVE_HTTP" = "auto" ]; then
        echo ""
        print_info "The coverage report has been generated successfully."
        echo ""
        echo "Would you like to start an HTTP server to view the reports?"
        echo "This is especially useful for SSH/remote environments."
        echo ""
        echo "Options:"
        echo "  [y] Yes, start HTTP server (recommended for SSH)"
        echo "  [n] No, just show file location"
        echo ""
        read -p "Start HTTP server? [y/n]: " -r serve_choice

        case $serve_choice in
            [Yy]|[Yy][Ee][Ss]|"")
                SERVE_HTTP=true
                ;;
            [Nn]|[Nn][Oo])
                SERVE_HTTP=false
                ;;
            *)
                print_warning "Invalid choice, defaulting to no HTTP server"
                SERVE_HTTP=false
                ;;
        esac
    fi
}

# Function to start HTTP server
start_http_server() {
    if [ "$SERVE_HTTP" != true ]; then
        return 0
    fi

    local html_dir="$COVERAGE_DIR/html"

    if [ ! -d "$html_dir" ]; then
        print_error "HTML directory not found: $html_dir"
        return 1
    fi

    # Check if Python 3 is available
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is required for HTTP server but not found"
        print_info "Please install Python 3 or use file:// URL to view reports"
        return 1
    fi

    # Find available port
    local actual_port
    actual_port=$(find_available_port $HTTP_PORT)
    if [ $? -ne 0 ]; then
        return 1
    fi

    if [ "$actual_port" != "$HTTP_PORT" ]; then
        print_info "Port $HTTP_PORT is busy, using port $actual_port instead"
    fi

    print_info "Starting HTTP server on port $actual_port..."
    print_verbose "Serving directory: $html_dir"

    # Get hostname/IP for display
    local hostname
    if command -v hostname >/dev/null 2>&1; then
        hostname=$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "localhost")
    else
        hostname="localhost"
    fi

    echo ""
    print_success "🌐 Coverage Report HTTP Server Started!"
    print_success "======================================"
    echo ""
    echo "📡 Server Details:"
    echo "   URL:        http://${hostname}:${actual_port}/"
    echo "   Local URL:  http://localhost:${actual_port}/"
    echo "   Port:       ${actual_port}"
    echo "   Directory:  ${html_dir}"
    echo ""
    echo "🔗 Access URLs:"
    echo "   Main Report: http://${hostname}:${actual_port}/"
    echo "   Local:       http://localhost:${actual_port}/"
    echo ""
    echo "📋 Instructions:"
    echo "   1. Open the URL above in your web browser"
    echo "   2. If using SSH, forward port: ssh -L ${actual_port}:localhost:${actual_port} user@host"
    echo "   3. Press Ctrl+C to stop the server"
    echo ""
    print_info "Server is ready! Press Ctrl+C to stop..."
    echo ""

    # Start the HTTP server
    cd "$html_dir"

    # Use Python's built-in HTTP server
    python3 -m http.server $actual_port 2>/dev/null || {
        print_error "Failed to start HTTP server"
        return 1
    }
}

# Function to cleanup temporary files
cleanup_temp_files() {
    print_verbose "Cleaning up temporary files..."
    rm -f /tmp/coverage_raw.info /tmp/coverage_filtered.info
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -b|--build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            -o|--output-dir)
                COVERAGE_DIR="$2"
                shift 2
                ;;
            -n|--no-browser)
                OPEN_BROWSER=false
                shift
                ;;
            -k|--keep-previous)
                CLEAN_PREVIOUS=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -s|--serve)
                SERVE_HTTP=true
                OPEN_BROWSER=false  # Disable browser when serving via HTTP
                shift
                ;;
            -p|--port)
                HTTP_PORT="$2"
                shift 2
                ;;
            --clean-only)
                clean_previous_reports
                print_success "Previous coverage reports cleaned"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    print_info "AOCL-DLP Code Coverage Report Generator"
    print_info "======================================="

    parse_arguments "$@"

    # Convert to absolute paths
    BUILD_DIR="$(cd "$BUILD_DIR" 2>/dev/null && pwd)" || {
        print_error "Invalid build directory: $BUILD_DIR"
        exit 1
    }
    COVERAGE_DIR="$(mkdir -p "$COVERAGE_DIR" && cd "$COVERAGE_DIR" && pwd)"

    print_verbose "Configuration:"
    print_verbose "  Project root: $PROJECT_ROOT"
    print_verbose "  Build directory: $BUILD_DIR"
    print_verbose "  Coverage output: $COVERAGE_DIR"
    print_verbose "  Open browser: $OPEN_BROWSER"
    print_verbose "  Clean previous: $CLEAN_PREVIOUS"
    print_verbose "  Serve HTTP: $SERVE_HTTP"
    print_verbose "  HTTP port: $HTTP_PORT"

    check_dependencies
    validate_directories
    clean_previous_reports
    create_coverage_directory
    generate_exclusion_patterns
    collect_coverage_data
    filter_coverage_data
    generate_html_report
    generate_coverage_summary
    cleanup_temp_files

    echo ""
    print_success "Coverage report generated successfully!"
    print_info "Report location: $COVERAGE_DIR/html/index.html"
    print_info "Summary: $COVERAGE_DIR/summary.txt"

    open_browser
    start_http_server
}

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Run main function with all arguments
main "$@"
