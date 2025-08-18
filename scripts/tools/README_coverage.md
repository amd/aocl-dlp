# AOCL-DLP Code Coverage Tools

This directory contains comprehensive code coverage tools for the AOCL-DLP project, designed to work seamlessly with the existing project infrastructure and CMake coverage support.

## Overview

The coverage tooling consists of two main components:

1. **`coverage_to_html.sh`** - Bash script for generating HTML coverage reports using gcov/lcov
2. **`coverage_analyzer.py`** - Python script for advanced coverage analysis and reporting

Both tools leverage the existing `scripts/common/` infrastructure and exclusion patterns, ensuring consistency with the project's coding standards and practices.

## Prerequisites

### System Dependencies

Install the required coverage tools:

```bash
# Ubuntu/Debian
sudo apt install lcov

# RHEL/CentOS/Fedora
sudo yum install lcov
# or
sudo dnf install lcov

# macOS (with Homebrew)
brew install lcov
```

### Project Setup

1. **Build with Coverage Support**:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
   cmake --build build
   ```

2. **Run Tests to Generate Coverage Data**:
   ```bash
   # Run your test suite to generate .gcda files
   ./build/tests/classic/test_gemm
   ./build/tests/classic/test_yaml_gemm
   # ... run other tests as needed
   ```

## Tool 1: HTML Coverage Report Generator (`coverage_to_html.sh`)

### Basic Usage

```bash
# Generate coverage report with defaults
./scripts/tools/coverage_to_html.sh

# Generate with verbose output, don't open browser
./scripts/tools/coverage_to_html.sh -v -n

# Generate and serve via HTTP server (perfect for SSH environments!)
./scripts/tools/coverage_to_html.sh -s

# Use custom build directory and serve on specific port
./scripts/tools/coverage_to_html.sh -b ./my_build_dir -s -p 9000

# Keep previous reports instead of cleaning them
./scripts/tools/coverage_to_html.sh -k
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-b, --build-dir DIR` | Specify build directory (default: `build`) |
| `-o, --output-dir DIR` | Specify coverage output directory (default: `coverage_reports`) |
| `-n, --no-browser` | Don't open browser automatically |
| `-k, --keep-previous` | Keep previous coverage reports |
| `-v, --verbose` | Enable verbose output |
| `-s, --serve` | Start HTTP server to serve reports (great for SSH environments) |
| `-p, --port PORT` | HTTP server port (default: 8080, auto-detects if busy) |
| `--clean-only` | Only clean previous reports and exit |

### Output Structure

The script generates:
```
coverage_reports/
├── coverage.info          # LCOV info file (intermediate)
├── summary.txt           # Text summary report
└── html/                 # HTML report directory
    ├── index.html        # Main coverage report
    ├── *.html           # Per-file coverage pages
    └── *.css, *.js      # Supporting files
```

### SSH/Remote Development Usage

The HTTP server feature is specifically designed for SSH and remote development environments:

#### Basic SSH Usage
```bash
# On the remote server, generate and serve coverage reports
./scripts/tools/coverage_to_html.sh -s

# The script will show output like:
# 🌐 Coverage Report HTTP Server Started!
# ======================================
# URL:        http://your-server:8080/
# Local URL:  http://localhost:8080/
```

#### SSH Port Forwarding
```bash
# Method 1: SSH with port forwarding (from your local machine)
ssh -L 8080:localhost:8080 user@remote-server

# Method 2: Add to your ~/.ssh/config
Host myserver
    HostName remote-server.example.com
    User myuser
    LocalForward 8080 localhost:8080

# Then connect and access http://localhost:8080 on your local machine
```

#### Automatic Port Detection
```bash
# If port 8080 is busy, the script automatically finds the next available port
./scripts/tools/coverage_to_html.sh -s
# Output: "Port 8080 is busy, using port 8081 instead"

# Use custom port range
./scripts/tools/coverage_to_html.sh -s -p 9000  # Start searching from port 9000
```

#### Server Management
```bash
# Start server in background (optional)
./scripts/tools/coverage_to_html.sh -s > coverage_server.log 2>&1 &

# Stop server when done
# Press Ctrl+C or kill the process
```

### Exclusion Patterns

The script automatically excludes:
- Build directories (`build/`, `install/`, `exp/`)
- Test code (`tests/`, `bench/`, `*_test.*`)
- Documentation (`docs/`, `scripts/`)
- Examples (`examples/`)
- External dependencies (`xbyak/`, `JIT/`, `external/`)
- System headers (`/usr/include/`, `/opt/*/include/`)
- Generated files (`CMakeFiles/`, `.cmake/`)

## Tool 2: Advanced Coverage Analyzer (`coverage_analyzer.py`)

### Basic Usage

```bash
# Analyze coverage from default location
./scripts/tools/coverage_analyzer.py

# Analyze specific LCOV file
./scripts/tools/coverage_analyzer.py --lcov coverage_reports/coverage.info

# Find coverage gaps below 70%
./scripts/tools/coverage_analyzer.py --threshold 70

# Export analysis to JSON
./scripts/tools/coverage_analyzer.py --json coverage_analysis.json
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--lcov PATH` | Path to LCOV info file for analysis |
| `--build-dir DIR` | Build directory (default: `build`) |
| `--output-dir DIR` | Output directory for reports (default: `coverage_reports`) |
| `--threshold PERCENT` | Coverage threshold for gap analysis (default: 80.0) |
| `--json PATH` | Export coverage data to JSON file |
| `--discover-sources` | Discover and list source files for coverage analysis |
| `--gaps-only` | Only show coverage gaps, not full report |

### Features

#### Source File Discovery
```bash
# List all source files that will be analyzed for coverage
./scripts/tools/coverage_analyzer.py --discover-sources
```

#### Coverage Gap Analysis
```bash
# Find files/functions with coverage below threshold
./scripts/tools/coverage_analyzer.py --threshold 85 --gaps-only
```

#### Detailed Reporting
The analyzer generates:
- **Summary Statistics**: Overall line, function, and branch coverage
- **File-by-File Analysis**: Coverage percentages for each source file
- **Uncovered Functions**: List of functions not executed during tests
- **Coverage Gaps**: Files and functions below specified threshold

#### JSON Export
```bash
# Export detailed coverage data for further analysis
./scripts/tools/coverage_analyzer.py --json detailed_coverage.json
```

## Complete Workflow Example

Here's a complete workflow for generating and analyzing coverage:

```bash
# 1. Clean build with coverage
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DBUILD_TESTING=ON
cmake --build build

# 2. Run test suite
./build/tests/classic/test_gemm
./build/tests/classic/test_yaml_gemm
./build/tests/classic/test_postops_yaml

# 3. Generate HTML coverage report
./scripts/tools/coverage_to_html.sh -v

# 4. Analyze coverage gaps
./scripts/tools/coverage_analyzer.py --threshold 80

# 5. Export detailed analysis
./scripts/tools/coverage_analyzer.py --json coverage_analysis.json

# 6. View results
# HTML report opens automatically, or visit: coverage_reports/html/index.html
# Text summary: coverage_reports/summary.txt
# Detailed analysis: coverage_reports/detailed_report.txt
```

## Integration with Existing Infrastructure

### Leverages `scripts/common/`

Both tools use the existing project infrastructure:

- **File Filtering**: Uses `scripts/common/utils.py` for consistent file filtering
- **Exclusion Patterns**: Extends `scripts/common/config.py` patterns for coverage-specific exclusions
- **Coding Standards**: Follows project's bash scripting and Python patterns

### CMake Integration

The tools work with the existing CMake coverage support:

```cmake
# Coverage flags are automatically applied when using Coverage build type
set(CMAKE_BUILD_TYPE Coverage)

# This enables:
# - Compiler flags: -g -O0 -fprofile-arcs -ftest-coverage -fprofile-abs-path
# - Linker flags: --coverage
```

## Troubleshooting

### Common Issues

1. **No coverage data found**
   ```
   Error: No coverage data found in build directory
   ```
   **Solution**: Ensure you built with `CMAKE_BUILD_TYPE=Coverage` and ran tests

2. **Missing dependencies**
   ```
   Error: Missing required dependencies: lcov
   ```
   **Solution**: Install lcov using your system package manager

3. **Permission denied**
   ```
   Permission denied: ./scripts/tools/coverage_to_html.sh
   ```
   **Solution**: Make script executable: `chmod +x scripts/tools/coverage_to_html.sh`

4. **Python import errors**
   ```
   Error: Could not import common modules
   ```
   **Solution**: Run from project root directory where `scripts/common/` is accessible

### Debugging Tips

- Use `-v` flag with `coverage_to_html.sh` for verbose output
- Check that `.gcno` and `.gcda` files exist in build directory
- Verify CMake was configured with `Coverage` build type
- Ensure tests actually execute the code you want to measure

## Advanced Usage

### Custom Exclusion Patterns

To modify exclusion patterns, edit:
- `scripts/common/config.py` for project-wide patterns
- Tool-specific patterns in the respective scripts

### CI/CD Integration

Example GitHub Actions workflow:
```yaml
- name: Generate Coverage Report
  run: |
    cmake -B build -DCMAKE_BUILD_TYPE=Coverage
    cmake --build build
    ./build/tests/classic/test_gemm
    ./scripts/tools/coverage_to_html.sh -n

- name: Upload Coverage Report
  uses: actions/upload-artifact@v3
  with:
    name: coverage-report
    path: coverage_reports/html/
```

### Integration with Memory Bank

When working with the project's memory bank system, update the relevant documentation:

```bash
# Update memory bank with coverage information
echo "Generated comprehensive coverage reports in coverage_reports/" >> memory-bank/progress.md
```

## File Locations

- **Tools**: `scripts/tools/coverage_to_html.sh`, `scripts/tools/coverage_analyzer.py`
- **Common Infrastructure**: `scripts/common/config.py`, `scripts/common/utils.py`
- **CMake Support**: `cmake/dlp_compiler_flags_linux.cmake`
- **Documentation**: This file (`scripts/tools/README_coverage.md`)

## Contributing

When modifying the coverage tools:

1. **Test thoroughly** with different build configurations
2. **Update documentation** if adding new features
3. **Follow existing patterns** from `scripts/common/`
4. **Consider exclusion patterns** for new directories or file types
5. **Update memory bank** documentation as appropriate
