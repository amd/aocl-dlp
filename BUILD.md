# Building AOCL-DLP (Deep Learning Primitives)

This document provides instructions for building the AOCL-DLP library from source code.

## 📋 System Requirements

Before building AOCL-DLP, ensure your system meets the following requirements:

### Software
- CMake (≥ 3.26)
- C/C++ compiler with C11/C++17 support (e.g., GCC 11+, Clang 14+)
- OpenMP and/or pthread libraries (for multi-threading)
- ninja-build (optional, for Ninja generator support)

**Note: GCC 11 introduced AVX512_BF16 support, which is required for bfloat16 GEMM.**

### Hardware
- x86 CPU with AVX2/FMA3 support
- AVX512 support for enhanced performance
- AVX512_VNNI support for int8 GEMM
- AVX512_BF16 support for bfloat16 GEMM

## Build Configuration Options

AOCL-DLP uses CMake for its build system with several configurable options:

| Option                        | Default      | Description                                                        |
|-------------------------------|--------------|--------------------------------------------------------------------|
| **General Build Options**     |              |                                                                    |
| BUILD_EXAMPLES                | OFF          | Build example programs                                             |
| BUILD_BENCHMARKS              | OFF          | Build benchmark programs                                           |
| BUILD_TESTING                 | OFF          | Build test programs (requires DLP_TESTING_CTEST_DISABLED=OFF for CTest) |
| BUILD_DOXYGEN                 | OFF          | Build Doxygen documentation                                        |
| BUILD_SPHINX                  | OFF          | Build Sphinx documentation                                         |
| CMAKE_EXPORT_COMPILE_COMMANDS | OFF          | Generate compile_commands.json for tooling                         |
| CMAKE_BUILD_TYPE              | Release      | Build type ("Release", "Debug", "RelWithDebInfo", "Coverage")      |
| CMAKE_INSTALL_PREFIX          | /usr/local   | Installation directory                                             |
|                               |              |                                                                    |
| **Compiler Options**          |              |                                                                    |
| CMAKE_CXX_COMPILER            | system       | Specify C++ compiler (e.g., g++)                                   |
| CMAKE_C_COMPILER              | system       | Specify C compiler (e.g., gcc)                                     |
|                               |              |                                                                    |
| **Threading & Sanitizers**    |              |                                                                    |
| DLP_THREADING_MODEL           | "none"       | Threading model ("none", "openmp", "pthread")                      |
| DLP_ENABLE_OPENMP             | ON           | Override OpenMP support (auto-enabled by threading model)         |
| DLP_OPENMP_ROOT               | ""           | Custom path to OpenMP installation                                 |
| DLP_USE_LLVM_OPENMP           | OFF          | Force using LLVM OpenMP implementation                             |
| DLP_ENABLE_ASAN               | OFF          | Enable AddressSanitizer                                            |
| DLP_ENABLE_TSAN               | OFF          | Enable ThreadSanitizer                                             |
| DLP_ENABLE_UBSAN              | OFF          | Enable UndefinedBehaviorSanitizer                                  |
| DLP_TESTING_CTEST_DISABLED    | ON           | Disable CTest integration (set to OFF to enable with BUILD_TESTING)|
|                               |              |                                                                    |
| **Testing Options**           |              |                                                                    |
| DLP_TESTING_LINK_STATIC       | ON           | Link tests with static AOCL-DLP library for better performance    |
| DLP_TESTING_ENABLE_DETAILED_DEBUG | OFF      | Enable detailed debug information for tests                        |
|                               |              |                                                                    |
| **Benchmarking Options**      |              |                                                                    |
| DLP_BENCHMARKS_LINK_STATIC    | ON           | Link benchmarks with static AOCL-DLP library for better performance |
|                               |              |                                                                    |
| **Build Target Options**      |              |                                                                    |
| DLP_EXAMPLES_LINK_STATIC      | ON           | Link examples with static AOCL-DLP library for better performance |
|                               |              |                                                                    |
| **Kernel Dispatch Table**     |              |                                                                    |
| DLP_KDT_TABLE_SIZE            | 16           | Set table size for the Kernel Dispatch Table                       |
| DLP_KDT_CHAIN_SIZE            | 128          | Set table chain size for the Kernel Dispatch Table                 |


**Note:**
- Options can be set via `-D<option>=<value>` when invoking `cmake`.
- Some options (like `-GNinja`) are passed as command-line arguments, not as variables.
- For a full list of options, see the modular cmake files: `cmake/dlp_core_options.cmake`, `cmake/dlp_testing.cmake`, `cmake/dlp_benchmark.cmake`, `cmake/dlp_build_options.cmake`, and `cmake/dlp_documentation.cmake`.

## Quick Start Build

### Linux

1. Clone and enter project:

   ```bash
   git clone <repository-url> && cd aocl-dlp
   ```

2. Create an out-of-source build directory:

   ```bash
   mkdir -p build && cd build
   ```

3. Configure (choose generator):
   ```bash
   # Default (GNU Make)
   cmake ..

   # Ninja (fast incremental builds)
   cmake -G Ninja ..
   ```

4. Build:
   ```bash
   # Make
   make -j$(nproc)

   # Ninja
   ninja
   ```

5. For installation instructions, see [INSTALL.md](INSTALL.md).

---

## Advanced Build Configuration

### Enabling Additional Components

To enable benchmarks:

```bash
cmake -DBUILD_BENCHMARKS=ON ..
```

To enable testing with full CTest integration:

```bash
cmake -DBUILD_TESTING=ON -DDLP_TESTING_CTEST_DISABLED=OFF ..
```

**Note:** Both `BUILD_TESTING=ON` and `DLP_TESTING_CTEST_DISABLED=OFF` are required for full CTest integration. Using only `BUILD_TESTING=ON` builds tests but uses traditional testing instead of Google Test discovery.

### Threading Model Configuration

AOCL-DLP supports multiple threading models:

```bash
# No threading (default)
cmake -DDLP_THREADING_MODEL=none ..

# Enable OpenMP threading
cmake -DDLP_THREADING_MODEL=openmp ..

# Enable Pthread threading
cmake -DDLP_THREADING_MODEL=pthread ..
```

**Note:** Setting `DLP_THREADING_MODEL=openmp` automatically enables OpenMP support. The separate `DLP_ENABLE_OPENMP` option (default: ON) provides additional control and can disable OpenMP entirely with `-DDLP_ENABLE_OPENMP=OFF`.

For custom OpenMP installation:

```bash
cmake -DDLP_THREADING_MODEL=openmp -DDLP_OPENMP_ROOT=/path/to/openmp ..
```

### Static vs Dynamic Linking

By default, tests, benchmarks, and examples are linked with the static AOCL-DLP library for better performance. You can control this behavior:

**Enable static linking (default):**
```bash
cmake -DBUILD_EXAMPLES=ON -DDLP_EXAMPLES_LINK_STATIC=ON ..
cmake -DBUILD_TESTING=ON -DDLP_TESTING_LINK_STATIC=ON ..
cmake -DBUILD_BENCHMARKS=ON -DDLP_BENCHMARKS_LINK_STATIC=ON ..
```

**Use dynamic linking:**
```bash
cmake -DBUILD_EXAMPLES=ON -DDLP_EXAMPLES_LINK_STATIC=OFF ..
cmake -DBUILD_TESTING=ON -DDLP_TESTING_LINK_STATIC=OFF ..
cmake -DBUILD_BENCHMARKS=ON -DDLP_BENCHMARKS_LINK_STATIC=OFF ..
```

**Verify linking with ldd:**
```bash
# Static linking - no libaocl-dlp.so should appear
ldd examples/classic/simple_gemm_f32

# Dynamic linking - libaocl-dlp.so should appear
ldd examples/classic/simple_gemm_f32
```

**Note:** Static linking provides better performance by eliminating dynamic library loading overhead, which is especially beneficial for benchmarks and performance testing.

### Specifying Build Type

You can specify different build types:

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (default)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

# Coverage build (for code coverage analysis)
cmake -DCMAKE_BUILD_TYPE=Coverage ..
```

### Configuring Kernel Dispatch Table Size

By default, the Kernel Dispatch Table (KDT) is configured with 16 buckets and 128 chains for optimal memory usage and quick kernel queries. If necessary, it can be manually configured as below:

```bash
cmake -DDLP_KDT_TABLE_SIZE=<number_of_buckets> -DDLP_KDT_CHAIN_SIZE=<number_of_chains> ..
```

## Benchmarking

Enable and run tests and benchmarks in one place:



## Developer Tips

- **Out-of-tree builds**: Always build in a separate `build/` directory to keep sources clean.
- **Custom install prefix**:
  ```bash
  cmake -DCMAKE_INSTALL_PREFIX=/opt/aocl-dlp ..
  ```
- **Verbose output**:
  ```bash
  # Make
  make VERBOSE=1

  # Ninja
  ninja -v
  ```
- **Clean cache**:
  ```bash
  rm -rf build/* && cd build && cmake ..
  ```
- **Parallel builds**: Leverage all cores with `-j$(nproc)` (Make) or default Ninja parallelism.

## CMake Build System Overview

AOCL-DLP uses a modern CMake build system structured as follows:

- Main `CMakeLists.txt`: Orchestrates the overall build process
- `cmake/dlp_variables.cmake`: Sets project variables, languages and standards
- `cmake/dlp_core_options.cmake`: Defines core library options and threading models
- `cmake/dlp_testing.cmake`: Defines testing options and infrastructure
- `cmake/dlp_benchmark.cmake`: Defines benchmarking options and infrastructure
- `cmake/dlp_build_options.cmake`: Defines build target options (examples, sanitizers)
- `cmake/dlp_documentation.cmake`: Defines documentation options and generation
- `cmake/dlp_dependencies.cmake`: Manages OpenMP and Pthread dependencies
- `cmake/dlp_compiler_flags_linux.cmake`: Sets compiler flags for Linux
- `cmake/dlp_compiler_flags_windows.cmake`: Sets compiler flags for Windows
- `cmake/dlp_install.cmake`: Manages installation rules
- `cmake/dlp_extensions.cmake`: Defines file extensions

## Troubleshooting

### Threading Model Issues

If you encounter issues with the selected threading model:

1. Make sure the required libraries are installed on your system:
   - For OpenMP: OpenMP development libraries
   - For Pthread: POSIX threads library

2. For OpenMP-specific issues:

```bash
cmake -DDLP_THREADING_MODEL=openmp -DDLP_OPENMP_ROOT=/path/to/openmp ..
```

### Compiler Requirements

Make sure your compiler supports:
- C11 standard for C code
- C++17 standard for C++ code

### Build Performance

To speed up the build process, use parallel compilation:

```bash
make -j$(nproc)  # Linux
```

## Known Issues

- Warnings may appear during compilation (-Werror is currently disabled)
- Some platforms may require specific environment setup for threading model detection
