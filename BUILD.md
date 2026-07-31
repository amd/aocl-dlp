# Building AOCL-DLP (Deep Learning Primitives)

This document provides instructions for building the AOCL-DLP library from source code.

## 🚀 Quick Start with CMake Presets

AOCL-DLP provides a modular set of CMake presets for standardized, easy-to-use build configurations. Presets are named `<flavor>-<compiler>[-make]` so you can pick a build flavor, a compiler (GCC/Clang on Linux, MSVC/clang-cl on Windows), and a generator (Ninja by default, or GNU Make via the `-make` suffix on Linux).

> **Requirements:** CMake presets require **CMake 3.26 or later**, a C/C++ compiler (**GCC** or **Clang** on Linux; **MSVC** or **clang-cl** on Windows), and a build tool. The presets use **[Ninja](https://ninja-build.org/)** by default; on Debian/Ubuntu install it with `sudo apt install ninja-build` (or use your platform's package manager). If you prefer GNU Make, use a `-make` preset (available for the `dev` and `release` flavors, Linux only) or the [manual build instructions](#quick-start-build) below.

> **Windows note:** Run the Windows presets from a **Developer Command Prompt / Developer PowerShell for Visual Studio** (or a shell where `vcvarsall.bat` has been sourced) so that the MSVC toolchain and `ninja` are on `PATH`. The `clang-cl` presets additionally require an LLVM/Clang installation providing `clang-cl`. Presets are host-gated by an OS condition, so only presets valid for your current platform are listed by `cmake --list-presets`.

### Available Presets

Flavors (`dev`, `sanitizers`, `all`, `release`) are combined with a compiler (`gcc`/`clang` on Linux, `msvc`/`clang-cl` on Windows) and, optionally, the Make generator (`-make`, Linux only).

**Linux presets** (available when building on Linux):

| Preset | Compiler | Generator | Description |
|--------|----------|-----------|-------------|
| `release-gcc` / `release-gcc-make` | GCC | Ninja / Make | Optimized release build for production use |
| `release-clang` / `release-clang-make` | Clang | Ninja / Make | Optimized release build for production use |
| `dev-gcc` / `dev-gcc-make` | GCC | Ninja / Make | Debug build with testing enabled |
| `dev-clang` / `dev-clang-make` | Clang | Ninja / Make | Debug build with testing enabled |
| `sanitizers-gcc` / `sanitizers-clang` | GCC / Clang | Ninja | Debug build with AddressSanitizer and UndefinedBehaviorSanitizer |
| `all-gcc` / `all-clang` | GCC / Clang | Ninja | Debug build with all features (tests, benchmarks, examples, OpenMP) |

**Windows presets** (available when building on Windows):

| Preset | Compiler | Generator | Description |
|--------|----------|-----------|-------------|
| `release-msvc` | MSVC | Ninja | Optimized release build for production use |
| `release-clang-cl` | clang-cl | Ninja | Optimized release build for production use |
| `dev-msvc` | MSVC | Ninja | Debug build with testing enabled |
| `dev-clang-cl` | clang-cl | Ninja | Debug build with testing enabled |
| `all-msvc` / `all-clang-cl` | MSVC / clang-cl | Ninja | Debug build with all features (tests, benchmarks, examples, OpenMP) |

### Usage Examples

> **Note:** Ninja-based presets require `ninja-build` to be installed. Use a `-make` preset to build with GNU Make instead.

```bash
# List all available presets (add =build, =test, or =workflow for other categories)
cmake --list-presets

# Release build with GCC (recommended for production)
cmake --preset=release-gcc
cmake --build --preset=release-gcc

# Debug build for development with Clang
cmake --preset=dev-clang
cmake --build --preset=dev-clang

# Build with sanitizers for debugging memory issues
cmake --preset=sanitizers-gcc
cmake --build --preset=sanitizers-gcc
ctest --preset=sanitizers-gcc

# Build with all features enabled
cmake --preset=all-gcc
cmake --build --preset=all-gcc
ctest --preset=all-gcc

# Release build with Clang using GNU Make instead of Ninja
cmake --preset=release-clang-make
cmake --build --preset=release-clang-make

# One-command workflow (configure + build, release with GCC)
cmake --workflow --preset=default

# Full workflow (configure + build + test, all features)
cmake --workflow --preset=full-gcc
```

On Windows, run the equivalent commands from a Developer Command Prompt / Developer PowerShell for Visual Studio:

```bat
:: Release build with MSVC (recommended for production)
cmake --preset=release-msvc
cmake --build --preset=release-msvc

:: Debug build for development with clang-cl
cmake --preset=dev-clang-cl
cmake --build --preset=dev-clang-cl

:: Build with all features enabled using MSVC
cmake --preset=all-msvc
cmake --build --preset=all-msvc
ctest --preset=all-msvc

:: Full workflow (configure + build + test, all features with clang-cl)
cmake --workflow --preset=full-clang-cl
```

For detailed preset documentation, see the [CMake Presets Reference](#cmake-presets-reference) section below.

---

## 📋 System Requirements

Before building AOCL-DLP, ensure your system meets the following requirements:

### Software
- CMake (≥ 3.26)
- C/C++ compiler with C11/C++17 support (e.g., GCC 11+, Clang 14+)
- OpenMP library (for multi-threading)
- ninja-build (default generator for the CMake presets; optional if you use a `-make` preset or the manual build)

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
| DLP_THREADING_MODEL           | "none"       | Threading model ("none", "openmp")                      |
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

AOCL-DLP supports the following threading models:

```bash
# No threading (default)
cmake -DDLP_THREADING_MODEL=none ..

# Enable OpenMP threading
cmake -DDLP_THREADING_MODEL=openmp ..
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
- `cmake/dlp_dependencies.cmake`: Manages OpenMP dependencies
- `cmake/dlp_compiler_flags_linux.cmake`: Sets compiler flags for Linux
- `cmake/dlp_compiler_flags_windows.cmake`: Sets compiler flags for Windows
- `cmake/dlp_install.cmake`: Manages installation rules
- `cmake/dlp_extensions.cmake`: Defines file extensions

## Troubleshooting

### Threading Model Issues

If you encounter issues with the selected threading model:

1. Make sure the required libraries are installed on your system:
   - For OpenMP: OpenMP development libraries

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

---

## CMake Presets Reference

AOCL-DLP uses a modular CMake preset system. Reusable fragments live under
`cmake/Presets/` (project options, base build flavors, architecture, OS,
compilers, and generators) and are assembled into the user-facing presets by
`cmake/Presets/Default.json`. The root `CMakePresets.json` simply includes that
file. This mirrors the layout used by AOCL-Utils and makes it easy to add new
compilers, generators, or flavors.

Presets are named `<flavor>-<compiler>[-make]`:

- **flavor** – `release` (optimized, extras off), `dev` (Debug + tests),
  `sanitizers` (Debug + ASan/UBSan + tests, Linux only), or `all` (Debug +
  tests + benchmarks + examples + OpenMP).
- **compiler** – `gcc` or `clang` on Linux; `msvc` or `clang-cl` on Windows.
- **generator** – Ninja by default; `-make` selects the GNU Make generator
  (provided for the `dev` and `release` flavors, Linux only).

Each preset carries an OS `condition` (via the `Os/Linux.json` or
`Os/Windows.json` fragments), so `cmake --list-presets` only shows presets that
match the current host. The modular fragments are assembled per platform:

- Linux: `x64-linux-gcc.json`, `x64-linux-llvm.json`
- Windows: `x64-windows-msvc.json`, `x64-windows-clang-cl.json`

New compiler fragments live under `cmake/Presets/Compilers/` (`Gcc.json`,
`Clang.json`, `Msvc.json`, `ClangCl.json`) and OS fragments under
`cmake/Presets/Os/`.

> **Windows note:** Run the Windows presets from a **Developer Command Prompt /
> Developer PowerShell for Visual Studio** so the MSVC toolchain and `ninja` are
> on `PATH`. `clang-cl` presets also require an LLVM/Clang install providing
> `clang-cl`. The Windows compiler fragments use MSVC-style flags
> (`/W4 /permissive- /EHsc`, `/Zi /Od /RTC1` for Debug, `/O2 /DNDEBUG` for
> Release).

### Configure & Build Presets

Each configure preset has a build preset of the same name.

**Linux presets:**

| Preset | Compiler | Generator | Description | Use Case |
|--------|----------|-----------|-------------|----------|
| `release-gcc` / `release-gcc-make` | GCC | Ninja / Make | Optimized release build | Production builds, general use |
| `release-clang` / `release-clang-make` | Clang | Ninja / Make | Optimized release build | Production builds, general use |
| `dev-gcc` / `dev-gcc-make` | GCC | Ninja / Make | Debug build with tests | Development, debugging, testing |
| `dev-clang` / `dev-clang-make` | Clang | Ninja / Make | Debug build with tests | Development, debugging, testing |
| `sanitizers-gcc` / `sanitizers-clang` | GCC / Clang | Ninja | Debug + ASAN + UBSAN + tests | Memory error and undefined behavior detection |
| `all-gcc` / `all-clang` | GCC / Clang | Ninja | Debug + tests + benchmarks + examples + OpenMP | Full development environment |

**Windows presets:**

| Preset | Compiler | Generator | Description | Use Case |
|--------|----------|-----------|-------------|----------|
| `release-msvc` | MSVC | Ninja | Optimized release build | Production builds, general use |
| `release-clang-cl` | clang-cl | Ninja | Optimized release build | Production builds, general use |
| `dev-msvc` | MSVC | Ninja | Debug build with tests | Development, debugging, testing |
| `dev-clang-cl` | clang-cl | Ninja | Debug build with tests | Development, debugging, testing |
| `all-msvc` / `all-clang-cl` | MSVC / clang-cl | Ninja | Debug + tests + benchmarks + examples + OpenMP | Full development environment |

```bash
# Configure then build
cmake --preset=<preset-name>
cmake --build --preset=<preset-name>
```

### Test Presets

Test presets exist for the flavors that build tests (`dev`, `sanitizers`, `all`),
for both compilers. Sanitizer test runs stop on the first failure.

| Preset | Description |
|--------|-------------|
| `dev-gcc` / `dev-clang` | Run tests in debug configuration (Linux) |
| `sanitizers-gcc` / `sanitizers-clang` | Run tests with sanitizers (Linux) |
| `all-gcc` / `all-clang` | Run all tests, full feature build (Linux) |
| `dev-msvc` / `dev-clang-cl` | Run tests in debug configuration (Windows) |
| `all-msvc` / `all-clang-cl` | Run all tests, full feature build (Windows) |

```bash
ctest --preset=dev-gcc
ctest --preset=sanitizers-clang
ctest --preset=all-gcc

# Windows
ctest --preset=dev-msvc
ctest --preset=all-clang-cl
```

### Workflow Presets

Workflows combine configure, build, and (optionally) test steps:

| Workflow | Steps | Description |
|----------|-------|-------------|
| `default` | configure → build (`release-gcc`) | Standard release build with GCC |
| `full-gcc` | configure → build → test (`all-gcc`) | Build and test with all features (GCC) |
| `full-clang` | configure → build → test (`all-clang`) | Build and test with all features (Clang) |
| `full-msvc` | configure → build → test (`all-msvc`) | Build and test with all features (MSVC, Windows) |
| `full-clang-cl` | configure → build → test (`all-clang-cl`) | Build and test with all features (clang-cl, Windows) |

### IDE Integration

CMake presets are supported by most modern IDEs:

- **Visual Studio Code**: With CMake Tools extension, presets appear in the CMake sidebar
- **CLion**: Presets are automatically detected and available in the CMake profile settings
- **Visual Studio**: Presets are integrated into the CMake configuration UI

### Tips for Using Presets

1. **First-time setup**: Run `cmake --list-presets` to see all available options
2. **Build location**: All presets use `build/<preset-name>` as the build directory
3. **Switching presets**: Each preset builds in its own directory, so you can switch without reconfiguring
4. **Inheritance**: Custom presets can inherit from existing presets using the `"inherits"` field
5. **Combining with flags**: You can still pass additional `-D` flags after the preset:
   ```bash
   cmake --preset=release-gcc -DCMAKE_INSTALL_PREFIX=/custom/path
   ```
