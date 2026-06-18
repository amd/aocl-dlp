@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Benchmark Script
:: Runs benchmark executables.
::
:: Usage:
::   bench.cmd [options] [-- benchmark_args...]
::
:: Options:
::   --build-dir DIR     Build directory (default: build)
::   --config TYPE       Build configuration (default: Release)
::   --target NAME       Run specific benchmark (e.g. bench_gemm, bench_batch_gemm)
::   --list              List available benchmark executables
::   --help              Show this help message
::
:: Everything after -- is passed directly to the benchmark executable.
::
:: Examples:
::   bench.cmd --target bench_gemm -- --benchmark_filter="BM_Gemm"
::   bench.cmd --target bench_gemm -- --benchmark_out=results.json --benchmark_out_format=json
:: ============================================================================

set "BUILD_DIR=build"
set "CONFIG=Release"
set "TARGET="
set "LIST_ONLY=0"
set "BENCH_ARGS="

:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--" ( shift & goto :collect_bench_args )
if /i "%~1"=="--build-dir" ( set "BUILD_DIR=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--config"    ( set "CONFIG=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--target"    ( set "TARGET=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--list"      ( set "LIST_ONLY=1"   & shift & goto :parse_args )
if /i "%~1"=="--help"      ( goto :show_help )
echo WARNING: Unknown option: %~1
shift
goto :parse_args

:collect_bench_args
if "%~1"=="" goto :done_args
set "BENCH_ARGS=!BENCH_ARGS! %~1"
shift
goto :collect_bench_args
:done_args

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo ERROR: Build directory "%BUILD_DIR%" is not configured.
    echo Run: configure.cmd --benchmarks
    exit /b 1
)

:: Check if benchmarks were enabled
findstr /c:"BUILD_BENCHMARKS:BOOL=ON" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Benchmarks were not enabled during configuration.
    echo Run: configure.cmd --benchmarks
    exit /b 1
)

:: Ensure the shared library is on PATH
set "PATH=%BUILD_DIR%\Release;%BUILD_DIR%\%CONFIG%;%BUILD_DIR%;%PATH%"

:: Find benchmark executables
set "BENCH_DIR=%BUILD_DIR%\bench\%CONFIG%"
if not exist "%BENCH_DIR%" set "BENCH_DIR=%BUILD_DIR%\bench"
if not exist "%BENCH_DIR%" set "BENCH_DIR=%BUILD_DIR%\%CONFIG%"

if "%LIST_ONLY%"=="1" (
    echo Available benchmarks in %BENCH_DIR%:
    for %%f in ("%BENCH_DIR%\bench_*.exe") do (
        echo   %%~nf
    )
    exit /b 0
)

if not "%TARGET%"=="" (
    :: Run specific benchmark
    set "BENCH_EXE=%BENCH_DIR%\%TARGET%.exe"
    if not exist "!BENCH_EXE!" (
        echo ERROR: Benchmark executable not found: !BENCH_EXE!
        echo Run: bench.cmd --list
        exit /b 1
    )
    echo Running: %TARGET% %BENCH_ARGS%
    echo.
    "!BENCH_EXE!" %BENCH_ARGS%
    exit /b !errorlevel!
) else (
    :: Run all benchmarks
    set "FOUND=0"
    set "FAILED=0"
    for %%f in ("%BENCH_DIR%\bench_*.exe") do (
        set "FOUND=1"
        echo.
        echo ============================================================
        echo  Running: %%~nf
        echo ============================================================
        "%%f" %BENCH_ARGS%
        if !errorlevel! neq 0 set "FAILED=1"
    )
    if "!FOUND!"=="0" (
        echo ERROR: No benchmark executables found in %BENCH_DIR%
        exit /b 1
    )
    if "!FAILED!"=="1" (
        echo.
        echo WARNING: Some benchmarks reported errors.
        exit /b 1
    )
    echo.
    echo All benchmarks completed.
    exit /b 0
)

:show_help
echo Usage: bench.cmd [options] [-- benchmark_args...]
echo.
echo Options:
echo   --build-dir DIR     Build directory (default: build)
echo   --config TYPE       Build configuration (default: Release)
echo   --target NAME       Run specific benchmark (e.g. bench_gemm)
echo   --list              List available benchmark executables
echo   --help              Show this help message
echo.
echo Everything after -- is passed directly to the benchmark executable.
echo.
echo Examples:
echo   bench.cmd --target bench_gemm
echo   bench.cmd --target bench_gemm -- --benchmark_out=results.json
exit /b 0
