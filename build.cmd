@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Build Script
:: Builds the project using the previously configured CMake build tree.
::
:: Usage:
::   build.cmd [options]
::
:: Options:
::   --build-dir DIR    Build directory (default: build)
::   --config TYPE      Build configuration for multi-config generators (default: Release)
::   --target TARGET    Specific target to build (default: all)
::   --parallel N       Number of parallel build jobs (default: auto)
::   --verbose          Enable verbose build output
::   --help             Show this help message
:: ============================================================================

set "BUILD_DIR=build"
set "CONFIG=Release"
set "TARGET="
set "PARALLEL="
set "VERBOSE="

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-dir" ( set "BUILD_DIR=%~2"  & shift & shift & goto :parse_args )
if /i "%~1"=="--config"    ( set "CONFIG=%~2"     & shift & shift & goto :parse_args )
if /i "%~1"=="--target"    ( set "TARGET=%~2"     & shift & shift & goto :parse_args )
if /i "%~1"=="--parallel"  ( set "PARALLEL=%~2"   & shift & shift & goto :parse_args )
if /i "%~1"=="--verbose"   ( set "VERBOSE=--verbose" & shift & goto :parse_args )
if /i "%~1"=="--help"      ( goto :show_help )
echo WARNING: Unknown option: %~1
shift
goto :parse_args
:done_args

:: Verify build directory exists and is configured
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo ERROR: Build directory "%BUILD_DIR%" is not configured.
    echo Run configure.cmd first.
    exit /b 1
)

:: Determine parallel jobs
set "PARALLEL_ARG="
if not "%PARALLEL%"=="" (
    set "PARALLEL_ARG=--parallel %PARALLEL%"
) else (
    set "PARALLEL_ARG=--parallel"
)

:: Build target argument
set "TARGET_ARG="
if not "%TARGET%"=="" (
    set "TARGET_ARG=--target %TARGET%"
)

echo.
echo ============================================================
echo  AOCL-DLP Build
echo ============================================================
echo  Build directory:  %BUILD_DIR%
echo  Configuration:    %CONFIG%
if not "%TARGET%"=="" echo  Target:           %TARGET%
echo ============================================================
echo.

cmake --build "%BUILD_DIR%" --config %CONFIG% %PARALLEL_ARG% %TARGET_ARG% %VERBOSE%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo Build successful.
exit /b 0

:show_help
echo Usage: build.cmd [options]
echo.
echo Options:
echo   --build-dir DIR    Build directory (default: build)
echo   --config TYPE      Build configuration: Release ^| Debug ^| RelWithDebInfo (default: Release)
echo   --target TARGET    Specific target to build (default: all)
echo   --parallel N       Number of parallel build jobs (default: auto)
echo   --verbose          Enable verbose build output
echo   --help             Show this help message
exit /b 0
