@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Clean Script
:: Removes build artifacts.
::
:: Usage:
::   clean.cmd [options]
::
:: Options:
::   --build-dir DIR    Build directory to clean (default: build)
::   --all              Also remove CMake cache (full reconfigure needed)
::   --help             Show this help message
:: ============================================================================

set "BUILD_DIR=build"
set "CLEAN_ALL=0"

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-dir" ( set "BUILD_DIR=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--all"       ( set "CLEAN_ALL=1"   & shift & goto :parse_args )
if /i "%~1"=="--help"      ( goto :show_help )
echo WARNING: Unknown option: %~1
shift
goto :parse_args
:done_args

if "%CLEAN_ALL%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Removing entire build directory: %BUILD_DIR%
        rmdir /s /q "%BUILD_DIR%"
        echo Done.
    ) else (
        echo Build directory "%BUILD_DIR%" does not exist. Nothing to clean.
    )
    exit /b 0
)

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Build directory "%BUILD_DIR%" is not configured. Nothing to clean.
    exit /b 0
)

echo Cleaning build artifacts in: %BUILD_DIR%
cmake --build "%BUILD_DIR%" --target clean

if %errorlevel% neq 0 (
    echo.
    echo WARNING: cmake --build clean returned an error.
    echo You may want to run: clean.cmd --all
    exit /b 1
)

echo.
echo Clean successful. Build cache preserved; run build.cmd to rebuild.
exit /b 0

:show_help
echo Usage: clean.cmd [options]
echo.
echo Options:
echo   --build-dir DIR    Build directory to clean (default: build)
echo   --all              Remove the entire build directory (full reconfigure needed)
echo   --help             Show this help message
exit /b 0
