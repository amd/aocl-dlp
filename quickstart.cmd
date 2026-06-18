@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Quick Start Script
:: One-command configure + build + test cycle.
::
:: Usage:
::   quickstart.cmd [options]
::
:: Options:
::   --build-dir DIR     Build directory (default: build)
::   --build-type TYPE   Release | Debug | RelWithDebInfo (default: Release)
::   --skip-tests        Skip running tests after build
::   --skip-examples     Do not build examples
::   --openmp            Enable OpenMP threading
::   --install           Also install after building
::   --clean             Start fresh (remove build dir)
::   --help              Show this help message
:: ============================================================================

set "BUILD_DIR=build"
set "BUILD_TYPE=Release"
set "SKIP_TESTS=0"
set "SKIP_EXAMPLES=0"
set "OPENMP="
set "DO_INSTALL=0"
set "DO_CLEAN="

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-dir"     ( set "BUILD_DIR=%~2"  & shift & shift & goto :parse_args )
if /i "%~1"=="--build-type"    ( set "BUILD_TYPE=%~2"  & shift & shift & goto :parse_args )
if /i "%~1"=="--skip-tests"    ( set "SKIP_TESTS=1"    & shift & goto :parse_args )
if /i "%~1"=="--skip-examples" ( set "SKIP_EXAMPLES=1"  & shift & goto :parse_args )
if /i "%~1"=="--openmp"        ( set "OPENMP=--openmp"  & shift & goto :parse_args )
if /i "%~1"=="--install"       ( set "DO_INSTALL=1"     & shift & goto :parse_args )
if /i "%~1"=="--clean"         ( set "DO_CLEAN=--clean"  & shift & goto :parse_args )
if /i "%~1"=="--help"          ( goto :show_help )
echo WARNING: Unknown option: %~1
shift
goto :parse_args
:done_args

set "CONFIGURE_FLAGS=--build-dir "%BUILD_DIR%" --build-type %BUILD_TYPE% --tests"
if "%SKIP_EXAMPLES%"=="0" set "CONFIGURE_FLAGS=%CONFIGURE_FLAGS% --examples"
if not "%OPENMP%"=="" set "CONFIGURE_FLAGS=%CONFIGURE_FLAGS% %OPENMP%"
if not "%DO_CLEAN%"=="" set "CONFIGURE_FLAGS=%CONFIGURE_FLAGS% %DO_CLEAN%"

echo.
echo ==============================================================
echo  AOCL-DLP Quick Start
echo ==============================================================
echo.

:: Step 1: Configure
echo [1/4] Configuring...
call "%~dp0configure.cmd" %CONFIGURE_FLAGS%
if %errorlevel% neq 0 (
    echo FAILED at configuration step.
    exit /b 1
)

:: Step 2: Build
echo.
echo [2/4] Building...
call "%~dp0build.cmd" --build-dir "%BUILD_DIR%" --config %BUILD_TYPE%
if %errorlevel% neq 0 (
    echo FAILED at build step.
    exit /b 1
)

:: Step 3: Test
if "%SKIP_TESTS%"=="0" (
    echo.
    echo [3/4] Running tests...
    call "%~dp0test.cmd" --build-dir "%BUILD_DIR%" --config %BUILD_TYPE%
    if !errorlevel! neq 0 (
        echo WARNING: Some tests failed, but build is complete.
    )
) else (
    echo.
    echo [3/4] Tests skipped.
)

:: Step 4: Install
if "%DO_INSTALL%"=="1" (
    echo.
    echo [4/4] Installing...
    call "%~dp0install.cmd" --build-dir "%BUILD_DIR%" --config %BUILD_TYPE%
    if !errorlevel! neq 0 (
        echo FAILED at install step.
        exit /b 1
    )
) else (
    echo.
    echo [4/4] Install skipped. Run install.cmd to install.
)

echo.
echo ==============================================================
echo  Quick start complete!
echo ==============================================================
exit /b 0

:show_help
echo Usage: quickstart.cmd [options]
echo.
echo Options:
echo   --build-dir DIR     Build directory (default: build)
echo   --build-type TYPE   Release ^| Debug ^| RelWithDebInfo (default: Release)
echo   --skip-tests        Skip running tests after build
echo   --skip-examples     Do not build examples
echo   --openmp            Enable OpenMP threading
echo   --install           Also install after building
echo   --clean             Start fresh (remove build dir)
echo   --help              Show this help message
exit /b 0
