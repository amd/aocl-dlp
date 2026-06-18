@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Configuration Script
:: Generates the CMake build tree in the specified build directory.
::
:: Usage:
::   configure.cmd [options]
::
:: Options:
::   --build-dir DIR        Build directory (default: build)
::   --install-dir DIR      Install prefix (default: build\install)
::   --build-type TYPE      Release | Debug | RelWithDebInfo (default: Release)
::   --generator GEN        CMake generator (default: auto-detect)
::   --tests                Enable tests
::   --benchmarks           Enable benchmarks
::   --examples             Enable examples
::   --all                  Enable tests + benchmarks + examples
::   --openmp               Enable OpenMP threading model
::   --static-tests         Link tests with static library
::   --static-benchmarks    Link benchmarks with static library
::   --static-examples      Link examples with static library
::   --logging              Enable DLP logging
::   --jit-debug            Enable JIT debugging
::   --docs                 Enable Doxygen + Sphinx docs
::   --clean                Remove build directory before configuring
::   --help                 Show this help message
::
:: Environment variables:
::   CMAKE_PREFIX_PATH      Additional CMake search paths
::   CC / CXX               Override C / C++ compilers
:: ============================================================================

set "BUILD_DIR=build"
set "INSTALL_DIR="
set "BUILD_TYPE=Release"
set "GENERATOR="
set "ENABLE_TESTS=OFF"
set "ENABLE_BENCHMARKS=OFF"
set "ENABLE_EXAMPLES=OFF"
set "THREADING_MODEL=none"
set "STATIC_TESTS=OFF"
set "STATIC_BENCHMARKS=OFF"
set "STATIC_EXAMPLES=OFF"
set "ENABLE_LOGGING=OFF"
set "ENABLE_JIT_DEBUG=OFF"
set "ENABLE_DOXYGEN=OFF"
set "ENABLE_SPHINX=OFF"
set "DO_CLEAN=0"
set "EXTRA_CMAKE_ARGS="

:: Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-dir"       ( set "BUILD_DIR=%~2"        & shift & shift & goto :parse_args )
if /i "%~1"=="--install-dir"     ( set "INSTALL_DIR=%~2"      & shift & shift & goto :parse_args )
if /i "%~1"=="--build-type"      ( set "BUILD_TYPE=%~2"       & shift & shift & goto :parse_args )
if /i "%~1"=="--generator"       ( set "GENERATOR=%~2"        & shift & shift & goto :parse_args )
if /i "%~1"=="--tests"           ( set "ENABLE_TESTS=ON"      & shift & goto :parse_args )
if /i "%~1"=="--benchmarks"      ( set "ENABLE_BENCHMARKS=ON" & shift & goto :parse_args )
if /i "%~1"=="--examples"        ( set "ENABLE_EXAMPLES=ON"   & shift & goto :parse_args )
if /i "%~1"=="--all"             ( set "ENABLE_TESTS=ON" & set "ENABLE_BENCHMARKS=ON" & set "ENABLE_EXAMPLES=ON" & shift & goto :parse_args )
if /i "%~1"=="--openmp"          ( set "THREADING_MODEL=openmp" & shift & goto :parse_args )
if /i "%~1"=="--static-tests"    ( set "STATIC_TESTS=ON"      & shift & goto :parse_args )
if /i "%~1"=="--static-benchmarks" ( set "STATIC_BENCHMARKS=ON" & shift & goto :parse_args )
if /i "%~1"=="--static-examples" ( set "STATIC_EXAMPLES=ON"   & shift & goto :parse_args )
if /i "%~1"=="--logging"         ( set "ENABLE_LOGGING=ON"    & shift & goto :parse_args )
if /i "%~1"=="--jit-debug"       ( set "ENABLE_JIT_DEBUG=ON"  & shift & goto :parse_args )
if /i "%~1"=="--docs"            ( set "ENABLE_DOXYGEN=ON" & set "ENABLE_SPHINX=ON" & shift & goto :parse_args )
if /i "%~1"=="--clean"           ( set "DO_CLEAN=1"           & shift & goto :parse_args )
if /i "%~1"=="--help"            ( goto :show_help )
:: Pass unknown args directly to CMake
set "EXTRA_CMAKE_ARGS=!EXTRA_CMAKE_ARGS! %~1"
shift
goto :parse_args
:done_args

:: Default install dir
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%BUILD_DIR%\install"

:: Clean if requested
if "%DO_CLEAN%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Removing existing build directory: %BUILD_DIR%
        rmdir /s /q "%BUILD_DIR%"
    )
)

:: Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: Determine generator
set "GEN_ARG="
set "ARCH_ARG="
if not "%GENERATOR%"=="" (
    set "GEN_ARG=-G "%GENERATOR%""
    :: Add -A x64 for Visual Studio generators
    echo "%GENERATOR%" | findstr /i "Visual Studio" >nul 2>&1
    if !errorlevel! equ 0 set "ARCH_ARG=-A x64"
) else (
    where ninja >nul 2>&1
    if !errorlevel! equ 0 (
        set "GEN_ARG=-G "Ninja""
        echo Auto-detected Ninja generator
    ) else (
        echo Using default CMake generator ^(Visual Studio^)
        set "ARCH_ARG=-A x64"
    )
)

:: Print configuration summary
echo.
echo ============================================================
echo  AOCL-DLP Configuration
echo ============================================================
echo  Build directory:    %BUILD_DIR%
echo  Install directory:  %INSTALL_DIR%
echo  Build type:         %BUILD_TYPE%
echo  Tests:              %ENABLE_TESTS%
echo  Benchmarks:         %ENABLE_BENCHMARKS%
echo  Examples:           %ENABLE_EXAMPLES%
echo  Threading model:    %THREADING_MODEL%
echo  Logging:            %ENABLE_LOGGING%
echo  JIT debugging:      %ENABLE_JIT_DEBUG%
echo ============================================================
echo.

:: Run CMake configure
cmake %GEN_ARG% %ARCH_ARG% ^
    -S "%~dp0." ^
    -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DBUILD_TESTING=%ENABLE_TESTS% ^
    -DBUILD_BENCHMARKS=%ENABLE_BENCHMARKS% ^
    -DBUILD_EXAMPLES=%ENABLE_EXAMPLES% ^
    -DDLP_THREADING_MODEL=%THREADING_MODEL% ^
    -DDLP_TESTING_LINK_STATIC=%STATIC_TESTS% ^
    -DDLP_BENCHMARKS_LINK_STATIC=%STATIC_BENCHMARKS% ^
    -DDLP_EXAMPLES_LINK_STATIC=%STATIC_EXAMPLES% ^
    -DDLP_ENABLE_LOGGING=%ENABLE_LOGGING% ^
    -DDLP_ENABLE_JIT_DEBUGGING=%ENABLE_JIT_DEBUG% ^
    -DBUILD_DOXYGEN=%ENABLE_DOXYGEN% ^
    -DBUILD_SPHINX=%ENABLE_SPHINX% ^
    %EXTRA_CMAKE_ARGS%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configuration failed.
    exit /b 1
)

echo.
echo Configuration successful. Run build.cmd to compile.
exit /b 0

:show_help
echo Usage: configure.cmd [options]
echo.
echo Options:
echo   --build-dir DIR          Build directory (default: build)
echo   --install-dir DIR        Install prefix (default: build\install)
echo   --build-type TYPE        Release ^| Debug ^| RelWithDebInfo (default: Release)
echo   --generator GEN          CMake generator (e.g. "Ninja", "Visual Studio 17 2022")
echo   --tests                  Enable test builds
echo   --benchmarks             Enable benchmark builds
echo   --examples               Enable example builds
echo   --all                    Enable tests + benchmarks + examples
echo   --openmp                 Enable OpenMP threading model
echo   --static-tests           Link tests with static library
echo   --static-benchmarks      Link benchmarks with static library
echo   --static-examples        Link examples with static library
echo   --logging                Enable DLP logging
echo   --jit-debug              Enable JIT debugging
echo   --docs                   Enable Doxygen + Sphinx documentation
echo   --clean                  Remove build directory before configuring
echo   --help                   Show this help message
echo.
echo Any unrecognized options are passed directly to CMake.
exit /b 0
