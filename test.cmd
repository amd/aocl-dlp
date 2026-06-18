@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AOCL-DLP Windows Test Script
:: Runs all tests via CTest.
::
:: Usage:
::   test.cmd [options]
::
:: Options:
::   --build-dir DIR    Build directory (default: build)
::   --config TYPE      Build configuration (default: Release)
::   --filter REGEX     Run only tests matching this regex
::   --parallel N       Number of parallel test jobs (default: auto)
::   --verbose          Enable verbose test output
::   --output-on-failure Show output only for failed tests (default)
::   --repeat N         Repeat each test N times
::   --help             Show this help message
:: ============================================================================

set "BUILD_DIR=build"
set "CONFIG=Release"
set "FILTER="
set "PARALLEL="
set "VERBOSE="
set "REPEAT="
set "OUTPUT_ON_FAILURE=--output-on-failure"

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--build-dir"          ( set "BUILD_DIR=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--config"             ( set "CONFIG=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--filter"             ( set "FILTER=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--parallel"           ( set "PARALLEL=%~2"  & shift & shift & goto :parse_args )
if /i "%~1"=="--verbose"            ( set "VERBOSE=-VV"   & shift & goto :parse_args )
if /i "%~1"=="--output-on-failure"  ( set "OUTPUT_ON_FAILURE=--output-on-failure" & shift & goto :parse_args )
if /i "%~1"=="--repeat"             ( set "REPEAT=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--help"               ( goto :show_help )
echo WARNING: Unknown option: %~1
shift
goto :parse_args
:done_args

if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo ERROR: Build directory "%BUILD_DIR%" is not configured.
    echo Run: configure.cmd --tests
    exit /b 1
)

:: Check if tests were enabled
findstr /c:"BUILD_TESTING:BOOL=ON" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Tests were not enabled during configuration.
    echo Run: configure.cmd --tests
    exit /b 1
)

set "FILTER_ARG="
if not "%FILTER%"=="" set "FILTER_ARG=-R "%FILTER%""

set "PARALLEL_ARG="
if not "%PARALLEL%"=="" (
    set "PARALLEL_ARG=-j %PARALLEL%"
) else (
    set "PARALLEL_ARG=-j"
)

set "REPEAT_ARG="
if not "%REPEAT%"=="" set "REPEAT_ARG=--repeat until-pass:%REPEAT%"

echo.
echo ============================================================
echo  AOCL-DLP Test
echo ============================================================
echo  Build directory:  %BUILD_DIR%
echo  Configuration:    %CONFIG%
if not "%FILTER%"=="" echo  Filter:           %FILTER%
echo ============================================================
echo.

:: Ensure the shared library is on PATH for test executables
set "PATH=%BUILD_DIR%\Release;%BUILD_DIR%\%CONFIG%;%BUILD_DIR%;%PATH%"

ctest --test-dir "%BUILD_DIR%" -C %CONFIG% %PARALLEL_ARG% %OUTPUT_ON_FAILURE% %VERBOSE% %FILTER_ARG% %REPEAT_ARG%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Some tests failed.
    exit /b 1
)

echo.
echo All tests passed.
exit /b 0

:show_help
echo Usage: test.cmd [options]
echo.
echo Options:
echo   --build-dir DIR    Build directory (default: build)
echo   --config TYPE      Build configuration (default: Release)
echo   --filter REGEX     Run only tests matching this regex
echo   --parallel N       Number of parallel test jobs (default: auto)
echo   --verbose          Enable verbose test output
echo   --repeat N         Repeat each test N times
echo   --help             Show this help message
exit /b 0
