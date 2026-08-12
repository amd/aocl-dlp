/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#pragma once
#include "classic/dlp_base_types.h"

#include "framework/types.hh"
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Forward declaration for UALType (VerbosityLevel is now fully defined via
// types.hh)
namespace dlp::testing::framework {
enum class UALType : uint32_t;
}

namespace dlp::testing::utils {

/**
 * @brief Simple argument parser for test utilities
 *
 * This utility provides a clean interface for parsing command line
 * arguments commonly used in test executables. It integrates well with
 * GoogleTest and can be easily extended for additional argument types.
 *
 * Example usage:
 * @code
 * int main(int argc, char** argv) {
 *     ArgParser parser(argc, argv);
 *
 *     std::string yaml_file = parser.getYamlFile("default_config.yaml");
 *     bool verbose = parser.hasFlag("--verbose");
 *
 *     // ... rest of test code
 * }
 * @endcode
 */
class ArgParser
{
  public:
    /**
     * @brief Construct argument parser with command line arguments
     * @param argc Argument count from main()
     * @param argv Argument values from main()
     */
    ArgParser(int argc, char** argv)
        : argc_(argc)
        , argv_(argv)
    {
        parseArguments();
    }

    /**
     * @brief Get YAML configuration file path
     * @param default_file Default file path if not specified via command
     * line
     * @return Path to YAML configuration file
     *
     * Supports both -f and --file flags:
     * - ./test_gemm -f config.yaml
     * - ./test_gemm --file config.yaml
     *
     * If the file path is relative, it's resolved relative to current
     * working directory. If the file doesn't exist, a warning is printed
     * but execution continues.
     */
    std::string getYamlFile(const std::string& default_file = "") const
    {
        if (!yaml_file_.empty()) {
            // Validate file exists
            if (!std::filesystem::exists(yaml_file_)) {
                std::cerr << "Warning: Specified YAML file '" << yaml_file_
                          << "' does not exist. Using default if available."
                          << std::endl;
                if (!default_file.empty()
                    && std::filesystem::exists(default_file)) {
                    return default_file;
                }
            }
            return yaml_file_;
        }

        return default_file;
    }

    /**
     * @brief Get all YAML configuration file paths
     * @param default_file Default file path if none specified via command line
     * @return Vector of YAML configuration file paths
     *
     * Supports specifying multiple YAML files in a single command using the
     * -f / --file flag. Two forms are accepted and can be combined:
     * - Repeated flags:      ./test_gemm -f a.yaml -f b.yaml -f c.yaml
     * - Comma-separated:     ./test_gemm -f a.yaml,b.yaml,c.yaml
     *
     * Non-existent files are reported with a warning and skipped so a single
     * bad path does not block the rest of the run. If no valid files are
     * specified (or none exist), the provided default_file is returned as the
     * sole entry (when non-empty).
     */
    std::vector<std::string> getYamlFiles(
        const std::string& default_file = "") const
    {
        std::vector<std::string> files;

        for (const auto& file : yaml_files_) {
            if (file.empty()) {
                continue;
            }
            if (!std::filesystem::exists(file)) {
                std::cerr << "Warning: Specified YAML file '" << file
                          << "' does not exist. Skipping." << std::endl;
                continue;
            }
            files.push_back(file);
        }

        // Only fall back to the built-in default when the user did NOT pass
        // any -f/--file flag. If the user explicitly specified file(s) but all
        // of them were invalid, we deliberately return an empty list so the
        // caller can error out instead of silently running the default suite.
        if (files.empty() && yaml_files_.empty() && !default_file.empty()) {
            if (std::filesystem::exists(default_file)) {
                files.push_back(default_file);
            } else {
                std::cerr << "Warning: Default YAML file '" << default_file
                          << "' does not exist. No YAML files will be used."
                          << std::endl;
            }
        }

        return files;
    }

    /**
     * @brief Check whether the user explicitly passed a -f/--file flag
     * @return true if at least one -f/--file value was provided on the command
     *         line, false otherwise
     *
     * Useful to distinguish "no config specified (use default)" from "config
     * specified but invalid (should be an error)".
     */
    bool hasYamlFileArg() const { return !yaml_files_.empty(); }

    /**
     * @brief Check if a flag is present in command line arguments
     * @param flag Flag to check (e.g., "--verbose", "-v")
     * @return true if flag is present, false otherwise
     */
    bool hasFlag(const std::string& flag) const
    {
        for (iter_t i = 1; i < argc_; ++i) {
            if (argv_[i] == flag) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Get value for a specific flag
     * @param flag Flag name (e.g., "--output", "-o")
     * @param default_value Default value if flag not found
     * @return Value following the flag, or default_value if not found
     */
    std::string getFlagValue(const std::string& flag,
                             const std::string& default_value = "") const
    {
        for (iter_t i = 1; i < argc_ - 1; ++i) {
            if (argv_[i] == flag) {
                return std::string(argv_[i + 1]);
            }
        }
        return default_value;
    }

    /**
     * @brief Print usage information
     * @param program_name Name of the program (usually argv[0])
     */
    void printUsage(const std::string& program_name = "test_program") const
    {
        std::cout << "DLPTestSuite Help\n";
        std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -f, --file <path>       Specify YAML configuration "
                     "file. May be given\n";
        std::cout << "                          multiple times, and/or as a "
                     "comma-separated\n";
        std::cout << "                          list, to run multiple YAML "
                     "files in one run\n";
        std::cout << "  -n <count>              Number of benchmark iterations "
                     "(overrides MinTime)\n";
        std::cout << "  --benchmark_min_time=<N>[s]\n";
        std::cout << "                          Bench-only: MinTime in seconds "
                     "(overrides\n";
        std::cout << "                          BENCH_MIN_TIME and the 3s "
                     "default)\n";
        std::cout << "  --ual-test <type>       UAL implementation to test "
                     "(DLP|REF|MKL|ONEDNN)\n";
        std::cout << "  --ual-ref <type>        UAL reference implementation "
                     "(DLP|REF|MKL|ONEDNN)\n";
        std::cout << "  --cold                  Bench-only: flush caches "
                     "between iterations\n";
        std::cout << "  --cold-passes <N>       Bench-only: number of flush "
                     "passes per iter (default 1)\n";
        std::cout << "  -h, --help              Show this help message\n";
        std::cout << "  -v, --verbose           Enable verbose/detailed debug "
                     "output\n";
        std::cout << "  -vv                     Print partial matrices (5x5 "
                     "elements)\n";
        std::cout
            << "  -vvv                    Print full matrices (up to 20x20)\n";
        std::cout << "\nVerbosity Levels:\n";
        std::cout << "  Level 0 (default):     No extra output\n";
        std::cout << "  Level 1 (-v):          Verbose comparison results\n";
        std::cout << "  Level 2 (-vv):         + Print partial matrices\n";
        std::cout << "  Level 3 (-vvv):        + Print full matrices\n";
        std::cout << "\nBenchmark Iteration Control:\n";
        std::cout << "  By default, benchmarks run for MinTime(3.0) seconds.\n";
        std::cout
            << "  Set BENCH_MIN_TIME=<N> for an environment fallback, or\n";
        std::cout << "  --benchmark_min_time=<N>[s] for a CLI override.\n";
        std::cout << "  Use -n to specify exact iteration count instead.\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << program_name << " -f my_config.yaml\n";
        std::cout << "  " << program_name << " -f my_config.yaml -n 1000\n";
        std::cout << "  " << program_name << " --ual-test DLP --ual-ref REF\n";
        std::cout
            << "  " << program_name
            << " --file /path/to/config.yaml --ual-test DLP --ual-ref MKL\n";
    }

    /**
     * @brief Check if help was requested
     * @return true if help flags (-h, --help) are present
     */
    bool helpRequested() const { return hasFlag("-h") || hasFlag("--help"); }

    /**
     * @brief Check if verbose mode was requested
     * @return true if verbose flags (-v, --verbose) are present
     */
    bool isVerbose() const
    {
        return getVerbosityLevel()
               > dlp::testing::framework::VerbosityLevel::SILENT;
    }

    /**
     * @brief Get verbosity level based on command line flags
     * @return VerbosityLevel enum value (SILENT, BASIC, PARTIAL_MATRIX,
     * FULL_MATRIX)
     *
     * Supports multiple verbosity levels:
     * - SILENT: No verbosity (default)
     * - BASIC: -v or --verbose → Verbose comparison results
     * - PARTIAL_MATRIX: -vv → Print partial matrices (5x5 elements)
     * - FULL_MATRIX: -vvv → Print full matrices (up to 50x50)
     *
     * Examples:
     * - ./test_gemm -v     → BASIC
     * - ./test_gemm -vv    → PARTIAL_MATRIX
     * - ./test_gemm -vvv   → FULL_MATRIX
     * - ./test_gemm --verbose → BASIC
     */
    dlp::testing::framework::VerbosityLevel getVerbosityLevel() const
    {
        int level = 0;

        for (iter_t i = 1; i < argc_; ++i) {
            std::string arg = argv_[i];

            // Count consecutive v's in -vv, -vvv, etc.
            if (arg.length() >= 2 && arg[0] == '-' && arg[1] == 'v') {
                // Count v's after the first one
                size_t v_count = 0;
                for (std::size_t j = 1; j < arg.length() && arg[j] == 'v';
                     ++j) {
                    v_count++;
                }
                level = std::max(level, static_cast<int>(v_count));
            }
            // Handle --verbose as level 1
            else if (arg == "--verbose") {
                level = std::max(level, 1);
            }
        }

        // Convert int to enum, clamping at FULL_MATRIX
        if (level >= 3) {
            return dlp::testing::framework::VerbosityLevel::FULL_MATRIX;
        } else if (level == 2) {
            return dlp::testing::framework::VerbosityLevel::PARTIAL_MATRIX;
        } else if (level == 1) {
            return dlp::testing::framework::VerbosityLevel::BASIC;
        } else {
            return dlp::testing::framework::VerbosityLevel::SILENT;
        }
    }

    /**
     * @brief Get UAL implementation to test
     * @param default_val Default UAL type if not specified (default: "DLP")
     * @return UAL type string for implementation under test
     *
     * Supports both --ual-test and --ual_test flags:
     * - ./test_gemm --ual-test DLP
     * - ./test_gemm --ual_test DLP
     */
    std::string getUalTest(const std::string& default_val = "DLP") const
    {
        return ual_test_.empty() ? default_val : ual_test_;
    }

    /**
     * @brief Get UAL reference implementation
     * @param default_val Default UAL type if not specified (default: "REF")
     * @return UAL type string for reference implementation
     *
     * Supports both --ual-ref and --ual_ref flags:
     * - ./test_gemm --ual-ref REF
     * - ./test_gemm --ual_ref REF
     */
    std::string getUalRef(const std::string& default_val = "REF") const
    {
        return ual_ref_.empty() ? default_val : ual_ref_;
    }

    /**
     * @brief Get number of iterations for benchmarking
     * @return Number of iterations if specified, -1 otherwise (use default
     * MinTime)
     *
     * When the -n flag is provided, the benchmark will run exactly that many
     * iterations instead of using the default MinTime(3.0) approach.
     *
     * Examples:
     * - ./bench -f config.yaml -n 1000  → returns 1000
     * - ./bench -f config.yaml          → returns -1 (use default MinTime)
     */
    int64_t getIterations() const { return iterations_; }

    /**
     * @brief Get MinTime budget for Google Benchmark, in seconds.
     * @param default_value Returned when neither CLI flag nor env var set
     *                      a positive value. Defaults to 3.0s, matching the
     *                      existing hardcoded ->MinTime(3.0) behaviour.
     * @return Resolved MinTime in seconds.
     *
     * Resolution order: --benchmark_min_time=N[s] CLI flag > BENCH_MIN_TIME
     * env var > default_value.
     *
     * Google Benchmark's --benchmark_min_time CLI flag is silently ignored
     * when a benchmark is registered with ->MinTime(X), since the chain
     * call wins. We peek at argv during parseArguments() and feed the
     * resolved value into the chain call so users can override the time
     * budget without recompiling.
     *
     * Iteration mode (--benchmark_min_time=Nx) is intentionally not handled
     * here; pass -n N for explicit iteration counts instead.
     */
    double getBenchMinTime(double default_value = 3.0) const
    {
        if (bench_min_time_cli_ > 0.0)
            return bench_min_time_cli_;
        const char* env = std::getenv("BENCH_MIN_TIME");
        if (env && *env) {
            double v = std::atof(env);
            if (v > 0.0)
                return v;
        }
        return default_value;
    }

    /**
     * @brief Cold-cache mode requested via --cold flag.
     *
     * When set, bench fixtures evict all caches between iterations so each
     * timed kernel call starts cold. Bench-only feature; tests ignore.
     */
    bool getColdCache() const { return cold_cache_; }

    /**
     * @brief Number of flush passes per iteration (--cold-passes <N>).
     *
     * Defaults to 1. Higher values insure against retention-resistant
     * replacement policies; useful only in pathological cases.
     */
    int getColdPasses() const { return cold_passes_; }

    /**
     * @brief Parse UAL type string to UALType enum
     * @param type_str String representation of UAL type
     * @return UALType enum value
     * @throws std::invalid_argument if type_str is not recognized
     *
     * Supported values: "DLP", "REF", "MKL", "ONEDNN" (case-sensitive)
     */
    static dlp::testing::framework::UALType parseUALType(
        const std::string& type_str);

    // Static helper functions
    /**
     * @brief Helper function to integrate with GoogleTest
     *
     * This function processes custom arguments before GoogleTest processes its
     * own. Use this pattern in main() to cleanly separate test-specific
     * arguments from GoogleTest arguments.
     *
     * @param argc Reference to argument count (will be modified)
     * @param argv Reference to argument array (will be modified)
     * @return ArgParser instance with parsed custom arguments
     *
     * Example usage:
     * @code
     * int main(int argc, char** argv) {
     *     auto parser = dlp::testing::parseTestArgs(argc, argv);
     *
     *     // Now argc/argv only contain GoogleTest arguments
     *     ::testing::InitGoogleTest(&argc, argv);
     *
     *     std::string yaml_file = parser.getYamlFile("default.yaml");
     *     // ... use yaml_file in tests
     *
     *     return RUN_ALL_TESTS();
     * }
     * @endcode
     */
    static ArgParser parseTestArgs(int& argc, char**& argv)
    {
        // Create parser with original arguments
        ArgParser parser(argc, argv);

        // Create new argument list without our custom flags
        static std::vector<char*> new_argv;
        new_argv.clear();
        new_argv.push_back(argv[0]); // Keep program name

        // Copy only non-custom arguments
        for (iter_t i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            // Skip our custom flags and their values
            if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
                ++i; // Skip both flag and value
                continue;
            }

            // Skip UAL test flag and value
            if ((arg == "--ual-test" || arg == "--ual_test") && i + 1 < argc) {
                ++i; // Skip both flag and value
                continue;
            }

            // Skip UAL ref flag and value
            if ((arg == "--ual-ref" || arg == "--ual_ref") && i + 1 < argc) {
                ++i; // Skip both flag and value
                continue;
            }

            // Skip iteration count flag and value. If the value is missing,
            // consume the flag anyway so it is not forwarded as an
            // unrecognized argument to GoogleTest/benchmark.
            if (arg == "-n") {
                if (i + 1 < argc) {
                    ++i; // Skip both flag and value
                } else {
                    std::cerr << "Warning: '-n' requires a value and will be "
                                 "ignored. Usage: -n <iterations>"
                              << std::endl;
                }
                continue;
            }

            // Skip bench-only --cold (boolean flag, no value).
            if (arg == "--cold") {
                continue;
            }

            // Skip bench-only --cold-passes <N>. Consume the flag whether or
            // not its value is present so it is not forwarded as an
            // unrecognized argument to Google Benchmark. A following flag is
            // not a value and must remain available to Google Benchmark.
            if (arg == "--cold-passes") {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    ++i; // Skip both flag and value
                }
                continue;
            }

            // Skip verbose flags (-v, -vv, -vvv, --verbose)
            if (arg == "--verbose"
                || (arg.length() >= 2 && arg[0] == '-'
                    && arg.substr(1).find_first_not_of('v')
                           == std::string::npos)) {
                continue;
            }

            // Keep help flags for GoogleTest to handle naturally
            new_argv.push_back(argv[i]);
        }

        // Update argc and argv for GoogleTest
        argc = static_cast<int>(new_argv.size());
        argv = new_argv.data();

        return parser;
    }

  private:
    void parseArguments()
    {
        for (iter_t i = 1; i < argc_; ++i) {
            std::string arg = argv_[i];

            // Handle YAML file specification. Supports multiple files via
            // repeated -f/--file flags and/or comma-separated values.
            if ((arg == "-f" || arg == "--file") && i + 1 < argc_) {
                std::string value = argv_[i + 1];

                // Split on commas to support "-f a.yaml,b.yaml" and append
                // every entry to the multi-file list.
                std::stringstream ss(value);
                std::string       token;
                while (std::getline(ss, token, ',')) {
                    // Trim surrounding whitespace from each token.
                    size_t start = token.find_first_not_of(" \t");
                    size_t end   = token.find_last_not_of(" \t");
                    if (start != std::string::npos
                        && end != std::string::npos) {
                        const std::string trimmed =
                            token.substr(start, end - start + 1);
                        // First specified file is retained in yaml_file_ for
                        // backward compatibility with getYamlFile().
                        if (yaml_file_.empty()) {
                            yaml_file_ = trimmed;
                        }
                        yaml_files_.push_back(trimmed);
                    }
                }

                ++i; // Skip the next argument (the file path)
            }

            // Handle UAL test specification: --ual-test VALUE or --ual_test
            // VALUE
            else if ((arg == "--ual-test" || arg == "--ual_test")
                     && i + 1 < argc_) {
                ual_test_ = argv_[i + 1];
                ++i; // Skip the next argument (the value)
            }

            // Handle UAL ref specification: --ual-ref VALUE or --ual_ref VALUE
            else if ((arg == "--ual-ref" || arg == "--ual_ref")
                     && i + 1 < argc_) {
                ual_ref_ = argv_[i + 1];
                ++i; // Skip the next argument (the value)
            }

            // Handle iteration count specification: -n VALUE
            else if (arg == "-n" && i + 1 < argc_) {
                try {
                    iterations_ = std::stoll(argv_[i + 1]);
                    if (iterations_ <= 0) {
                        std::cerr
                            << "Warning: Invalid iteration count '"
                            << argv_[i + 1]
                            << "'. Must be a positive integer. Using default."
                            << std::endl;
                        iterations_ = -1;
                    }
                } catch (const std::exception&) {
                    std::cerr
                        << "Warning: Invalid iteration count '" << argv_[i + 1]
                        << "'. Must be an integer. Using default." << std::endl;
                    iterations_ = -1;
                }
                ++i; // Skip the next argument (the value)
            }

            // Cold-cache flag (bench-only; tests ignore the parsed value).
            else if (arg == "--cold") {
                cold_cache_ = true;
            }

            // --cold-passes <N>. Do not consume a following flag as the value;
            // it belongs to Google Benchmark or another parser.
            else if (arg == "--cold-passes") {
                if (i + 1 < argc_ && argv_[i + 1][0] != '-') {
                    try {
                        int n = std::stoi(argv_[i + 1]);
                        if (n > 0)
                            cold_passes_ = n;
                    } catch (const std::exception&) {
                        // Silently keep default; bad value is operator error.
                    }
                    ++i;
                } else {
                    std::cerr
                        << "Warning: '--cold-passes' requires a value and will "
                           "be ignored. Usage: --cold-passes <N>"
                        << std::endl;
                }
            }

            // Peek at Google Benchmark's --benchmark_min_time=N[s] flag so
            // benchmarks can feed the value into ->MinTime() (which would
            // otherwise dominate the CLI flag). Iteration mode (=Nx) is left
            // for gbench / -n to handle.
            else if (arg.rfind("--benchmark_min_time=", 0) == 0) {
                std::string val =
                    arg.substr(std::string("--benchmark_min_time=").length());
                if (!val.empty() && val.back() != 'x') {
                    if (val.back() == 's')
                        val.pop_back();
                    try {
                        double v = std::stod(val);
                        if (v > 0.0)
                            bench_min_time_cli_ = v;
                    } catch (const std::exception&) {
                        // Silent — gbench will produce its own diagnostic.
                    }
                }
            }
        }
    }

    int                      argc_;
    char**                   argv_;
    std::string              yaml_file_;
    std::vector<std::string> yaml_files_;
    std::string              ual_test_;
    std::string              ual_ref_;
    int64_t                  iterations_ = -1; // -1 means use default MinTime
                                               // behavior
    double bench_min_time_cli_ =
        0.0; // 0 means CLI flag not specified; fallback to env/default
    bool cold_cache_  = false; // --cold flag (bench-only, default off)
    int  cold_passes_ = 1;     // --cold-passes <N> (default 1)
};

} // namespace dlp::testing::utils
