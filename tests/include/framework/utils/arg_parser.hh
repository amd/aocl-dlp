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
#include <filesystem>
#include <iostream>
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
     * @brief Get all non-flag arguments (positional arguments)
     * @return Vector of positional arguments
     */
    std::vector<std::string> getPositionalArgs() const
    {
        std::vector<std::string> positional;

        for (iter_t i = 1; i < argc_; ++i) {
            std::string arg = argv_[i];

            // Skip flags and their values
            if (!arg.empty() && arg[0] == '-') {
                // If this flag takes a value, skip the next argument too
                if ((arg == "-f" || arg == "--file") && i + 1 < argc_) {
                    ++i; // Skip the value
                }
                continue;
            }

            // Check if this is a value for a previous flag
            bool is_flag_value = false;
            if (i > 1) {
                std::string prev_arg = argv_[i - 1];
                if (prev_arg == "-f" || prev_arg == "--file") {
                    is_flag_value = true;
                }
            }

            if (!is_flag_value) {
                positional.push_back(arg);
            }
        }

        return positional;
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
        std::cout
            << "  -f, --file <path>       Specify YAML configuration file\n";
        std::cout << "  -n <count>              Number of benchmark iterations "
                     "(overrides MinTime)\n";
        std::cout << "  --ual-test <type>       UAL implementation to test "
                     "(DLP|REF|MKL|ONEDNN)\n";
        std::cout << "  --ual-ref <type>        UAL reference implementation "
                     "(DLP|REF|MKL|ONEDNN)\n";
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

            // Handle YAML file specification
            if ((arg == "-f" || arg == "--file") && i + 1 < argc_) {
                yaml_file_ = argv_[i + 1];
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
        }
    }

    int         argc_;
    char**      argv_;
    std::string yaml_file_;
    std::string ual_test_;
    std::string ual_ref_;
    int64_t     iterations_ = -1; // -1 means use default MinTime behavior
};

} // namespace dlp::testing::utils
