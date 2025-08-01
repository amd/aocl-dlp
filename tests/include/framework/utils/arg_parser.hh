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

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
        for (int i = 1; i < argc_; ++i) {
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
        for (int i = 1; i < argc_ - 1; ++i) {
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

        for (int i = 1; i < argc_; ++i) {
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
        std::cout << "  -f, --file <path>    Specify YAML configuration file\n";
        std::cout << "  -h, --help           Show this help message\n";
        std::cout << "  --verbose            Enable verbose output\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << program_name << " -f my_config.yaml\n";
        std::cout << "  " << program_name
                  << " --file /path/to/config.yaml --verbose\n";
    }

    /**
     * @brief Check if help was requested
     * @return true if help flags (-h, --help) are present
     */
    bool helpRequested() const { return hasFlag("-h") || hasFlag("--help"); }

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
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            // Skip our custom flags and their values
            if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
                ++i; // Skip both flag and value
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
        for (int i = 1; i < argc_; ++i) {
            std::string arg = argv_[i];

            // Handle YAML file specification
            if ((arg == "-f" || arg == "--file") && i + 1 < argc_) {
                yaml_file_ = argv_[i + 1];
                ++i; // Skip the next argument (the file path)
            }
        }
    }

  private:
    int         argc_;
    char**      argv_;
    std::string yaml_file_;
};

} // namespace dlp::testing::utils
