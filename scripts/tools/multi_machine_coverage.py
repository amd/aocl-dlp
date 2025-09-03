#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (
# INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Multi-Machine Coverage Aggregator for AOCL-DLP

This script collects coverage data from multiple machines and aggregates them
into a unified HTML coverage report using lcov. It leverages the existing
project infrastructure and maintains consistency with the single-machine tools.
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import shutil
import tempfile
import hashlib
import socket
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging

# Add the parent directory to the path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from common.config import FILE_FORMATS, EXCLUDE_PATTERNS
    from common.utils import filter_file
except ImportError as e:
    print(f"Error: Could not import common modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("multi_machine_coverage")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color


def print_colored(message: str, color: str = Colors.NC) -> None:
    """Print a message with color."""
    print(f"{color}{message}{Colors.NC}")


def print_info(message: str) -> None:
    """Print an info message."""
    print_colored(f"[INFO] {message}", Colors.BLUE)


def print_success(message: str) -> None:
    """Print a success message."""
    print_colored(f"[SUCCESS] {message}", Colors.GREEN)


def print_warning(message: str) -> None:
    """Print a warning message."""
    print_colored(f"[WARNING] {message}", Colors.YELLOW)


def print_error(message: str) -> None:
    """Print an error message."""
    print_colored(f"[ERROR] {message}", Colors.RED)


@dataclass
class CoverageSource:
    """Configuration for a coverage data source."""
    name: str
    type: str  # "remote" or "local"
    build_path: str
    host: Optional[str] = None
    username: Optional[str] = None
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    ssh_proxy_jump: Optional[str] = None
    ssh_extra_opts: List[str] = field(default_factory=list)
    compression: bool = True
    archive_mode: bool = True
    rsync_extra_opts: List[str] = field(default_factory=list)
    validate_connection: bool = True

    def __post_init__(self):
        """Validate source configuration."""
        if self.type == "remote" and (not self.host or not self.username):
            raise ValueError(
                f"Remote source '{self.name}' must have host and username")

        if self.type not in ["remote", "local"]:
            raise ValueError(
                f"Source type must be 'remote' or 'local', got '{self.type}'")


@dataclass
class AggregationConfig:
    """Configuration for coverage aggregation."""
    temp_dir: str = "/tmp/aocl_coverage_aggregate"
    output_dir: str = "coverage_reports_aggregated"
    cleanup_temp: bool = True
    rsync_timeout: int = 300
    parallel_limit: int = 3
    ssh_timeout: int = 30
    merge_strategy: str = "add"
    normalize_paths: bool = True
    extra_exclusions: List[str] = field(default_factory=list)
    coverage_threshold: float = 80.0
    preserve_individual_reports: bool = False
    retry_failed: bool = True
    max_retries: int = 3
    retry_delay: int = 10
    verify_checksums: bool = True
    log_level: str = "INFO"
    working_directory: Optional[str] = None
    max_download_size: int = 0  # 0 = unlimited


class MultiMachineCoverageAggregator:
    """Main class for aggregating coverage data from multiple machines."""

    def __init__(self, config_file: Path, project_root: Path):
        """Initialize the aggregator."""
        self.config_file = config_file
        self.project_root = project_root
        self.config = self._load_config()
        self.logger = setup_logging(self.config.log_level)
        self.sources: List[CoverageSource] = []
        self.temp_dir: Optional[Path] = None
        self.collected_data: Dict[str, Path] = {}

        # Statistics
        self.stats = {
            'sources_processed': 0,
            'sources_failed': 0,
            'total_size_downloaded': 0,
            'processing_time': 0
        }

        self._parse_sources()

    def _load_config(self) -> AggregationConfig:
        """Load and validate configuration from YAML file."""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Extract global configuration
        global_config = config_data.get('global', {})
        aggregation_config = config_data.get('aggregation', {})
        advanced_config = config_data.get('advanced', {})

        # Merge configurations
        merged_config = {**global_config, **
                         aggregation_config, **advanced_config}

        return AggregationConfig(**merged_config)

    def _parse_sources(self) -> None:
        """Parse coverage sources from configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

        sources_config = config_data.get('sources', [])

        for source_data in sources_config:
            try:
                source = CoverageSource(**source_data)
                self.sources.append(source)
                self.logger.info(
                    f"Added source: {source.name} ({source.type})")
            except (TypeError, ValueError) as e:
                self.logger.error(f"Invalid source configuration: {e}")
                raise

    def _setup_temp_directory(self) -> None:
        """Setup temporary directory for coverage collection."""
        self.temp_dir = Path(self.config.temp_dir)

        if self.temp_dir.exists():
            if self.config.cleanup_temp:
                print_info(
                    f"Cleaning existing temp directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
            else:
                print_warning(
                    f"Reusing existing temp directory: {self.temp_dir}")

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Created temp directory: {self.temp_dir}")

    def _check_dependencies(self) -> None:
        """Check required dependencies."""
        required_tools = ['rsync', 'lcov', 'genhtml']

        missing_tools = []
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            raise RuntimeError(
                f"Missing required tools: {', '.join(missing_tools)}")

        print_info("All required dependencies found")

    def _validate_connection(self, source: CoverageSource) -> bool:
        """Validate connection to a remote source."""
        if source.type != "remote":
            return True

        if not source.validate_connection:
            return True

        try:
            # Check if host is reachable
            print_info(f"Validating connection to {source.host}...")

            # Try SSH connection test
            ssh_cmd = self._build_ssh_command(
                source, ["echo", "connection_test"])
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.ssh_timeout
            )

            if result.returncode == 0:
                self.logger.info(
                    f"Connection to {source.name} validated successfully")
                return True
            else:
                self.logger.error(
                    f"SSH connection failed for {source.name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error(f"Connection timeout for {source.name}")
            return False
        except Exception as e:
            self.logger.error(
                f"Connection validation failed for {source.name}: {e}")
            return False

    def _build_ssh_command(self, source: CoverageSource, remote_cmd: List[str]) -> List[str]:
        """Build SSH command for remote execution."""
        ssh_cmd = ["ssh"]

        # Add SSH options
        ssh_cmd.extend(["-o", f"ConnectTimeout={self.config.ssh_timeout}"])
        ssh_cmd.extend(["-o", "BatchMode=yes"])  # Non-interactive

        if source.ssh_key:
            ssh_cmd.extend(["-i", os.path.expanduser(source.ssh_key)])

        if source.ssh_port != 22:
            ssh_cmd.extend(["-p", str(source.ssh_port)])

        if source.ssh_proxy_jump:
            ssh_cmd.extend(["-J", source.ssh_proxy_jump])

        # Add extra SSH options
        ssh_cmd.extend(source.ssh_extra_opts)

        # Add host and command
        ssh_cmd.append(f"{source.username}@{source.host}")
        ssh_cmd.extend(remote_cmd)

        return ssh_cmd

    def _build_rsync_command(self, source: CoverageSource, local_dest: Path) -> List[str]:
        """Build rsync command for data transfer."""
        rsync_cmd = ["rsync"]

        # Basic options
        if source.archive_mode:
            rsync_cmd.append("-a")  # Archive mode
        else:
            rsync_cmd.append("-r")  # Recursive only

        if source.compression:
            rsync_cmd.append("-z")  # Compression

        # Verbose and progress
        rsync_cmd.extend(["-v", "--progress"])

        # Timeout
        rsync_cmd.extend(["--timeout", str(self.config.rsync_timeout)])

        # SSH options for remote sources
        if source.type == "remote":
            ssh_opts = f"ssh -o ConnectTimeout={self.config.ssh_timeout} -o BatchMode=yes"

            if source.ssh_key:
                ssh_opts += f" -i {os.path.expanduser(source.ssh_key)}"

            if source.ssh_port != 22:
                ssh_opts += f" -p {source.ssh_port}"

            if source.ssh_proxy_jump:
                ssh_opts += f" -J {source.ssh_proxy_jump}"

            # Add extra SSH options
            for opt in source.ssh_extra_opts:
                ssh_opts += f" {opt}"

            rsync_cmd.extend(["-e", ssh_opts])

        # Add extra rsync options
        rsync_cmd.extend(source.rsync_extra_opts)

        # Source and destination
        if source.type == "remote":
            rsync_src = f"{source.username}@{source.host}:{source.build_path}/"
        else:
            rsync_src = f"{source.build_path}/"

        rsync_cmd.extend([rsync_src, str(local_dest)])

        return rsync_cmd

    def _collect_from_source(self, source: CoverageSource) -> Tuple[bool, Optional[Path]]:
        """Collect coverage data from a single source."""
        source_dir = self.temp_dir / source.name
        source_dir.mkdir(exist_ok=True)

        self.logger.info(f"Collecting coverage data from {source.name}...")

        try:
            # Validate connection first
            if not self._validate_connection(source):
                return False, None

            # Check if source has coverage data
            if source.type == "remote":
                # Check remote directory for coverage files
                check_cmd = self._build_ssh_command(
                    source,
                    ["find", source.build_path, "-name", "*.gcno",
                        "-o", "-name", "*.gcda", "|", "head", "-1"]
                )
                result = subprocess.run(
                    check_cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0 or not result.stdout.strip():
                    self.logger.warning(
                        f"No coverage data found in {source.name}:{source.build_path}")
                    return False, None
            else:
                # Check local directory
                build_path = Path(source.build_path)
                if not build_path.exists():
                    self.logger.error(
                        f"Local build path does not exist: {build_path}")
                    return False, None

                # Check for coverage files
                coverage_files = list(build_path.rglob(
                    "*.gcno")) + list(build_path.rglob("*.gcda"))
                if not coverage_files:
                    self.logger.warning(
                        f"No coverage data found in {build_path}")
                    return False, None

            # Perform the data transfer
            rsync_cmd = self._build_rsync_command(source, source_dir)

            self.logger.debug(f"Running rsync command: {' '.join(rsync_cmd)}")

            start_time = time.time()
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.rsync_timeout
            )
            transfer_time = time.time() - start_time

            if result.returncode == 0:
                # Calculate transferred size
                transferred_size = self._calculate_directory_size(source_dir)
                self.stats['total_size_downloaded'] += transferred_size

                print_success(f"Successfully collected from {source.name} "
                              f"({transferred_size / (1024*1024):.1f} MB in {transfer_time:.1f}s)")

                # Verify coverage data was actually transferred
                coverage_files = list(source_dir.rglob(
                    "*.gcno")) + list(source_dir.rglob("*.gcda"))
                if not coverage_files:
                    self.logger.warning(
                        f"No coverage files found after transfer from {source.name}")
                    return False, None

                return True, source_dir
            else:
                self.logger.error(
                    f"rsync failed for {source.name}: {result.stderr}")
                return False, None

        except subprocess.TimeoutExpired:
            self.logger.error(f"Transfer timeout for {source.name}")
            return False, None
        except Exception as e:
            self.logger.error(f"Failed to collect from {source.name}: {e}")
            return False, None

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except OSError:
            pass
        return total_size

    def _collect_coverage_data(self) -> None:
        """Collect coverage data from all sources in parallel."""
        print_info(
            f"Collecting coverage data from {len(self.sources)} sources...")

        successful_collections = {}
        failed_sources = []

        with ThreadPoolExecutor(max_workers=self.config.parallel_limit) as executor:
            # Submit all collection tasks
            future_to_source = {
                executor.submit(self._collect_from_source, source): source
                for source in self.sources
            }

            # Process completed tasks
            for future in as_completed(future_to_source):
                source = future_to_source[future]

                try:
                    success, data_path = future.result()

                    if success and data_path:
                        successful_collections[source.name] = data_path
                        self.stats['sources_processed'] += 1
                    else:
                        failed_sources.append(source.name)
                        self.stats['sources_failed'] += 1

                        if self.config.retry_failed:
                            # Retry logic
                            for retry in range(self.config.max_retries):
                                print_warning(
                                    f"Retrying {source.name} (attempt {retry + 1}/{self.config.max_retries})")
                                time.sleep(self.config.retry_delay)

                                success, data_path = self._collect_from_source(
                                    source)
                                if success and data_path:
                                    successful_collections[source.name] = data_path
                                    self.stats['sources_processed'] += 1
                                    failed_sources.remove(source.name)
                                    self.stats['sources_failed'] -= 1
                                    break

                except Exception as e:
                    self.logger.error(
                        f"Exception during collection from {source.name}: {e}")
                    failed_sources.append(source.name)
                    self.stats['sources_failed'] += 1

        self.collected_data = successful_collections

        if failed_sources:
            print_warning(
                f"Failed to collect from: {', '.join(failed_sources)}")

        if not successful_collections:
            raise RuntimeError(
                "No coverage data could be collected from any source")

        print_success(
            f"Successfully collected from {len(successful_collections)} sources")

    def _generate_individual_reports(self) -> None:
        """Generate individual coverage reports for each source."""
        if not self.config.preserve_individual_reports:
            return

        print_info("Generating individual coverage reports...")

        for source_name, data_path in self.collected_data.items():
            try:
                individual_output = Path(
                    self.config.output_dir) / f"individual_{source_name}"
                individual_output.mkdir(parents=True, exist_ok=True)

                # Generate lcov info file for this source
                lcov_cmd = [
                    "lcov",
                    "--ignore-errors", "inconsistent",
                    "--ignore-errors", "corrupt",
                    "--capture",
                    "--directory", str(data_path),
                    "--output-file", str(individual_output / "coverage.info"),
                    "--base-directory", str(self.project_root),
                    "--no-external"
                ]

                result = subprocess.run(
                    lcov_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Generate HTML report
                    genhtml_cmd = [
                        "genhtml", str(individual_output / "coverage.info"),
                        "--ignore-errors", "inconsistent",
                        "--ignore-errors", "corrupt",
                        "--ignore-errors", "deprecated",
                        "--output-directory", str(individual_output / "html"),
                        "--title", f"AOCL-DLP Coverage Report - {source_name}",
                        "--show-details", "--branch-coverage", "--function-coverage"
                    ]

                    subprocess.run(genhtml_cmd, capture_output=True)
                    print_info(
                        f"Individual report generated for {source_name}")

            except Exception as e:
                self.logger.warning(
                    f"Failed to generate individual report for {source_name}: {e}")

    def _aggregate_coverage_data(self) -> Path:
        """Aggregate coverage data from all sources using lcov."""
        print_info("Aggregating coverage data...")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual lcov info files
        info_files = []

        for source_name, data_path in self.collected_data.items():
            info_file = self.temp_dir / f"{source_name}_coverage.info"

            # Generate lcov info file for this source
            lcov_cmd = [
                "lcov",
                "--ignore-errors", "inconsistent",
                "--ignore-errors", "corrupt",
                "--capture",
                "--directory", str(data_path),
                "--output-file", str(info_file),
                "--base-directory", str(self.project_root),
                "--no-external",
                "--quiet"
            ]

            result = subprocess.run(lcov_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                info_files.append(info_file)
                self.logger.info(f"Generated lcov info for {source_name}")
            else:
                self.logger.warning(
                    f"Failed to generate lcov info for {source_name}: {result.stderr}")

        if not info_files:
            raise RuntimeError("Failed to generate any lcov info files")

        # Merge all info files
        merged_info = output_dir / "merged_coverage.info"

        if len(info_files) == 1:
            # Only one file, just copy it
            shutil.copy2(info_files[0], merged_info)
        else:
            # Merge multiple files
            merge_cmd = [
                "lcov",
                "--ignore-errors", "inconsistent",
                "--ignore-errors", "corrupt"
            ]

            for info_file in info_files:
                merge_cmd.extend(["--add-tracefile", str(info_file)])

            merge_cmd.extend(["--output-file", str(merged_info)])

            result = subprocess.run(merge_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to merge coverage data: {result.stderr}")

        # Apply exclusion patterns
        filtered_info = output_dir / "coverage.info"

        # Build exclusion patterns - using both relative and absolute patterns
        exclusion_patterns = [
            # Build directories (all variations)
            "*/build/*", "*build/*", "build/*", "/*/build/*", "*/*/build/*",
            # Test directories (all variations)
            "*/tests/*", "*/test/*", "*tests/*", "*test/*", "tests/*", "test/*",
            "/*/tests/*", "/*/test/*", "*/*/tests/*", "*/*/test/*",
            # Dependency directories (all variations)
            "*/_deps/*", "*deps/*", "_deps/*", "deps/*",
            "/*/build/_deps/*", "*/build/_deps/*", "**/build/_deps/*",
            "*/build/*/_deps/*", "**/build/*/_deps/*",
            # Google Test (comprehensive patterns)
            "*/gtest/*", "*/googletest/*", "*gtest*", "*googletest*",
            "*/build/_deps/gtest*", "**/gtest*", "**/googletest*",
            "*/build/_deps/googletest*", "*/build/**/gtest*", "*/build/**/googletest*",
            # YAML-CPP (comprehensive patterns)
            "*/yaml-cpp/*", "*yaml-cpp*", "*/build/_deps/yaml-cpp*",
            "**/yaml-cpp*", "*/build/**/yaml-cpp*",
            # CMake files (all variations)
            "*/CMakeFiles/*", "*CMakeFiles*", "CMakeFiles/*",
            "/*/CMakeFiles/*", "*/*/CMakeFiles/*", "**/CMakeFiles/*",
            "*/.cmake/*", "*.cmake/*", ".cmake/*",
            # Other exclusions
            "*/bench/*", "*/docs/*", "*/scripts/*", "*/examples/*",
            "*/external/*", "*/third_party/*", "*/xbyak/*",
            # System includes
            "/usr/include/*", "/usr/local/include/*"
        ]
        exclusion_patterns.extend(self.config.extra_exclusions)

        # Apply exclusions
        self.logger.info(
            f"Applying {len(exclusion_patterns)} exclusion patterns...")
        self.logger.debug(f"Exclusion patterns: {exclusion_patterns}")

        # First, let's see what files are actually in the merged coverage data
        list_cmd = ["lcov", "--list", str(merged_info)]
        list_result = subprocess.run(list_cmd, capture_output=True, text=True)
        if list_result.returncode == 0 and self.logger.level <= 10:  # DEBUG level
            self.logger.debug("Files in merged coverage before exclusion:")
            # Show first 20 lines
            for line in list_result.stdout.split('\n')[:20]:
                if line.strip():
                    self.logger.debug(f"  {line}")
            if len(list_result.stdout.split('\n')) > 20:
                self.logger.debug("  ... (truncated)")

        exclude_cmd = [
            "lcov",
            "--ignore-errors", "inconsistent",
            "--ignore-errors", "corrupt",
            "--ignore-errors", "unused",
            "--remove", str(merged_info)
        ]
        exclude_cmd.extend(exclusion_patterns)
        exclude_cmd.extend(["--output-file", str(filtered_info)])

        self.logger.debug(f"Exclusion command: {' '.join(exclude_cmd)}")
        result = subprocess.run(exclude_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.warning(f"Failed to apply exclusions: {result.stderr}")
            # Use unfiltered version if exclusion fails
            shutil.copy2(merged_info, filtered_info)
        else:
            self.logger.info("Exclusion patterns applied successfully")
            if result.stdout.strip():
                self.logger.debug(f"Exclusion output: {result.stdout}")

            # Show what files remain after exclusion (in debug mode)
            if self.logger.level <= 10:  # DEBUG level
                list_filtered_cmd = ["lcov", "--list", str(filtered_info)]
                list_filtered_result = subprocess.run(
                    list_filtered_cmd, capture_output=True, text=True)
                if list_filtered_result.returncode == 0:
                    self.logger.debug("Files remaining after exclusion:")
                    # Show first 20 lines
                    for line in list_filtered_result.stdout.split('\n')[:20]:
                        if line.strip():
                            self.logger.debug(f"  {line}")
                    if len(list_filtered_result.stdout.split('\n')) > 20:
                        self.logger.debug("  ... (truncated)")

        print_success("Coverage data aggregated successfully")
        return filtered_info

    def _generate_html_report(self, lcov_info: Path) -> None:
        """Generate HTML coverage report."""
        print_info("Generating HTML coverage report...")

        output_dir = Path(self.config.output_dir)
        html_dir = output_dir / "html"

        # Generate HTML report
        genhtml_cmd = [
            "genhtml", str(lcov_info),
            "--ignore-errors", "inconsistent",
            "--ignore-errors", "corrupt",
            "--ignore-errors", "deprecated",
            "--output-directory", str(html_dir),
            "--title", "AOCL-DLP Multi-Machine Coverage Report",
            "--show-details",
            "--legend",
            "--branch-coverage",
            "--function-coverage",
            "--sort",
            "--num-spaces", "4"
        ]

        result = subprocess.run(genhtml_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate HTML report: {result.stderr}")

        print_success(f"HTML report generated: {html_dir / 'index.html'}")

    def _generate_summary_report(self, lcov_info: Path) -> None:
        """Generate summary report with statistics."""
        print_info("Generating summary report...")

        output_dir = Path(self.config.output_dir)
        summary_file = output_dir / "summary.txt"

        # Get lcov summary
        lcov_summary_cmd = [
            "lcov",
            "--ignore-errors", "inconsistent",
            "--ignore-errors", "corrupt",
            "--summary", str(lcov_info)
        ]
        result = subprocess.run(
            lcov_summary_cmd, capture_output=True, text=True)

        # Create summary content
        summary_content = [
            "AOCL-DLP Multi-Machine Coverage Summary",
            "=" * 50,
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {self.config_file}",
            f"Project root: {self.project_root}",
            "",
            "Sources Processed:",
            "-" * 20
        ]

        for source_name in self.collected_data.keys():
            summary_content.append(f"  ✓ {source_name}")

        if self.stats['sources_failed'] > 0:
            summary_content.extend([
                "",
                "Failed Sources:",
                "-" * 15,
                f"  {self.stats['sources_failed']} sources failed"
            ])

        summary_content.extend([
            "",
            "Transfer Statistics:",
            "-" * 20,
            f"  Total data transferred: {self.stats['total_size_downloaded'] / (1024*1024):.1f} MB",
            f"  Sources processed: {self.stats['sources_processed']}",
            f"  Processing time: {self.stats['processing_time']:.1f} seconds",
            "",
            "Coverage Statistics:",
            "-" * 20
        ])

        if result.returncode == 0:
            summary_content.append(result.stdout)
        else:
            summary_content.append("Failed to generate coverage statistics")

        # Write summary file
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_content))

        print_success(f"Summary report generated: {summary_file}")

        # Display key metrics
        if result.returncode == 0:
            print("\nCoverage Summary:")
            for line in result.stdout.split('\n'):
                if 'lines' in line.lower() or 'functions' in line.lower() or 'branches' in line.lower():
                    print(f"  {line.strip()}")

    def _cleanup_temp_directory(self) -> None:
        """Clean up temporary directory."""
        if self.config.cleanup_temp and self.temp_dir and self.temp_dir.exists():
            print_info("Cleaning up temporary directory...")
            shutil.rmtree(self.temp_dir)
            print_info("Temporary directory cleaned")

    def run(self) -> None:
        """Run the complete multi-machine coverage aggregation process."""
        start_time = time.time()

        try:
            print_colored(
                "AOCL-DLP Multi-Machine Coverage Aggregator", Colors.WHITE)
            print_colored("=" * 50, Colors.WHITE)

            # Setup and validation
            self._check_dependencies()
            self._setup_temp_directory()

            # Collect coverage data
            self._collect_coverage_data()

            # Generate individual reports if requested
            self._generate_individual_reports()

            # Aggregate and generate reports
            lcov_info = self._aggregate_coverage_data()
            self._generate_html_report(lcov_info)
            self._generate_summary_report(lcov_info)

            self.stats['processing_time'] = time.time() - start_time

            print_success(
                "Multi-machine coverage aggregation completed successfully!")
            print_info(
                f"Results available in: {Path(self.config.output_dir).absolute()}")

        except Exception as e:
            print_error(f"Coverage aggregation failed: {e}")
            self.logger.exception("Full error details:")
            raise
        finally:
            self._cleanup_temp_directory()


def create_example_config(output_file: Path) -> None:
    """Create an example configuration file."""
    example_config = {
        'global': {
            'temp_dir': '/tmp/aocl_coverage_aggregate',
            'output_dir': 'coverage_reports_aggregated',
            'cleanup_temp': True,
            'rsync_timeout': 300,
            'parallel_limit': 3,
            'ssh_timeout': 30
        },
        'sources': [
            {
                'name': 'build-server-1',
                'type': 'remote',
                'host': '192.168.1.100',
                'username': 'builduser',
                'build_path': '/path/to/aocl-dlp/build',
                'compression': True,
                'archive_mode': True,
                'validate_connection': True
            },
            {
                'name': 'local-build',
                'type': 'local',
                'build_path': './build',
                'archive_mode': True
            }
        ],
        'aggregation': {
            'merge_strategy': 'add',
            'normalize_paths': True,
            'coverage_threshold': 80.0,
            'preserve_individual_reports': False
        },
        'advanced': {
            'retry_failed': True,
            'max_retries': 3,
            'retry_delay': 10,
            'log_level': 'INFO'
        }
    }

    with open(output_file, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)

    print_success(f"Example configuration created: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Multi-Machine Coverage Aggregator for AOCL-DLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config coverage_machines.yaml
  %(prog)s --config my_config.yaml --verbose
  %(prog)s --create-example-config example.yaml
  %(prog)s --config coverage.yaml --output-dir ./aggregated_coverage
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--create-example-config',
        type=Path,
        help='Create an example configuration file and exit'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--temp-dir',
        type=Path,
        help='Override temporary directory from config'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )

    args = parser.parse_args()

    # Handle example config creation
    if args.create_example_config:
        create_example_config(args.create_example_config)
        return

    # Validate arguments
    if not args.config:
        print_error(
            "Configuration file is required. Use --config <file> or --create-example-config <file>")
        parser.print_help()
        sys.exit(1)

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = (script_dir / '..' / '..').resolve()

    try:
        # Create aggregator
        aggregator = MultiMachineCoverageAggregator(args.config, project_root)

        # Override config values from command line
        if args.output_dir:
            aggregator.config.output_dir = str(args.output_dir)

        if args.temp_dir:
            aggregator.config.temp_dir = str(args.temp_dir)

        if args.verbose:
            aggregator.config.log_level = "DEBUG"
            aggregator.logger.setLevel(logging.DEBUG)

        if args.dry_run:
            print_info("DRY RUN MODE - No actual operations will be performed")
            print_info(
                f"Would collect from {len(aggregator.sources)} sources:")
            for source in aggregator.sources:
                if source.type == "remote":
                    print_info(
                        f"  {source.name}: {source.username}@{source.host}:{source.build_path}")
                else:
                    print_info(f"  {source.name}: {source.build_path}")
            return

        # Run aggregation
        aggregator.run()

    except KeyboardInterrupt:
        print_warning("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to run coverage aggregation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
