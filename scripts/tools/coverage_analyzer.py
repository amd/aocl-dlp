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
Advanced Code Coverage Analyzer for AOCL-DLP

This script provides advanced analysis capabilities for code coverage data,
leveraging the existing project infrastructure and exclusion patterns.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

# Add the parent directory to the path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from common.config import FILE_FORMATS, EXCLUDE_PATTERNS
    from common.utils import filter_file
except ImportError as e:
    print(f"Error: Could not import common modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


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


class CoverageAnalyzer:
    """Advanced coverage analysis for AOCL-DLP."""

    def __init__(self, project_root: Path, build_dir: Path):
        """Initialize the coverage analyzer."""
        self.project_root = project_root
        self.build_dir = build_dir
        self.coverage_data = {}

        # Extended exclusion patterns for coverage analysis
        self.coverage_exclusions = EXCLUDE_PATTERNS + [
            "tests", "test", "bench", "benchmark", "docs", "scripts",
            "tools", "examples", "example", "JIT", "external",
            "third_party", "3rdparty", "CMakeFiles", ".cmake"
        ]

    def discover_source_files(self) -> List[Path]:
        """
        Discover all source files in the project using the existing filter logic.

        Returns:
            List of source file paths that should be analyzed for coverage.
        """
        print_info("Discovering source files...")

        source_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in str(Path(root) / d) for pattern in self.coverage_exclusions)]

            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.project_root)

                # Use the existing filter logic from common.utils
                if filter_file(str(file_path)):
                    # Additional check for coverage-specific exclusions
                    if not any(pattern in str(relative_path) for pattern in self.coverage_exclusions):
                        source_files.append(file_path)

        print_info(f"Found {len(source_files)} source files for coverage analysis")
        return source_files

    def parse_lcov_file(self, lcov_file: Path) -> Dict:
        """
        Parse an LCOV info file and extract coverage data.

        Args:
            lcov_file: Path to the LCOV info file

        Returns:
            Dictionary containing parsed coverage data
        """
        print_info(f"Parsing LCOV file: {lcov_file}")

        coverage_data = {
            'files': {},
            'summary': {
                'total_lines': 0,
                'covered_lines': 0,
                'total_functions': 0,
                'covered_functions': 0,
                'total_branches': 0,
                'covered_branches': 0
            }
        }

        if not lcov_file.exists():
            print_error(f"LCOV file not found: {lcov_file}")
            return coverage_data

        current_file = None
        current_file_data = {}

        with open(lcov_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('SF:'):
                    # Source file
                    if current_file and current_file_data:
                        coverage_data['files'][current_file] = current_file_data

                    current_file = line[3:]  # Remove 'SF:'
                    current_file_data = {
                        'lines': {},
                        'functions': {},
                        'branches': {},
                        'line_coverage': 0.0,
                        'function_coverage': 0.0,
                        'branch_coverage': 0.0
                    }

                elif line.startswith('DA:'):
                    # Line data (DA:line_number,execution_count)
                    parts = line[3:].split(',')
                    if len(parts) >= 2:
                        line_num = int(parts[0])
                        exec_count = int(parts[1])
                        current_file_data['lines'][line_num] = exec_count

                elif line.startswith('FN:'):
                    # Function definition (FN:line_number,function_name)
                    parts = line[3:].split(',', 1)
                    if len(parts) >= 2:
                        line_num = int(parts[0])
                        func_name = parts[1]
                        current_file_data['functions'][func_name] = {
                            'line': line_num,
                            'executed': False
                        }

                elif line.startswith('FNDA:'):
                    # Function execution data (FNDA:execution_count,function_name)
                    parts = line[5:].split(',', 1)
                    if len(parts) >= 2:
                        exec_count = int(parts[0])
                        func_name = parts[1]
                        if func_name in current_file_data['functions']:
                            current_file_data['functions'][func_name]['executed'] = exec_count > 0
                            current_file_data['functions'][func_name]['exec_count'] = exec_count

                elif line.startswith('BDA:'):
                    # Branch data (BDA:line_number,block_number,branch_number,taken_count)
                    parts = line[4:].split(',')
                    if len(parts) >= 4:
                        line_num = int(parts[0])
                        block_num = int(parts[1])
                        branch_num = int(parts[2])
                        taken_count = int(parts[3])

                        branch_key = f"{line_num}:{block_num}:{branch_num}"
                        current_file_data['branches'][branch_key] = taken_count

        # Add the last file
        if current_file and current_file_data:
            coverage_data['files'][current_file] = current_file_data

        # Calculate summary statistics
        self._calculate_summary_stats(coverage_data)

        return coverage_data

    def _calculate_summary_stats(self, coverage_data: Dict) -> None:
        """Calculate summary statistics for the coverage data."""
        summary = coverage_data['summary']

        for file_path, file_data in coverage_data['files'].items():
            # Line coverage
            total_lines = len(file_data['lines'])
            covered_lines = sum(1 for count in file_data['lines'].values() if count > 0)

            if total_lines > 0:
                file_data['line_coverage'] = (covered_lines / total_lines) * 100
                summary['total_lines'] += total_lines
                summary['covered_lines'] += covered_lines

            # Function coverage
            total_functions = len(file_data['functions'])
            covered_functions = sum(1 for func in file_data['functions'].values() if func.get('executed', False))

            if total_functions > 0:
                file_data['function_coverage'] = (covered_functions / total_functions) * 100
                summary['total_functions'] += total_functions
                summary['covered_functions'] += covered_functions

            # Branch coverage
            total_branches = len(file_data['branches'])
            covered_branches = sum(1 for count in file_data['branches'].values() if count > 0)

            if total_branches > 0:
                file_data['branch_coverage'] = (covered_branches / total_branches) * 100
                summary['total_branches'] += total_branches
                summary['covered_branches'] += covered_branches

    def generate_detailed_report(self, coverage_data: Dict, output_file: Optional[Path] = None) -> None:
        """
        Generate a detailed coverage report.

        Args:
            coverage_data: Parsed coverage data
            output_file: Optional output file path
        """
        print_info("Generating detailed coverage report...")

        summary = coverage_data['summary']

        report_lines = [
            "AOCL-DLP Detailed Coverage Analysis",
            "=" * 50,
            "",
            "Summary Statistics:",
            "-" * 20
        ]

        # Overall coverage percentages
        if summary['total_lines'] > 0:
            line_coverage = (summary['covered_lines'] / summary['total_lines']) * 100
            report_lines.append(f"Line Coverage:     {line_coverage:.2f}% ({summary['covered_lines']}/{summary['total_lines']})")

        if summary['total_functions'] > 0:
            func_coverage = (summary['covered_functions'] / summary['total_functions']) * 100
            report_lines.append(f"Function Coverage: {func_coverage:.2f}% ({summary['covered_functions']}/{summary['total_functions']})")

        if summary['total_branches'] > 0:
            branch_coverage = (summary['covered_branches'] / summary['total_branches']) * 100
            report_lines.append(f"Branch Coverage:   {branch_coverage:.2f}% ({summary['covered_branches']}/{summary['total_branches']})")

        report_lines.extend([
            "",
            "File-by-File Coverage:",
            "-" * 25
        ])

        # Sort files by coverage percentage (lowest first)
        files_by_coverage = sorted(
            coverage_data['files'].items(),
            key=lambda x: x[1]['line_coverage']
        )

        for file_path, file_data in files_by_coverage:
            # Make path relative to project root for readability
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
            except ValueError:
                rel_path = Path(file_path)

            line_cov = file_data['line_coverage']
            func_cov = file_data['function_coverage']
            branch_cov = file_data['branch_coverage']

            # Color code based on coverage level
            if line_cov >= 80:
                status = "GOOD"
            elif line_cov >= 60:
                status = "FAIR"
            else:
                status = "POOR"

            report_lines.append(
                f"{str(rel_path):<60} Lines: {line_cov:5.1f}%  Funcs: {func_cov:5.1f}%  Branches: {branch_cov:5.1f}%  [{status}]"
            )

        # Identify uncovered functions
        uncovered_functions = []
        for file_path, file_data in coverage_data['files'].items():
            for func_name, func_data in file_data['functions'].items():
                if not func_data.get('executed', False):
                    try:
                        rel_path = Path(file_path).relative_to(self.project_root)
                    except ValueError:
                        rel_path = Path(file_path)
                    uncovered_functions.append((str(rel_path), func_name, func_data['line']))

        if uncovered_functions:
            report_lines.extend([
                "",
                "Uncovered Functions:",
                "-" * 20
            ])

            for file_path, func_name, line_num in sorted(uncovered_functions)[:20]:  # Limit to top 20
                report_lines.append(f"{file_path}:{line_num} - {func_name}")

            if len(uncovered_functions) > 20:
                report_lines.append(f"... and {len(uncovered_functions) - 20} more")

        report_content = "\n".join(report_lines)

        if output_file:
            print_info(f"Writing detailed report to: {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_content)
        else:
            print(report_content)

    def find_coverage_gaps(self, coverage_data: Dict, threshold: float = 80.0) -> List[Dict]:
        """
        Find files and functions with coverage below the threshold.

        Args:
            coverage_data: Parsed coverage data
            threshold: Coverage threshold percentage

        Returns:
            List of coverage gaps
        """
        print_info(f"Finding coverage gaps below {threshold}%...")

        gaps = []

        for file_path, file_data in coverage_data['files'].items():
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
            except ValueError:
                rel_path = Path(file_path)

            if file_data['line_coverage'] < threshold:
                gaps.append({
                    'type': 'file',
                    'path': str(rel_path),
                    'coverage': file_data['line_coverage'],
                    'lines_total': len(file_data['lines']),
                    'lines_covered': sum(1 for count in file_data['lines'].values() if count > 0)
                })

            # Find uncovered functions
            for func_name, func_data in file_data['functions'].items():
                if not func_data.get('executed', False):
                    gaps.append({
                        'type': 'function',
                        'path': str(rel_path),
                        'function': func_name,
                        'line': func_data['line'],
                        'coverage': 0.0
                    })

        return gaps

    def export_coverage_json(self, coverage_data: Dict, output_file: Path) -> None:
        """Export coverage data to JSON format."""
        print_info(f"Exporting coverage data to: {output_file}")

        # Convert paths to relative paths for JSON export
        json_data = {
            'summary': coverage_data['summary'],
            'files': {}
        }

        for file_path, file_data in coverage_data['files'].items():
            try:
                rel_path = str(Path(file_path).relative_to(self.project_root))
            except ValueError:
                rel_path = file_path

            json_data['files'][rel_path] = file_data

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced Code Coverage Analyzer for AOCL-DLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --lcov coverage_reports/coverage.info
  %(prog)s --lcov coverage.info --threshold 70 --json coverage.json
  %(prog)s --discover-sources
        """
    )

    parser.add_argument(
        '--lcov',
        type=Path,
        help='Path to LCOV info file for analysis'
    )

    parser.add_argument(
        '--build-dir',
        type=Path,
        default=Path('build'),
        help='Build directory (default: build)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('coverage_reports'),
        help='Output directory for reports (default: coverage_reports)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=80.0,
        help='Coverage threshold for gap analysis (default: 80.0)'
    )

    parser.add_argument(
        '--json',
        type=Path,
        help='Export coverage data to JSON file'
    )

    parser.add_argument(
        '--discover-sources',
        action='store_true',
        help='Discover and list source files for coverage analysis'
    )

    parser.add_argument(
        '--gaps-only',
        action='store_true',
        help='Only show coverage gaps, not full report'
    )

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = (script_dir / '..' / '..').resolve()

    print_colored("AOCL-DLP Advanced Coverage Analyzer", Colors.WHITE)
    print_colored("=" * 40, Colors.WHITE)

    analyzer = CoverageAnalyzer(project_root, args.build_dir)

    if args.discover_sources:
        source_files = analyzer.discover_source_files()
        print("\nDiscovered source files:")
        for file_path in sorted(source_files):
            rel_path = file_path.relative_to(project_root)
            print(f"  {rel_path}")
        return

    if not args.lcov:
        # Try to find default LCOV file
        default_lcov = args.output_dir / 'coverage.info'
        if default_lcov.exists():
            args.lcov = default_lcov
            print_info(f"Using default LCOV file: {default_lcov}")
        else:
            print_error("No LCOV file specified and default not found")
            print_info("Please specify --lcov <file> or run coverage generation first")
            sys.exit(1)

    # Parse coverage data
    coverage_data = analyzer.parse_lcov_file(args.lcov)

    if not coverage_data['files']:
        print_error("No coverage data found in LCOV file")
        sys.exit(1)

    # Generate reports
    if not args.gaps_only:
        analyzer.generate_detailed_report(coverage_data, args.output_dir / 'detailed_report.txt')

    # Find coverage gaps
    gaps = analyzer.find_coverage_gaps(coverage_data, args.threshold)

    if gaps:
        print_warning(f"Found {len(gaps)} coverage gaps below {args.threshold}%:")

        file_gaps = [g for g in gaps if g['type'] == 'file']
        func_gaps = [g for g in gaps if g['type'] == 'function']

        if file_gaps:
            print_colored("\nFiles with low coverage:", Colors.YELLOW)
            for gap in sorted(file_gaps, key=lambda x: x['coverage'])[:10]:
                print(f"  {gap['path']}: {gap['coverage']:.1f}% ({gap['lines_covered']}/{gap['lines_total']} lines)")

        if func_gaps:
            print_colored(f"\nUncovered functions ({len(func_gaps)} total):", Colors.YELLOW)
            for gap in sorted(func_gaps, key=lambda x: x['path'])[:15]:
                print(f"  {gap['path']}:{gap['line']} - {gap['function']}")
    else:
        print_success(f"All files have coverage above {args.threshold}%!")

    # Export JSON if requested
    if args.json:
        analyzer.export_coverage_json(coverage_data, args.json)

    print_success("Coverage analysis complete!")


if __name__ == '__main__':
    main()
