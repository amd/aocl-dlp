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
#
import os
import sys
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import FILE_FORMATS, EXCLUDE_PATTERNS
from common.utils import filter_file

'''
    This script will format all the source files in the current directory using the ".clang-format"
'''

def format_file(file_path):
    try:
        subprocess.run(["clang-format", "-i", file_path], check=True)
        return f"Formatted: {file_path}", True
    except subprocess.CalledProcessError as e:
        return f"Failed to format {file_path}: {e}", False

def find_files_to_format():
    current_dir = os.getcwd()
    files_to_format = []

    for root, _, files in os.walk(current_dir):
        for file in files:
            if filter_file(file):
                file_path = os.path.join(root, file)
                files_to_format.append(file_path)

    return files_to_format

def main():
    # Find all files to format
    files_to_format = find_files_to_format()

    if not files_to_format:
        print("No files found to format.")
        return

    # Display preview
    print(f"Found {len(files_to_format)} files to format:")
    for i, file_path in enumerate(files_to_format, 1):
        print(f"{i}. {file_path}")

    # Ask for confirmation
    response = input("\nDo you want to continue with formatting? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Formatting cancelled.")
        return

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"\nUsing {num_cores} CPU cores for parallel formatting...")

    # Use process pool to parallelize formatting
    successful_formats = 0
    failed_formats = 0

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(format_file, files_to_format))

        for message, success in results:
            print(message)
            if success:
                successful_formats += 1
            else:
                failed_formats += 1

    print(f"\nFormatting completed: {successful_formats} successful, {failed_formats} failed")

if __name__ == "__main__":
    main()
