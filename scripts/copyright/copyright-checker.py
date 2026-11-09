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

import sys
import os
from datetime import datetime
import subprocess
import re

REPORT = True

# Set module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import FILE_FORMATS, EXCLUDE_PATTERNS
from common.utils import filter_file

'''
    This script will check copyright headers of source files
'''

COPYRIGHT_HEADER = """
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
"""

# Clause starts and the disclaimer ending. A header that drops or rewrites one
# of these fails, whatever its line wrapping.
REQUIRED_LICENSE_CLAUSES = [
    "1. Redistributions of source code must retain the above copyright notice",
    "2. Redistributions in binary form must reproduce the above copyright notice",
    "3. Neither the name of the copyright holder nor the names of its contributors",
    "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE",
]

def get_file_creation_year(file_path):
    """
    Get the year the file was first committed to git.
    Returns the year as an integer.
    """
    try:
        # Use --reverse to get the oldest commit first
        git_log = subprocess.check_output(['git', 'log', '--reverse', '--pretty=format:%ad', '--date=short', file_path],
                                         stderr=subprocess.DEVNULL)
        if git_log:
            # Extract year from the first (oldest) entry
            start_year_str = git_log.decode().split('\n')[0].split('-')[0]
            return int(start_year_str)
    except (subprocess.CalledProcessError, IndexError, ValueError):
        pass

    # Fallback to current year if git information is unavailable
    return datetime.now().year

def extract_copyright_year(content):
    """
    Extract the copyright year from the file content.
    Handles both single year and range formats.
    Returns tuple (start_year, end_year) as integers, or (None, None) if not found.
    """
    # Look for the copyright line with the C-style comment format
    copyright_pattern = r'Copyright\s+©\s+Advanced Micro Devices,\s+Inc\.,\s+or\s+its\s+affiliates'
    match = re.search(copyright_pattern, content)

    if not match:
        return None, None

    # Since our copyright doesn't include a year, return the current year as both start and end
    current_year = datetime.now().year
    return current_year, current_year

def check_copyright(file_path):
    """
    Check if the file has the correct copyright header.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()

            # Check if the copyright notice is present
            copyright_text = 'Copyright © Advanced Micro Devices, Inc., or its affiliates.'
            if copyright_text not in content:
                print(f"File {file_path} is missing the copyright header")
                return False

            # Layout varies across files, so compare the license clauses in
            # whitespace-normalized form rather than byte-for-byte.
            header = " ".join(
                re.sub(r'^\s*(/\*|\*/|\*|//)?', '', line)
                for line in content[content.index(copyright_text):].split('\n')[:40]
            )
            header = " ".join(header.split())
            missing = [clause for clause in REQUIRED_LICENSE_CLAUSES if clause not in header]
            if missing:
                print(f"File {file_path} has a non-standard license header, missing: {missing}")
                return False

            return True

    except Exception as e:
        print(f"Error checking copyright for {file_path}: {e}")
        return False

def main():
    files = sys.argv[1:]
    if not files:  # If no files are provided as arguments
        print("No files provided to check")
        return

    failed = False
    for file in files:
        if filter_file(file):
            if not check_copyright(file):
                failed = True
        else:
            print(f"File {file} is not a supported C/C++ file type")

    if failed and REPORT:
        sys.exit(1)

if __name__ == "__main__":
    main()
