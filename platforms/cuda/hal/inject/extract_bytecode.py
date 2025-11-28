#!/usr/bin/env python3
"""
Extract hexadecimal bytecode from CUDA assembly files and generate C++ array.

Usage:
    python3 extract_bytecode.py <input.asm> [--function <function_name>] [--output <output.cpp>]

Example:
    python3 extract_bytecode.py inject_sm70.asm --function check_preempt
"""

import re
import argparse
import sys
from pathlib import Path


def extract_bytecode_from_asm(asm_file: Path, function_name: str = None):
    """
    Extract hexadecimal bytecode from CUDA assembly file.
    
    Args:
        asm_file: Path to the .asm file
        function_name: Optional function name to extract (if None, extracts all)
    
    Returns:
        List of hex strings (without 0x prefix)
    """
    bytecodes = []
    in_target_function = function_name is None
    
    with open(asm_file, 'r') as f:
        for line in f:
            # Check if we're entering a function
            # Use exact match: "Function : check_preempt" should match exactly, not "check_preempt_trap"
            if function_name:
                # Match pattern: "Function : <function_name>" with word boundaries
                # This ensures "check_preempt" doesn't match "check_preempt_trap"
                function_pattern = rf'\s+Function\s+:\s+{re.escape(function_name)}\s*$'
                if re.search(function_pattern, line):
                    in_target_function = True
                    continue
            
            # Check if we're leaving the function (next function starts)
            # Only check this if we're already in the target function
            if in_target_function and function_name:
                # Check if this line starts a new function
                if re.match(r'\s+Function\s+:', line):
                    # Make sure it's not the same function (could be duplicate)
                    function_pattern = rf'\s+Function\s+:\s+{re.escape(function_name)}\s*$'
                    if not re.search(function_pattern, line):
                        break
            
            # Only extract hex bytecode if we're in the target function
            if in_target_function:
                # Extract hex bytecode from comments
                # Pattern: /* 0x... */ or /* 0x... */
                hex_pattern = r'/\*\s*(0x[0-9a-fA-F]+)\s*\*/'
                matches = re.findall(hex_pattern, line)
                
                if matches:
                    for match in matches:
                        # Remove 0x prefix and store
                        hex_value = match[2:] if match.startswith('0x') else match
                        bytecodes.append(hex_value)
    
    return bytecodes


def hex_to_uint64_array(hex_strings):
    """
    Convert list of hex strings to uint64_t array format.
    
    Args:
        hex_strings: List of hex strings (without 0x prefix)
    
    Returns:
        List of formatted uint64_t values
    """
    uint64_values = []
    
    for hex_str in hex_strings:
        # Convert hex string to uint64_t
        value = int(hex_str, 16)
        # Format as 0x... with proper padding
        formatted = f"0x{hex_str.lower()}"
        uint64_values.append(formatted)
    
    return uint64_values


def generate_cpp_code(uint64_values, array_name="instructions", indent="    "):
    """
    Generate C++ code with uint64_t array.
    
    Args:
        uint64_values: List of formatted hex values
        array_name: Name of the array variable
        indent: Indentation string
    
    Returns:
        C++ code string
    """
    if not uint64_values:
        return ""
    
    lines = [f"static const uint64_t {array_name}[] =", "{"]
    
    # Group values with proper formatting
    for i, value in enumerate(uint64_values):
        comma = "," if i < len(uint64_values) - 1 else ""
        lines.append(f"{indent}{value}{comma}")
    
    lines.append("};")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract hexadecimal bytecode from CUDA assembly files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input .asm file"
    )
    parser.add_argument(
        "--function",
        "-f",
        type=str,
        default=None,
        help="Function name to extract (if not specified, extracts all)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file (if not specified, prints to stdout)"
    )
    parser.add_argument(
        "--array-name",
        "-a",
        type=str,
        default="instructions",
        help="Name of the C++ array variable (default: instructions)"
    )
    
    args = parser.parse_args()
    
    asm_file = Path(args.input)
    if not asm_file.exists():
        print(f"Error: File '{asm_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Extract bytecodes
    bytecodes = extract_bytecode_from_asm(asm_file, args.function)
    
    if not bytecodes:
        print(f"Warning: No bytecodes found", file=sys.stderr)
        if args.function:
            print(f"  Function: {args.function}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to uint64_t array
    uint64_values = hex_to_uint64_array(bytecodes)
    
    # Generate C++ code
    cpp_code = generate_cpp_code(uint64_values, args.array_name)
    
    # Output
    if args.output:
        output_file = Path(args.output)
        output_file.write_text(cpp_code + "\n")
        print(f"Generated {len(uint64_values)} instructions in '{output_file}'")
    else:
        print(cpp_code)
    
    print(f"Total instructions: {len(uint64_values)}", file=sys.stderr)


if __name__ == "__main__":
    main()

