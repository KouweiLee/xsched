# CUDA Code to Assembly to Bytecode Conversion Guide

This guide explains how to convert CUDA code (`.cu` files) to assembly and then extract hexadecimal bytecode for use in C++ code.

## Overview

The process involves three steps:
1. **Compile** CUDA code to binary (`.cubin`)
2. **Disassemble** binary to assembly (`.asm`)
3. **Extract** hexadecimal bytecode from assembly

## Step-by-Step Process

### Step 1: Compile CUDA Code to Binary

Use `nvcc` to compile your `.cu` file to a `.cubin` file:

```bash
nvcc -cubin inject.cu -o inject.cubin -arch=sm_70 -rdc=true
```

**Important**: Use `-rdc=true` (relocatable device code) to preserve all `__device__` functions even if they are not called. Without this option, unused functions may be optimized away and won't appear in the assembly output.

Replace `sm_70` with your target architecture (e.g., `sm_35`, `sm_86`).

### Step 2: Disassemble Binary to Assembly

Use `cuobjdump` to disassemble the `.cubin` file:

```bash
cuobjdump -sass inject.cubin > inject_sm70.asm
```

This generates an assembly file with instructions and their hexadecimal bytecode.

### Step 3: Extract Hexadecimal Bytecode

Use the provided Python script to extract bytecode from the assembly file:

```bash
python3 extract_bytecode.py inject_sm70.asm --function check_preempt --array-name guardian_instructions
```

Options:
- `--function <name>`: Extract bytecode for a specific function (optional, extracts all if not specified)
- `--array-name <name>`: Name of the C++ array variable (default: `instructions`)
- `--output <file>`: Output file (default: prints to stdout)

### Step 4: Use in C++ Code

The extracted bytecode can be used in C++ code like this:

```cpp
void GuardianSM70::GetGuardianInstructions(const void **guardian_instr, size_t *size)
{
    static const uint64_t guardian_instructions[] = 
    {
        0x000000fffffff389,
        0x000fe200000e00ff,
        0x0000000000ff7355,
        // ... more instructions ...
    };
    *guardian_instr = guardian_instructions;
    *size = sizeof(guardian_instructions);
}
```

## Example Workflow

Here's a complete example for extracting `check_preempt` function:

```bash
# 1. Compile (with -rdc=true to preserve all functions)
nvcc -cubin inject.cu -o inject.cubin -arch=sm_70 -rdc=true

# 2. Disassemble
cuobjdump -sass inject.cubin > inject_sm70.asm

# 3. Extract bytecode
python3 extract_bytecode.py inject_sm70.asm \
    --function check_preempt \
    --array-name guardian_instructions \
    --output guardian_instructions.cpp
```

## Understanding the Assembly Format

The assembly file format looks like this:

```
        /*0000*/              @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ ;               /* 0x000000fffffff389 */
                                                                               /* 0x000fe200000e00ff */
```

Each instruction has:
- An address comment: `/*0000*/`
- The assembly instruction
- Hexadecimal bytecode in comments: `/* 0x... */`

The extract script finds all hexadecimal values in comments and converts them to a C++ array.

## Notes

- Each instruction is 16 bytes (128 bits), represented as two `uint64_t` values
- The bytecode is architecture-specific (sm_35, sm_70, sm_86, etc.)
- Make sure to use the correct architecture when compiling

