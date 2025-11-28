GPU_GEN_CODE = $(subst .,,${shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader})

# Default target: compile inject.cu to cubin file (binary)
# Use -rdc=true to preserve all device functions even if unused
bin: inject.cu
	nvcc -cubin $< -o inject_sm${GPU_GEN_CODE}.cubin -arch=sm_${GPU_GEN_CODE} -rdc=true

# Disassemble cubin to assembly
asm: inject_sm${GPU_GEN_CODE}.cubin
	cuobjdump -sass $< > inject_sm${GPU_GEN_CODE}.asm

# Extract bytecode for check_preempt function
extract-check-preempt: inject_sm${GPU_GEN_CODE}.asm
	python3 extract_bytecode.py $< --function check_preempt --array-name guardian_instructions

# Extract bytecode for restore_exec function
extract-restore-exec: inject_sm${GPU_GEN_CODE}.asm
	python3 extract_bytecode.py $< --function restore_exec --array-name resume_instructions

# Extract bytecode for check_preempt_trap function
extract-check-preempt-trap: inject_sm${GPU_GEN_CODE}.asm
	python3 extract_bytecode.py $< --function check_preempt_trap --array-name trap_inject_instrs

# Extract all bytecodes
extract-all: inject_sm${GPU_GEN_CODE}.asm
	python3 extract_bytecode.py $< --array-name all_instructions

# Complete workflow: compile -> disassemble -> extract check_preempt
all: bin asm extract-check-preempt

.PHONY: clean
clean:
	@rm -f inject_sm*.cubin inject_sm*.asm
