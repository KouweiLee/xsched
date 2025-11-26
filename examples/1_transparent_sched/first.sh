# the process will be scheduled according to the global (GLB) scheduler, i.e., the xserver
export XSCHED_SCHEDULER=GLB

# automatically create an XQueue for each created HwQueue (in this case, CUDA stream)
export XSCHED_AUTO_XQUEUE=ON

# automatically set the priority of the created XQueue to 1 (bigger is higher priority)
export XSCHED_AUTO_XQUEUE_PRIORITY=1

# automatically enable Level-1 preemption for the created XQueue
# higher levels can achieve faster preemption but needs corresponding implementation
export XSCHED_AUTO_XQUEUE_LEVEL=1

# automatically set the threshold of the created XQueue to 16 (default is 16)
# threshold is the number of in-flight commands in the XQueue, >= 1
# smaller threshold leads to faster preemption but higher execution overhead
export XSCHED_AUTO_XQUEUE_THRESHOLD=16

# automatically set the command batch size of the created XQueue to 8 (default is 8)
# batch size is the number of commands that XSched will launch at a time,
# >= 1 && <= threshold, recommended to be half of the threshold
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8

# Intercept the CUDA calls using the shim library (libcuda.so -> libshimcuda.so).
# For cuda, libshimcuda.so implements all the symbols in libcuda.so, and we set
# LD_LIBRARY_PATH to the path of the XSched library to intercept the CUDA calls.
# Replace <install_path> with the path of the XSched installation directory.
export LD_LIBRARY_PATH=/home/lgw/study/phd/xsched/output/lib:$LD_LIBRARY_PATH
# For other platforms like opencl, we may use LD_PRELOAD to intercept the calls.
export LD_PRELOAD=/home/lgw/study/phd/xsched/output/lib/libOpenCL.so:$LD_PRELOAD

# run the app
./app