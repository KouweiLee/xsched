export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=0
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=4
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=2
# replace <install_path> with the path of the XSched installation directory.
export LD_LIBRARY_PATH=/home/lgw/study/phd/xsched/output/lib:$LD_LIBRARY_PATH
./app