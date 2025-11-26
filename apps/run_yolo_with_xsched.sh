#!/bin/bash
# Launcher script for YOLO inference with XSched GPU scheduling
# This script configures XSched transparently, so the YOLO program
# doesn't need to know about XSched at all.

set -e

# Get script directory and xsched root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSCHED_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XSCHED_INSTALL="${XSCHED_ROOT}/output"

# Default values
PRIORITY=0
PREEMPT_LEVEL=1
SCHEDULER="GLB"  # GLB for global scheduler (xserver), or APP for app-managed

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --priority)
            PRIORITY="$2"
            shift 2
            ;;
        --preempt-level)
            PREEMPT_LEVEL="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] -- [YOLO_ARGS...]"
            echo ""
            echo "XSched Options:"
            echo "  --priority PRIORITY     Priority for GPU scheduling (-255 to 255, default: 0)"
            echo "  --preempt-level LEVEL   Preemption level (1=Block, 2=Deactivate, 3=Interrupt, default: 1)"
            echo "  --scheduler SCHEDULER    Scheduler type (GLB or APP, default: GLB)"
            echo ""
            echo "Examples:"
            echo "  # Run with high priority (using default model, auto-downloaded)"
            echo "  $0 --priority 100 -- python3 yolo_xsched.py image.jpg"
            echo ""
            echo "  # Run with custom model and high priority"
            echo "  $0 --priority 100 -- python3 yolo_xsched.py yolov8s.pt image.jpg"
            echo ""
            echo "  # Run with low priority and 100 iterations"
            echo "  $0 --priority -100 -- python3 yolo_xsched.py image.jpg --iterations 100"
            echo ""
            echo "  # Run with app-managed scheduler"
            echo "  $0 --scheduler APP --priority 50 -- python3 yolo_xsched.py image.jpg"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if XSched is built
if [ ! -d "${XSCHED_INSTALL}/lib" ]; then
    echo "Error: XSched is not built. Please build it first:"
    echo "  cd ${XSCHED_ROOT} && make cuda INSTALL_PATH=./output"
    exit 1
fi

# Set LD_LIBRARY_PATH for XSched libraries
export LD_LIBRARY_PATH="${XSCHED_INSTALL}/lib:${LD_LIBRARY_PATH}"

# Configure XSched environment variables for transparent scheduling
export XSCHED_SCHEDULER="${SCHEDULER}"
export XSCHED_AUTO_XQUEUE="ON"
export XSCHED_AUTO_XQUEUE_PRIORITY="${PRIORITY}"
export XSCHED_AUTO_XQUEUE_LEVEL="${PREEMPT_LEVEL}"
export XSCHED_AUTO_XQUEUE_THRESHOLD="16"
export XSCHED_AUTO_XQUEUE_BATCH_SIZE="8"

# Check if xserver is running (for global scheduler)
if [ "${SCHEDULER}" = "GLB" ] && ! pgrep -x "xserver" > /dev/null; then
    echo "Warning: xserver is not running."
    echo "For global scheduling, start xserver first:"
    echo "  ${XSCHED_INSTALL}/bin/xserver HPF 50000"
    echo ""
    echo "Continuing anyway (will use local scheduling if xserver unavailable)..."
fi

echo "XSched Configuration:"
echo "  Scheduler: ${SCHEDULER}"
echo "  Priority: ${PRIORITY}"
echo "  Preempt Level: ${PREEMPT_LEVEL}"
echo "  Library Path: ${XSCHED_INSTALL}/lib"
echo ""

# Run the YOLO program with remaining arguments
if [ $# -eq 0 ]; then
    echo "Error: No YOLO program arguments provided"
    echo "Usage: $0 [XSCHED_OPTIONS] -- yolo_xsched.py [YOLO_ARGS...]"
    exit 1
fi

exec "$@"

