#!/bin/bash
# Example script to run YOLO with XSched
# This demonstrates how to use the launcher script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example 1: Run YOLO with high priority (infinite loop, default model auto-downloaded)
IMAGE_PATH="${1:-test.jpg}"     # Default to test.jpg
PRIORITY="${2:-100}"            # Default priority: 100 (high)
MODEL_PATH="${3:-}"              # Optional model path (default: yolov8n.pt)

echo "Example: Running YOLO with XSched (high priority)"
if [ -n "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_PATH} (will auto-download if not found)"
else
    echo "Model: yolov8n.pt (default, will auto-download if not found)"
fi
echo "Image: ${IMAGE_PATH}"
echo "Priority: ${PRIORITY}"
echo ""

# Use the launcher script to configure XSched transparently
if [ -n "${MODEL_PATH}" ]; then
    "${SCRIPT_DIR}/run_yolo_with_xsched.sh" \
        --priority "${PRIORITY}" \
        -- \
        python3 "${SCRIPT_DIR}/yolo_xsched.py" \
            "${MODEL_PATH}" \
            "${IMAGE_PATH}"
else
    "${SCRIPT_DIR}/run_yolo_with_xsched.sh" \
        --priority "${PRIORITY}" \
        -- \
        python3 "${SCRIPT_DIR}/yolo_xsched.py" \
            "${IMAGE_PATH}"
fi

# Example 2: Run with 100 iterations
# "${SCRIPT_DIR}/run_yolo_with_xsched.sh" \
#     --priority 50 \
#     -- \
#     python3 "${SCRIPT_DIR}/yolo_xsched.py" \
#         "${IMAGE_PATH}" \
#         --iterations 100 \
#         --interval 0.1


