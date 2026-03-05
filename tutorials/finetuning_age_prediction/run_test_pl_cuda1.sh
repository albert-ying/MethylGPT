#!/bin/bash

echo "=== Testing PyTorch Lightning with CUDA device 1 ==="
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo

# Check GPU status before test
echo "=== GPU Status Before Test ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
echo

# Test 1: Direct device specification (should use physical GPU 1)
echo "=== Test 1: Direct device specification ==="
echo "Running: python test_pl_cuda1.py"
python test_pl_cuda1.py
echo

# Check GPU status after test
echo "=== GPU Status After Test ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
echo

echo "=== Test completed ==="
