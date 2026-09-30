#!/usr/bin/env bash
# Wait until a Kaggle kernel finishes:  tools/poll_kernel.sh canhquang/bev-real-flow [interval_seconds]
K=$1; T=${2:-45}; D=$(dirname "$0")
for i in $(seq 1 600); do
  s=$("$D/kgl.sh" kernels status "$K" 2>&1 | tail -1 | grep -o 'KernelWorkerStatus.[A-Z]*')
  echo "$(date +%H:%M:%S) $K $s"
  echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL" && break
  sleep "$T"
done
