#!/usr/bin/env bash
# Kaggle CLI wrapper. The API token stays in D:\kaggle_mcp\.env (NOT copied into this workspace); it is read at call time.
#   tools/kgl.sh kernels status canhquang/bev-real-flow
#   tools/kgl.sh kernels output canhquang/bev-real-flow -p some_dir [--file-pattern REGEX]
#   tools/kgl.sh kernels push -p notebooks/bev-real-flow      tools/kgl.sh quota
ENVF=/d/kaggle_mcp/.env
[ -f "$ENVF" ] || { echo "missing $ENVF (Kaggle token)"; exit 1; }
export KAGGLE_API_TOKEN=$(grep '^KAGGLE_API_TOKEN=' "$ENVF" | cut -d= -f2)
export PYTHONUTF8=1
exec /d/kaggle_mcp/venv/Scripts/kaggle.exe "$@"
