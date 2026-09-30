import json
fus = open('late_fusion.py', encoding='utf-8').read()
ev = open('fusion_eval.py', encoding='utf-8').read()
c1 = "import os\nos.makedirs('/tmp/l3/fusion', exist_ok=True)\nopen('/tmp/l3/fusion/late_fusion.py','w',encoding='utf-8').write(%r)\nopen('/tmp/l3/fusion_eval.py','w',encoding='utf-8').write(%r)\nprint('written')" % (fus, ev)
c2 = r'''%%bash
set -euo pipefail
export W=/tmp/l3; export MPLBACKEND=Agg
TAR=$(find /kaggle/input -maxdepth 6 -name 'env_bevfusion.tar.*' | head -1); echo "$TAR"
mkdir -p $W/fe /kaggle/working/outputs
case "$TAR" in *.zst) zstd -dc "$TAR" | tar -C $W -xf - envs/bevfusion ;; *.gz) tar -C $W -xzf "$TAR" envs/bevfusion ;; esac
PY=$W/envs/bevfusion/bin/python
$PY -c "import nuscenes, scipy, numpy; print('devkit', nuscenes.__file__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
$PY -W ignore $W/fusion_eval.py 2>&1 | tail -40
ls -la /kaggle/working/outputs/fusion_eval /kaggle/working/outputs/fusion_eval/submissions
'''
cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)} for c in (c1, c2)]
json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}, open('bev-real-fusion-eval.ipynb', 'w'), indent=1)
json.dump({"id": "canhquang/bev-real-fusion-eval", "title": "bev_real_fusion_eval", "code_file": "bev-real-fusion-eval.ipynb", "language": "python", "kernel_type": "notebook",
  "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "dataset_sources": ["aadimator/nuscenes-mini"], "competition_sources": [],
  "kernel_sources": ["canhquang/bev-real-freeze", "canhquang/bev-real-l3c", "canhquang/bev-real-train-b", "canhquang/bev-real-cr-infer"]}, open('kernel-metadata.json', 'w'), indent=2)
print('ok')
