import json
src = open('viz_l3.py', encoding='utf-8').read()
cell = r'''%%bash
set -euo pipefail
V=$(find /kaggle/input -maxdepth 6 -type d -name v1.0-mini | head -1); ROOT=$(dirname "$V")
RL=$(find /kaggle/input -name results_nusc.json -path '*L3C_lidar_only*' | head -1)
RC=$(find /kaggle/input -name results_nusc.json -path '*L3D_lidar_camera*' | head -1)
echo "ROOT=$ROOT"; echo "RL=$RL"; echo "RC=$RC"
[ -n "$ROOT" ] && [ -n "$RL" ] && [ -n "$RC" ] || { find /kaggle/input -maxdepth 4 | head -50; exit 1; }
mkdir -p /kaggle/working/viz
python3 /kaggle/working/viz_l3.py --root "$ROOT" --res_lidar "$RL" --res_cam "$RC" --out /kaggle/working/viz --thr 0.3 2>&1 | tail -30
ls /kaggle/working/viz | wc -l
'''
w = "import os\nopen('/kaggle/working/viz_l3.py','w',encoding='utf-8').write(" + repr(src) + ")\nprint('written', os.path.getsize('/kaggle/working/viz_l3.py'))"
cells = [
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": w.splitlines(keepends=True)},
  {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell.splitlines(keepends=True)}]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open('bev-real-viz.ipynb', 'w', encoding='utf-8'), indent=1)
json.dump({"id": "canhquang/bev-real-viz", "title": "bev_real_viz", "code_file": "bev-real-viz.ipynb", "language": "python", "kernel_type": "notebook",
  "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "dataset_sources": ["aadimator/nuscenes-mini"],
  "competition_sources": [], "kernel_sources": ["canhquang/bev-real-l3c"]}, open('kernel-metadata.json', 'w'), indent=2)
print('ok')
