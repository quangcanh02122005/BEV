import json

cfg_src = open('our_bevfusion_mini.py', encoding='utf-8').read()
drv_src = open('train_driver.py', encoding='utf-8').read()
viz_src = open('viz_l3.py', encoding='utf-8').read()
cells = []


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})


def bash(s):
    code("%%bash\n" + s.strip("\n"))


bash(r'''
set -euo pipefail
cat > /kaggle/working/l3_env.sh <<'EOF'
export W=/tmp/l3
export ENVDIR=$W/envs/bevfusion
export PY=$ENVDIR/bin/python
export C=$W/constraints.txt
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export OUT=/kaggle/working/outputs
[ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda
true
EOF
source /kaggle/working/l3_env.sh
mkdir -p $W $OUT
TAR=$(find /kaggle/input -maxdepth 6 -name 'env_bevfusion.tar.*' | head -1)
[ -n "$TAR" ] || { echo "khong tim thay env tar"; find /kaggle/input -maxdepth 4 | head -40; exit 1; }
ls -la "$TAR"
case "$TAR" in *.zst) zstd -dc "$TAR" | tar -C $W -xf - ;; *.gz) tar -C $W -xzf "$TAR" ;; esac
V=$(find /kaggle/input -maxdepth 6 -type d -name v1.0-mini | head -1); ROOT=$(dirname "$V"); echo "nuScenes root: $ROOT"
DST=$W/mmdetection3d/data/nuscenes
for d in samples sweeps maps v1.0-mini; do ln -sfn "$ROOT/$d" "$DST/$d"; done
LC=$(find /kaggle/input -name our_lidar_mini.pth | head -1)
[ -n "$LC" ] || { echo "khong thay our_lidar_mini.pth cua Stage A"; find /kaggle/input -maxdepth 5 | head -60; exit 1; }
ls -la "$LC"; echo "$LC" > $W/our_lidar_ckpt_path.txt
$PY --version
''')

code("import os\nfiles = {\n"
     "  '/tmp/l3/mmdetection3d/projects/BEVFusion/configs/our_bevfusion_mini.py': %r,\n" % cfg_src +
     "  '/tmp/l3/train_driver.py': %r,\n" % drv_src +
     "  '/tmp/l3/viz_l3.py': %r,\n" % viz_src +
     "}\nfor p, s in files.items():\n    open(p, 'w', encoding='utf-8').write(s); print('written', p, len(s))\n")

PATCH_CELL = '\nimport re, glob, subprocess, os\npat = re.compile(r"\\bnp\\.(long|int|float|bool|object)\\b")\nrep = {"long": "np.int64", "int": "int", "float": "float", "bool": "bool", "object": "object"}\nchanged = []\nfor f in (glob.glob("/tmp/l3/mmdetection3d/mmdet3d/**/*.py", recursive=True) + glob.glob("/tmp/l3/mmdetection3d/projects/BEVFusion/**/*.py", recursive=True)\n          + glob.glob("/tmp/l3/mmdetection3d/tools/**/*.py", recursive=True)):\n    s = open(f, encoding="utf-8").read()\n    n = pat.sub(lambda m: rep[m.group(1)], s)\n    if n != s:\n        open(f, "w", encoding="utf-8").write(n); changed.append(f)\nprint("numpy-alias patch (np.long->np.int64; np.int/float/bool/object -> builtins) applied to:", changed)\nos.makedirs("/kaggle/working/outputs/config", exist_ok=True)\nd = subprocess.run(["git", "-C", "/tmp/l3/mmdetection3d", "diff"], capture_output=True, text=True).stdout\nopen("/kaggle/working/outputs/config/numpy_alias_patch.diff", "w").write(d)\nprint(d[:3000])\n'
code(PATCH_CELL)

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
export OUR_LIDAR_CKPT=$(cat $W/our_lidar_ckpt_path.txt)
cd $W/mmdetection3d
export PYTHONPATH=$PWD
$PY -W ignore - <<'EOF'
import torch, numpy, mmengine, mmcv, mmdet, mmdet3d, spconv, os
from mmdet3d.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
import projects.BEVFusion.bevfusion
v = dict(numpy=numpy.__version__, torch=torch.__version__, mmengine=mmengine.__version__, mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmdet3d=mmdet3d.__version__, spconv=spconv.__version__)
print(v)
assert v == dict(numpy='1.26.4', torch='2.1.0+cu121', mmengine='0.10.3', mmcv='2.1.0', mmdet='3.3.0', mmdet3d='1.4.0', spconv='2.3.6'), v
assert IS_SPCONV2_AVAILABLE and torch.cuda.is_available()
from mmengine import Config
c = Config.fromfile('projects/BEVFusion/configs/our_bevfusion_mini.py')
import re
pths = re.findall(r"[\w./:\-]+\.pth", c.pretty_text)
print('all .pth strings in config:', pths)
assert c.load_from == os.environ['OUR_LIDAR_CKPT'] and os.path.basename(c.load_from) == 'our_lidar_mini.pth'
assert all(('swin_tiny_patch4_window7_224.pth' in p) or p == c.load_from or p.endswith('our_lidar_mini.pth') for p in pths), pths
assert not any('bevfusion_lidar' in p and 'openmmlab' in p for p in pths)
print('img backbone init:', c.model.img_backbone.init_cfg)
print('train_cfg', c.train_cfg, '| lr', c.optim_wrapper.optimizer.lr, 'accum', c.optim_wrapper.accumulative_counts)
ck = torch.load(c.load_from, map_location='cpu')
print('our LiDAR ckpt tensors:', len(ck['state_dict']), '| note:', ck.get('meta', {}).get('our_note'))
print('STAGE B GATE PASS')
EOF
''')

bash(r'''
set -uo pipefail
source /kaggle/working/l3_env.sh
mkdir -p $OUT/logs
trap 'cp -r $W/logs/. $OUT/logs/ 2>/dev/null || true' EXIT
export OUR_LIDAR_CKPT=$(cat $W/our_lidar_ckpt_path.txt)
export OUR_CFG=projects/BEVFusion/configs/our_bevfusion_mini.py OUR_NAME=our_bevfusion_mini OUR_EXPECT_LOAD=our_lidar_mini.pth
export OUR_EMIN=6 OUR_EMAX=24 OUR_BUDGET_S=10800
$PY $W/train_driver.py 2>&1 | tee $OUT/logs/driver_stdout.log
rc=${PIPESTATUS[0]}
echo "driver exit code: $rc"
exit $rc
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
D=$OUT/our_bevfusion_mini
OURS=$(find $D/predictions -name results_nusc.json | head -1)
OFF=$(find /kaggle/input -name results_nusc.json -path '*L3D_lidar_camera*' | head -1)
V=$(find /kaggle/input -maxdepth 6 -type d -name v1.0-mini | head -1); ROOT=$(dirname "$V")
echo "ours=$OURS official=$OFF"
mkdir -p $D/demo_images
$PY $W/viz_l3.py --root "$ROOT" --tables "$ROOT/v1.0-mini" --res_lidar "$OURS" --res_cam "$OFF" --out $D/demo_images --every 10 --thr 0.3 --dpi 70 \
  --label_l "OUR LiDAR+Camera (trained on mini TRAIN)" --label_c "official LiDAR+Camera pretrained" \
  --note "left BEV = OUR model, right BEV = official (trained on trainval -> contaminated on mini VAL); mini VAL is NOT a full-nuScenes benchmark" 2>&1 | tail -12
cd $OUT && du -sh . && find . -maxdepth 3 | sort | head -60
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open('bev-real-train-b.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump({"id": "canhquang/bev-real-train-b", "title": "bev_real_train_b", "code_file": "bev-real-train-b.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4", "enable_internet": "true",
           "dataset_sources": ["aadimator/nuscenes-mini"], "competition_sources": [],
           "kernel_sources": ["canhquang/bev-real-freeze", "canhquang/bev-real-l3c", "canhquang/bev-real-train-a"]},
          open('kernel-metadata.json', 'w'), indent=2)
print('cells', len(cells))
