import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def bash(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": ("%%bash\n" + s.strip("\n")).splitlines(keepends=True)})


md("# BEV_real L3 (auto): env -> nuScenes-mini -> L3C LiDAR-only -> L3D LiDAR+Camera\n"
   "Pretrained BEVFusion inference + nuScenes eval. Heavy files live in /kaggle/temp (not saved as output); "
   "results go to /kaggle/working/outputs.")

bash(r'''
set -euo pipefail
cat > /kaggle/working/l3_env.sh <<'EOF'
export B=/kaggle/temp; [ -d "$B" ] || B=/tmp
export W=$B/l3
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
mkdir -p $W/logs $W/exports $W/checkpoints $OUT
echo "W=$W"; df -h $W /kaggle/working | cat
nvidia-smi -L
which nvcc && nvcc --version | tail -2
git clone --depth 1 --branch v1.4.0 https://github.com/open-mmlab/mmdetection3d.git $W/mmdetection3d 2>&1 | tail -2
ls $W/mmdetection3d/projects/BEVFusion/configs
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
{
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $W/miniconda.sh
bash $W/miniconda.sh -b -p $W/miniconda3 > /dev/null
$W/miniconda3/bin/conda --version
$W/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
$W/miniconda3/bin/conda tos accept || true
$W/miniconda3/bin/conda create -y --override-channels -c conda-forge -p $ENVDIR python=3.10 pip
$PY --version
} 2>&1 | tee $W/logs/step_conda.log
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cat > $C <<'EOF'
numpy==1.26.4
torch==2.1.0+cu121
torchvision==0.16.0+cu121
EOF
{
$PY -m pip install -q -U pip wheel
$PY -m pip install -c $C torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install -c $C "numpy==1.26.4" ninja
$PY -m pip install -c $C "mmengine==0.10.3"
$PY -m pip install -c $C "mmcv==2.1.0" --only-binary=mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
$PY -m pip install -c $C spconv-cu120
$PY -m pip install "setuptools==69.5.1"
$PY -c "
import spconv, spconv.pytorch
print('spconv', spconv.__version__)
import torch, numpy, mmengine, mmcv, setuptools
from mmcv.ops import nms
assert numpy.__version__=='1.26.4' and torch.__version__=='2.1.0+cu121' and mmcv.__version__=='2.1.0' and setuptools.__version__=='69.5.1'
b = torch.tensor([[0,0,10,10],[1,1,11,11],[50,50,60,60]], dtype=torch.float32, device='cuda')
s = torch.tensor([0.9,0.8,0.7], device='cuda')
_, keep = nms(b, s, 0.5); assert keep.tolist()==[0,2], keep
print('ENV GATE PASS', torch.__version__, numpy.__version__, mmengine.__version__, mmcv.__version__, torch.cuda.get_device_name(0))
"
} 2>&1 | tee $W/logs/step_env.log
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cd $W/mmdetection3d
{
$PY -m pip install -c $C "mmdet==3.3.0"
$PY -m pip install -c $C lyft_dataset_sdk "networkx>=2.5" numba nuscenes-devkit plyfile scikit-image tensorboard trimesh terminaltables
$PY -m pip install -c $C --no-deps --no-build-isolation -e .
$PY -W ignore -c "
import numpy, torch, setuptools, mmengine, mmcv, mmdet, mmdet3d, mmcv._ext
from mmdet3d.utils import register_all_modules
register_all_modules(init_default_scope=True)
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
print('IS_SPCONV2_AVAILABLE =', IS_SPCONV2_AVAILABLE)
assert IS_SPCONV2_AVAILABLE, 'mmdet3d khong dung spconv2 -> checkpoint se khong nap duoc'
v = dict(numpy=numpy.__version__, torch=torch.__version__, setuptools=setuptools.__version__, mmengine=mmengine.__version__, mmcv=mmcv.__version__, mmdet=mmdet.__version__, mmdet3d=mmdet3d.__version__)
print(v)
assert v['numpy']=='1.26.4' and v['torch']=='2.1.0+cu121' and v['mmcv']=='2.1.0' and v['mmdet']=='3.3.0' and v['mmdet3d']=='1.4.0', v
print('MMDET3D GATE PASS')
"
$PY -m pip check || true
} 2>&1 | tee $W/logs/step_mmdet3d.log
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cd $W/mmdetection3d
LOG=$W/logs/step_ops.log
{
NVCC_MAJOR=$(nvcc --version | grep -oP 'release \K[0-9]+')
[ "$NVCC_MAJOR" = "12" ] || { echo "nvcc major=$NVCC_MAJOR != 12"; exit 1; }
export TORCH_CUDA_ARCH_LIST="7.5" FORCE_CUDA=1 MAX_JOBS=4
$PY -W ignore projects/BEVFusion/setup.py develop
} > $LOG 2>&1 || { grep -nE "error|Error|FAILED" $LOG | head -40; tail -40 $LOG; exit 1; }
tail -3 $LOG
$PY -W ignore -c "
import projects.BEVFusion.bevfusion
from mmdet3d.registry import MODELS
assert all(n in MODELS.module_dict for n in ['BEVFusion','DepthLSSTransform','TransFusionHead'])
print('BEVFUSION OPS GATE PASS')
" 2>&1 | tee -a $LOG
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cd $W/mmdetection3d
export PYTHONPATH=$PWD:${PYTHONPATH:-}
V=$(find /kaggle/input -maxdepth 5 -type d -name v1.0-mini | head -1)
[ -n "$V" ] || { echo "khong thay v1.0-mini trong /kaggle/input"; find /kaggle/input -maxdepth 3 | head -30; exit 1; }
ROOT=$(dirname "$V"); echo "nuScenes-mini root: $ROOT"
DST=$PWD/data/nuscenes; mkdir -p $DST
for d in samples sweeps maps v1.0-mini; do ln -sfn "$ROOT/$d" "$DST/$d"; done
ls -la $DST
{
$PY -W ignore - <<'EOF'
import sys, os, inspect, time
sys.path.insert(0, os.getcwd())
sys.argv = ['create_data.py', 'nuscenes', '--root-path', './data/nuscenes', '--out-dir', './data/nuscenes',
            '--extra-tag', 'nuscenes', '--version', 'v1.0-mini']
from mmengine.registry import init_default_scope
init_default_scope('mmdet3d')
import tools.create_data as cd
sig = inspect.signature(cd.nuscenes_data_prep)
kw = dict(root_path='./data/nuscenes', info_prefix='nuscenes', version='v1.0-mini', out_dir='./data/nuscenes', max_sweeps=10)
if 'dataset_name' in sig.parameters: kw['dataset_name'] = 'NuScenesDataset'
kw = {k: v for k, v in kw.items() if k in sig.parameters}
print('call kwargs:', kw)
t = time.time(); cd.nuscenes_data_prep(**kw); print('done in', round(time.time() - t, 1), 's')
import mmengine
for s in ['train', 'val']:
    d = mmengine.load(f'data/nuscenes/nuscenes_infos_{s}.pkl')
    n = [len(x.get('lidar_sweeps', []) or []) for x in d['data_list']]
    print(s, len(d['data_list']), d['metainfo'], 'sweeps min/max', min(n), max(n))
EOF
test -f data/nuscenes/nuscenes_infos_val.pkl && echo "CREATE_DATA PASS"
} 2>&1 | tee $W/logs/step_data.log
''')

RUN_TEMPLATE = r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cd $W/mmdetection3d
export PYTHONPATH=$PWD:${PYTHONPATH:-}
TAG=@TAG@
CFG=projects/BEVFusion/configs/@CFG@
CKPT=$W/checkpoints/$(basename @URL@)
[ -f "$CKPT" ] || wget -q @URL@ -O "$CKPT"
ls -la "$CKPT"
rm -rf $W/exports/$TAG $W/work_dirs/$TAG
$PY -W ignore tools/test.py $CFG $CKPT --work-dir $W/work_dirs/$TAG \
   --cfg-options test_evaluator.jsonfile_prefix=$W/exports/$TAG test_dataloader.dataset.metainfo.version=v1.0-mini 2>&1 | tee $W/logs/test_$TAG.log | grep -vE "^\s*$" | tail -60
mkdir -p $OUT/$TAG
if grep -q "size mismatch\|missing keys in source" $W/logs/test_$TAG.log; then echo "CHECKPOINT LOAD MISMATCH -> ket qua vo nghia"; grep -m5 "size mismatch\|missing keys" $W/logs/test_$TAG.log; fi
find $W/exports/$TAG -name "results_nusc.json" -exec cp {} $OUT/$TAG/ \;
find $W/exports/$TAG -name "metrics_summary.json" -exec cp {} $OUT/$TAG/ \;
cp $W/logs/test_$TAG.log $OUT/$TAG/
ls -la $OUT/$TAG
$PY - <<'EOF'
import json, glob
f = glob.glob('/kaggle/working/outputs/@TAG@/metrics_summary.json')
assert f, 'khong co metrics_summary.json -> inference/eval that bai'
m = json.load(open(f[0]))
print('@TAG@  mAP=%.4f  NDS=%.4f' % (m['mean_ap'], m['nd_score']))
log = open('/kaggle/working/outputs/@TAG@/test_@TAG@.log', errors='replace').read()
assert 'size mismatch' not in log, 'checkpoint size mismatch'
assert m['mean_ap'] > 0.2, 'mAP qua thap -> pipeline sai, khong phai PASS'
print('@TAG@ INFERENCE PASS')
EOF
'''


def run_cell(tag, cfg, url):
    bash(RUN_TEMPLATE.replace("@TAG@", tag).replace("@CFG@", cfg).replace("@URL@", url))


BASE = "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/"
run_cell("L3C_lidar_only", "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
         BASE + "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth")
run_cell("L3D_lidar_camera", "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
         BASE + "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth")

bash(r'''
source /kaggle/working/l3_env.sh
$PY - <<'EOF'
import json
res = {}
for tag in ['L3C_lidar_only', 'L3D_lidar_camera']:
    try:
        m = json.load(open(f'/kaggle/working/outputs/{tag}/metrics_summary.json'))
        tp = m['tp_errors']
        res[tag] = dict(mAP=m['mean_ap'], NDS=m['nd_score'], mATE=tp['trans_err'], mASE=tp['scale_err'], mAOE=tp['orient_err'], mAVE=tp['vel_err'])
    except Exception as e:
        res[tag] = f'FAILED: {e}'
json.dump(res, open('/kaggle/working/outputs/summary.json', 'w'), indent=2)
print(json.dumps(res, indent=2))
EOF
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open('bev-real-l3c.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump({
    "id": "canhquang/bev-real-l3c", "title": "bev_real_l3c", "code_file": "bev-real-l3c.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": "true",
    "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4", "enable_internet": "true",
    "dataset_sources": ["aadimator/nuscenes-mini"], "competition_sources": [], "kernel_sources": []
}, open('kernel-metadata.json', 'w'), indent=2)
print('cells:', len(cells))
