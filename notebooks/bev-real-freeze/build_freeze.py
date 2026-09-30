import json
nb = json.load(open('../../frozen/L3C_L3E_v1/notebook_v5/bev-real-l3c.ipynb', encoding='utf-8'))
cells = nb['cells']
assert len(cells) == 10
cells = cells[:9]   # drop the v5 summary cell; keep env/ops/data/L3C/L3D verbatim (frozen pipeline)


def bash(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": ("%%bash\n" + s.strip("\n")).splitlines(keepends=True)})


bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
$PY - <<'EOF'
import json
ref = {'L3C_lidar_only': (0.5431496439080831, 0.5625808309394464), 'L3D_lidar_camera': (0.5702214005656601, 0.5755269346010203)}
for k, (m0, n0) in ref.items():
    m = json.load(open(f'/kaggle/working/outputs/{k}/metrics_summary.json'))
    d1, d2 = abs(m['mean_ap'] - m0), abs(m['nd_score'] - n0)
    print(k, 'mAP', round(m['mean_ap'], 5), 'NDS', round(m['nd_score'], 5), '| delta', round(d1, 5), round(d2, 5))
    assert d1 < 5e-3 and d2 < 5e-3, 'REGRESSION: rebuilt env does not reproduce frozen L3C/L3D'
print('FREEZE REGRESSION PASS')
EOF
''')

bash(r'''
source /kaggle/working/l3_env.sh
{
echo "== date (UTC)"; date -u
echo "== host"; uname -a; head -3 /etc/os-release; python3 --version
echo "== GPU"; nvidia-smi; nvcc --version | tail -2
echo "== env python"; $PY --version
echo "== torch/cuda"; $PY -c "import torch,spconv;print('torch',torch.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'spconv',spconv.__version__)"
echo "== mmdetection3d git"; git -C $W/mmdetection3d log -1 --format='%H %d' ; git -C $W/mmdetection3d describe --tags 2>/dev/null || true
echo "== constraints"; cat $C
echo "== env vars"; echo "MPLBACKEND=$MPLBACKEND CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_HOME=${CUDA_HOME:-}"
echo "== conda list --export"; $W/miniconda3/bin/conda list -p $ENVDIR --export
echo "== pip freeze (bevfusion env)"; $PY -m pip freeze
echo "== pip check"; $PY -m pip check || true
} > $OUT/environment.txt 2>&1
wc -l $OUT/environment.txt; grep -E "^(torch|numpy|mmcv|mmdet|mmdet3d|mmengine|spconv|setuptools)==|^mmdet3d" $OUT/environment.txt | head -20
''')

bash(r'''
set -euo pipefail
source /kaggle/working/l3_env.sh
cd $W
if command -v zstd >/dev/null 2>&1; then CMP="zstd -T0 -3"; EXT=zst; else CMP="gzip -1"; EXT=gz; fi
echo "compressor: $CMP"
tar -cf - envs/bevfusion mmdetection3d constraints.txt logs | $CMP > /kaggle/working/env_bevfusion.tar.$EXT
ls -la /kaggle/working/env_bevfusion.tar.$EXT
sha256sum /kaggle/working/env_bevfusion.tar.$EXT | tee $OUT/SHA256SUMS.txt
if [ "$EXT" = zst ]; then zstd -t /kaggle/working/env_bevfusion.tar.zst && echo "archive integrity OK"; else gzip -t /kaggle/working/env_bevfusion.tar.gz && echo "archive integrity OK"; fi
echo "FREEZE PACK PASS"
''')

nb['cells'] = cells
json.dump(nb, open('bev-real-freeze.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump({"id": "canhquang/bev-real-freeze", "title": "bev_real_freeze", "code_file": "bev-real-freeze.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4", "enable_internet": "true",
           "dataset_sources": ["aadimator/nuscenes-mini"], "competition_sources": [], "kernel_sources": []},
          open('kernel-metadata.json', 'w'), indent=2)
print('cells', len(cells))
