import csv, glob, hashlib, json, os, shutil, subprocess, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, '..', 'notebooks')
A = os.path.join(NB, 'bev-real-train-a', 'out', 'outputs')
B = os.path.join(NB, 'bev-real-train-b', 'out', 'outputs')
FZ = os.path.join(NB, 'bev-real-freeze', 'out', 'outputs')
FROZEN = os.path.join(ROOT, '..', 'frozen', 'L3C_L3E_v1', 'l3c_l3d_outputs')
DST = os.path.join(ROOT, 'our_bevfusion_mini')
if os.path.exists(DST):
    shutil.rmtree(DST)
for d in ['config', 'training_logs/lidar_stageA', 'training_logs/fusion_stageB', 'training_curves', 'demo_images/stageA_lidar',
          'demo_images/stageB_lidar_camera', 'predictions', 'epoch_ckpts']:
    os.makedirs(os.path.join(DST, d), exist_ok=True)


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)


sa, sb = os.path.join(A, 'our_lidar_mini'), os.path.join(B, 'our_bevfusion_mini')
# checkpoints
cp(f'{sa}/our_lidar_mini.pth', f'{DST}/our_lidar_mini.pth')
cp(f'{sb}/our_bevfusion_mini.pth', f'{DST}/our_bevfusion_mini.pth')
cp(f'{sa}/our_lidar_mini_best_by_val_NDS.pth', f'{DST}/epoch_ckpts/our_lidar_mini_best_by_val_NDS.pth')
cp(f'{sb}/our_bevfusion_mini_best_by_val_NDS.pth', f'{DST}/epoch_ckpts/our_bevfusion_mini_best_by_val_NDS.pth')
for f in glob.glob(f'{sa}/epoch_ckpts/*.pth'):
    cp(f, f'{DST}/epoch_ckpts/stageA_lidar_{os.path.basename(f)}')
for f in glob.glob(f'{sb}/epoch_ckpts/*.pth'):
    cp(f, f'{DST}/epoch_ckpts/stageB_fusion_{os.path.basename(f)}')
# configs (ours + the exact upstream base configs they inherit from, mmdet3d v1.4.0)
NB_A, NB_B = os.path.join(NB, 'bev-real-train-a'), os.path.join(NB, 'bev-real-train-b')
cp(f'{NB_A}/our_lidar_mini.py', f'{DST}/config/our_lidar_mini.py')
cp(f'{NB_B}/our_bevfusion_mini.py', f'{DST}/config/our_bevfusion_mini.py')
cp(f'{A}/config/numpy_alias_patch.diff', f'{DST}/config/numpy_alias_patch.diff')
base = 'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v1.4.0/'
for rel in ['projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
            'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py',
            'configs/_base_/default_runtime.py']:
    urllib.request.urlretrieve(base + rel, f'{DST}/config/upstream_v1.4.0_{os.path.basename(rel)}')
cp(f'{NB_A}/train_driver.py', f'{DST}/config/train_driver_stageA.py')
cp(f'{NB_B}/train_driver.py', f'{DST}/config/train_driver_stageB.py')
# logs / curves / demo
for tag, src, out in (('A', sa, 'lidar_stageA'), ('B', sb, 'fusion_stageB')):
    for f in ['train_full.log', 'scalars.json', 'smoke_report.json', 'full_train_report.json']:
        cp(f'{src}/training_logs/{f}', f'{DST}/training_logs/{out}/{f}')
    cp(f'{A if tag == "A" else B}/logs/driver_stdout.log', f'{DST}/training_logs/{out}/driver_stdout.log')
    for f in glob.glob(f'{src}/training_curves/*.png'):
        cp(f, f'{DST}/training_curves/stage{tag}_{os.path.basename(f)}')
for f in glob.glob(f'{sa}/demo_images/*'):
    cp(f, f'{DST}/demo_images/stageA_lidar/{os.path.basename(f)}')
for f in glob.glob(f'{sb}/demo_images/*'):
    cp(f, f'{DST}/demo_images/stageB_lidar_camera/{os.path.basename(f)}')
for tag, src in (('our_lidar_mini', sa), ('our_bevfusion_mini', sb)):
    for k in ('final_last', 'final_best'):
        cp(f'{src}/predictions/{k}/results_nusc.json', f'{DST}/predictions/{tag}_{k}_results_nusc.json')
# environment freeze
cp(f'{FZ}/environment.txt', f'{DST}/environment.txt')
cp(f'{FZ}/SHA256SUMS.txt', f'{DST}/environment_archive_SHA256.txt')

# ---------------- metrics + comparison table ----------------
mA, mB = json.load(open(f'{sa}/metrics.json')), json.load(open(f'{sb}/metrics.json'))


def off(tag):
    m = json.load(open(f'{FROZEN}/{tag}/metrics_summary.json'))
    return dict(mAP=m['mean_ap'], NDS=m['nd_score'], mATE=m['tp_errors']['trans_err'], mASE=m['tp_errors']['scale_err'],
                mAOE=m['tp_errors']['orient_err'], mAVE=m['tp_errors']['vel_err'], mAAE=m['tp_errors']['attr_err'],
                per_class_AP_2m=m['mean_dist_aps'])


def rerun(tag):
    m = json.load(open(f'{FZ}/{tag}/metrics_summary.json'))
    return dict(mAP=m['mean_ap'], NDS=m['nd_score'])


rows = [
    ('Official pretrained LiDAR', 'official (trainval) - CONTAMINATED on mini VAL', off('L3C_lidar_only'), rerun('L3C_lidar_only')),
    ('Official pretrained LiDAR+Camera', 'official (trainval) - CONTAMINATED on mini VAL', off('L3D_lidar_camera'), rerun('L3D_lidar_camera')),
    ('Our trained LiDAR (last epoch, 15)', 'ours: mini TRAIN only, from scratch', mA['final_last_epoch'], None),
    ('Our trained LiDAR+Camera (last epoch, 7)', 'ours: LiDAR init from our LiDAR, Swin-T ImageNet init', mB['final_last_epoch'], None),
]
extra = [
    ('Our LiDAR (best-by-val epoch, selected on VAL -> optimistic)', 'ours', mA['best_by_val'], None),
    ('Our LiDAR+Camera (best-by-val epoch, selected on VAL -> optimistic)', 'ours', mB['best_by_val'], None),
]
CLS = ['car', 'truck', 'bus', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'trailer', 'construction_vehicle', 'barrier']
out = dict(
    note='nuScenes-mini VAL only (81 samples, 2 scenes). NOT a full nuScenes benchmark. Official checkpoints were trained on nuScenes trainval, '
         'which contains the mini scenes -> their mini-VAL numbers are contaminated (optimistic) and not comparable to ours.',
    rows=[dict(model=r[0], training=r[1], **{k: r[2][k] for k in ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE')},
               per_class_AP_2m=r[2]['per_class_AP_2m'], rerun_in_frozen_env=r[3]) for r in rows + extra],
    stageA=dict(epochs=mA['epochs'], train_secs=mA['train_secs'], iters_per_epoch=mA['iters_per_epoch'], params=mA['params'],
                per_epoch_val=mA['per_epoch_val'], initial_train_loss=mA['initial_train_loss'], final_train_loss=mA['final_train_loss']),
    stageB=dict(epochs=mB['epochs'], train_secs=mB['train_secs'], iters_per_epoch=mB['iters_per_epoch'], params=mB['params'],
                per_epoch_val=mB['per_epoch_val'], initial_train_loss=mB['initial_train_loss'], final_train_loss=mB['final_train_loss']))
json.dump(out, open(f'{DST}/metrics.json', 'w'), indent=2)

with open(f'{DST}/comparison_table.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'training', 'mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE'] + [f'AP@2m_{c}' for c in CLS])
    for r in out['rows']:
        w.writerow([r['model'], r['training'], *[round(r[k], 4) for k in ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE')],
                    *[round(r['per_class_AP_2m'][c], 3) for c in CLS]])

md = ['| Model | Training data | mAP | NDS | mATE | mASE | mAOE | mAVE |', '|---|---|---|---|---|---|---|---|']
for r in out['rows'][:4]:
    md.append(f"| {r['model']} | {r['training']} | {r['mAP']:.3f} | {r['NDS']:.3f} | {r['mATE']:.2f} | {r['mASE']:.2f} | {r['mAOE']:.2f} | {r['mAVE']:.2f} |")
md2 = ['| Model | ' + ' | '.join(CLS) + ' |', '|---|' + '---|' * len(CLS)]
for r in out['rows'][:4]:
    md2.append(f"| {r['model']} | " + ' | '.join(f"{r['per_class_AP_2m'][c]:.2f}" for c in CLS) + ' |')
open(f'{DST}/_table.md', 'w').write('\n'.join(md) + '\n\n' + '\n'.join(md2) + '\n')

# checksums
lines = []
for p in sorted(glob.glob(f'{DST}/**/*', recursive=True)):
    if os.path.isfile(p) and not p.endswith('SHA256SUMS.txt'):
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        lines.append(f'{h}  {os.path.relpath(p, DST)}')
open(f'{DST}/SHA256SUMS.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(md)); print(); print('\n'.join(md2))
print('files:', len(lines), '| size MB:', round(sum(os.path.getsize(p) for p in glob.glob(f"{DST}/**/*", recursive=True) if os.path.isfile(p)) / 1e6, 1))
