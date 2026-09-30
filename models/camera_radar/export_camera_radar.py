"""Stage C export: CRN Camera+Radar predictions (nuScenes global frame) -> common ego format (same convention as L3E)."""
import csv, json, os, sys, zipfile
import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, 'tables')
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'notebooks', 'bev-real-cr-infer', 'out', 'outputs', 'crn_r18_pretrained', 'results_nusc.json')


def load_table(name):
    p = os.path.join(TAB, name + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


def qR(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


GEN2DET = {'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.trailer': 'trailer',
           'vehicle.construction': 'construction_vehicle', 'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
           'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
           'human.pedestrian.police_officer': 'pedestrian', 'movable_object.trafficcone': 'traffic_cone', 'movable_object.barrier': 'barrier'}

sample = {s['token']: s for s in load_table('sample')}
sd = load_table('sample_data')
ego = {e['token']: e for e in load_table('ego_pose')}
lidar_kf = {d['sample_token']: d for d in sd if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
# keep the SAME sample_idx as the existing L3E export (samples.csv)
rows = list(csv.DictReader(open(os.path.join(HERE, 'samples.csv'))))
tokens = [r['sample_token'] for r in rows]
tidx = {t: i for i, t in enumerate(tokens)}
pose = {t: (np.array(ego[lidar_kf[t]['ego_pose_token']]['translation']), qR(ego[lidar_kf[t]['ego_pose_token']]['rotation'])) for t in tokens}

res = json.load(open(SRC))
assert set(res['results'].keys()) == set(tokens), 'CRN results cover different samples than L3E'
HDR = ['sample_idx', 'sample_token', 'class', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'score']
out = []
for t in tokens:
    tr, R = pose[t]
    for b in res['results'][t]:
        c = R.T @ (np.array(b['translation']) - tr)
        Rb = R.T @ qR(b['rotation'])
        yaw = float(np.arctan2(Rb[1, 0], Rb[0, 0]))
        v = (R.T @ np.array([b['velocity'][0], b['velocity'][1], 0.0]))[:2]
        out.append([tidx[t], t, b['detection_name'], *c, *b['size'], yaw, v[0], v[1], b['detection_score']])

with open(os.path.join(HERE, 'pred_camera_radar_ego.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(HDR); w.writerows(out)

a = np.array([[r[i] for i in (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] for r in out], dtype=float)
gt = {k: v for k, v in __import__('scipy.io', fromlist=['loadmat']).loadmat(os.path.join(HERE, 'l3e_predictions.mat'), squeeze_me=False).items() if not k.startswith('__')}
mat = dict(pred_camera_radar=dict(sample_idx=a[:, 0] + 1, cls=np.array([r[2] for r in out], dtype=object).reshape(-1, 1), x=a[:, 1], y=a[:, 2], z=a[:, 3],
                                  w=a[:, 4], l=a[:, 5], h=a[:, 6], yaw=a[:, 7], vx=a[:, 8], vy=a[:, 9], score=a[:, 10]),
           samples=gt['samples'], gt=gt['gt'])
savemat(os.path.join(HERE, 'camera_radar_predictions.mat'), mat, do_compression=True)

# --- validation vs GT in the ego frame (same check as the LiDAR/Camera exports) ---
G = list(csv.DictReader(open(os.path.join(HERE, 'gt_ego.csv'))))
gxy = np.array([[float(g['x']), float(g['y'])] for g in G]); gi = np.array([int(g['sample_idx']) for g in G]); gc = np.array([g['class'] for g in G])
for name, rr in (('camera_radar', out),):
    p = [r for r in rr if r[-1] > 0.3]
    hit = 0
    for r in p:
        mk = (gi == r[0]) & (gc == r[2])
        if mk.any() and np.min(np.hypot(gxy[mk, 0] - r[3], gxy[mk, 1] - r[4])) < 2.0:
            hit += 1
    xy = np.array([[r[3], r[4]] for r in rr if r[-1] > 0.05])
    sp = np.array([np.hypot(r[10], r[11]) for r in p])
    print(f'{name}: {len(rr)} boxes | score>0.3: {len(p)} ({len(p) / len(tokens):.1f}/sample) | matched to GT (same class, <2 m, ego frame): {hit / max(1, len(p)):.1%} '
          f'| x range [{xy[:, 0].min():.0f},{xy[:, 0].max():.0f}] y range [{xy[:, 1].min():.0f},{xy[:, 1].max():.0f}] | speed(score>0.3) median {np.median(sp):.2f} p95 {np.percentile(sp, 95):.1f} m/s')
print('wrote pred_camera_radar_ego.csv and camera_radar_predictions.mat |', len(out), 'rows')
