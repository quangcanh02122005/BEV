import json, zipfile, csv, os
import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, 'tables')
RES = os.path.join(HERE, '..', '..', 'notebooks', 'bev-real-l3c', 'out', 'outputs')
MODELS = {'lidar_only': 'L3C_lidar_only', 'lidar_camera': 'L3D_lidar_camera'}


def load_table(name):
    p = os.path.join(TAB, name + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


GEN2DET = {
    'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer', 'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian', 'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone', 'movable_object.barrier': 'barrier'}

sample = {s['token']: s for s in load_table('sample')}
sd = {d['token']: d for d in load_table('sample_data')}
ego = {e['token']: e for e in load_table('ego_pose')}
cs = {c['token']: c for c in load_table('calibrated_sensor')}
scene = {s['token']: s for s in load_table('scene')}
inst = {i['token']: i for i in load_table('instance')}
cat = {c['token']: c['name'] for c in load_table('category')}
ann_by_sample = {}
for a in load_table('sample_annotation'):
    ann_by_sample.setdefault(a['sample_token'], []).append(a)

results = {m: json.load(open(os.path.join(RES, tag, 'results_nusc.json'))) for m, tag in MODELS.items()}
tokens = sorted(results['lidar_only']['results'].keys(),
                key=lambda t: (sample[t]['scene_token'], sample[t]['timestamp']))
tidx = {t: i for i, t in enumerate(tokens)}

lidar_kf = {d['sample_token']: d for d in sd.values() if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
pose = {}
rows_s = []
for t in tokens:
    s = sample[t]
    lid = lidar_kf[t]
    e = ego[lid['ego_pose_token']]
    R = quat_to_R(e['rotation'])
    pose[t] = (np.array(e['translation']), R)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    rows_s.append([tidx[t], t, scene[s['scene_token']]['name'], s['timestamp'] / 1e6,
                   e['translation'][0], e['translation'][1], e['translation'][2], yaw])


def to_ego(t, center, rot_q, vel=None):
    tr, R = pose[t]
    c = R.T @ (np.array(center) - tr)
    Rb = R.T @ quat_to_R(rot_q)
    yaw = float(np.arctan2(Rb[1, 0], Rb[0, 0]))
    v = None
    if vel is not None:
        v = (R.T @ np.array([vel[0], vel[1], 0.0]))[:2]
    return c, yaw, v


HDR = ['sample_idx', 'sample_token', 'class', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'score']


def pred_rows(model):
    out = []
    for t in tokens:
        for b in results[model]['results'][t]:
            c, yaw, v = to_ego(t, b['translation'], b['rotation'], b['velocity'])
            out.append([tidx[t], t, b['detection_name'], *c, *b['size'], yaw, v[0], v[1], b['detection_score']])
    return out


gt_rows = []
for t in tokens:
    for a in ann_by_sample.get(t, []):
        name = cat[inst[a['instance_token']]['category_token']]
        if name not in GEN2DET:
            continue
        c, yaw, _ = to_ego(t, a['translation'], a['rotation'])
        gt_rows.append([tidx[t], t, GEN2DET[name], *c, *a['size'], yaw, np.nan, np.nan, 1.0])


def write_csv(name, hdr, rows):
    with open(os.path.join(HERE, name), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)


def to_mat(rows):
    n = len(rows)
    a = np.array([[r[i] for i in (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] for r in rows], dtype=float).reshape(n, 11)
    return dict(
        sample_idx=a[:, 0] + 1,
        cls=np.array([r[2] for r in rows], dtype=object).reshape(-1, 1),
        x=a[:, 1], y=a[:, 2], z=a[:, 3], w=a[:, 4], l=a[:, 5], h=a[:, 6], yaw=a[:, 7], vx=a[:, 8], vy=a[:, 9],
        score=a[:, 10])


write_csv('samples.csv', ['sample_idx', 'sample_token', 'scene', 'timestamp_s', 'ego_x_global', 'ego_y_global',
                          'ego_z_global', 'ego_yaw_global'], rows_s)
write_csv('gt_ego.csv', HDR, gt_rows)
for m in MODELS:
    write_csv(f'pred_{m}_ego.csv', HDR, pred_rows(m))

mat = {'samples': dict(sample_idx=np.arange(1, len(tokens) + 1, dtype=float),
                       token=np.array(tokens, dtype=object).reshape(-1, 1),
                       scene=np.array([r[2] for r in rows_s], dtype=object).reshape(-1, 1),
                       timestamp_s=np.array([r[3] for r in rows_s]),
                       ego_global=np.array([[r[4], r[5], r[6], r[7]] for r in rows_s])),
       'gt': to_mat(gt_rows)}
for m in MODELS:
    mat['pred_' + m] = to_mat(pred_rows(m))
savemat(os.path.join(HERE, 'l3e_predictions.mat'), mat, do_compression=True)

print('samples', len(tokens), '| GT boxes', len(gt_rows))
gt = np.array([[r[3], r[4]] for r in gt_rows]); gi = np.array([r[0] for r in gt_rows]); gc = np.array([r[2] for r in gt_rows])
for m in MODELS:
    pr = pred_rows(m)
    p = [r for r in pr if r[-1] > 0.3]
    hit = 0
    for r in p:
        mk = (gi == r[0]) & (gc == r[2])
        if mk.any() and np.min(np.hypot(gt[mk, 0] - r[3], gt[mk, 1] - r[4])) < 2.0:
            hit += 1
    xy = np.array([[r[3], r[4]] for r in pr])
    print(f'{m}: {len(pr)} boxes | score>0.3: {len(p)} | matched to GT (same class, <2m, ego frame): {hit / max(1, len(p)):.1%}'
          f' | x range [{xy[:, 0].min():.0f},{xy[:, 0].max():.0f}] y range [{xy[:, 1].min():.0f},{xy[:, 1].max():.0f}]')
