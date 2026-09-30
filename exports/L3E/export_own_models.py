"""Export OUR trained models (Stage A LiDAR-only, Stage B Camera+LiDAR) to the common ego format (same convention and sample_idx as the other L3E exports)."""
import csv, json, os, zipfile
import numpy as np
from scipy.io import loadmat, savemat

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, '..', '..')
SRC = {'our_lidar': os.path.join(W, 'models', 'lidar_only', 'predictions', 'final_last_results_nusc.json'),
       'our_bevfusion': os.path.join(W, 'models', 'camera_lidar', 'predictions', 'final_last_results_nusc.json')}


def tab(n):
    p = os.path.join(HERE, 'tables', n + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


def qR(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


sd = tab('sample_data'); ego = {e['token']: e for e in tab('ego_pose')}; kf = {d['sample_token']: d for d in sd if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
tokens = [r['sample_token'] for r in csv.DictReader(open(os.path.join(HERE, 'samples.csv')))]; tidx = {t: i for i, t in enumerate(tokens)}
pose = {t: (np.array(ego[kf[t]['ego_pose_token']]['translation']), qR(ego[kf[t]['ego_pose_token']]['rotation'])) for t in tokens}
HDR = ['sample_idx', 'sample_token', 'class', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'score']
lm = loadmat(os.path.join(HERE, 'l3e_predictions.mat'))
mat = dict(samples=lm['samples'], gt=lm['gt'])
for name, path in SRC.items():
    res = json.load(open(path))['results']; assert set(res) == set(tokens); rows = []
    for t in tokens:
        tr, R = pose[t]
        for b in res[t]:
            c = R.T @ (np.array(b['translation']) - tr); Rb = R.T @ qR(b['rotation']); v = (R.T @ np.array([b['velocity'][0], b['velocity'][1], 0.0]))[:2]
            rows.append([tidx[t], t, b['detection_name'], *c, *b['size'], float(np.arctan2(Rb[1, 0], Rb[0, 0])), v[0], v[1], b['detection_score']])
    with open(os.path.join(HERE, f'pred_{name}_ego.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)
    a = np.array([[r[i] for i in (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] for r in rows], dtype=float)
    mat['pred_' + name] = dict(sample_idx=a[:, 0] + 1, cls=np.array([r[2] for r in rows], dtype=object).reshape(-1, 1), x=a[:, 1], y=a[:, 2], z=a[:, 3], w=a[:, 4], l=a[:, 5], h=a[:, 6], yaw=a[:, 7], vx=a[:, 8], vy=a[:, 9], score=a[:, 10])
    xy = a[a[:, 10] > 0.3]
    print(f'{name}: {len(rows)} boxes, {len(xy)} with score > 0.3, x range [{a[:, 1].min():.0f},{a[:, 1].max():.0f}] y range [{a[:, 2].min():.0f},{a[:, 2].max():.0f}]')
savemat(os.path.join(HERE, 'own_models_predictions.mat'), mat, do_compression=True)
print('wrote pred_our_lidar_ego.csv, pred_our_bevfusion_ego.csv, own_models_predictions.mat')
