"""Stage E step 1: extract the 81 mini-VAL LiDAR/radar keyframes from the local nuScenes-mini archive and write inputs for the MATLAB classical pipeline."""
import csv, json, os, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = r'D:\Vinuni\endgame\cam_lidar_radar\docs\archive.zip'
E = os.path.join(HERE, '..', 'exports', 'L3E')
OUT = os.path.join(HERE, 'data')
RADARS = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

z = zipfile.ZipFile(ARCHIVE)


def tab(n):
    return json.loads(z.read(f'v1.0-mini/{n}.json'))


sd = tab('sample_data'); cs = {c['token']: c for c in tab('calibrated_sensor')}; ego = {e['token']: e for e in tab('ego_pose')}
sample = {s['token']: s for s in tab('sample')}
kf = {}
for d in sd:
    if d['is_key_frame']:
        kf[(d['sample_token'], d['filename'].split('/')[1])] = d


def qR(q):
    w, x, y, zz = q
    return np.array([[1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)], [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)],
                     [2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)]])


rows = list(csv.DictReader(open(os.path.join(E, 'samples.csv'))))
inputs = []
for r in rows:
    t = r['sample_token']
    rec = dict(sample_idx=int(r['sample_idx']), token=t, scene=r['scene'], timestamp=sample[t]['timestamp'] / 1e6)
    for ch, key in [('LIDAR_TOP', 'lidar')] + [(c, c) for c in RADARS]:
        d = kf[(t, ch)]
        dst = os.path.join(OUT, d['filename'])
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, 'wb').write(z.read(d['filename']))
        c = cs[d['calibrated_sensor_token']]
        rec[key] = dict(file=d['filename'], R=qR(c['rotation']).flatten(order='F').tolist(), t=c['translation'], timestamp=d['timestamp'] / 1e6)
    inputs.append(rec)
json.dump(inputs, open(os.path.join(OUT, 'stage_e_inputs.json'), 'w'))
# GT (ego frame) with visibility info for a fair evaluation: keep annotation point counts
ann = tab('sample_annotation'); inst = {i['token']: i for i in tab('instance')}; cat = {c['token']: c['name'] for c in tab('category')}
GEN2DET = {'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.trailer': 'trailer', 'vehicle.construction': 'construction_vehicle',
           'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle', 'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
           'human.pedestrian.construction_worker': 'pedestrian', 'human.pedestrian.police_officer': 'pedestrian', 'movable_object.trafficcone': 'traffic_cone', 'movable_object.barrier': 'barrier'}
tok2idx = {r['sample_token']: int(r['sample_idx']) for r in rows}
pose = {}
for r in rows:
    d = kf[(r['sample_token'], 'LIDAR_TOP')]
    e = ego[d['ego_pose_token']]
    pose[r['sample_token']] = (np.array(e['translation']), qR(e['rotation']))
with open(os.path.join(OUT, 'gt_visibility_ego.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['sample_idx', 'class', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'num_lidar_pts', 'num_radar_pts', 'visibility'])
    n = 0
    for a in ann:
        t = a['sample_token']
        if t not in tok2idx:
            continue
        name = cat[inst[a['instance_token']]['category_token']]
        if name not in GEN2DET:
            continue
        tr, R = pose[t]
        c = R.T @ (np.array(a['translation']) - tr); Rb = R.T @ qR(a['rotation'])
        w.writerow([tok2idx[t], GEN2DET[name], *c, *a['size'], float(np.arctan2(Rb[1, 0], Rb[0, 0])), a['num_lidar_pts'], a['num_radar_pts'], a['visibility_token']]); n += 1
print('samples', len(inputs), '| GT boxes', n, '| files extracted under', OUT)

# ---- radar rows (devkit default filter: invalid_state==0 and ambig_state==3), sensor frame, for the MATLAB radar rasteriser ----
TY = {('F', 4): '<f4', ('F', 8): '<f8', ('I', 1): 'i1', ('I', 2): '<i2', ('I', 4): '<i4', ('U', 1): 'u1', ('U', 2): '<u2', ('U', 4): '<u4'}


def read_pcd(p):
    with open(p, 'rb') as f:
        h = {}
        while True:
            k, *v = f.readline().decode().strip().split(); h[k] = v
            if k == 'DATA':
                break
        dt = np.dtype([(n, TY[(t, int(s))]) for n, t, s in zip(h['FIELDS'], h['TYPE'], h['SIZE'])])
        return np.frombuffer(f.read(), dtype=dt, count=int(h['POINTS'][0]))


radar = []
tot = {c: 0 for c in RADARS}; raw = {c: 0 for c in RADARS}
for rec in inputs:
    per = {}
    for c in RADARS:
        a = read_pcd(os.path.join(OUT, rec[c]['file']))
        raw[c] += len(a)
        m = (a['invalid_state'] == 0) & (a['ambig_state'] == 3)
        a = a[m]; tot[c] += len(a)
        per[c] = np.stack([a['x'], a['y'], a['z'], a['vx'], a['vy']], 1).astype(float).round(4).tolist()
    radar.append(per)
json.dump(radar, open(os.path.join(OUT, 'stage_e_radar.json'), 'w'))
print('radar points kept/raw per radar:', {c: (tot[c], raw[c]) for c in RADARS})
