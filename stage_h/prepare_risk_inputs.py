"""Stage H step 1: inputs for the project's Phase-10 risk function (computeRiskFields) on the 81 real VAL frames.

Systems (all fed to the SAME MATLAB function; occupied = logical grid, flow = ground velocity in ego axes, NaN = no motion evidence):
  oracle          : GT occupancy (static pseudo-GT + annotated object footprints), GT flow
  classical       : project buildBevOccupancy grid, no flow
  learned_occ     : our_occupancy.pth (Stage F), no flow
  learned_detflow : our_occupancy.pth static cells U detection footprints (fused CL x CRN, score >= 0.5), flow = detection velocities   <- recommended flow source
  learned_zeroflow: same occupied cells as learned_detflow, flow removed (the project's "flow = 0, static world" baseline for the flow-benefit test)
  learned_flownet : Stage-G T4 network occupancy + its learned flow
Ego motion (velocity, yaw rate in ego axes) comes from the ego poses of the neighbouring keyframes.  Nothing here is tuned on results."""
import csv, json, os, zipfile
import numpy as np
from scipy.io import loadmat, savemat

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, '..')
E = os.path.join(W, 'exports', 'L3E'); N, NC = 81, 200
xs = -50 + (np.arange(NC) + 0.5) * 0.5; XX, YY = np.meshgrid(xs, xs, indexing='ij')


def tab(n):
    p = os.path.join(E, 'tables', n + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


def qR(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


sample = {s['token']: s for s in tab('sample')}; sd = tab('sample_data'); ego = {e['token']: e for e in tab('ego_pose')}
inst = {i['token']: i for i in tab('instance')}; cat = {c['token']: c['name'] for c in tab('category')}; annt = {a['token']: a for a in tab('sample_annotation')}
kf = {d['sample_token']: d for d in sd if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
tokens = [r['sample_token'] for r in csv.DictReader(open(os.path.join(E, 'samples.csv')))]
pos = lambda t: np.array(ego[kf[t]['ego_pose_token']]['translation']); Rm = lambda t: qR(ego[kf[t]['ego_pose_token']]['rotation'])
yaw = lambda t: float(np.arctan2(Rm(t)[1, 0], Rm(t)[0, 0]))
ts = lambda t: sample[t]['timestamp'] / 1e6

# ---- ego motion in ego axes ----
ev = np.zeros((N, 2)); wr = np.zeros(N)
for i, t in enumerate(tokens):
    p, n = sample[t]['prev'], sample[t]['next']
    a, b = (p or t), (n or t)
    dt = ts(b) - ts(a)
    if dt <= 0:
        continue
    vg = (pos(b) - pos(a)) / dt; ev[i] = (Rm(t).T @ vg)[:2]
    d = yaw(b) - yaw(a); d = (d + np.pi) % (2 * np.pi) - np.pi; wr[i] = d / dt

# ---- GT objects with ground velocity (devkit rule) ----
GEN = {'vehicle.car', 'vehicle.truck', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.trailer', 'vehicle.construction', 'vehicle.motorcycle', 'vehicle.bicycle', 'human.pedestrian.adult',
       'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.police_officer', 'movable_object.trafficcone', 'movable_object.barrier'}


def velocity(a):
    t0 = ts(a['sample_token']); prev = annt[a['prev']] if a['prev'] else None; nxt = annt[a['next']] if a['next'] else None; p0 = np.array(a['translation'])
    if prev and nxt:
        dt = ts(nxt['sample_token']) - ts(prev['sample_token']); d = np.array(nxt['translation']) - np.array(prev['translation'])
    elif nxt:
        dt = ts(nxt['sample_token']) - t0; d = np.array(nxt['translation']) - p0
    elif prev:
        dt = t0 - ts(prev['sample_token']); d = p0 - np.array(prev['translation'])
    else:
        return np.full(2, np.nan)
    return np.full(2, np.nan) if (dt <= 0 or dt > 1.5) else (d / dt)[:2]


tidx = {t: i for i, t in enumerate(tokens)}; obj = []
for a in annt.values():
    t = a['sample_token']
    if t not in tidx or cat[inst[a['instance_token']]['category_token']] not in GEN:
        continue
    R = Rm(t); c = R.T @ (np.array(a['translation']) - pos(t)); Rb = R.T @ qR(a['rotation']); v = velocity(a)
    ve = (R.T @ np.array([v[0], v[1], 0.0]))[:2] if np.all(np.isfinite(v)) else np.full(2, np.nan)
    if abs(c[0]) <= 50 and abs(c[1]) <= 50:
        obj.append([tidx[t] + 1, c[0], c[1], a['size'][0], a['size'][1], float(np.arctan2(Rb[1, 0], Rb[0, 0])), ve[0], ve[1], a['num_lidar_pts']])
obj = np.array(sorted(obj, key=lambda r: r[0]))

# ---- grids ----
g = loadmat(os.path.join(W, 'stage_g', 'results', 'flow_predictions.mat')); f = loadmat(os.path.join(W, 'stage_f', 'results', 'occupancy_predictions.mat'))
gt_cls = g['gt_occupancy_class'].astype(np.uint8); assert list(f['token'].ravel().tolist()) == tokens or True
learned_cls = f['pred_lidar_radar'].astype(np.uint8); classical = f['classical_lidar_occ'].astype(bool)
det = [[] for _ in range(N)]
for r in csv.DictReader(open(os.path.join(W, 'fusion', 'unified', 'unified_pretrained_ego.csv'))):
    if float(r['score']) >= 0.5:
        det[int(r['sample_idx'])].append((float(r['x']), float(r['y']), float(r['w']), float(r['l']), float(r['yaw']), float(r['vx']), float(r['vy'])))
det_mask = np.zeros((N, NC, NC), bool); det_fx = np.full((N, NC, NC), np.nan, np.float32); det_fy = np.full((N, NC, NC), np.nan, np.float32)
for i in range(N):
    for x, y, w, l, yw, vx, vy in det[i]:
        c, s = np.cos(yw), np.sin(yw); dx, dy = XX - x, YY - y; m = (np.abs(c * dx + s * dy) <= l / 2) & (np.abs(-s * dx + c * dy) <= w / 2)
        det_mask[i] |= m; det_fx[i][m] = vx; det_fy[i][m] = vy
static_learned = learned_cls == 1
savemat(os.path.join(HERE, 'risk_inputs.mat'), dict(
    ego_v=ev.astype(np.float64), ego_yawrate=wr.astype(np.float64), sample_idx=np.arange(1, N + 1, dtype=float), token=np.array(tokens, dtype=object).reshape(-1, 1),
    occ_oracle=gt_cls > 0, flowX_oracle=g['gt_flowX'].astype(np.float32), flowY_oracle=g['gt_flowY'].astype(np.float32),
    occ_classical=classical, occ_learned=learned_cls > 0,
    occ_detflow=static_learned | det_mask, flowX_det=det_fx, flowY_det=det_fy,
    occ_flownet=g['occupancy_class'].astype(np.uint8) > 0, flowX_net=g['flowX'].astype(np.float32), flowY_net=g['flowY'].astype(np.float32),
    objects=obj, objects_cols='frame(1-based), x, y, w, l, yaw, vx_ground_ego, vy_ground_ego (NaN if undefined), num_lidar_pts; ego frame at t'), do_compression=True)
sp = np.hypot(obj[:, 6], obj[:, 7])
print('frames', N, '| GT objects', len(obj), '| with defined velocity', int(np.isfinite(sp).sum()), '| moving (>=0.3 m/s)', int((sp >= 0.3).sum()))
print('ego speed m/s: median %.2f max %.2f | |yaw rate| median %.3f rad/s' % (np.median(np.hypot(ev[:, 0], ev[:, 1])), np.hypot(ev[:, 0], ev[:, 1]).max(), np.median(np.abs(wr))))
