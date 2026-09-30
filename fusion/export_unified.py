"""Convert late-fusion submissions (global frame) to the unified ego-frame object list (+ MATLAB .mat) and print a quick GT check."""
import csv, json, os, sys, zipfile
import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, '..', 'exports', 'L3E')
TAB = os.path.join(E, 'tables')
SUB = os.path.join(HERE, '..', 'notebooks', 'bev-real-fusion-eval', 'out', 'outputs', 'fusion_eval', 'submissions')
OUTD = os.path.join(HERE, 'unified')
os.makedirs(OUTD, exist_ok=True)
GATE = {'car': 2.5, 'truck': 3.5, 'bus': 4.0, 'trailer': 4.0, 'construction_vehicle': 4.0, 'pedestrian': 1.5, 'motorcycle': 2.0, 'bicycle': 1.5, 'traffic_cone': 1.0, 'barrier': 1.5}


def load_table(n):
    p = os.path.join(TAB, n + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


def qR(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


sd = load_table('sample_data'); ego = {e['token']: e for e in load_table('ego_pose')}
lidar_kf = {d['sample_token']: d for d in sd if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
tokens = [r['sample_token'] for r in csv.DictReader(open(os.path.join(E, 'samples.csv')))]
tidx = {t: i for i, t in enumerate(tokens)}
pose = {t: (np.array(ego[lidar_kf[t]['ego_pose_token']]['translation']), qR(ego[lidar_kf[t]['ego_pose_token']]['rotation'])) for t in tokens}
G = list(csv.DictReader(open(os.path.join(E, 'gt_ego.csv'))))
gxy = np.array([[float(g['x']), float(g['y'])] for g in G]); gi = np.array([int(g['sample_idx']) for g in G]); gc = np.array([g['class'] for g in G])

HDR = ['sample_idx', 'sample_token', 'class', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'score', 'source', 'score_cl', 'score_cr']


def to_rows(sub, src):
    rows = []
    for t in tokens:
        tr, R = pose[t]
        for b, (s, sa, sb) in zip(sub['results'][t], src[t]):
            c = R.T @ (np.array(b['translation']) - tr)
            Rb = R.T @ qR(b['rotation'])
            v = (R.T @ np.array([b['velocity'][0], b['velocity'][1], 0.0]))[:2]
            rows.append([tidx[t], t, b['detection_name'], *c, *b['size'], float(np.arctan2(Rb[1, 0], Rb[0, 0])), v[0], v[1], b['detection_score'], s, sa, sb])
    return rows


def gt_check(rows, thr=0.3):
    p = [r for r in rows if r[12] > thr]
    hit = 0; matched = set()
    for r in sorted(p, key=lambda r: -r[12]):
        mk = np.where((gi == r[0]) & (gc == r[2]))[0]
        if len(mk):
            d = np.hypot(gxy[mk, 0] - r[3], gxy[mk, 1] - r[4]); k = int(np.argmin(d))
            if d[k] < GATE[r[2]] and mk[k] not in matched:
                hit += 1; matched.add(mk[k])
    return len(p), hit / max(1, len(p)), len(matched) / len(gi)


def clean_rows(res, thr):
    """apply the same greedy check for single-branch reference systems."""
    return res


def single_rows(path):
    sub = json.load(open(path))
    return to_rows(sub, {t: [('cl' if 'L3D' in path or 'OURB' in path else 'cr', b['detection_score'], 0.0) for b in sub['results'][t]] for t in tokens})


ALL = {}
SYSTEMS = {'pretrained': 'FUSED_official_baseline', 'ours': 'FUSED_ours_baseline'}
report = {}
for name, stem in SYSTEMS.items():
    sub = json.load(open(os.path.join(SUB, stem + '.json'))); src = json.load(open(os.path.join(SUB, stem + '_sources.json')))
    rows = to_rows(sub, src); ALL[name] = rows
    with open(os.path.join(OUTD, f'unified_{name}_ego.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)
    a = np.array([[r[0], *r[3:13], r[14], r[15]] for r in rows], dtype=float)
    code = {'both': 3, 'cl': 1, 'cr': 2}
    mat = dict(sample_idx=a[:, 0] + 1, cls=np.array([r[2] for r in rows], dtype=object).reshape(-1, 1), x=a[:, 1], y=a[:, 2], z=a[:, 3], w=a[:, 4], l=a[:, 5], h=a[:, 6],
               yaw=a[:, 7], vx=a[:, 8], vy=a[:, 9], score=a[:, 10], source_code=np.array([code[r[13]] for r in rows], dtype=float), score_cl=a[:, 11], score_cr=a[:, 12])
    savemat(os.path.join(OUTD, f'unified_{name}.mat'), dict(unified=mat, source_legend='1=Camera+LiDAR only, 2=Camera+Radar only, 3=both sensors agree'), do_compression=True)
    n, prec, rec = gt_check(rows)
    srcs = {s: sum(1 for r in rows if r[12] > 0.3 and r[13] == s) for s in ('both', 'cl', 'cr')}
    report[name] = dict(rows=len(rows), objects_score_gt_0_3=n, per_sample=round(n / len(tokens), 1), precision_vs_GT=round(prec, 3), recall_vs_GT=round(rec, 3), by_source_score_gt_0_3=srcs)
    print(name, json.dumps(report[name]))

# reference single-branch systems with the same quick check
ref = {'CL_official': os.path.join(HERE, '..', 'frozen', 'L3C_L3E_v1', 'l3c_l3d_outputs', 'L3D_lidar_camera', 'results_nusc.json'),
       'CL_ours': os.path.join(HERE, '..', 'models', 'our_bevfusion_mini', 'predictions', 'our_bevfusion_mini_final_last_results_nusc.json'),
       'CR_crn': os.path.join(HERE, '..', 'models', 'camera_radar', 'crn_r18_pretrained', 'results_nusc.json')}
for k, p in ref.items():
    n, prec, rec = gt_check(single_rows(p.replace('L3D', 'L3D') if 'CR' not in k else p) if 'CR' not in k else to_rows(json.load(open(p)), {t: [('cr', b['detection_score'], b['detection_score']) for b in json.load(open(p))['results'][t]] for t in tokens}))
    ALL[k] = None
    report[k] = dict(objects_score_gt_0_3=n, per_sample=round(n / len(tokens), 1), precision_vs_GT=round(prec, 3), recall_vs_GT=round(rec, 3))
    print(k, json.dumps(report[k]))
json.dump(report, open(os.path.join(OUTD, 'unified_gt_check.json'), 'w'), indent=2)
