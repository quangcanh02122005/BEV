"""Stage D: late (object-level) fusion of Camera+LiDAR and Camera+Radar detections. No learning; all constants are priors fixed before evaluation.

Input/Output: nuScenes detection-result json (global frame).  Association: Hungarian on BEV centre distance, same class only, class-dependent gate.
Matched pair : geometry/yaw/attributes weighted toward Camera+LiDAR, velocity = score-weighted mean, score = noisy-OR.
Unmatched    : kept, score multiplied by a penalty (a single-sensor detection is less trustworthy than an agreed one)."""
import argparse, json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = {'car': 2.5, 'truck': 3.5, 'bus': 4.0, 'trailer': 4.0, 'construction_vehicle': 4.0, 'pedestrian': 1.5, 'motorcycle': 2.0,
        'bicycle': 1.5, 'traffic_cone': 1.0, 'barrier': 1.5}
DEFAULTS = dict(s_min=0.05, w_a=0.7, w_b=0.3, pen_a=0.85, pen_b=0.6, vel='mean', max_boxes=500)


def fuse_sample(A, B, p):
    A = [b for b in A if b['detection_score'] >= p['s_min']]
    B = [b for b in B if b['detection_score'] >= p['s_min']]
    used_b, out = set(), []
    matched_a = {}
    for cls in GATE:
        ia = [i for i, b in enumerate(A) if b['detection_name'] == cls]
        ib = [j for j, b in enumerate(B) if b['detection_name'] == cls]
        if not ia or not ib:
            continue
        ca = np.array([A[i]['translation'][:2] for i in ia]); cb = np.array([B[j]['translation'][:2] for j in ib])
        d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) / GATE[cls]
        cost = np.where(d <= 1.0, d, 1e3)
        r, c = linear_sum_assignment(cost)
        for a, b in zip(r, c):
            if cost[a, b] < 1e3:
                matched_a[ia[a]] = ib[b]; used_b.add(ib[b])
    for i, a in enumerate(A):
        if i in matched_a:
            b = B[matched_a[i]]
            sa, sb = a['detection_score'], b['detection_score']
            wa, wb = p['w_a'] * sa, p['w_b'] * sb
            t = (wa * np.array(a['translation']) + wb * np.array(b['translation'])) / (wa + wb)
            if p['vel'] == 'mean':
                v = (sa * np.array(a['velocity']) + sb * np.array(b['velocity'])) / (sa + sb)
            elif p['vel'] == 'a':
                v = np.array(a['velocity'])
            else:
                v = np.array(b['velocity'])
            f = dict(a); f.update(translation=t.tolist(), velocity=v.tolist(), detection_score=float(1 - (1 - sa) * (1 - sb)))
            f['_src'] = 'both'; f['_sa'] = sa; f['_sb'] = sb
            out.append(f)
        else:
            f = dict(a); f['detection_score'] = a['detection_score'] * p['pen_a']; f['_src'] = 'cl'; f['_sa'] = a['detection_score']; f['_sb'] = 0.0
            out.append(f)
    for j, b in enumerate(B):
        if j not in used_b:
            f = dict(b); f['detection_score'] = b['detection_score'] * p['pen_b']; f['_src'] = 'cr'; f['_sa'] = 0.0; f['_sb'] = b['detection_score']
            out.append(f)
    out.sort(key=lambda x: -x['detection_score'])
    return out[:p['max_boxes']]


def fuse(resA, resB, **kw):
    p = dict(DEFAULTS); p.update(kw)
    assert set(resA['results']) == set(resB['results']), 'sample sets differ'
    res = {t: fuse_sample(resA['results'][t], resB['results'][t], p) for t in resA['results']}
    return res, p


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='Camera+LiDAR results json'); ap.add_argument('--b', required=True, help='Camera+Radar results json')
    ap.add_argument('--out', required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument('--' + k, type=type(v), default=v)
    a = ap.parse_args()
    A = json.load(open(a.a)); B = json.load(open(a.b))
    res, p = fuse(A, B, **{k: getattr(a, k) for k in DEFAULTS})
    meta = dict(use_camera=True, use_lidar=True, use_radar=True, use_map=False, use_external=False)
    sub = dict(meta=meta, results={t: [{k: v for k, v in b.items() if not k.startswith('_')} for b in bs] for t, bs in res.items()})
    json.dump(sub, open(a.out, 'w'))
    json.dump({t: [(b['_src'], b['_sa'], b['_sb']) for b in bs] for t, bs in res.items()}, open(a.out.replace('.json', '_sources.json'), 'w'))
    n = {s: sum(1 for bs in res.values() for b in bs if b['_src'] == s) for s in ('both', 'cl', 'cr')}
    print('fused', sum(len(v) for v in res.values()), 'boxes', n, 'params', p)
