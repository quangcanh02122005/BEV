"""Stage E: Classical (project MATLAB pipeline) vs Learned on the SAME 81 nuScenes-mini VAL samples, same GT, same criteria.
All settings below are fixed a priori (before the first evaluation):
 GT evaluated : annotated objects with num_lidar_pts >= 5 inside the 100 x 100 m BEV (x,y in [-50,50]).  Other GT are 'don't care'.
 Matching     : Hungarian on ego-BEV centre distance, class-agnostic, gate = per-GT-class gate (car 2.5 m, truck 3.5, bus/trailer/constr. 4.0,
                pedestrian 1.5, motorcycle 2.0, bicycle 1.5, cone 1.0, barrier 1.5).  A prediction that matches only a 'don't care' GT is ignored.
 Learned preds: boxes with score >= 0.5 (0.3 / 0.7 also reported).  Classical preds: clusters accepted as vehicle/pedestrian (no score exists).
 Occupancy    : 200 x 200 cells, 0.5 m, 1-cell tolerance (the tolerance used by the project's own occupancy contract)."""
import csv, json, os, re, sys, zipfile
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, '..', 'exports', 'L3E')
RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
GATE = {'car': 2.5, 'truck': 3.5, 'bus': 4.0, 'trailer': 4.0, 'construction_vehicle': 4.0, 'pedestrian': 1.5, 'motorcycle': 2.0, 'bicycle': 1.5, 'traffic_cone': 1.0, 'barrier': 1.5}
GROUP = {'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle', 'construction_vehicle': 'vehicle', 'motorcycle': 'vehicle', 'bicycle': 'vehicle',
         'pedestrian': 'pedestrian', 'traffic_cone': 'static_small', 'barrier': 'static_small'}
N = 81; R = 100.0; RES_M = 0.5; NC = 200


def read_csv(p):
    return list(csv.DictReader(open(p)))


# ---------------- GT ----------------
gt_all = [dict(i=int(r['sample_idx']), c=r['class'], x=float(r['x']), y=float(r['y']), w=float(r['w']), l=float(r['l']), yaw=float(r['yaw']), nl=int(r['num_lidar_pts']))
          for r in read_csv(os.path.join(HERE, 'data', 'gt_visibility_ego.csv'))]
inb = lambda g: abs(g['x']) <= 50 and abs(g['y']) <= 50
gt_eval = [g for g in gt_all if inb(g) and g['nl'] >= 5]
print('GT boxes: all', len(gt_all), '| evaluated (in BEV, >=5 LiDAR pts):', len(gt_eval))


# ---------------- predictions ----------------
def learned_from_csv(path, thr):
    out = {i: [] for i in range(N)}
    for r in read_csv(path):
        s = float(r['score'])
        if s >= thr:
            out[int(r['sample_idx'])].append(dict(x=float(r['x']), y=float(r['y']), w=float(r['w']), l=float(r['l']), yaw=float(r['yaw']), c=r['class'], s=s))
    return out


def qR(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


TAB = os.path.join(E, 'tables')


def load_table(n):
    p = os.path.join(TAB, n + '.json')
    if os.path.exists(p):
        return json.load(open(p, encoding='utf-8'))
    with zipfile.ZipFile(p + '.zip') as z:
        return json.load(z.open(z.namelist()[0]))


_sd = load_table('sample_data'); _ego = {e['token']: e for e in load_table('ego_pose')}
_kf = {d['sample_token']: d for d in _sd if d['is_key_frame'] and '/LIDAR_TOP/' in d['filename']}
_tokens = [r['sample_token'] for r in read_csv(os.path.join(E, 'samples.csv'))]
_pose = {t: (np.array(_ego[_kf[t]['ego_pose_token']]['translation']), qR(_ego[_kf[t]['ego_pose_token']]['rotation'])) for t in _tokens}


def learned_from_json(path, thr):
    res = json.load(open(path))['results']; out = {i: [] for i in range(N)}
    for i, t in enumerate(_tokens):
        tr, Rm = _pose[t]
        for b in res[t]:
            if b['detection_score'] < thr:
                continue
            c = Rm.T @ (np.array(b['translation']) - tr); Rb = Rm.T @ qR(b['rotation'])
            out[i].append(dict(x=c[0], y=c[1], w=b['size'][0], l=b['size'][1], yaw=float(np.arctan2(Rb[1, 0], Rb[0, 0])), c=b['detection_name'], s=b['detection_score']))
    return out


def classical_preds(cond):
    cl = json.load(open(os.path.join(HERE, 'classical_out', f'classical_{cond}_clusters.json')))
    out = {}
    for i in range(N):
        items = cl[i] if i < len(cl) else []
        if isinstance(items, dict):
            items = [items]
        objs = []
        for c in items or []:
            if c.get('accepted') and c['class'] in ('vehicle', 'pedestrian'):
                cx, cy = c['center'][0], c['center'][1]
                objs.append(dict(x=cx, y=cy, w=c['width_m'], l=c['depth_m'], yaw=0.0, c=c['class'], s=1.0, corners=c['corners']))
        out[i] = objs
    return out


# ---------------- detection metrics ----------------
def match(preds, gt_e, gt_a):
    """returns tp, fp, ignored, matched GT set (eval subset), centre errors."""
    tp = fp = ign = 0; errs = []; hit = set(); per_group = {g: [0, 0] for g in ('vehicle', 'pedestrian', 'static_small')}; per_dist = {k: [0, 0] for k in ('0-20', '20-35', '35-50')}
    by_i_all = {}; by_i_eval = {}
    for k, g in enumerate(gt_a): by_i_all.setdefault(g['i'], []).append(k)
    ev_ids = {id(g) for g in gt_e}
    for i in range(N):
        P = preds.get(i, []); GA = [gt_a[k] for k in by_i_all.get(i, [])]
        if not P or not GA:
            fp += len(P); continue
        d = np.array([[np.hypot(p['x'] - g['x'], p['y'] - g['y']) / GATE[g['c']] for g in GA] for p in P])
        cost = np.where(d <= 1.0, d, 1e3); r, c = linear_sum_assignment(cost)
        matched = {a: b for a, b in zip(r, c) if cost[a, b] < 1e3}
        for a in range(len(P)):
            if a in matched:
                g = GA[matched[a]]
                if id(g) in ev_ids:
                    tp += 1; hit.add(id(g)); errs.append(np.hypot(P[a]['x'] - g['x'], P[a]['y'] - g['y']))
                else:
                    ign += 1
            else:
                fp += 1
    for g in gt_e:
        rng_ = np.hypot(g['x'], g['y']); dk = '0-20' if rng_ < 20 else ('20-35' if rng_ < 35 else '35-50')
        per_group[GROUP[g['c']]][1] += 1; per_dist[dk][1] += 1
        if id(g) in hit:
            per_group[GROUP[g['c']]][0] += 1; per_dist[dk][0] += 1
    prec = tp / max(1, tp + fp); rec = len(hit) / max(1, len(gt_e))
    return dict(tp=tp, fp=fp, ignored=ign, precision=round(prec, 3), recall=round(rec, 3), f1=round(2 * prec * rec / max(1e-9, prec + rec), 3), centre_err_m=round(float(np.mean(errs)) if errs else float('nan'), 3),
                n_pred=sum(len(v) for v in preds.values()), per_image=round(sum(len(v) for v in preds.values()) / N, 1),
                recall_by_group={k: round(v[0] / max(1, v[1]), 3) for k, v in per_group.items()}, recall_by_range={k: round(v[0] / max(1, v[1]), 3) for k, v in per_dist.items()})


# ---------------- occupancy ----------------
xs = -50 + (np.arange(NC) + 0.5) * RES_M; XX, YY = np.meshgrid(xs, xs, indexing='ij')   # rows = x, cols = y


def raster_boxes(boxes):
    m = np.zeros((NC, NC), bool)
    for b in boxes:
        dx, dy = XX - b['x'], YY - b['y']; c, s = np.cos(b['yaw']), np.sin(b['yaw'])
        u = c * dx + s * dy; v = -s * dx + c * dy
        m |= (np.abs(u) <= b['l'] / 2) & (np.abs(v) <= b['w'] / 2)
    return m


gt_by_i_all = {i: [] for i in range(N)}; gt_by_i_eval = {i: [] for i in range(N)}
for g in gt_all: gt_by_i_all[g['i']].append(g)
for g in gt_eval: gt_by_i_eval[g['i']].append(g)
GT_ALL = np.stack([raster_boxes(gt_by_i_all[i]) for i in range(N)]); GT_EVAL = np.stack([raster_boxes(gt_by_i_eval[i]) for i in range(N)])
DIL = np.ones((3, 3), bool)
dil = lambda a: np.stack([binary_dilation(x, DIL) for x in a])
GT_ALL_D = dil(GT_ALL)


def occ_metrics(P):
    Pd = dil(P)
    rec = (Pd & GT_EVAL).sum() / max(1, GT_EVAL.sum()); expl = (P & GT_ALL_D).sum() / max(1, P.sum())
    return dict(object_cell_recall=round(float(rec), 3), explained_by_labelled_objects=round(float(expl), 3), occupied_cells_per_frame=round(float(P.sum() / N), 0),
                occupied_area_m2_per_frame=round(float(P.sum() / N * RES_M ** 2), 0))


# ---------------- run ----------------
out = dict(setup=dict(gt_evaluated=len(gt_eval), gt_all=len(gt_all), n_samples=N), detection={}, occupancy={}, robustness={}, latency={})
sys_learned = {
    'Learned: official LiDAR-only (L3C)': ('csv', os.path.join(E, 'pred_lidar_only_ego.csv')),
    'Learned: official Camera+LiDAR (L3D)': ('csv', os.path.join(E, 'pred_lidar_camera_ego.csv')),
    'Learned: Camera+Radar CRN (pretrained)': ('csv', os.path.join(E, 'pred_camera_radar_ego.csv')),
    'Learned: fused CL+CR (official CL + CRN)': ('csv', os.path.join(HERE, '..', 'fusion', 'unified', 'unified_pretrained_ego.csv')),
    'Ours: LiDAR-only (Stage A, mini-trained)': ('json', os.path.join(HERE, '..', 'models', 'our_bevfusion_mini', 'predictions', 'our_lidar_mini_final_last_results_nusc.json')),
    'Ours: Camera+LiDAR (Stage B, mini-trained)': ('json', os.path.join(HERE, '..', 'models', 'our_bevfusion_mini', 'predictions', 'our_bevfusion_mini_final_last_results_nusc.json')),
}
cl_full = classical_preds('full')
out['detection']['Classical: project LiDAR pipeline (Level 1 code)'] = match(cl_full, gt_eval, gt_all)
for name, (kind, path) in sys_learned.items():
    for thr in (0.5, 0.3, 0.7):
        preds = learned_from_csv(path, thr) if kind == 'csv' else learned_from_json(path, thr)
        key = name if thr == 0.5 else f'{name} @score>={thr}'
        out['detection'][key] = match(preds, gt_eval, gt_all)
        if thr == 0.5:
            sys_learned[name] = (kind, path, preds)
print(f"{'system':60s} {'P':>5s} {'R':>5s} {'F1':>5s} {'n/img':>6s} {'veh R':>6s} {'ped R':>6s} {'cerr':>5s}")
for k, v in out['detection'].items():
    if '@' in k: continue
    print(f"{k:60s} {v['precision']:5.2f} {v['recall']:5.2f} {v['f1']:5.2f} {v['per_image']:6.1f} {v['recall_by_group']['vehicle']:6.2f} {v['recall_by_group']['pedestrian']:6.2f} {v['centre_err_m']:5.2f}")

mfull = loadmat(os.path.join(HERE, 'classical_out', 'classical_full.mat'))
occ_c = mfull['occ'].astype(bool); rhit = mfull['rhit'].astype(bool)
out['occupancy']['Classical LiDAR occupancy (project buildBevOccupancy)'] = occ_metrics(occ_c)
out['occupancy']['Classical Radar hits (project rasterizeRadarBev, 5 radars)'] = occ_metrics(rhit)
out['occupancy']['Classical LiDAR U Radar (plain union)'] = occ_metrics(occ_c | rhit)
for name, v in sys_learned.items():
    P = np.stack([raster_boxes(v[2][i]) for i in range(N)])
    out['occupancy'][name + ' (box footprints, score>=0.5)'] = occ_metrics(P)
for k, v in out['occupancy'].items():
    print(f"{k:72s} recall {v['object_cell_recall']:.2f} explained {v['explained_by_labelled_objects']:.2f} cells/frame {v['occupied_cells_per_frame']:.0f}")

# robustness (classical only)
for cond in ('full', 'keep50', 'keep25', 'range30'):
    m = loadmat(os.path.join(HERE, 'classical_out', f'classical_{cond}.mat'))
    d = match(classical_preds(cond), gt_eval, gt_all); o = occ_metrics(m['occ'].astype(bool))
    out['robustness'][cond] = dict(detection=dict(precision=d['precision'], recall=d['recall'], f1=d['f1'], per_image=d['per_image']), occupancy=o,
                                   lidar_path_s=round(float(np.nanmean(m['tLidar'])), 3))
    print(cond, out['robustness'][cond]['detection'], o['object_cell_recall'])

# latency
def last_time(path):
    t = [float(x) for x in re.findall(r'Epoch\(test\)[^\n]*?(?<![a-z_])time: ([0-9.]+)', open(path, errors='replace').read())]
    return (round(float(np.mean(t)), 3), len(t)) if t else None


out['latency'] = dict(
    classical_lidar_pipeline_s_per_frame_matlab_cpu=round(float(np.nanmean(mfull['tLidar'])), 3),
    classical_radar_rasterise_s_per_frame_matlab_cpu=round(float(np.nanmean(mfull['tRadar'])), 3),
    learned_official_lidar_only_s_per_frame_T4=last_time(os.path.join(HERE, '..', 'frozen', 'L3C_L3E_v1', 'l3c_l3d_outputs', 'L3C_lidar_only', 'test_L3C_lidar_only.log')),
    learned_official_camera_lidar_s_per_frame_T4=last_time(os.path.join(HERE, '..', 'frozen', 'L3C_L3E_v1', 'l3c_l3d_outputs', 'L3D_lidar_camera', 'test_L3D_lidar_camera.log')),
    note='different hardware (local CPU MATLAB with interpreted shims vs Kaggle T4 incl. data loading); indicative only. CRN latency not measured here.')
print(out['latency'])
json.dump(out, open(os.path.join(RES, 'stage_e_metrics.json'), 'w'), indent=2)

# ---------------- figures ----------------
def draw_box(ax, b, col, lw=1.2):
    c, s = np.cos(b['yaw']), np.sin(b['yaw']); cx = np.array([1, 1, -1, -1, 1]) * b['l'] / 2; cy = np.array([1, -1, -1, 1, 1]) * b['w'] / 2
    ax.plot(-(b['y'] + s * cx + c * cy), b['x'] + c * cx - s * cy, color=col, lw=lw)


fused = sys_learned['Learned: fused CL+CR (official CL + CRN)'][2]
for idx in (4, 30, 60):
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    titles = ['GT (green) + Classical LiDAR occupancy (grey) + radar hits (blue)', 'Classical clusters (orange) vs GT', 'Learned fused CL+CR (red, score>=0.5) vs GT']
    for a, t in zip(ax, titles):
        a.set_xlim(-50, 50); a.set_ylim(-50, 50); a.set_aspect('equal'); a.grid(alpha=.3); a.set_title(t, fontsize=10)
        for g in gt_by_i_all[idx]:
            draw_box(a, dict(x=g['x'], y=g['y'], w=g['w'], l=g['l'], yaw=g['yaw']), '#1fa31f')
        a.plot(0, 0, 'k^')
    ii, jj = np.where(occ_c[idx]); ax[0].scatter(-xs[jj], xs[ii], s=6, c='0.55', marker='s')
    ii, jj = np.where(rhit[idx]); ax[0].scatter(-xs[jj], xs[ii], s=6, c='tab:blue', marker='.')
    for b in cl_full[idx]:
        cc = np.array(b['corners']); cc = np.vstack([cc, cc[:1]]); ax[1].plot(-cc[:, 1], cc[:, 0], color='darkorange', lw=1.5)
    for b in fused[idx]:
        draw_box(ax[2], b, '#d62728', 1.5)
    fig.suptitle(f'Stage E | nuScenes-mini VAL sample {idx + 1}/81 | classical = project MATLAB code, learned = pretrained (contaminated)', fontsize=11)
    fig.savefig(os.path.join(RES, f'stage_e_compare_sample{idx + 1}.png'), dpi=70, bbox_inches='tight'); plt.close(fig)

names = ['Classical LiDAR'] + [k.replace('Learned: ', 'L: ').replace('Ours: ', 'Ours: ') for k in sys_learned]
keys = ['Classical: project LiDAR pipeline (Level 1 code)'] + list(sys_learned.keys())
fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27; xsb = np.arange(len(keys))
ax.bar(xsb - w, [out['detection'][k]['precision'] for k in keys], w, label='precision'); ax.bar(xsb, [out['detection'][k]['recall'] for k in keys], w, label='recall'); ax.bar(xsb + w, [out['detection'][k]['f1'] for k in keys], w, label='F1')
ax.set_xticks(xsb); ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=7, rotation=20); ax.legend(); ax.set_title('Object detection on the same 81 VAL samples (class-agnostic, centre gate; learned at score>=0.5)'); ax.grid(alpha=.3, axis='y')
fig.tight_layout(); fig.savefig(os.path.join(RES, 'stage_e_detection_bars.png'), dpi=90); plt.close(fig)
print('wrote', RES)
