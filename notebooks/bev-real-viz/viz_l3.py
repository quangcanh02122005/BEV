"""Visualize REAL nuScenes-mini val samples: 6 cameras + 3D pred/GT boxes + LiDAR BEV, for two official models.
No nuscenes-devkit needed: reads the json tables directly. Usage: see argparse below."""
import argparse, csv, json, os, sys, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='nuScenes-mini root (has samples/ sweeps/ v1.0-mini/)')
ap.add_argument('--tables', default=None, help='dir with json tables (default: <root>/v1.0-mini)')
ap.add_argument('--res_lidar', required=True)
ap.add_argument('--res_cam', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--thr', type=float, default=0.3)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--mock', action='store_true', help='local code-path test: fake images/points if files are missing')
ap.add_argument('--dpi', type=int, default=75)
args = ap.parse_args()
TAB = args.tables or os.path.join(args.root, 'v1.0-mini')
os.makedirs(args.out, exist_ok=True)


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


sample = {s['token']: s for s in load_table('sample')}
sd_all = load_table('sample_data')
ego = {e['token']: e for e in load_table('ego_pose')}
cs = {c['token']: c for c in load_table('calibrated_sensor')}
scene = {s['token']: s for s in load_table('scene')}
inst = {i['token']: i for i in load_table('instance')}
cat = {c['token']: c['name'] for c in load_table('category')}
kf = {}
for d in sd_all:
    if d['is_key_frame']:
        kf[(d['sample_token'], d['filename'].split('/')[1])] = d
ann = {}
for a in load_table('sample_annotation'):
    ann.setdefault(a['sample_token'], []).append(a)

GEN2DET = {'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
           'vehicle.trailer': 'trailer', 'vehicle.construction': 'construction_vehicle',
           'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
           'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
           'human.pedestrian.construction_worker': 'pedestrian', 'human.pedestrian.police_officer': 'pedestrian',
           'movable_object.trafficcone': 'traffic_cone', 'movable_object.barrier': 'barrier'}

R_l = json.load(open(args.res_lidar))['results']
R_c = json.load(open(args.res_cam))['results']
tokens = sorted(R_l.keys(), key=lambda t: (sample[t]['scene_token'], sample[t]['timestamp']))
assert set(R_l) == set(R_c), 'two result files cover different samples'
if args.limit:
    tokens = tokens[:args.limit]

XC = np.array([1, 1, 1, 1, -1, -1, -1, -1.])
YC = np.array([1, -1, -1, 1, 1, -1, -1, 1.])
ZC = np.array([1, 1, -1, -1, 1, 1, -1, -1.])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def corners(center, size, quat):
    w, l, h = size
    c = np.stack([XC * l / 2, YC * w / 2, ZC * h / 2])
    return (qR(quat) @ c).T + np.array(center)


def gt_boxes(t):
    out = []
    for a in ann.get(t, []):
        n = cat[inst[a['instance_token']]['category_token']]
        if n in GEN2DET:
            out.append(dict(translation=a['translation'], size=a['size'], rotation=a['rotation'],
                            detection_name=GEN2DET[n], detection_score=1.0))
    return out


def to_ego(P, e):
    return (qR(e['rotation']).T @ (P - np.array(e['translation'])).T).T


def to_cam(Pe, c):
    return (qR(c['rotation']).T @ (Pe - np.array(c['translation'])).T).T


def draw_cam(ax, img, boxes, sd, color, lw, label_scores):
    c = cs[sd['calibrated_sensor_token']]
    e = ego[sd['ego_pose_token']]
    K = np.array(c['camera_intrinsic'])
    H, W = img.shape[:2]
    s = W / 1600.0
    if not ax.images:
        ax.imshow(img)
    n_vis = 0
    for b in boxes:
        P = corners(b['translation'], b['size'], b['rotation'])
        Pc = to_cam(to_ego(P, e), c)
        if np.all(Pc[:, 2] < 0.1):
            continue
        uv = (K @ Pc.T).T
        z = np.maximum(uv[:, 2:3], 1e-3)
        uv = uv[:, :2] / z * s
        drawn = False
        for i, j in EDGES:
            if Pc[i, 2] > 0.1 and Pc[j, 2] > 0.1:
                ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color=color, lw=lw)
                drawn = True
        if drawn:
            n_vis += 1
            if label_scores:
                ok = np.where(Pc[:, 2] > 0.1)[0]
                p = uv[ok].mean(0)
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    ax.text(p[0], p[1], '%s %.2f' % (b['detection_name'][:4], b['detection_score']), color='yellow',
                            fontsize=5, ha='center', va='center')
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    return n_vis


def draw_bev(ax, pts, boxes_pred, boxes_gt, e, title):
    ax.set_facecolor('black')
    ax.scatter(-pts[:, 1], pts[:, 0], s=0.25, c=np.clip(pts[:, 2], -3, 3), cmap='Greys_r', vmin=-3, vmax=3, linewidths=0)
    for boxes, col, lw in ((boxes_gt, '#39ff14', 1.0), (boxes_pred, '#ff3030', 1.1)):
        for b in boxes:
            P = to_ego(corners(b['translation'], b['size'], b['rotation']), e)
            poly = P[[0, 1, 5, 4, 0]]
            ax.plot(-poly[:, 1], poly[:, 0], color=col, lw=lw)
            f = (P[0] + P[1]) / 2
            ax.plot([-(P[:4, 1].mean()), -f[1]], [P[:4, 0].mean(), f[0]], color=col, lw=lw)
    ax.plot(0, 0, marker='^', color='cyan', ms=6)
    ax.set_xlim(-54, 54); ax.set_ylim(-54, 54); ax.set_aspect('equal')
    ax.tick_params(labelsize=6, colors='gray')
    ax.set_title(title, fontsize=8)


CAMS = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'], ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]
stats = []
for idx, t in enumerate(tokens, 1):
    s = sample[t]
    sc = scene[s['scene_token']]
    lid = kf[(t, 'LIDAR_TOP')]
    e_l = ego[lid['ego_pose_token']]
    c_l = cs[lid['calibrated_sensor_token']]
    fp = os.path.join(args.root, lid['filename'])
    if os.path.exists(fp):
        pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 5)[:, :3]
    elif args.mock:
        pts = np.random.randn(20000, 3) * [20, 20, 1]
    else:
        raise FileNotFoundError(fp)
    pts_e = (qR(c_l['rotation']) @ pts.T).T + np.array(c_l['translation'])
    pl = [b for b in R_l[t] if b['detection_score'] >= args.thr]
    pc = [b for b in R_c[t] if b['detection_score'] >= args.thr]
    gt = gt_boxes(t)

    fig = plt.figure(figsize=(24, 9.6))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1.05, 1.05], wspace=0.04, hspace=0.06)
    n_img, n_pred_vis, cam_ok = 0, 0, 0
    for r in range(2):
        for k in range(3):
            ch = CAMS[r][k]
            d = kf[(t, ch)]
            ip = os.path.join(args.root, d['filename'])
            ax = fig.add_subplot(gs[r, k])
            if os.path.exists(ip):
                im = Image.open(ip).convert('RGB')
                im = im.resize((640, 360))
                img = np.asarray(im); cam_ok += 1
            elif args.mock:
                img = np.full((360, 640, 3), 90, np.uint8)
            else:
                raise FileNotFoundError(ip)
            draw_cam(ax, img, gt, d, '#39ff14', 0.6, False)
            n_pred_vis += draw_cam(ax, img, pc, d, '#ff3030', 1.0, True)
            ax.set_title(ch, fontsize=8, color='black', pad=2)
    ax_l = fig.add_subplot(gs[:, 3])
    draw_bev(ax_l, pts_e, pl, gt, e_l, 'L3C official LiDAR-only | %d pred (score>=%.2f) vs %d GT' % (len(pl), args.thr, len(gt)))
    ax_c = fig.add_subplot(gs[:, 4])
    draw_bev(ax_c, pts_e, pc, gt, e_l, 'L3D official LiDAR+Camera | %d pred (score>=%.2f) vs %d GT' % (len(pc), args.thr, len(gt)))
    fig.suptitle('sample %d/%d | %s (%s) | token %s | t=%.3fs | LiDAR pts %d\n'
                 'cameras: GT green, L3D pred red (yellow = class score) | BEV: GT green, pred red | official pretrained BEVFusion on nuScenes-mini VAL (contaminated, not a benchmark)'
                 % (idx, len(tokens), sc['name'], sc['description'][:38], t[:8], s['timestamp'] / 1e6, len(pts)), fontsize=9)
    out = os.path.join(args.out, 'sample_%03d_%s.png' % (idx, sc['name']))
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    stats.append(dict(idx=idx, token=t, scene=sc['name'], cams_found=cam_ok, lidar_pts=int(len(pts)), gt=len(gt),
                      pred_lidar=len(pl), pred_cam=len(pc), file=os.path.basename(out)))
    if idx % 10 == 0 or idx == len(tokens):
        print('rendered', idx, '/', len(tokens), flush=True)

with open(os.path.join(args.out, 'viz_index.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
    w.writeheader(); w.writerows(stats)
summary = dict(n_samples=len(stats), thr=args.thr, cams_missing=sum(6 - x['cams_found'] for x in stats),
               mean_gt=float(np.mean([x['gt'] for x in stats])), mean_pred_lidar=float(np.mean([x['pred_lidar'] for x in stats])),
               mean_pred_cam=float(np.mean([x['pred_cam'] for x in stats])), mock=args.mock)
json.dump(summary, open(os.path.join(args.out, 'viz_summary.json'), 'w'), indent=2)
pick = [stats[i]['file'] for i in np.linspace(0, len(stats) - 1, min(6, len(stats))).astype(int)]
ims = [Image.open(os.path.join(args.out, f)) for f in pick]
Wm = 1200
ims = [im.resize((Wm, int(im.height * Wm / im.width))) for im in ims]
sheet = Image.new('RGB', (2 * Wm, sum(im.height for im in ims[0::2]) + 0), 'white')
y = [0, 0]
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 2) * Wm, y[i % 2])); y[i % 2] += im.height
sheet.save(os.path.join(args.out, 'overview_6_samples.png'))
print(json.dumps(summary, indent=2))
ok = summary['n_samples'] == (args.limit or 81) and (args.mock or summary['cams_missing'] == 0)
print('VIZ PASS' if ok else 'VIZ FAIL')
