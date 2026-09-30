"""Stage H: how much do perception errors change the Level-1 risk score? (there is NO ground-truth risk on nuScenes-mini)
Protocol = the project's Phase-10 evaluation, fixed before running: object score = max risk of the system's cells within 0.75 m of the object footprint (0 if none);
ORACLE = the same risk function fed with GT occupancy (P=1) and GT flow / ego motion.  Metrics: MAE vs oracle, Spearman vs oracle, Spearman vs -DCA_truth (DCA from pure kinematics,
independent of the risk formula), high-risk (>= 0.35) precision/recall vs oracle, flow benefit (moving objects: MAE with detection flow vs flow = 0; the project asks for >= 25% relative reduction).
This measures the propagation of perception quality into the score, NOT the correctness of risk."""
import json, os
import numpy as np
from scipy.io import loadmat
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
R = loadmat(os.path.join(HERE, 'risk_out.mat')); I = loadmat(os.path.join(HERE, 'risk_inputs.mat'))
names = ['oracle', 'classical', 'learned_occ', 'learned_detflow', 'learned_zeroflow', 'learned_flownet']
risk = {s: R['risk_' + s].astype(np.float32) for s in names}; ttc = {s: R['ttc_' + s].astype(np.float32) for s in names}
obj = I['objects']; frame = obj[:, 0].astype(int) - 1; N = 81
xs = -50 + (np.arange(200) + 0.5) * 0.5; XX, YY = np.meshgrid(xs, xs, indexing='ij'); OFF = 1.3
ego_v, ego_w = I['ego_v'], I['ego_yawrate'].ravel()


def rect_dist(win_x, win_y, cx, cy, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw); dx, dy = win_x - cx, win_y - cy; u = c * dx + s * dy; v = -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(u) - l / 2, 0), np.maximum(np.abs(v) - w / 2, 0))


def object_scores(rk):
    out = np.zeros(len(obj))
    for j, (f, cx, cy, w, l, yaw, *_r) in enumerate(obj):
        f = int(f) - 1; r0 = max(0, int((cx - l - w - 2 + 50) / 0.5)); r1 = min(200, int((cx + l + w + 2 + 50) / 0.5) + 1); c0 = max(0, int((cy - l - w - 2 + 50) / 0.5)); c1 = min(200, int((cy + l + w + 2 + 50) / 0.5) + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        d = rect_dist(XX[r0:r1, c0:c1], YY[r0:r1, c0:c1], cx, cy, w, l, yaw); m = d <= 0.75
        out[j] = rk[f, r0:r1, c0:c1][m].max() if m.any() else 0.0
    return out


def dca_truth():
    """closest approach of the object centre to the ego centre within 3 s, constant ground velocity vs constant ego velocity / yaw rate (body-centred coordinates)."""
    times = np.arange(0, 3.0001, 0.25); out = np.full(len(obj), np.nan)
    for j, (f, cx, cy, w, l, yaw, vx, vy, _n) in enumerate(obj):
        f = int(f) - 1
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        v = ego_v[f]; sp = np.hypot(*v); om = ego_w[f]; d0 = np.arctan2(v[1], v[0]) if sp > 1e-6 else 0.0
        if abs(om) < 1e-4:
            ec = sp * times[:, None] * np.array([np.cos(d0), np.sin(d0)])
        else:
            a = d0 + om * times; ec = sp / om * np.stack([np.sin(a) - np.sin(d0), -(np.cos(a) - np.cos(d0))], 1)
        oc = np.array([cx - OFF, cy]) + times[:, None] * np.array([vx, vy]); out[j] = np.hypot(*(oc - ec).T).min()
    return out


sc = {s: object_scores(risk[s]) for s in names}; dca = dca_truth()
speed = np.hypot(obj[:, 6], obj[:, 7]); moving = np.isfinite(speed) & (speed >= 0.3); near = np.hypot(obj[:, 1], obj[:, 2]) <= 30
sets = {'all objects': np.ones(len(obj), bool), 'objects within 30 m': near, 'moving objects (>=0.3 m/s)': moving}
o = sc['oracle']; out = dict(n_objects=int(len(obj)), n_moving=int(moving.sum()), oracle_high_risk_objects=int((o >= 0.35).sum()), oracle_mean_score=round(float(o.mean()), 4), systems={})
sp_or = spearmanr(o[np.isfinite(dca)], -dca[np.isfinite(dca)])[0]
out['oracle_spearman_vs_minus_DCA'] = round(float(sp_or), 3)
for s in names[1:]:
    r = {}
    for nm, m in sets.items():
        r[f'MAE_vs_oracle[{nm}]'] = round(float(np.abs(sc[s][m] - o[m]).mean()), 4)
    r['Spearman_vs_oracle'] = round(float(spearmanr(sc[s], o)[0]), 3)
    ok = np.isfinite(dca); r['Spearman_vs_minus_DCA'] = round(float(spearmanr(sc[s][ok], -dca[ok])[0]), 3)
    hi_o = o >= 0.35; hi_s = sc[s] >= 0.35
    r['high_risk_precision'] = round(float((hi_o & hi_s).sum() / max(1, hi_s.sum())), 3); r['high_risk_recall'] = round(float((hi_o & hi_s).sum() / max(1, hi_o.sum())), 3)
    mx = risk[s].reshape(N, -1).max(1); mo = risk['oracle'].reshape(N, -1).max(1)
    r['per_frame_max_risk_pearson_vs_oracle'] = round(float(np.corrcoef(mx, mo)[0, 1]), 3)
    tm = ttc[s].reshape(N, -1).min(1); to = ttc['oracle'].reshape(N, -1).min(1); fin = np.isfinite(to) & np.isfinite(tm)
    r['min_TTC_frames_oracle_finite'] = int(np.isfinite(to).sum()); r['min_TTC_frames_system_finite_too'] = int(fin.sum()); r['min_TTC_MAE_s(where both finite)'] = round(float(np.abs(tm[fin] - to[fin]).mean()), 3) if fin.any() else None
    out['systems'][s] = r
mae_f = np.abs(sc['learned_detflow'][moving] - o[moving]).mean(); mae_0 = np.abs(sc['learned_zeroflow'][moving] - o[moving]).mean()
out['flow_benefit_moving_objects'] = dict(MAE_with_detection_flow=round(float(mae_f), 4), MAE_flow_zero=round(float(mae_0), 4), relative_reduction=round(float((mae_0 - mae_f) / mae_0), 3), project_threshold=0.25)
out['ego'] = dict(median_speed_mps=round(float(np.median(np.hypot(ego_v[:, 0], ego_v[:, 1]))), 2), note='ego moves at ~5 m/s in most frames: closing speed and overlap are dominated by ego motion')
json.dump(out, open(os.path.join(RES, 'risk_metrics.json'), 'w'), indent=2)
print('objects', out['n_objects'], 'moving', out['n_moving'], '| oracle high-risk objects', out['oracle_high_risk_objects'], '| oracle Spearman vs -DCA', out['oracle_spearman_vs_minus_DCA'])
for s, r in out['systems'].items():
    print(f"{s:17s} MAE all {r['MAE_vs_oracle[all objects]']:.3f} near30 {r['MAE_vs_oracle[objects within 30 m]']:.3f} moving {r['MAE_vs_oracle[moving objects (>=0.3 m/s)]']:.3f} | rho_oracle {r['Spearman_vs_oracle']:.2f} rho_-DCA {r['Spearman_vs_minus_DCA']:.2f} | hiP {r['high_risk_precision']:.2f} hiR {r['high_risk_recall']:.2f} | frame-max r {r['per_frame_max_risk_pearson_vs_oracle']:.2f}")
print('flow benefit', out['flow_benefit_moving_objects'])

# ---- figures ----
pick = [int(i) for i in np.argsort(-risk['oracle'].reshape(N, -1).max(1))[[0, 20, 40]]]
for k in pick:
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))
    for a, s, t in zip(ax, ['oracle', 'classical', 'learned_detflow', 'learned_flownet'], ['ORACLE input (GT occupancy + GT flow)', 'Classical occupancy, no flow', 'Learned occupancy + detection flow', 'Learned occupancy + learned flow net']):
        im = a.imshow(risk[s][k].T if False else np.where(risk[s][k] > 0, risk[s][k], np.nan), origin='lower', extent=[-50, 50, -50, 50], cmap="inferno", vmin=0, vmax=0.6); a.set_facecolor('#151515')
        a.invert_xaxis(); a.set_title(t, fontsize=10); a.set_xlabel('y left [m]'); a.set_ylabel('x forward [m]'); a.plot(0, 0, 'c^')
    fig.colorbar(im, ax=ax, fraction=0.012, label='uncalibrated risk score')
    fig.suptitle(f'Stage H risk bridge | nuScenes-mini VAL frame {k + 1}/81 | Level-1 computeRiskFields on real data | ego {np.hypot(*ego_v[k]):.1f} m/s | risk = uncalibrated score, not collision probability', fontsize=11)
    fig.savefig(os.path.join(RES, f'stage_h_risk_frame{k + 1}.png'), dpi=60, bbox_inches='tight'); plt.close(fig)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for a, s in zip(ax, ['classical', 'learned_detflow', 'learned_flownet']):
    a.scatter(o, sc[s], s=6, alpha=.4); a.plot([0, 1], [0, 1], 'r--'); a.set_xlabel('oracle object score'); a.set_ylabel(f'{s} object score'); a.set_title(f"{s}: rho={out['systems'][s]['Spearman_vs_oracle']}, MAE={out['systems'][s]['MAE_vs_oracle[all objects]']}")
fig.tight_layout(); fig.savefig(os.path.join(RES, 'stage_h_risk_scatter.png'), dpi=80)
print('wrote', RES)

# ---- non-trivial objects (oracle score >= 0.1, the project's oracleRangeMin) + bootstrap over frames (post-hoc addition, declared) ----
nt = o >= 0.1; rng = np.random.default_rng(0); frames = np.unique(frame); B = 1000
def stat(fr_sample, s, kind):
    idx = np.concatenate([np.where((frame == f) & nt)[0] for f in fr_sample]);
    if len(idx) < 5: return np.nan
    return np.abs(sc[s][idx] - o[idx]).mean() if kind == 'mae' else spearmanr(sc[s][idx], o[idx])[0]
boot = {}
for s in names[1:]:
    m = [stat(rng.choice(frames, len(frames)), s, 'mae') for _ in range(B)]; r = [stat(rng.choice(frames, len(frames)), s, 'rho') for _ in range(B)]
    boot[s] = dict(n_objects=int(nt.sum()), MAE=round(float(np.abs(sc[s][nt] - o[nt]).mean()), 4), MAE_CI95=[round(float(np.nanpercentile(m, 2.5)), 4), round(float(np.nanpercentile(m, 97.5)), 4)],
                   Spearman=round(float(spearmanr(sc[s][nt], o[nt])[0]), 3), Spearman_CI95=[round(float(np.nanpercentile(r, 2.5)), 3), round(float(np.nanpercentile(r, 97.5)), 3)])
paired = {}
for a_, b_ in (('learned_detflow', 'learned_zeroflow'), ('learned_detflow', 'classical'), ('learned_detflow', 'learned_flownet')):
    d = []
    for _ in range(B):
        fs = rng.choice(frames, len(frames)); d.append(stat(fs, a_, 'mae') - stat(fs, b_, 'mae'))
    paired[f'MAE({a_}) - MAE({b_})'] = dict(mean=round(float(np.nanmean(d)), 4), CI95=[round(float(np.nanpercentile(d, 2.5)), 4), round(float(np.nanpercentile(d, 97.5)), 4)])
out['non_trivial_objects_oracle_ge_0.1'] = boot; out['paired_bootstrap_MAE_difference'] = paired
mov_nt = moving & nt
out['flow_benefit_non_trivial_moving'] = dict(n=int(mov_nt.sum()), MAE_det=round(float(np.abs(sc['learned_detflow'][mov_nt] - o[mov_nt]).mean()), 4) if mov_nt.any() else None, MAE_zero=round(float(np.abs(sc['learned_zeroflow'][mov_nt] - o[mov_nt]).mean()), 4) if mov_nt.any() else None)
json.dump(out, open(os.path.join(RES, 'risk_metrics.json'), 'w'), indent=2)
print('\nobjects with oracle score >= 0.1:', int(nt.sum()))
for s, r in boot.items(): print(f"{s:17s} MAE {r['MAE']:.3f} {r['MAE_CI95']} | Spearman {r['Spearman']:.2f} {r['Spearman_CI95']}")
for k, v in paired.items(): print(k, v)
print('flow benefit on non-trivial moving objects', out['flow_benefit_non_trivial_moving'])
