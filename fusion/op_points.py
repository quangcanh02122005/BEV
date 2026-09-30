import runpy, json, os, sys
import numpy as np
g = runpy.run_path('export_unified.py')
rows_f = g['ALL']; gt_check = g['gt_check']; to_rows = g['to_rows']; tokens = g['tokens']
H = os.path.dirname(os.path.abspath('export_unified.py'))
def load(p, tag):
    sub = json.load(open(p)); return to_rows(sub, {t: [(tag, b['detection_score'], b['detection_score']) for b in sub['results'][t]] for t in tokens})
systems = {'CL_official': load(os.path.join(H, '..', 'frozen', 'L3C_L3E_v1', 'l3c_l3d_outputs', 'L3D_lidar_camera', 'results_nusc.json'), 'cl'),
           'CR_crn': load(os.path.join(H, '..', 'models', 'camera_radar', 'crn_r18_pretrained', 'results_nusc.json'), 'cr'),
           'FUSED_pretrained': rows_f['pretrained'], 'CL_ours': load(os.path.join(H, '..', 'models', 'our_bevfusion_mini', 'predictions', 'our_bevfusion_mini_final_last_results_nusc.json'), 'cl'),
           'FUSED_ours': rows_f['ours']}
print(f"{'system':18s} " + ' | '.join(f'thr={t}: n/img P R' for t in (0.3, 0.4, 0.5, 0.6, 0.7)))
out = {}
for k, r in systems.items():
    cells = []
    out[k] = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        n, p, rc = gt_check(r, t); cells.append(f'{n / 81:5.1f} {p:.2f} {rc:.2f}'); out[k][str(t)] = dict(per_sample=round(n / 81, 1), precision=round(p, 3), recall=round(rc, 3))
    print(f'{k:18s} ' + ' | '.join(cells))
json.dump(out, open('unified/operating_points.json', 'w'), indent=2)
