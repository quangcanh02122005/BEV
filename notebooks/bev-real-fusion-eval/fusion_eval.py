"""Official nuScenes detection eval (mini_val) for single-branch systems and late-fusion variants."""
import glob, json, os, sys, time
sys.path.insert(0, '/tmp/l3/fusion')
from late_fusion import fuse
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

OUT = '/kaggle/working/outputs/fusion_eval'
os.makedirs(OUT, exist_ok=True)
V = None
for r, d, f in os.walk('/kaggle/input'):
    if 'v1.0-mini' in d:
        V = r; break
print('dataroot', V)
nusc = NuScenes(version='v1.0-mini', dataroot=V, verbose=False)
cfg = config_factory('detection_cvpr_2019')


def find(sub):
    r = [p for p in glob.glob('/kaggle/input/**/results_nusc.json', recursive=True) if sub in p]
    assert r, 'missing ' + sub
    return r[0]


P = dict(L3D=find('L3D_lidar_camera'), OURB=find('our_bevfusion_mini/predictions/final_last'), CRN=find('crn_r18_pretrained'))
print(json.dumps(P, indent=1))
R = {k: json.load(open(v)) for k, v in P.items()}
META = dict(use_camera=True, use_lidar=True, use_radar=True, use_map=False, use_external=False)


def evaluate(name, results, save=False):
    sub = dict(meta=META, results={t: [{k: v for k, v in b.items() if not k.startswith('_')} for b in bs] for t, bs in results.items()})
    d = f'/tmp/l3/fe/{name}'; os.makedirs(d, exist_ok=True)
    p = f'{d}/results_nusc.json'; json.dump(sub, open(p, 'w'))
    t0 = time.time()
    NuScenesEval(nusc, config=cfg, result_path=p, eval_set='mini_val', output_dir=d, verbose=False).main(render_curves=False)
    m = json.load(open(f'{d}/metrics_summary.json'))
    row = dict(name=name, mAP=m['mean_ap'], NDS=m['nd_score'], mATE=m['tp_errors']['trans_err'], mASE=m['tp_errors']['scale_err'],
               mAOE=m['tp_errors']['orient_err'], mAVE=m['tp_errors']['vel_err'], mAAE=m['tp_errors']['attr_err'], AP2m=m['mean_dist_aps'],
               n_boxes=sum(len(v) for v in results.values()), secs=round(time.time() - t0, 1))
    if save:
        os.makedirs(f'{OUT}/submissions', exist_ok=True)
        json.dump(sub, open(f'{OUT}/submissions/{name}.json', 'w'))
        json.dump({t: [(b['_src'], b['_sa'], b['_sb']) for b in bs] for t, bs in results.items() if bs and '_src' in bs[0]}, open(f'{OUT}/submissions/{name}_sources.json', 'w'))
    print(f"{name:34s} mAP={row['mAP']:.4f} NDS={row['NDS']:.4f} mATE={row['mATE']:.3f} mAOE={row['mAOE']:.3f} mAVE={row['mAVE']:.3f} boxes={row['n_boxes']} ({row['secs']}s)", flush=True)
    return row


rows = []
plain = lambda k: {t: v for t, v in R[k]['results'].items()}
rows.append(evaluate('CL_official_pretrained', plain('L3D')))
rows.append(evaluate('CL_ours_trained', plain('OURB')))
rows.append(evaluate('CR_crn_pretrained', plain('CRN')))
for tag, A in (('official', 'L3D'), ('ours', 'OURB')):
    variants = [('baseline', {}, True), ('no_penalty', dict(pen_a=1.0, pen_b=1.0), False), ('vel_from_CL', dict(vel='a'), False),
                ('vel_from_CR', dict(vel='b'), False), ('equal_weights', dict(w_a=0.5, w_b=0.5), False),
                ('strong_penalty', dict(pen_a=0.7, pen_b=0.4), False)]
    for vn, kw, save in variants:
        res, p = fuse(R[A], R['CRN'], **kw)
        rows.append(evaluate(f'FUSED_{tag}_{vn}', res, save=save))
json.dump(rows, open(f'{OUT}/fusion_eval.json', 'w'), indent=2)
print('FUSION EVAL PASS', len(rows), 'systems')
