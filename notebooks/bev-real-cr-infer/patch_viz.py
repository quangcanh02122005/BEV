import ast, shutil
shutil.copy('../bev-real-train-a/viz_l3.py', 'viz_l3.py')
p = 'viz_l3.py'
s = open(p, encoding='utf-8').read()
s = s.replace("ap.add_argument('--every', type=int, default=1)",
              "ap.add_argument('--every', type=int, default=1)\nap.add_argument('--cam_pred', choices=['l', 'c'], default='c', help='which model is drawn on the camera panels')")
old = "            n_pred_vis += draw_cam(ax, img, pc, d, '#ff3030', 1.0, True)"
assert old in s
s = s.replace(old, "            n_pred_vis += draw_cam(ax, img, (pl if args.cam_pred == 'l' else pc), d, '#ff3030', 1.0, True)")
old2 = "'cameras: GT green, right-panel model pred red (yellow = class score) | BEV: GT green, pred red | '"
assert old2 in s
s = s.replace(old2, "'cameras: GT green, ' + ('left' if args.cam_pred == 'l' else 'right') + '-panel model pred red (yellow = class score) | BEV: GT green, pred red | '")
open(p, 'w', encoding='utf-8').write(s)
ast.parse(s)
print('viz patched ok')
