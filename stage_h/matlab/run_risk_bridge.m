function run_risk_bridge()
%RUN_RISK_BRIDGE Level-1 Phase-10 risk heatmap (computeRiskFields, unmodified) on REAL nuScenes-mini VAL frames.
% Same function, same weights/features/tolerances as Level 1; only the physical rig parameters change (set before running, not tuned):
%   grid +-50 m @0.5 m; ego rectangle 4.084 x 1.730 m (Renault Zoe); grid coordinates are shifted by -1.3 m in x so that the ego BODY CENTRE
%   (not the rear-axle origin of the nuScenes ego frame) is the origin the risk code assumes.
% Risk is an UNCALIBRATED score in [0,1], not a collision probability. Occupied cells of the non-oracle systems use confidence 0.9 (the project's uncalibrated label).
here = fileparts(mfilename('fullpath')); root = fileparts(here);
proj = 'D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radar_ws';
addpath(genpath(fullfile(proj, 'src'))); addpath(fullfile(proj, 'config'));
S = load(fullfile(root, 'risk_inputs.mat'));
cfg = riskMapConfig();
cfg.bev.xBounds_m = [-51.3 48.7]; cfg.bev.yBounds_m = [-50 50]; cfg.bev.resolution_m = 0.5;
cfg.risk.ego.length_m = 4.084; cfg.risk.ego.width_m = 1.730;
names = {'oracle', 'classical', 'learned_occ', 'learned_detflow', 'learned_zeroflow', 'learned_flownet'};
occField = {'occ_oracle', 'occ_classical', 'occ_learned', 'occ_detflow', 'occ_detflow', 'occ_flownet'};
fxField = {'flowX_oracle', '', '', 'flowX_det', '', 'flowX_net'}; fyField = {'flowY_oracle', '', '', 'flowY_det', '', 'flowY_net'};
prob = [1 0.9 0.9 0.9 0.9 0.9];
n = size(S.occ_oracle, 1); out = struct(); secs = zeros(1, numel(names));
for s = 1:numel(names)
    out.(['risk_' names{s}]) = zeros(n, 200, 200, 'single'); out.(['ttc_' names{s}]) = inf(n, 200, 200, 'single');
end
for k = 1:n
    em = struct('velocity', S.ego_v(k, :), 'yawRate', S.ego_yawrate(k));
    for s = 1:numel(names)
        occ = logical(reshape(S.(occField{s})(k, :, :), 200, 200));
        if isempty(fxField{s}), fx = nan(200, 200); fy = nan(200, 200); else, fx = reshape(S.(fxField{s})(k, :, :), 200, 200); fy = reshape(S.(fyField{s})(k, :, :), 200, 200); end
        t0 = tic; f = computeRiskFields(cfg, occ, prob(s) * ones(200, 200), fx, fy, em); secs(s) = secs(s) + toc(t0);
        out.(['risk_' names{s}])(k, :, :) = f.risk; out.(['ttc_' names{s}])(k, :, :) = f.ttcSeconds;
    end
    if mod(k, 20) == 0, fprintf('frame %d/%d\n', k, n); end
end
out.system_names = names; out.seconds_per_frame = secs / n; out.body_center_shift_m = 1.3;
save(fullfile(here, '..', 'risk_out.mat'), '-struct', 'out', '-v7');
for s = 1:numel(names)
    r = out.(['risk_' names{s}]); fprintf('%-18s mean max-risk/frame %.3f | occupied-cell mean risk %.3f | %.1f ms/frame\n', names{s}, mean(reshape(max(max(r, [], 3), [], 2), [], 1)), mean(r(r > 0)), 1000 * secs(s) / n);
end
fprintf('RISK BRIDGE DONE\n');
end
