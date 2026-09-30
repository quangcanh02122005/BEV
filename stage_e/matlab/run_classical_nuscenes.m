function run_classical_nuscenes(varargin)
%RUN_CLASSICAL_NUSCENES Chạy pipeline CLASSICAL của dự án (Level 1, không sửa mã) trên 81 mẫu nuScenes-mini VAL.
% Dùng nguyên các hàm: processLidarScan, buildBevOccupancy, extractLidarDetections, rasterizeRadarBev (project/src).
% Chỉ thay THÔNG SỐ VẬT LÝ của rig nuScenes (đặt trước khi chạy, không chỉnh theo kết quả):
%   lưới BEV +-50 m @0.5 m, extrinsic LiDAR/radar thật, hộp thân xe ego nuScenes (Renault Zoe), LiDAR FOV ngang 360 deg.
% Mọi ngưỡng thuật toán (ground RANSAC, cluster 0.8 m, minPoints, maxHeight 2.5 m, pedestrian, ...) giữ nguyên giá trị Level 1.
here = fileparts(mfilename('fullpath')); root = fileparts(here);
proj = 'D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radar_ws';
addpath(genpath(fullfile(proj, 'src'))); addpath(fullfile(proj, 'config'));
% Options: 'Shims' = 'auto' (default: base-MATLAB stand-ins ONLY if a toolbox function is missing) | true | false ; 'OutDir' = folder name under stage_e.
ip = inputParser; addParameter(ip, 'Shims', 'auto'); addParameter(ip, 'OutDir', 'classical_out'); parse(ip, varargin{:});
have = all(cellfun(@(f) exist(f, 'file') > 0, {'pointCloud', 'pcfitplane', 'pcsegdist', 'objectDetection'}));
useShims = strcmp(ip.Results.Shims, 'auto') && ~have || isequal(ip.Results.Shims, true);
if useShims, addpath(fullfile(here, 'shims')); end
fprintf('toolbox functions present: %d | using shims: %d\n', have, useShims);
outDir = fullfile(root, ip.Results.OutDir); if ~exist(outDir, 'dir'), mkdir(outDir); end

cfg = unifiedBevConfig();
cfg.bev.xBounds_m = [-50 50]; cfg.bev.yBounds_m = [-50 50]; cfg.bev.resolution_m = 0.5;
cfg.occupancy.self.bounds_m = [-1.0 3.7; -1.1 1.1; -0.3 1.9];
cfg.urban.lidar.horizontalFov_deg = 360;
validateattributes(cfg.bev.resolution_m, {'numeric'}, {'positive'});

inputs = jsondecode(fileread(fullfile(root, 'data', 'stage_e_inputs.json')));
radar = jsondecode(fileread(fullfile(root, 'data', 'stage_e_radar.json')));
radarNames = {'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'};
conditions = struct('name', {'full', 'keep50', 'keep25', 'range30'}, 'keep', {1, 0.5, 0.25, 1}, 'maxRange', {inf, inf, inf, 30});
egoPose = createPose('ego', 'world', eye(3), [0 0 0]);
n = numel(inputs);
for ci = 1:numel(conditions)
    cond = conditions(ci); rng(20260925 + ci);
    occ = false(n, 200, 200); rhit = false(n, 200, 200); clusters = cell(n, 1); status = strings(n, 1); tLidar = nan(n, 1); tRadar = nan(n, 1);
    for k = 1:n
        rec = inputs(k);
        fid = fopen(fullfile(root, 'data', strrep(rec.lidar.file, '/', filesep)), 'rb'); raw = fread(fid, [5 Inf], 'float32')'; fclose(fid);
        xyz = double(raw(:, 1:3));
        if cond.keep < 1, xyz = xyz(rand(size(xyz, 1), 1) < cond.keep, :); end
        if isfinite(cond.maxRange), xyz = xyz(vecnorm(xyz(:, 1:2) - rec.lidar.t(1:2)', 2, 2) <= cond.maxRange, :); end
        cfgS = cfg; cfgS.sensors.lidar.translation_m = reshape(rec.lidar.t, 1, 3); cfgS.sensors.lidar.rotationMatrix = reshape(rec.lidar.R, 3, 3);
        t0 = tic;
        cloud = processLidarScan(struct('points', xyz, 'timestamp', rec.timestamp), cfgS);
        status(k) = string(cloud.status);
        if cloud.valid
            grid = buildBevOccupancy(cloud, cfgS);
            [~, cl] = extractLidarDetections(cloud, egoPose, cfgS, rec.timestamp);
            tLidar(k) = toc(t0);
            occ(k, :, :) = grid.occupied; clusters{k} = cl;
        else
            clusters{k} = struct('center', {}, 'width_m', {}, 'depth_m', {}, 'height_m', {}, 'points', {}, 'class', {}, 'corners', {}, 'truncated', {}, 'accepted', {});
        end
        if ci == 1
            t1 = tic; hit = false(200, 200);
            for r = 1:5
                rows = radar(k).(radarNames{r}); rr = [];
                if ~isempty(rows)
                    rows = reshape(rows, [], 5); m = size(rows, 1); rr = [ones(m, 1) (1:m)' rows(:, 1:3) rows(:, 4:5) zeros(m, 1)];
                end
                cfgR = cfg; cfgR.sensors.radar.translation_m = reshape(rec.(radarNames{r}).t, 1, 3); cfgR.sensors.radar.rotationMatrix = reshape(rec.(radarNames{r}).R, 3, 3);
                if ~isempty(rr), ch = rasterizeRadarBev(rr, [0 0], cfgR); hit = hit | ch.radarHit; end
            end
            rhit(k, :, :) = hit; tRadar(k) = toc(t1);
        end
        if mod(k, 20) == 0, fprintf('[%s] %d/%d\n', cond.name, k, n); end
    end
    save(fullfile(outDir, ['classical_' cond.name '.mat']), 'occ', 'rhit', 'status', 'tLidar', 'tRadar', '-v7');
    fid = fopen(fullfile(outDir, ['classical_' cond.name '_clusters.json']), 'w'); fwrite(fid, jsonencode(clusters)); fclose(fid);
    fprintf('condition %s: valid scans %d/%d | mean lidar path %.3f s | mean occupied cells %.0f\n', cond.name, nnz(status == "COMPLETED"), n, mean(tLidar, 'omitnan'), mean(sum(occ, [2 3])));
end
fprintf('CLASSICAL NUSCENES RUN DONE\n');
end
