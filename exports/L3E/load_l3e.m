% L3E -> MATLAB: load BEVFusion predictions + GT (ego frame) and plot a BEV frame.
% Frame: ego/vehicle, x forward, y left, z up, yaw (rad) about z, CCW from +x. Boxes: l along heading, w lateral.
here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'l3e_predictions.mat'));   % S.samples, S.gt, S.pred_lidar_only, S.pred_lidar_camera

idx = 1;               % sample index (1-based, ordered by scene then time)
thr = 0.3;             % score threshold for predictions
f = figure('Visible', 'off', 'Position', [100 100 1100 520]);
sets = {S.pred_lidar_only, 'L3C LiDAR-only'; S.pred_lidar_camera, 'L3D LiDAR+Camera'};
for k = 1:2
    subplot(1, 2, k); hold on; axis equal; grid on;
    drawBoxes(S.gt, idx, [0 0.6 0], 0);
    drawBoxes(sets{k, 1}, idx, [0.85 0.1 0.1], thr);
    plot(0, 0, 'k^', 'MarkerFaceColor', 'k');
    xlim([-50 50]); ylim([-50 50]); xlabel('-y (left is -)  [m]'); ylabel('x forward [m]');
    n = nnz(sets{k, 1}.sample_idx == idx & sets{k, 1}.score >= thr);
    title(sprintf('%s | sample %d (%s) | %d pred, GT green', sets{k, 2}, idx, S.samples.scene{idx}, n));
end
exportgraphics(f, fullfile(here, 'l3e_bev_sample1.png'), 'Resolution', 120);
fprintf('samples=%d gt=%d pred_lidar_only=%d pred_lidar_camera=%d\n', numel(S.samples.sample_idx), ...
    numel(S.gt.x), numel(S.pred_lidar_only.x), numel(S.pred_lidar_camera.x));

function drawBoxes(T, idx, col, thr)
    m = find(T.sample_idx == idx & T.score >= thr);
    for j = reshape(m, 1, [])
        c = cos(T.yaw(j)); s = sin(T.yaw(j));
        cx = [1 1 -1 -1 1] * T.l(j) / 2; cy = [1 -1 -1 1 1] * T.w(j) / 2;
        X = T.x(j) + c * cx - s * cy; Y = T.y(j) + s * cx + c * cy;
        plot(-Y, X, '-', 'Color', col, 'LineWidth', 1);
    end
end
