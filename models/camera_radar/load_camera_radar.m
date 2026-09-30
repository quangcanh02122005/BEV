% Stage C: load CRN Camera+Radar predictions (ego frame: x fwd, y left, z up, yaw CCW rad) and compare with GT and Camera+LiDAR.
here = fileparts(mfilename('fullpath'));
C = load(fullfile(here, 'camera_radar_predictions.mat'));   % C.samples, C.gt, C.pred_camera_radar
L = load(fullfile(here, 'l3e_predictions.mat'));            % L.pred_lidar_camera (official BEVFusion)
idx = 5; thr = 0.3;
f = figure('Visible', 'off', 'Position', [100 100 1100 520]);
sets = {C.pred_camera_radar, 'CRN Camera+Radar (pretrained)'; L.pred_lidar_camera, 'BEVFusion Camera+LiDAR (pretrained)'};
for k = 1:2
    subplot(1, 2, k); hold on; axis equal; grid on;
    drawBoxes(C.gt, idx, [0 0.6 0], 0);
    drawBoxes(sets{k, 1}, idx, [0.85 0.1 0.1], thr);
    plot(0, 0, 'k^', 'MarkerFaceColor', 'k'); xlim([-50 50]); ylim([-50 50]);
    xlabel('-y (left is -) [m]'); ylabel('x forward [m]');
    title(sprintf('%s | sample %d | %d pred', sets{k, 2}, idx, nnz(sets{k, 1}.sample_idx == idx & sets{k, 1}.score >= thr)));
end
exportgraphics(f, fullfile(here, 'camera_radar_bev_sample5.png'), 'Resolution', 110);
fprintf('samples=%d gt=%d crn_boxes=%d (score>=%.1f: %d)\n', numel(C.samples.sample_idx), numel(C.gt.x), numel(C.pred_camera_radar.x), thr, nnz(C.pred_camera_radar.score >= thr));

function drawBoxes(T, idx, col, thr)
    m = find(T.sample_idx == idx & T.score >= thr);
    for j = reshape(m, 1, [])
        c = cos(T.yaw(j)); s = sin(T.yaw(j));
        cx = [1 1 -1 -1 1] * T.l(j) / 2; cy = [1 -1 -1 1 1] * T.w(j) / 2;
        plot(-(T.y(j) + s * cx + c * cy), T.x(j) + c * cx - s * cy, '-', 'Color', col, 'LineWidth', 1);
    end
end
