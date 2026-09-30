% Stage D: Unified Object List (Camera+LiDAR  x  Camera+Radar late fusion), ego frame: x fwd, y left, z up, yaw CCW rad.
% source_code: 1 = Camera+LiDAR only, 2 = Camera+Radar only, 3 = both sensors agree.
here = fileparts(mfilename('fullpath'));
U = load(fullfile(here, 'unified_pretrained.mat')); U = U.unified;      % or unified_ours.mat
G = load(fullfile(here, '..', '..', 'exports', 'L3E', 'l3e_predictions.mat')); G = G.gt;
idx = 5; thr = 0.5;
f = figure('Visible', 'off', 'Position', [100 100 700 700]); hold on; axis equal; grid on;
drawBoxes(G, idx, [0 0.65 0], 0, 1.0);
col = {[0.85 0.1 0.1], [0.1 0.3 0.9], [0.6 0.1 0.7]};
names = {'Camera+LiDAR only', 'Camera+Radar only', 'both agree'};
for c = 1:3
    m = U.sample_idx == idx & U.score >= thr & U.source_code == c;
    sub = struct('x', U.x(m), 'y', U.y(m), 'l', U.l(m), 'w', U.w(m), 'yaw', U.yaw(m), 'sample_idx', U.sample_idx(m), 'score', U.score(m));
    drawBoxes(sub, idx, col{c}, 0, 1.6);
    plot(NaN, NaN, '-', 'Color', col{c}, 'LineWidth', 1.6, 'DisplayName', sprintf('%s (%d)', names{c}, nnz(m)));
end
plot(NaN, NaN, '-', 'Color', [0 0.65 0], 'DisplayName', 'GT'); legend('Location', 'southoutside');
plot(0, 0, 'k^', 'MarkerFaceColor', 'k'); xlim([-50 50]); ylim([-50 50]); xlabel('-y [m]'); ylabel('x forward [m]');
title(sprintf('Unified object list | sample %d | score >= %.1f | %d objects', idx, thr, nnz(U.sample_idx == idx & U.score >= thr)));
exportgraphics(f, fullfile(here, 'unified_bev_sample5.png'), 'Resolution', 110);
fprintf('objects=%d  (score>=%.1f: %d)  both=%d cl=%d cr=%d\n', numel(U.x), thr, nnz(U.score >= thr), nnz(U.score >= thr & U.source_code == 3), nnz(U.score >= thr & U.source_code == 1), nnz(U.score >= thr & U.source_code == 2));

function drawBoxes(T, idx, col, thr, lw)
    m = find(T.sample_idx == idx & T.score >= thr);
    for j = reshape(m, 1, [])
        c = cos(T.yaw(j)); s = sin(T.yaw(j));
        cx = [1 1 -1 -1 1] * T.l(j) / 2; cy = [1 -1 -1 1 1] * T.w(j) / 2;
        plot(-(T.y(j) + s * cx + c * cy), T.x(j) + c * cx - s * cy, '-', 'Color', col, 'LineWidth', lw, 'HandleVisibility', 'off');
    end
end
