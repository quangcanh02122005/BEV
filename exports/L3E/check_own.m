S = load(fullfile(fileparts(mfilename('fullpath')), 'own_models_predictions.mat'));
fprintf('own LiDAR boxes=%d (score>=0.3: %d) | own Camera+LiDAR boxes=%d (score>=0.3: %d) | GT=%d | samples=%d\n', numel(S.pred_our_lidar.x), nnz(S.pred_our_lidar.score>=0.3), numel(S.pred_our_bevfusion.x), nnz(S.pred_our_bevfusion.score>=0.3), numel(S.gt.x), numel(S.samples.sample_idx));
