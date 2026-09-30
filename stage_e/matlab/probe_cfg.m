proj = 'D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radar_ws';
addpath(genpath(fullfile(proj,'src'))); addpath(fullfile(proj,'config'));
names = {'trackingConfig','unifiedBevConfig','ablationConfig','riskMapConfig','finalDemoConfig'};
for k = 1:numel(names)
    try
        c = feval(names{k});
        fprintf('%-18s tracking=%d bevUnified=%d occupancy=%d bev=%d sensors.lidar=%d sensors.radar=%d\n', names{k}, isfield(c,'tracking'), isfield(c,'bevUnified'), isfield(c,'occupancy'), isfield(c,'bev'), isfield(c.sensors,'lidar'), isfield(c.sensors,'radar'));
    catch e
        fprintf('%-18s ERROR %s\n', names{k}, e.message);
    end
end
c = ablationConfig();
disp(c.bev); disp(c.sensors.lidar); disp(c.sensors.radar);
fprintf('footprint vehicle=%s pedestrian=%s\n', mat2str(c.frames.footprint_m.vehicle), mat2str(c.frames.footprint_m.pedestrian));
