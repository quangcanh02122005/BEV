v = ver; names = string({v.Name});
fprintf('%s | %s\n', version, matlabroot);
need = {'Computer Vision','Lidar','Sensor Fusion','Navigation','Radar','Automated Driving','Simulink 3D','Image Processing','Statistics','Signal Processing'};
for k = 1:numel(need), fprintf('%-20s installed=%d\n', need{k}, any(contains(names, need{k}))); end
fprintf('functions: pointCloud=%d pcfitplane=%d pcsegdist=%d objectDetection=%d trackerGNN=%d occupancyMap3D=%d\n', exist('pointCloud'), exist('pcfitplane'), exist('pcsegdist'), exist('objectDetection'), exist('trackerGNN'), exist('occupancyMap3D'));
fprintf('licenses: CV=%d Lidar=%d SFT=%d\n', license('test','Video_and_Image_Blockset'), license('test','Lidar_Toolbox'), license('test','Sensor_Fusion_and_Tracking'));
disp(names');
