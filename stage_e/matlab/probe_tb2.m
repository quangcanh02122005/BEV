v = ver; names = string({v.Name});
fprintf('%s | %s\n', version, matlabroot);
fprintf('CV=%d Lidar=%d SensorFusion=%d Navigation=%d Radar=%d | pointCloud=%d pcfitplane=%d pcsegdist=%d objectDetection=%d\n', any(contains(names,'Computer Vision')), any(contains(names,'Lidar')), any(contains(names,'Sensor Fusion')), any(contains(names,'Navigation')), any(contains(names,'Radar')), exist('pointCloud'), exist('pcfitplane'), exist('pcsegdist'), exist('objectDetection'));
fprintf('%s\n', strjoin(names, ', '));
