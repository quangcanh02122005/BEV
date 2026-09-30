v = ver; names = string({v.Name}); disp(names');
fprintf('CV toolbox: %d | Lidar toolbox: %d | pointCloud: %d | pcfitplane: %d | pcsegdist: %d\n', any(contains(names,'Computer Vision')), any(contains(names,'Lidar')), exist('pointCloud'), exist('pcfitplane'), exist('pcsegdist'));
