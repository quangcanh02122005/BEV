function pc = pointCloud(location)
%POINTCLOUD (shim, base MATLAB) Minimal stand-in for Computer Vision/Lidar Toolbox pointCloud: only .Location and .Count.
% Used because the toolbox is not installed on this machine; see stage_e/README.md.
validateattributes(location, {'numeric'}, {'2d', 'ncols', 3});
pc = struct('Location', double(location), 'Count', size(location, 1));
end
