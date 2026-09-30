function d = objectDetection(time, measurement, varargin)
%OBJECTDETECTION (shim) Plain struct with the same fields the project reads; Sensor Fusion and Tracking Toolbox not installed.
d = struct('Time', time, 'Measurement', measurement);
for k = 1:2:numel(varargin), d.(varargin{k}) = varargin{k + 1}; end
end
