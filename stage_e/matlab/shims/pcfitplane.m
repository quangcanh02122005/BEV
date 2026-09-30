function [model, inlierIndices, outlierIndices] = pcfitplane(pc, maxDistance, referenceVector, maxAngularDistance)
%PCFITPLANE (shim, base MATLAB) RANSAC plane fit: keeps the model with most inliers (|distance| <= maxDistance) whose normal is
% within maxAngularDistance degrees of referenceVector. Returns model.Parameters = [a b c d] (unit normal). Uses the global rng
% (the project seeds it), 3-point samples, adaptive iteration count (99% confidence, max 1000).
P = pc.Location; n = size(P, 1);
if nargin < 3 || isempty(referenceVector), referenceVector = [0 0 0]; end
if nargin < 4, maxAngularDistance = 5; end
useRef = norm(referenceVector) > 0; if useRef, referenceVector = referenceVector(:)' / norm(referenceVector); end
cosMax = cosd(maxAngularDistance);
best = 0; bestParams = [NaN NaN NaN NaN]; bestInl = false(n, 1); iter = 0; maxIter = 1000;
while iter < maxIter
    iter = iter + 1;
    idx = randperm(n, 3); p1 = P(idx(1), :); v = cross(P(idx(2), :) - p1, P(idx(3), :) - p1); nv = norm(v);
    if nv < 1e-9, continue; end
    v = v / nv;
    if useRef && abs(dot(v, referenceVector)) < cosMax, continue; end
    d = -v * p1'; inl = abs(P * v' + d) <= maxDistance; c = nnz(inl);
    if c > best
        best = c; bestParams = [v d]; bestInl = inl;
        ratio = c / n; if ratio > 0, maxIter = min(1000, max(20, ceil(log(0.01) / log(1 - ratio^3 + eps)))); end
    end
end
model = struct('Parameters', bestParams);
inlierIndices = find(bestInl); outlierIndices = find(~bestInl);
end
