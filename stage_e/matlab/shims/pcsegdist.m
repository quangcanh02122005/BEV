function [labels, numClusters] = pcsegdist(pc, minDistance, varargin)
%PCSEGDIST (shim, base MATLAB) Euclidean single-linkage clustering: points closer than or equal to minDistance are the same cluster.
% 'NumClusterPoints' [min max]: clusters outside the range get label 0; surviving clusters are numbered 1..K consecutively.
P = pc.Location; n = size(P, 1); nr = [1 inf];
for k = 1:2:numel(varargin)
    if strcmpi(varargin{k}, 'NumClusterPoints'), nr = varargin{k + 1}; if isscalar(nr), nr = [nr inf]; end; end
end
labels = zeros(n, 1); numClusters = 0;
if n == 0, return; end
cell = max(minDistance, eps); g = floor((P - min(P, [], 1)) / cell) + 1; dims = max(g, [], 1) + 2;
lin = g(:, 1) + (g(:, 2) - 1) * dims(1) + (g(:, 3) - 1) * dims(1) * dims(2);
[sl, order] = sort(lin); [uk, first] = unique(sl, 'first'); [~, last] = unique(sl, 'last');
map = containers.Map('KeyType', 'double', 'ValueType', 'any'); for i = 1:numel(uk), map(uk(i)) = [first(i) last(i)]; end
parent = 1:n;
find_ = @(x) x;  %#ok<NASGU>
for i = 1:n
    gi = g(i, :);
    for dx = -1:1, for dy = -1:1, for dz = -1:1
        key = (gi(1) + dx) + (gi(2) + dy - 1) * dims(1) + (gi(3) + dz - 1) * dims(1) * dims(2);
        if ~isKey(map, key), continue; end
        span = map(key); cand = order(span(1):span(2));
        cand = cand(cand > i);
        if isempty(cand), continue; end
        close = cand(vecnorm(P(cand, :) - P(i, :), 2, 2) <= minDistance);
        for j = close'
            ri = i; while parent(ri) ~= ri, parent(ri) = parent(parent(ri)); ri = parent(ri); end
            rj = j; while parent(rj) ~= rj, parent(rj) = parent(parent(rj)); rj = parent(rj); end
            if ri ~= rj, parent(rj) = ri; end
        end
    end, end, end
end
root = zeros(n, 1);
for i = 1:n, r = i; while parent(r) ~= r, r = parent(r); end; root(i) = r; end
[ur, ~, comp] = unique(root); sizes = accumarray(comp, 1);
ok = find(sizes >= nr(1) & sizes <= nr(2));
remap = zeros(numel(ur), 1); remap(ok) = 1:numel(ok);
labels = remap(comp); numClusters = numel(ok);
end
